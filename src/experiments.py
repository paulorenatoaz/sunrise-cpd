"""Unified multi-sensor sunrise change-point detection experiment.

The experiment is a single multi-sensor model with three resource
regimes (low / medium / high). Each regime is defined by *two*
explicit numeric budgets:

    sum_{i in S(t)} C_i  <=  sensing_budget B
    sum_{i in S(t)} T_i  <=  transmission_budget C

The active subset ``S(t)`` is selected dynamically at every timestamp
by the two-budget greedy policy in :mod:`src.budget`. The first
selected sensor at each timestamp is the local/reference sensor and
pays no transmission cost; the rest are cooperative sensors and pay
their nominal transmission cost.

The detector (:func:`src.detector.cusum_detect_dynamic_budget`)
accumulates the per-sensor Gaussian log-likelihood ratio averaged over
``S(t)`` in a one-sided CUSUM. Per-sensor Gaussian parameters are
fixed global empirical estimates loaded from
:data:`src.paths.SENSOR_INFORMATIVENESS_JSON`.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import paths
from .budget import (
    BudgetPolicyConfig,
    SensorCost,
    build_sensor_ranking,
    default_sensor_costs,
    load_sensor_costs,
    regime_budgets,
    write_sensor_costs,
)
from .detector import (
    DynamicBudgetLLRConfig,
    cusum_detect_dynamic_budget,
)
from .informativeness import load_global_sensor_parameters

logger = logging.getLogger(__name__)


LOW_BUDGET_RESULTS_JSON = paths.JSON_DIR / "sunrise_low_budget_results.json"
MEDIUM_BUDGET_RESULTS_JSON = (
    paths.JSON_DIR / "sunrise_medium_budget_results.json"
)
HIGH_BUDGET_RESULTS_JSON = paths.JSON_DIR / "sunrise_high_budget_results.json"

_REGIME_OUTPUTS = {
    "low": LOW_BUDGET_RESULTS_JSON,
    "medium": MEDIUM_BUDGET_RESULTS_JSON,
    "high": HIGH_BUDGET_RESULTS_JSON,
}

_REGIME_DESCRIPTIONS = {
    "low": (
        "Low budget. The dynamic two-budget policy enforces "
        "sensing_budget B = 1.0 and transmission_budget C = 0.0, so at "
        "most one sensor (the local/reference sensor) can be selected "
        "at each timestamp; no cooperative sensors are allowed."
    ),
    "medium": (
        "Medium budget. The dynamic two-budget policy enforces "
        "sensing_budget B = 3.0 and transmission_budget C = 2.0, so up "
        "to three sensors can be selected at each timestamp: one "
        "local/reference sensor plus up to two cooperative sensors."
    ),
    "high": (
        "High budget. The dynamic two-budget policy uses sensing_budget "
        "B equal to the number of valid sensors and transmission_budget "
        "C equal to the number of valid sensors minus one, so all "
        "available sensors may be selected at every timestamp."
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify(detected: datetime | None,
              sunrise: datetime,
              window_start: datetime,
              window_end: datetime,
              tolerance_minutes: int
              ) -> tuple[str, bool, bool, bool]:
    """Classify a per-day detection outcome.

    Returns ``(status, on_time_flag, late_flag, false_alarm_flag,
    missed_flag)`` collapsed as ``(status, false_alarm, missed, late)``
    for storage convenience.
    """
    tol = timedelta(minutes=tolerance_minutes)
    if detected is None or not (window_start <= detected <= window_end):
        return "missed_detection", False, True, False
    if detected < sunrise - tol:
        return "false_alarm", True, False, False
    if abs(detected - sunrise) <= tol:
        return "on_time", False, False, False
    # detected > sunrise + tol
    return "late_detection", False, False, True


def _aggregate_metrics(per_day: list[dict]) -> dict:
    """Compute aggregate metrics across per-day results."""
    n_days = len(per_day)
    detections = [p for p in per_day
                  if p["detected_change_point_utc"] is not None
                  and not p["missed_detection"]]
    delays = np.array([p["signed_delay_minutes"] for p in detections
                       if p["signed_delay_minutes"] is not None], dtype=float)
    abs_errors = np.array([p["absolute_error_minutes"] for p in detections
                           if p["absolute_error_minutes"] is not None],
                          dtype=float)
    positive_delays = (np.maximum(delays, 0.0)
                       if delays.size else np.array([], dtype=float))
    n_missed = sum(1 for p in per_day if p["missed_detection"])
    n_false = sum(1 for p in per_day if p["false_alarm"])
    n_late = sum(1 for p in per_day
                 if p.get("status") == "late_detection")
    n_on_time = sum(1 for p in per_day if p.get("status") == "on_time")

    def _maybe(fn, arr):
        return float(fn(arr)) if arr.size else None

    avg_active = [p.get("average_active_sensors_per_timestamp")
                  for p in per_day
                  if p.get("average_active_sensors_per_timestamp") is not None]
    avg_avail = [p.get("average_available_sensors_per_timestamp")
                 for p in per_day
                 if p.get("average_available_sensors_per_timestamp") is not None]
    avg_sense = [p.get("average_sensing_cost_used") for p in per_day
                 if p.get("average_sensing_cost_used") is not None]
    avg_trans = [p.get("average_transmission_cost_used") for p in per_day
                 if p.get("average_transmission_cost_used") is not None]
    frac2 = [p.get("fraction_timestamps_at_least_2_available")
             for p in per_day
             if p.get("fraction_timestamps_at_least_2_available") is not None]
    frac3 = [p.get("fraction_timestamps_at_least_3_available")
             for p in per_day
             if p.get("fraction_timestamps_at_least_3_available") is not None]

    return {
        "valid_days_count": n_days,
        "detected_days_count": int(len(detections)),
        "on_time_count": int(n_on_time),
        "late_detection_count": int(n_late),
        "missed_detection_count": int(n_missed),
        "false_alarm_count": int(n_false),
        "out_of_tolerance_count": int(n_late + n_false),
        "on_time_rate": (float(n_on_time) / float(n_days)
                         if n_days else None),
        "mean_signed_delay_minutes": _maybe(np.mean, delays),
        "median_signed_delay_minutes": _maybe(np.median, delays),
        "mean_absolute_error_minutes": _maybe(np.mean, abs_errors),
        "median_absolute_error_minutes": _maybe(np.median, abs_errors),
        "std_signed_delay_minutes": (
            float(np.std(delays, ddof=1)) if delays.size > 1 else None
        ),
        "min_signed_delay_minutes": _maybe(np.min, delays),
        "max_signed_delay_minutes": _maybe(np.max, delays),
        "ADD_like_positive_delay_minutes": _maybe(np.mean, positive_delays),
        "median_positive_delay_minutes": _maybe(np.median, positive_delays),
        "mean_available_sensors_per_timestamp": (
            float(np.mean(avg_avail)) if avg_avail else None),
        "mean_active_sensors_per_timestamp": (
            float(np.mean(avg_active)) if avg_active else None),
        "mean_sensing_cost_used": (
            float(np.mean(avg_sense)) if avg_sense else None),
        "mean_transmission_cost_used": (
            float(np.mean(avg_trans)) if avg_trans else None),
        "mean_fraction_at_least_2_available": (
            float(np.mean(frac2)) if frac2 else None),
        "mean_fraction_at_least_3_available": (
            float(np.mean(frac3)) if frac3 else None),
    }


def _load_valid_sensors() -> list[str]:
    """Load the full valid sensor set from the preprocessing summary."""
    if not paths.PREPROCESSING_SUMMARY_JSON.exists():
        raise FileNotFoundError(
            f"Preprocessing summary missing: "
            f"{paths.PREPROCESSING_SUMMARY_JSON}. Run preprocessing first."
        )
    with open(paths.PREPROCESSING_SUMMARY_JSON, "r", encoding="utf-8") as fh:
        prep = json.load(fh)
    return [str(s) for s in prep.get("valid_sensors") or []]


def _build_window_matrix(sensor_df: pd.DataFrame,
                         candidate_sensors: list[str],
                         win_start: pd.Timestamp,
                         win_end: pd.Timestamp,
                         freq: str = "2min",
                         ) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    """Build a sensor-aligned matrix on a regular common time grid.

    Sensors are aligned to a regular ``freq``-spaced grid (default
    2 minutes) by averaging all observations that fall inside each
    grid bin (``floor`` rule). Bins with no observation are left as
    NaN. Using a common grid lets multiple sensors be active at the
    same timestamp.
    """
    sub = sensor_df[
        (sensor_df["timestamp_utc"] >= win_start)
        & (sensor_df["timestamp_utc"] <= win_end)
        & (sensor_df["station_id"].isin(candidate_sensors))
    ]
    grid = pd.date_range(
        start=win_start.floor(freq),
        end=win_end.ceil(freq),
        freq=freq, tz="UTC", inclusive="both",
    )
    if sub.empty:
        return grid, {sid: np.full(len(grid), np.nan)
                      for sid in candidate_sensors}
    sub = sub.copy()
    sub["bin"] = sub["timestamp_utc"].dt.floor(freq)
    values_by_sensor: dict[str, np.ndarray] = {}
    for sid in candidate_sensors:
        s = sub.loc[sub["station_id"] == sid, ["bin", "value"]]
        if s.empty:
            values_by_sensor[sid] = np.full(len(grid), np.nan)
            continue
        series = (s.groupby("bin")["value"].mean()
                   .astype(float).sort_index().reindex(grid))
        values_by_sensor[sid] = series.to_numpy(dtype=float)
    return grid, values_by_sensor


# ---------------------------------------------------------------------------
# Unified multi-sensor runner
# ---------------------------------------------------------------------------

def run_experiment(regime: str,
                   k: int = 3,             # accepted for backward compat
                   pre_window_minutes: int = 120,
                   post_window_minutes: int = 120,
                   tolerance_minutes: int = 15,
                   threshold: float = 5.0,
                   drift_k: float = 0.0,   # accepted for backward compat
                   sync_freq: str = "2min",
                   timezone_name: str = paths.GSB_TIMEZONE,
                   detector_mode: str = "global_gaussian_llr",
                   sensing_budget: float | None = None,
                   transmission_budget: float | None = None,
                   ) -> dict:
    """Run the unified multi-sensor sunrise CPD experiment.

    Both budgets default to the values produced by
    :func:`src.budget.regime_budgets` for the given regime.
    """
    regime = regime.lower()
    if regime not in _REGIME_OUTPUTS:
        raise ValueError(f"Unknown regime: {regime!r}")
    if detector_mode != "global_gaussian_llr":
        raise ValueError(
            f"Unsupported detector_mode: {detector_mode!r}. The dynamic "
            "two-budget policy only supports 'global_gaussian_llr'."
        )
    if not paths.SYNCHRONIZED_PARQUET.exists():
        raise FileNotFoundError(
            f"Processed dataset missing: {paths.SYNCHRONIZED_PARQUET}."
        )
    if not paths.SUNRISE_GROUND_TRUTH_JSON.exists():
        raise FileNotFoundError(
            f"Sunrise ground truth missing: {paths.SUNRISE_GROUND_TRUTH_JSON}."
        )

    valid_sensors = _load_valid_sensors()
    if not valid_sensors:
        raise RuntimeError("Preprocessing summary lists no valid sensors.")

    # Cost model: load if a sensor_costs.json exists; otherwise create
    # default homogeneous-cost records and persist them.
    sensor_costs: dict[str, SensorCost] = load_sensor_costs()
    missing_costs = [sid for sid in valid_sensors if sid not in sensor_costs]
    if missing_costs or not sensor_costs:
        for sid in missing_costs:
            sensor_costs[sid] = SensorCost(sensor_id=sid)
        write_sensor_costs({sid: sensor_costs[sid] for sid in valid_sensors})

    # Global empirical Gaussian parameters per sensor.
    global_params_all = load_global_sensor_parameters(
        paths.SENSOR_INFORMATIVENESS_JSON
    )
    candidate_params = {
        sid: global_params_all[sid]
        for sid in valid_sensors if sid in global_params_all
    }
    missing_params = [sid for sid in valid_sensors
                      if sid not in candidate_params]
    if not candidate_params:
        raise RuntimeError(
            "No global Gaussian parameters available for the valid "
            "sensors; run 'rank-sensors' first."
        )
    if missing_params:
        logger.warning(
            "Valid sensors without global parameters (excluded): %s",
            missing_params,
        )

    informativeness_map = {
        sid: float(candidate_params[sid].get("D_i") or 0.0)
        for sid in candidate_params
    }
    candidate_sensors = sorted(candidate_params.keys())

    # Two-budget configuration.
    default_b, default_c = regime_budgets(regime, len(candidate_sensors))
    if sensing_budget is None:
        sensing_budget = default_b
    if transmission_budget is None:
        transmission_budget = default_c
    policy_config = BudgetPolicyConfig(
        regime=regime,
        sensing_budget=float(sensing_budget),
        transmission_budget=float(transmission_budget),
        unit_cost_assumption=False,
        homogeneous_cost_assumption=all(
            c.cost_source == "homogeneous_synthetic_cost_assumption"
            for c in sensor_costs.values()
        ),
        score_name="D_i",
    )
    detector_config = DynamicBudgetLLRConfig(
        threshold=threshold,
        sensing_budget=float(sensing_budget),
        transmission_budget=float(transmission_budget),
    )

    sensor_ranking = build_sensor_ranking(
        candidate_sensors, informativeness_map, sensor_costs)

    aggregation_description = (
        "At each timestamp t, the dynamic two-budget policy ranks the "
        "available sensors by D_i and greedily selects the local/"
        "reference sensor (effective transmission cost 0) plus any "
        "cooperative sensors (full transmission cost) that fit within "
        "sensing_budget B and transmission_budget C. The detector "
        "computes the per-sensor Gaussian LLR "
        "llr_{i,t} = ((x_{i,t} - mu_{0,i})^2 - (x_{i,t} - mu_{1,i})^2) "
        "/ (2 sigma_i^2) using fixed global empirical parameters and "
        "averages over the sensors in S(t) that report a finite "
        "reading. The aggregated evidence drives a one-sided CUSUM "
        "S_t = max(0, S_{t-1} + L_t); detection occurs when S_t >= h."
    )

    # Load processed sensor data for the full candidate sensor set.
    long_df = pd.read_parquet(paths.SYNCHRONIZED_PARQUET)
    long_df["timestamp_utc"] = pd.to_datetime(long_df["timestamp_utc"],
                                              utc=True)
    long_df["station_id"] = long_df["station_id"].astype(str)
    sensor_df = long_df[long_df["station_id"].isin(candidate_sensors)].copy()
    sensor_df = sensor_df.sort_values("timestamp_utc").reset_index(drop=True)

    with open(paths.SUNRISE_GROUND_TRUTH_JSON, "r", encoding="utf-8") as fh:
        gt = json.load(fh)
    sunrise_records = gt.get("records", [])

    tz_local = ZoneInfo(timezone_name)
    pre_td = timedelta(minutes=pre_window_minutes)
    post_td = timedelta(minutes=post_window_minutes)

    per_day: list[dict] = []
    for rec in sunrise_records:
        sunrise_utc = pd.Timestamp(rec["sunrise_utc"]).tz_convert("UTC")
        win_start = sunrise_utc - pre_td
        win_end = sunrise_utc + post_td
        grid, values_by_sensor = _build_window_matrix(
            sensor_df, candidate_sensors, win_start, win_end, freq=sync_freq,
        )

        result = cusum_detect_dynamic_budget(
            timestamps=list(grid),
            values_by_sensor=values_by_sensor,
            sensor_params=candidate_params,
            sensor_costs=sensor_costs,
            sensing_budget=float(sensing_budget),
            transmission_budget=float(transmission_budget),
            config=detector_config,
        )

        detected_utc = (
            pd.Timestamp(result.detected_time).tz_convert("UTC")
            if result.detected_time is not None else None
        )
        delay_minutes = (
            (detected_utc - sunrise_utc).total_seconds() / 60.0
            if detected_utc is not None else None
        )
        abs_err = abs(delay_minutes) if delay_minutes is not None else None
        status, false_alarm, missed, late = _classify(
            detected_utc.to_pydatetime() if detected_utc is not None else None,
            sunrise_utc.to_pydatetime(),
            win_start.to_pydatetime(), win_end.to_pydatetime(),
            tolerance_minutes,
        )

        sel_sum = result.selection_summary or {}
        local_counts = dict(sel_sum.get("local_sensor_counts") or {})
        coop_counts = dict(sel_sum.get("cooperative_sensor_counts") or {})
        most_local = max(local_counts.items(), key=lambda kv: kv[1],
                         default=(None, 0))
        per_day_entry = {
            "date": rec["date"],
            "true_change_point_utc": sunrise_utc.isoformat(),
            "true_change_point_local": (
                sunrise_utc.tz_convert(tz_local).isoformat()),
            "detected_change_point_utc": (
                detected_utc.isoformat() if detected_utc is not None else None),
            "detected_change_point_local": (
                detected_utc.tz_convert(tz_local).isoformat()
                if detected_utc is not None else None),
            "signed_delay_minutes": delay_minutes,
            "absolute_error_minutes": abs_err,
            "status": status,
            "false_alarm": false_alarm,
            "missed_detection": missed,
            "late_detection": late,
            "candidate_sensors": list(sel_sum.get("candidate_sensors")
                                      or candidate_sensors),
            "sensors_ever_selected": list(
                sel_sum.get("sensors_ever_selected") or []),
            "sensor_selection_counts": dict(
                sel_sum.get("sensor_selection_counts") or {}),
            "local_sensor_counts": local_counts,
            "cooperative_sensor_counts": coop_counts,
            "most_frequent_local_sensor": most_local[0],
            "most_frequent_local_sensor_count": int(most_local[1]),
            "average_available_sensors_per_timestamp": sel_sum.get(
                "average_available_sensors_per_timestamp"),
            "average_active_sensors_per_timestamp": sel_sum.get(
                "average_active_sensors_per_timestamp"),
            "average_sensing_cost_used": sel_sum.get(
                "average_sensing_cost_used"),
            "average_transmission_cost_used": sel_sum.get(
                "average_transmission_cost_used"),
            "fraction_timestamps_at_least_2_available": sel_sum.get(
                "fraction_timestamps_at_least_2_available"),
            "fraction_timestamps_at_least_3_available": sel_sum.get(
                "fraction_timestamps_at_least_3_available"),
            "timestamps_without_selection": sel_sum.get(
                "timestamps_without_selection"),
            "evaluated_timestamps": sel_sum.get("evaluated_timestamps"),
            "grid_points": int(result.n_observations),
            "statistic_at_detection": result.statistic_at_detection,
            "notes": result.notes,
        }
        per_day.append(per_day_entry)

    aggregate = _aggregate_metrics(per_day)

    # Cross-day selection frequency summary.
    union_counts: dict[str, int] = {sid: 0 for sid in candidate_sensors}
    union_local: dict[str, int] = {sid: 0 for sid in candidate_sensors}
    union_coop: dict[str, int] = {sid: 0 for sid in candidate_sensors}
    for p in per_day:
        for sid, c in (p.get("sensor_selection_counts") or {}).items():
            union_counts[sid] = union_counts.get(sid, 0) + int(c)
        for sid, c in (p.get("local_sensor_counts") or {}).items():
            union_local[sid] = union_local.get(sid, 0) + int(c)
        for sid, c in (p.get("cooperative_sensor_counts") or {}).items():
            union_coop[sid] = union_coop.get(sid, 0) + int(c)

    sensors_ever_selected = sorted(
        sid for sid, c in union_counts.items() if c > 0)
    most_freq_total = sorted(union_counts.items(),
                             key=lambda kv: kv[1], reverse=True)
    most_freq_local = sorted(union_local.items(),
                             key=lambda kv: kv[1], reverse=True)
    most_freq_coop = sorted(union_coop.items(),
                            key=lambda kv: kv[1], reverse=True)

    payload = {
        "scenario_name": f"{regime}_budget",
        "budget_regime": regime,
        "budget_description": _REGIME_DESCRIPTIONS[regime],
        "sensing_budget": float(sensing_budget),
        "transmission_budget": float(transmission_budget),
        "budget_constraints": [
            "sum_{i in S(t)} C_i <= sensing_budget B",
            "sum_{i in S(t)} T_i <= transmission_budget C",
        ],
        "cost_model": "homogeneous_synthetic_cost_assumption",
        "homogeneous_cost_assumption": (
            policy_config.homogeneous_cost_assumption),
        "unit_cost_assumption": False,
        "dynamic_selection": True,
        "selection_policy": (
            "Two-budget dynamic selection. At each timestamp t, the "
            "policy ranks the available sensors by D_i and greedily "
            "selects the local/reference sensor (effective transmission "
            "cost 0) plus any cooperative sensors (full transmission "
            "cost) under the constraints "
            "sum_{i in S(t)} C_i <= B and "
            "sum_{i in S(t)} T_i <= C."
        ),
        "selection_policy_config": policy_config.to_dict(),
        "full_candidate_sensors": list(candidate_sensors),
        "sensor_ranking": sensor_ranking,
        "sensor_costs": [sensor_costs[sid].to_dict()
                         for sid in candidate_sensors],
        "selected_sensors_summary": {
            "sensors_ever_selected": sensors_ever_selected,
            "most_frequently_selected_sensors": [
                {"sensor_id": sid, "selection_count": int(c)}
                for sid, c in most_freq_total if c > 0
            ],
            "most_frequently_selected_local_sensors": [
                {"sensor_id": sid, "selection_count": int(c)}
                for sid, c in most_freq_local if c > 0
            ],
            "most_frequently_selected_cooperative_sensors": [
                {"sensor_id": sid, "selection_count": int(c)}
                for sid, c in most_freq_coop if c > 0
            ],
        },
        "detector_input_mode": "multi_sensor",
        "detector_mode": detector_mode,
        "detector_name": detector_config.name,
        "aggregation": detector_config.aggregation,
        "evidence_type": detector_config.evidence_type,
        "parameter_source": "global_empirical_sensor_parameters",
        "global_parameter_source": "global_empirical_sensor_parameters",
        "global_parameter_file": str(
            paths.SENSOR_INFORMATIVENESS_JSON.relative_to(paths.ROOT_DIR)),
        "detector_aggregation_description": aggregation_description,
        "detector_parameters": {
            "evidence_type": detector_config.evidence_type,
            "selection_policy": detector_config.selection_policy,
            "formula": (
                "S(t) = greedy_select_by(D_i) subject to "
                "sum C_i <= B and sum T_i <= C, with the first "
                "selected sensor as local/reference (effective "
                "transmission cost = 0); "
                "llr_{i,t} = ((x_{i,t}-mu_{0,i})^2 - "
                "(x_{i,t}-mu_{1,i})^2) / (2 sigma_i^2); "
                "L_t = mean_{i in S(t)} llr_{i,t}; "
                "S_t = max(0, S_{t-1} + L_t); detect when S_t >= h."
            ),
        },
        "threshold": threshold,
        "tolerance_minutes": tolerance_minutes,
        "analysis_window": {
            "pre_window_minutes": pre_window_minutes,
            "post_window_minutes": post_window_minutes,
            "sync_freq": sync_freq,
            "alignment_method": (
                "regular common time grid with 'floor' binning by "
                "sync_freq; bins without observations are NaN"),
        },
        "number_of_days": len(per_day),
        "number_of_detected_days": aggregate["detected_days_count"],
        "number_of_on_time_detections": aggregate["on_time_count"],
        "number_of_late_detections": aggregate["late_detection_count"],
        "number_of_missed_detections": aggregate["missed_detection_count"],
        "number_of_false_alarms": aggregate["false_alarm_count"],
        "aggregate_metrics": aggregate,
        "per_day_results": per_day,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return payload


def write_results(payload: dict, out_path: Path | None = None) -> Path:
    """Persist an experiment payload to JSON."""
    if out_path is None:
        out_path = _REGIME_OUTPUTS[payload["budget_regime"]]
    paths.ensure_dirs()
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
    logger.info("Results written to %s", out_path)
    return out_path


# Backwards-compatible thin wrappers used by older imports / tests.
def run_low_budget(**kwargs) -> dict:
    """Run the unified experiment with the low-budget regime."""
    kwargs.pop("sensor_id", None)  # silently ignore: chosen by policy now
    return run_experiment(regime="low", **kwargs)


def write_low_budget_results(payload: dict,
                             out_path: Path = LOW_BUDGET_RESULTS_JSON) -> Path:
    return write_results(payload, out_path)
