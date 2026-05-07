"""Unified multi-sensor sunrise change-point detection experiment.

The experiment is a single multi-sensor model with three resource
regimes (low / medium / high). Each regime is defined by a numeric
budget ``B`` over the sensing/communication cost ``C_i + T_i``; the
active subset ``S(t)`` is selected dynamically at every timestamp by
the budget policy in :mod:`src.budget` (greedy information-per-cost).
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
    load_sensor_costs,
    regime_budget,
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
        "Low budget. The same multi-sensor network is available at every "
        "timestamp. The dynamic budget policy enforces "
        "sum_{i in S(t)} (C_i + T_i) <= B with B = 1.0 under the "
        "unit-cost approximation, so at most one available sensor is "
        "activated per timestamp. The chosen sensor may vary across "
        "timestamps and days depending on availability and "
        "information-per-cost score."
    ),
    "medium": (
        "Medium budget. The dynamic budget policy enforces "
        "sum_{i in S(t)} (C_i + T_i) <= B with B = 3.0 under the "
        "unit-cost approximation, so up to three available sensors are "
        "activated per timestamp. The active subset may vary over time."
    ),
    "high": (
        "High budget. The dynamic budget policy uses B equal to the "
        "number of valid sensors, so all available sensors may be "
        "activated at every timestamp."
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
              ) -> tuple[str, bool, bool]:
    """Classify a per-day detection outcome."""
    tol = timedelta(minutes=tolerance_minutes)
    if detected is None or not (window_start <= detected <= window_end):
        return "missed_detection", False, True
    if detected < sunrise - tol:
        return "false_alarm", True, False
    if abs(detected - sunrise) <= tol:
        return "on_time", False, False
    return "detected", False, False


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
    n_missed = sum(1 for p in per_day if p["missed_detection"])
    n_false = sum(1 for p in per_day if p["false_alarm"])

    def _maybe(fn, arr):
        return float(fn(arr)) if arr.size else None

    avg_active = [p.get("average_active_sensors_per_timestamp")
                  for p in per_day
                  if p.get("average_active_sensors_per_timestamp") is not None]
    avg_budget = [p.get("average_budget_used") for p in per_day
                  if p.get("average_budget_used") is not None]

    return {
        "valid_days_count": n_days,
        "detected_days_count": int(len(detections)),
        "missed_detection_count": int(n_missed),
        "false_alarm_count": int(n_false),
        "mean_signed_delay_minutes": _maybe(np.mean, delays),
        "median_signed_delay_minutes": _maybe(np.median, delays),
        "mean_absolute_error_minutes": _maybe(np.mean, abs_errors),
        "median_absolute_error_minutes": _maybe(np.median, abs_errors),
        "std_signed_delay_minutes": (
            float(np.std(delays, ddof=1)) if delays.size > 1 else None
        ),
        "min_signed_delay_minutes": _maybe(np.min, delays),
        "max_signed_delay_minutes": _maybe(np.max, delays),
        "mean_active_sensors_per_timestamp": (
            float(np.mean(avg_active)) if avg_active else None),
        "mean_budget_used_per_timestamp": (
            float(np.mean(avg_budget)) if avg_budget else None),
    }


def _load_valid_sensors_and_informativeness() -> tuple[list[str], list[dict]]:
    """Load the full valid sensor set and the empirical D_i ranking."""
    if not paths.PREPROCESSING_SUMMARY_JSON.exists():
        raise FileNotFoundError(
            f"Preprocessing summary missing: "
            f"{paths.PREPROCESSING_SUMMARY_JSON}. Run preprocessing first."
        )
    with open(paths.PREPROCESSING_SUMMARY_JSON, "r", encoding="utf-8") as fh:
        prep = json.load(fh)
    valid_sensors = [str(s) for s in prep.get("valid_sensors") or []]
    info_records: list[dict] = []
    if paths.SENSOR_INFORMATIVENESS_JSON.exists():
        with open(paths.SENSOR_INFORMATIVENESS_JSON, "r",
                  encoding="utf-8") as fh:
            info_records = json.load(fh).get("sensors", [])
    return valid_sensors, info_records


def _build_window_matrix(sensor_df: pd.DataFrame,
                         candidate_sensors: list[str],
                         win_start: pd.Timestamp,
                         win_end: pd.Timestamp,
                         freq: str = "2min"
                         ) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    """Build a sensor-aligned matrix for one daily window."""
    sub = sensor_df[
        (sensor_df["timestamp_utc"] >= win_start)
        & (sensor_df["timestamp_utc"] <= win_end)
        & (sensor_df["station_id"].isin(candidate_sensors))
    ]
    if sub.empty:
        empty = pd.DatetimeIndex([], tz="UTC")
        return empty, {sid: np.array([]) for sid in candidate_sensors}
    grid = pd.DatetimeIndex(
        sorted(sub["timestamp_utc"].unique()), tz="UTC"
    )
    values_by_sensor: dict[str, np.ndarray] = {}
    for sid in candidate_sensors:
        s = sub.loc[sub["station_id"] == sid, ["timestamp_utc", "value"]]
        if s.empty:
            values_by_sensor[sid] = np.full(len(grid), np.nan)
            continue
        series = (s.groupby("timestamp_utc")["value"].mean()
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
                   budget: float | None = None,
                   ) -> dict:
    """Run the unified multi-sensor sunrise CPD experiment.

    The full set of valid sensors is always passed to the detector. The
    detector then consults the dynamic budget policy at every timestamp
    to obtain ``S(t)`` under ``sum_{i in S(t)} (C_i + T_i) <= B``. The
    numeric budget ``B`` for the regime is computed by
    :func:`src.budget.regime_budget` unless overridden via ``budget``.
    """
    regime = regime.lower()
    if regime not in _REGIME_OUTPUTS:
        raise ValueError(f"Unknown regime: {regime!r}")
    if detector_mode != "global_gaussian_llr":
        raise ValueError(
            f"Unsupported detector_mode: {detector_mode!r}. The dynamic "
            "budget policy currently only supports 'global_gaussian_llr'."
        )
    if not paths.SYNCHRONIZED_PARQUET.exists():
        raise FileNotFoundError(
            f"Processed dataset missing: {paths.SYNCHRONIZED_PARQUET}."
        )
    if not paths.SUNRISE_GROUND_TRUTH_JSON.exists():
        raise FileNotFoundError(
            f"Sunrise ground truth missing: {paths.SUNRISE_GROUND_TRUTH_JSON}."
        )

    valid_sensors, _info_records = _load_valid_sensors_and_informativeness()
    if not valid_sensors:
        raise RuntimeError("Preprocessing summary lists no valid sensors.")

    # Cost model: load if a sensor_costs.json exists; otherwise create
    # default unit-cost records and persist them so the JSON is always
    # available alongside the result files.
    sensor_costs: dict[str, SensorCost] = load_sensor_costs()
    missing_costs = [sid for sid in valid_sensors if sid not in sensor_costs]
    if missing_costs:
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
            "Valid sensors without global parameters (excluded by the "
            "detector candidate pool): %s", missing_params,
        )

    informativeness_map = {
        sid: float(candidate_params[sid].get("D_i") or 0.0)
        for sid in candidate_params
    }
    candidate_sensors = sorted(candidate_params.keys())

    # Budget value and policy configuration.
    if budget is None:
        budget = regime_budget(regime, len(candidate_sensors))
    policy_config = BudgetPolicyConfig(
        regime=regime,
        budget=float(budget),
        unit_cost_assumption=all(
            c.cost_source == "unit_cost_assumption"
            for c in sensor_costs.values()
        ),
        score_name="D_i_per_total_cost",
    )
    detector_config = DynamicBudgetLLRConfig(
        threshold=threshold, budget=float(budget),
    )

    sensor_ranking = build_sensor_ranking(
        candidate_sensors, informativeness_map, sensor_costs)

    aggregation_description = (
        "At each timestamp t, the dynamic budget policy selects "
        "S(t) by greedy information-per-cost ranking under the "
        "constraint sum_{i in S(t)} (C_i + T_i) <= B. The detector then "
        "computes the per-sensor Gaussian LLR "
        "llr_{i,t} = ((x_{i,t} - mu_{0,i})^2 - (x_{i,t} - mu_{1,i})^2) "
        "/ (2 sigma_i^2) using fixed global empirical parameters and "
        "averages over the sensors in S(t) that report a finite "
        "reading. The aggregated evidence drives a one-sided CUSUM "
        "S_t = max(0, S_{t-1} + L_t) with no drift; detection occurs "
        "when S_t >= h."
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
            budget=float(budget),
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
        status, false_alarm, missed = _classify(
            detected_utc.to_pydatetime() if detected_utc is not None else None,
            sunrise_utc.to_pydatetime(),
            win_start.to_pydatetime(), win_end.to_pydatetime(),
            tolerance_minutes,
        )

        sel_sum = result.selection_summary or {}
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
            "candidate_sensors": list(sel_sum.get("candidate_sensors")
                                      or candidate_sensors),
            "sensors_ever_selected": list(
                sel_sum.get("sensors_ever_selected") or []),
            "sensor_selection_counts": dict(
                sel_sum.get("sensor_selection_counts") or {}),
            "average_active_sensors_per_timestamp": sel_sum.get(
                "average_active_sensors_per_timestamp"),
            "average_budget_used": sel_sum.get("average_budget_used"),
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
    for p in per_day:
        for sid, c in (p.get("sensor_selection_counts") or {}).items():
            union_counts[sid] = union_counts.get(sid, 0) + int(c)
    sensors_ever_selected = sorted(
        sid for sid, c in union_counts.items() if c > 0
    )
    most_frequent = sorted(
        union_counts.items(), key=lambda kv: kv[1], reverse=True
    )

    payload = {
        "scenario_name": f"{regime}_budget",
        "budget_regime": regime,
        "budget_description": _REGIME_DESCRIPTIONS[regime],
        "budget_value": float(budget),
        "budget_constraint": "sum_{i in S(t)} (C_i + T_i) <= B",
        "cost_model": "unit_cost_assumption",
        "unit_cost_assumption": policy_config.unit_cost_assumption,
        "dynamic_selection": True,
        "selection_policy": (
            "Dynamic information-per-cost selection. At each timestamp t, "
            "the policy ranks the available sensors by D_i / (C_i + T_i) "
            "and greedily selects S(t) under the budget constraint."
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
                for sid, c in most_frequent if c > 0
            ],
        },
        "detector_input_mode": "multi_sensor",
        "detector_mode": detector_mode,
        "detector_name": detector_config.name,
        "aggregation": detector_config.aggregation,
        "evidence_type": detector_config.evidence_type,
        "parameter_source": "global_empirical_sensor_parameters",
        "global_parameter_file": str(
            paths.SENSOR_INFORMATIVENESS_JSON.relative_to(paths.ROOT_DIR)),
        "detector_aggregation_description": aggregation_description,
        "detector_parameters": {
            "evidence_type": detector_config.evidence_type,
            "selection_policy": detector_config.selection_policy,
            "formula": (
                "S(t) = greedy_select_by(D_i/(C_i+T_i)) "
                "subject to sum (C_i+T_i) <= B; "
                "llr_{i,t} = ((x_{i,t}-mu_{0,i})^2 - "
                "(x_{i,t}-mu_{1,i})^2) / (2 sigma_i^2); "
                "L_t = mean_{i in S(t)} llr_{i,t}; "
                "S_t = max(0, S_{t-1} + L_t); "
                "detect when S_t >= h."
            ),
        },
        "threshold": threshold,
        "tolerance_minutes": tolerance_minutes,
        "analysis_window": {
            "pre_window_minutes": pre_window_minutes,
            "post_window_minutes": post_window_minutes,
            "sync_freq": sync_freq,
        },
        "number_of_days": len(per_day),
        "number_of_detected_days": aggregate["detected_days_count"],
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
