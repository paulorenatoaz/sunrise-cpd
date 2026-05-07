"""Unified multi-sensor sunrise change-point detection experiment.

The experiment is one model with three resource regimes (low / medium /
high). The runner always loads the *full* set of valid sensors plus the
per-sensor informativeness ranking, asks the budget policy in
:mod:`src.budget` to pick an active subset, then forwards that subset to
the multi-sensor CUSUM detector in :mod:`src.detector`.
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
from .budget import BudgetSelection, select_sensors_by_budget
from .detector import (
    DetectorConfig,
    GlobalLLRConfig,
    cusum_detect_multi_sensor,
    cusum_detect_multi_sensor_global_params,
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
        "Low budget. The same multi-sensor network is available, but the "
        "sampling policy may activate only one sensor. The budget policy "
        "selects the single most informative sensor (top-1 by D_i) from "
        "the full valid sensor set."
    ),
    "medium": (
        "Medium budget. The sampling policy may activate a small subset "
        "of sensors (top-k by D_i) from the full valid sensor set."
    ),
    "high": (
        "High budget. The sampling policy may activate every valid sensor "
        "from the multi-sensor network."
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
    """Classify a per-day detection outcome.

    Returns ``(status, false_alarm_flag, missed_detection_flag)``.
    """
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
                         selected_sensors: list[str],
                         win_start: pd.Timestamp,
                         win_end: pd.Timestamp,
                         freq: str = "2min"
                         ) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    """Build a sensor-aligned matrix for one daily window.

    The processed long table stores raw (non-grid-aligned) timestamps,
    so we use the *union* of the actual observation timestamps of the
    selected sensors inside the window as the common axis. For
    single-sensor (low) budgets this reduces exactly to that sensor's
    own timestamps. ``freq`` is unused here but kept for forward
    compatibility with grid-resampling strategies.
    """
    sub = sensor_df[
        (sensor_df["timestamp_utc"] >= win_start)
        & (sensor_df["timestamp_utc"] <= win_end)
        & (sensor_df["station_id"].isin(selected_sensors))
    ]
    if sub.empty:
        empty = pd.DatetimeIndex([], tz="UTC")
        return empty, {sid: np.array([]) for sid in selected_sensors}
    grid = pd.DatetimeIndex(
        sorted(sub["timestamp_utc"].unique()), tz="UTC"
    )
    values_by_sensor: dict[str, np.ndarray] = {}
    for sid in selected_sensors:
        s = sub.loc[sub["station_id"] == sid, ["timestamp_utc", "value"]]
        if s.empty:
            values_by_sensor[sid] = np.full(len(grid), np.nan)
            continue
        # Average duplicates if any, then reindex to the union grid.
        series = (s.groupby("timestamp_utc")["value"].mean()
                   .astype(float).sort_index().reindex(grid))
        values_by_sensor[sid] = series.to_numpy(dtype=float)
    return grid, values_by_sensor


# ---------------------------------------------------------------------------
# Unified multi-sensor runner
# ---------------------------------------------------------------------------

def run_experiment(regime: str,
                   k: int = 3,
                   pre_window_minutes: int = 120,
                   post_window_minutes: int = 120,
                   tolerance_minutes: int = 15,
                   threshold: float = 5.0,
                   drift_k: float = 0.5,
                   sync_freq: str = "2min",
                   timezone_name: str = paths.GSB_TIMEZONE,
                   detector_mode: str = "global_gaussian_llr",
                   ) -> dict:
    """Run the unified multi-sensor sunrise CPD experiment.

    The main detector mode (``"global_gaussian_llr"``) uses fixed
    global empirical Gaussian parameters per sensor, loaded from
    :data:`paths.SENSOR_INFORMATIVENESS_JSON`. The legacy mode
    ``"daily_baseline_zscore"`` is kept as a diagnostic.

    Steps:
        1. Load every valid sensor and the empirical D_i ranking.
        2. Apply :func:`select_sensors_by_budget` for the requested
           ``regime`` to obtain the active subset.
        3. For each valid day, build a synchronized per-sensor matrix on
           the ``[sunrise - pre, sunrise + post]`` window.
        4. Run the chosen detector on the active subset.
        5. Classify and aggregate results.

    Returns:
        JSON-serializable payload describing the experiment.
    """
    regime = regime.lower()
    if regime not in _REGIME_OUTPUTS:
        raise ValueError(f"Unknown regime: {regime!r}")
    if detector_mode not in {"global_gaussian_llr", "daily_baseline_zscore"}:
        raise ValueError(f"Unknown detector_mode: {detector_mode!r}")
    if not paths.SYNCHRONIZED_PARQUET.exists():
        raise FileNotFoundError(
            f"Processed dataset missing: {paths.SYNCHRONIZED_PARQUET}."
        )
    if not paths.SUNRISE_GROUND_TRUTH_JSON.exists():
        raise FileNotFoundError(
            f"Sunrise ground truth missing: {paths.SUNRISE_GROUND_TRUTH_JSON}."
        )

    valid_sensors, info_records = _load_valid_sensors_and_informativeness()
    if not valid_sensors:
        raise RuntimeError("Preprocessing summary lists no valid sensors.")

    selection: BudgetSelection = select_sensors_by_budget(
        valid_sensors=valid_sensors,
        sensor_informativeness=info_records,
        regime=regime,
        k=k,
    )
    selected_sensors = selection.selected_sensors
    if not selected_sensors:
        raise RuntimeError("Budget policy selected no sensors.")
    logger.info("Budget regime '%s' selected sensors: %s",
                regime, selected_sensors)

    long_df = pd.read_parquet(paths.SYNCHRONIZED_PARQUET)
    long_df["timestamp_utc"] = pd.to_datetime(long_df["timestamp_utc"],
                                              utc=True)
    long_df["station_id"] = long_df["station_id"].astype(str)
    sensor_df = long_df[long_df["station_id"].isin(selected_sensors)].copy()
    sensor_df = sensor_df.sort_values("timestamp_utc").reset_index(drop=True)

    with open(paths.SUNRISE_GROUND_TRUTH_JSON, "r", encoding="utf-8") as fh:
        gt = json.load(fh)
    sunrise_records = gt.get("records", [])

    # Detector setup. The main mode uses fixed global empirical
    # Gaussian parameters per sensor; the legacy mode uses a same-day
    # pre-sunrise baseline and is kept as a diagnostic.
    if detector_mode == "global_gaussian_llr":
        global_params_all = load_global_sensor_parameters(
            paths.SENSOR_INFORMATIVENESS_JSON
        )
        sensor_params = {
            sid: global_params_all[sid]
            for sid in selected_sensors if sid in global_params_all
        }
        missing_params = [sid for sid in selected_sensors
                          if sid not in sensor_params]
        if not sensor_params:
            raise RuntimeError(
                "No global Gaussian parameters available for the "
                "selected sensors; run 'rank-sensors' first."
            )
        if missing_params:
            logger.warning(
                "Selected sensors without global parameters (skipped by "
                "the detector): %s", missing_params,
            )
        llr_config = GlobalLLRConfig(threshold=threshold)
        detector_name = llr_config.name
        aggregation_name = llr_config.aggregation
        evidence_type = llr_config.evidence_type
        aggregation_description = (
            "At each timestamp, compute the per-sensor Gaussian "
            "log-likelihood ratio "
            "llr_{i,t} = ((x_{i,t} - mu_{0,i})^2 - (x_{i,t} - "
            "mu_{1,i})^2) / (2 sigma_i^2) using fixed global empirical "
            "parameters, then average across active sensors with finite "
            "readings. Apply the one-sided CUSUM "
            "S_t = max(0, S_{t-1} + L_t) and detect when S_t >= h."
        )
        detector_parameters = {
            "evidence_type": evidence_type,
            "formula": (
                "llr_{i,t} = ((x_{i,t} - mu_{0,i})^2 - "
                "(x_{i,t} - mu_{1,i})^2) / (2 sigma_i^2); "
                "L_t = mean_{i in S_t} llr_{i,t}; "
                "S_t = max(0, S_{t-1} + L_t); "
                "detect when S_t >= h."
            ),
        }
        parameter_source = "global_empirical_sensor_parameters"
    else:
        legacy_config = DetectorConfig(threshold=threshold, drift_k=drift_k)
        sensor_params = {}
        detector_name = legacy_config.name
        aggregation_name = legacy_config.aggregation
        evidence_type = "per_day_pre_sunrise_baseline_zscore"
        aggregation_description = (
            "At each timestamp, compute the per-sensor standardized "
            "evidence z_{i,t} = (x_{i,t} - mu_{0,i,daily}) / "
            "sigma_{0,i,daily} from the same-day pre-sunrise baseline, "
            "then average across active sensors with finite readings. "
            "Apply the one-sided Page CUSUM "
            "S_t = max(0, S_{t-1} + Z_t - k) and detect when S_t >= h."
        )
        detector_parameters = {
            "drift_k": legacy_config.drift_k,
            "min_baseline_samples": legacy_config.min_baseline_samples,
            "fallback_sigma": legacy_config.fallback_sigma,
            "evidence_type": evidence_type,
            "formula": (
                "z_{i,t} = (x_{i,t} - mu_{0,i})/sigma_{0,i}; "
                "Z_t = mean_{i in S_t} z_{i,t}; "
                "S_t = max(0, S_{t-1} + Z_t - k); "
                "detect when S_t >= h."
            ),
        }
        parameter_source = "per_day_pre_sunrise_baseline"

    tz_local = ZoneInfo(timezone_name)
    pre_td = timedelta(minutes=pre_window_minutes)
    post_td = timedelta(minutes=post_window_minutes)

    per_day: list[dict] = []
    for rec in sunrise_records:
        sunrise_utc = pd.Timestamp(rec["sunrise_utc"]).tz_convert("UTC")
        win_start = sunrise_utc - pre_td
        win_end = sunrise_utc + post_td
        grid, values_by_sensor = _build_window_matrix(
            sensor_df, selected_sensors, win_start, win_end, freq=sync_freq,
        )

        if detector_mode == "global_gaussian_llr":
            result = cusum_detect_multi_sensor_global_params(
                timestamps=list(grid),
                values_by_sensor=values_by_sensor,
                sensor_params=sensor_params,
                config=llr_config,
            )
        else:
            result = cusum_detect_multi_sensor(
                timestamps=list(grid),
                values_by_sensor=values_by_sensor,
                sunrise_time=sunrise_utc,
                config=legacy_config,
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

        used_sensors = [
            sid for sid, b in result.baselines.items() if b.get("used")
        ]
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
            "active_sensors": list(selected_sensors),
            "used_sensors": used_sensors,
            "grid_points": int(result.n_observations),
            "statistic_at_detection": result.statistic_at_detection,
            "notes": result.notes,
        }
        if detector_mode == "global_gaussian_llr":
            per_day_entry["sensor_parameters_used"] = result.baselines
        else:
            per_day_entry["baselines"] = result.baselines
        per_day.append(per_day_entry)


    aggregate = _aggregate_metrics(per_day)
    payload = {
        "scenario_name": f"{regime}_budget",
        "budget_regime": regime,
        "budget_description": _REGIME_DESCRIPTIONS[regime],
        "selection_policy": (
            "Rank candidate sensors by D_i (Gaussian divergence at sunrise) "
            "and pick the top entries allowed by the budget. Costs C_i and "
            "T_i are not available, so unit cost is assumed; this is "
            "equivalent to ranking by D_i / (C_i + T_i) with C_i + T_i = 1."
        ),
        "all_valid_sensors": list(valid_sensors),
        "selected_sensors": list(selected_sensors),
        "selection_reason": selection.selection_reason,
        "sensor_ranking": selection.ranking_used,
        "unit_cost_assumption": selection.unit_cost_assumption,
        "k": selection.k,
        "detector_input_mode": "multi_sensor",
        "detector_mode": detector_mode,
        "detector_name": detector_name,
        "aggregation": aggregation_name,
        "detector_aggregation": aggregation_name,
        "evidence_type": evidence_type,
        "parameter_source": parameter_source,
        "global_parameter_file": (
            str(paths.SENSOR_INFORMATIVENESS_JSON.relative_to(paths.ROOT_DIR))
            if detector_mode == "global_gaussian_llr" else None
        ),
        "global_sensor_parameters_used": (
            {sid: sensor_params[sid] for sid in sensor_params}
            if detector_mode == "global_gaussian_llr" else None
        ),
        "detector_aggregation_description": aggregation_description,
        "detector_parameters": detector_parameters,
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


# Backwards-compatible thin wrapper used by older imports / tests.
def run_low_budget(**kwargs) -> dict:
    """Run the unified experiment with the low-budget regime."""
    kwargs.pop("sensor_id", None)  # silently ignore: chosen by policy now
    return run_experiment(regime="low", **kwargs)


def write_low_budget_results(payload: dict,
                             out_path: Path = LOW_BUDGET_RESULTS_JSON) -> Path:
    return write_results(payload, out_path)
