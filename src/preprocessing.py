"""Preprocessing of the selected SensorScope deployment.

Loads station files into a unified dataframe, parses the timestamp,
normalizes timezone, filters invalid measurements of the chosen
light-related variable, and synchronizes readings into a sensor-by-time
matrix.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from .inspection import (
    ColumnDefinition,
    DeploymentInspection,
)

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """Lightweight container for preprocessing outputs."""

    long_df: pd.DataFrame
    matrix: pd.DataFrame
    summary: dict


# ---------------------------------------------------------------------------
# Column resolution helpers
# ---------------------------------------------------------------------------

_TS_NAME_RE = re.compile(r"time\s*since.*epoch|unix|epoch", re.IGNORECASE)


def find_timestamp_column_index(defs: list[ColumnDefinition]) -> int | None:
    """Return the 0-based index of the most likely timestamp column.

    Prefers a "time since the epoch [s]" style column. Falls back to the
    first column whose name contains ``time`` or ``date``.
    """
    for cd in defs:
        if _TS_NAME_RE.search(cd.name):
            return cd.index - 1
    for cd in defs:
        low = cd.name.lower()
        if "time" in low or "date" in low or "timestamp" in low:
            return cd.index - 1
    return None


def find_station_column_index(defs: list[ColumnDefinition]) -> int | None:
    """Return the 0-based index of the station/sensor identifier column."""
    for cd in defs:
        low = cd.name.lower()
        if low.strip() in ("station id", "station_id", "node id", "node_id",
                           "sensor id", "sensor_id"):
            return cd.index - 1
    for cd in defs:
        low = cd.name.lower()
        if "station" in low or "node" in low:
            return cd.index - 1
    return None


def find_variable_column_index(defs: list[ColumnDefinition],
                               variable_name: str) -> int | None:
    """Return the 0-based index of the named variable."""
    for cd in defs:
        if cd.name == variable_name:
            return cd.index - 1
    # Loose match.
    target = variable_name.lower().strip()
    for cd in defs:
        if cd.name.lower().strip() == target:
            return cd.index - 1
    return None


def find_year_month_day_indices(defs: list[ColumnDefinition]
                                ) -> tuple[int | None, ...]:
    """Return 0-based indices for (year, month, day, hour, minute, second).

    Any element may be None when the corresponding column is not declared.
    """
    fields = {"year": None, "month": None, "day": None,
              "hour": None, "minute": None, "second": None}
    for cd in defs:
        low = cd.name.lower().strip()
        if low in fields and fields[low] is None:
            fields[low] = cd.index - 1
    return tuple(fields[k] for k in
                 ("year", "month", "day", "hour", "minute", "second"))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_station_file(path: Path, n_columns_expected: int | None = None
                      ) -> pd.DataFrame:
    """Load a single SensorScope station file into a numeric dataframe.

    Lines are whitespace-separated. Non-parseable values become NaN.

    Args:
        path: File path.
        n_columns_expected: Expected number of columns. When provided, lines
            with a different column count are dropped.

    Returns:
        Dataframe with numeric columns ``c0, c1, ...``.
    """
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        engine="python",
        on_bad_lines="skip",
        comment="#",
        dtype=str,
    )
    if n_columns_expected is not None and df.shape[1] != n_columns_expected:
        # Try to keep only rows with the expected number of fields.
        if df.shape[1] > n_columns_expected:
            df = df.iloc[:, :n_columns_expected]
        else:
            for k in range(df.shape[1], n_columns_expected):
                df[k] = np.nan
    df.columns = [f"c{i}" for i in range(df.shape[1])]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_deployment_long(insp: DeploymentInspection,
                         variable_name: str,
                         deployment_timezone: str
                         ) -> tuple[pd.DataFrame, dict]:
    """Load all station files into a long-format dataframe.

    The output dataframe has columns:
    ``timestamp_utc, timestamp_local, date, station_id, value``.

    Args:
        insp: Deployment inspection result.
        variable_name: Name of the selected light-related variable (must be
            present in ``insp.column_definitions``).
        deployment_timezone: IANA timezone name for the deployment.

    Returns:
        Tuple ``(long_df, diagnostics_dict)``.
    """
    defs = insp.column_definitions
    if not defs:
        raise ValueError("No column definitions available for deployment.")
    ts_idx = find_timestamp_column_index(defs)
    sta_idx = find_station_column_index(defs)
    var_idx = find_variable_column_index(defs, variable_name)
    ymd_idx = find_year_month_day_indices(defs)

    if var_idx is None:
        raise ValueError(
            f"Variable '{variable_name}' not found in column definitions.")

    n_expected = len(defs)
    frames: list[pd.DataFrame] = []
    raw_row_count = 0
    per_station_raw: dict[str, int] = {}

    for station_id, fp in sorted(insp.station_files.items(),
                                 key=lambda kv: int(kv[0])):
        try:
            df = load_station_file(fp, n_columns_expected=n_expected)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load %s: %s", fp, exc)
            continue
        if df.empty:
            continue
        raw_row_count += len(df)
        per_station_raw[station_id] = len(df)
        out = pd.DataFrame()
        out["station_id"] = (
            df[f"c{sta_idx}"].astype("Int64").astype(str)
            if sta_idx is not None else station_id
        )
        # Override with the file-derived station id when sta_idx absent
        # to guarantee a usable identifier.
        if sta_idx is None:
            out["station_id"] = station_id
        out["value"] = df[f"c{var_idx}"]
        if ts_idx is not None:
            out["epoch_s"] = df[f"c{ts_idx}"]
        else:
            out["epoch_s"] = np.nan
        # Optional Y/M/D fallback.
        for label, idx in zip(
                ("year", "month", "day", "hour", "minute", "second"),
                ymd_idx):
            if idx is not None:
                out[label] = df[f"c{idx}"]
        frames.append(out)

    if not frames:
        raise RuntimeError("No station data could be loaded.")

    long_df = pd.concat(frames, ignore_index=True)

    # ---- Timestamp construction ----
    ts = pd.Series(pd.NaT, index=long_df.index, dtype="datetime64[ns, UTC]")
    if "epoch_s" in long_df.columns and long_df["epoch_s"].notna().any():
        # Epoch seconds; treat as UTC.
        epoch = pd.to_numeric(long_df["epoch_s"], errors="coerce")
        ts_from_epoch = pd.to_datetime(epoch, unit="s", utc=True,
                                       errors="coerce")
        ts = ts_from_epoch
    needs_fallback = ts.isna()
    if needs_fallback.any() and "year" in long_df.columns:
        ymd_cols = [c for c in
                    ("year", "month", "day", "hour", "minute", "second")
                    if c in long_df.columns]
        if {"year", "month", "day"}.issubset(set(ymd_cols)):
            sub = long_df.loc[needs_fallback, ymd_cols].copy()
            for c in ("hour", "minute", "second"):
                if c not in sub.columns:
                    sub[c] = 0
            built = pd.to_datetime(
                sub.rename(columns={"year": "year", "month": "month",
                                    "day": "day", "hour": "hour",
                                    "minute": "minute", "second": "second"}),
                errors="coerce", utc=True,
            )
            ts.loc[needs_fallback] = built

    long_df["timestamp_utc"] = ts
    long_df = long_df.dropna(subset=["timestamp_utc", "value"])
    long_df["timestamp_local"] = long_df["timestamp_utc"].dt.tz_convert(
        deployment_timezone)
    long_df["date"] = long_df["timestamp_local"].dt.date

    long_df = long_df[["timestamp_utc", "timestamp_local", "date",
                       "station_id", "value"]]

    diagnostics = {
        "raw_row_count": int(raw_row_count),
        "raw_rows_per_station": {k: int(v) for k, v in per_station_raw.items()},
        "timestamp_strategy": (
            "epoch_seconds_utc" if ts_idx is not None else "year_month_day_local"
        ),
        "timestamp_column_index": ts_idx,
        "station_column_index": sta_idx,
        "variable_column_index": var_idx,
    }
    return long_df, diagnostics


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_values(long_df: pd.DataFrame, lower: float = -50.0,
                 upper: float = 5000.0) -> tuple[pd.DataFrame, dict]:
    """Filter physically implausible values for a generic radiation reading.

    Values outside ``[lower, upper]`` are dropped. The defaults are chosen
    to be permissive enough for any radiation/light unit (W/m^2, lux, mV)
    while excluding obvious sensor faults.

    Args:
        long_df: Long-format dataframe with a ``value`` column.
        lower: Inclusive lower bound.
        upper: Inclusive upper bound.

    Returns:
        Tuple ``(filtered_df, exclusions_summary)``.
    """
    n_before = len(long_df)
    mask_finite = np.isfinite(long_df["value"].to_numpy())
    mask_range = (long_df["value"] >= lower) & (long_df["value"] <= upper)
    keep = mask_finite & mask_range.to_numpy()
    out = long_df.loc[keep].copy()
    n_after = len(out)
    return out, {
        "filter_lower_bound": lower,
        "filter_upper_bound": upper,
        "rows_before": int(n_before),
        "rows_after": int(n_after),
        "rows_dropped": int(n_before - n_after),
    }


def estimate_sampling_interval_seconds(long_df: pd.DataFrame) -> float | None:
    """Return the median inter-sample interval in seconds across stations."""
    if long_df.empty:
        return None
    diffs = []
    for _sid, g in long_df.groupby("station_id"):
        ts = g["timestamp_utc"].sort_values().to_numpy()
        if len(ts) > 1:
            d = np.diff(ts).astype("timedelta64[s]").astype(float)
            d = d[(d > 0) & (d < 3600 * 6)]
            if len(d):
                diffs.append(np.median(d))
    if not diffs:
        return None
    return float(np.median(diffs))


def synchronize_matrix(long_df: pd.DataFrame,
                       freq: str = "2min") -> pd.DataFrame:
    """Build a time-by-sensor matrix at a fixed sampling cadence.

    The cadence defaults to 2 minutes which is a typical SensorScope rate.
    Values are aggregated by mean within each bin.

    Args:
        long_df: Long-format dataframe.
        freq: Pandas offset string for the resampling cadence.

    Returns:
        Dataframe indexed by UTC timestamp with one column per station.
    """
    if long_df.empty:
        return pd.DataFrame()
    df = long_df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.set_index("timestamp_utc")
    pivot = (
        df.groupby([pd.Grouper(freq=freq), "station_id"])["value"]
        .mean()
        .unstack("station_id")
        .sort_index()
    )
    return pivot


def select_valid_sensors(matrix: pd.DataFrame,
                         max_missing_rate: float = 0.5
                         ) -> tuple[list[str], dict]:
    """Filter sensors by missing-data rate.

    Args:
        matrix: Sensor matrix from :func:`synchronize_matrix`.
        max_missing_rate: Maximum allowed fraction of missing samples.

    Returns:
        Tuple ``(valid_sensor_ids, missing_rate_per_sensor)``.
    """
    if matrix.empty:
        return [], {}
    missing = matrix.isna().mean().to_dict()
    valid = [s for s, m in missing.items() if m <= max_missing_rate]
    return valid, {str(k): float(v) for k, v in missing.items()}


def select_valid_days(long_df: pd.DataFrame,
                      min_observations_per_day: int = 60
                      ) -> list[str]:
    """Return the set of dates with at least ``min_observations_per_day`` rows."""
    if long_df.empty:
        return []
    counts = long_df.groupby("date").size()
    valid = counts[counts >= min_observations_per_day].index
    return [d.isoformat() for d in sorted(valid)]


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def preprocess(insp: DeploymentInspection,
               variable_name: str,
               deployment_timezone: str,
               sync_freq: str = "2min",
               max_sensor_missing_rate: float = 0.5,
               value_lower: float = -50.0,
               value_upper: float = 5000.0
               ) -> PreprocessingResult:
    """Run the full preprocessing pipeline.

    Args:
        insp: Deployment inspection result.
        variable_name: Name of the selected light-related variable.
        deployment_timezone: IANA timezone for the deployment.
        sync_freq: Resampling cadence for the synchronized matrix.
        max_sensor_missing_rate: Sensor exclusion threshold.
        value_lower: Lower bound for plausible values.
        value_upper: Upper bound for plausible values.

    Returns:
        PreprocessingResult.
    """
    long_df, ts_diag = load_deployment_long(
        insp, variable_name, deployment_timezone)
    n_sensors_before = long_df["station_id"].nunique()
    days_before = long_df["date"].nunique()

    cleaned, clean_diag = clean_values(long_df, value_lower, value_upper)
    matrix = synchronize_matrix(cleaned, freq=sync_freq)
    valid_sensors, missing_by_sensor = select_valid_sensors(
        matrix, max_missing_rate=max_sensor_missing_rate)
    valid_days = select_valid_days(cleaned)

    matrix_filtered = matrix[valid_sensors] if valid_sensors else matrix.iloc[:, 0:0]
    sampling_seconds = estimate_sampling_interval_seconds(cleaned)

    summary = {
        "selected_variable": variable_name,
        "deployment_name": insp.deployment_name,
        "deployment_timezone": deployment_timezone,
        "raw_row_count": ts_diag["raw_row_count"],
        "cleaned_row_count": int(len(cleaned)),
        "n_sensors_before_filtering": int(n_sensors_before),
        "n_sensors_after_filtering": int(len(valid_sensors)),
        "n_days_before_filtering": int(days_before),
        "n_valid_days": int(len(valid_days)),
        "valid_days": valid_days,
        "valid_sensors": valid_sensors,
        "missing_rate_by_sensor": missing_by_sensor,
        "synchronized_matrix_shape": list(matrix_filtered.shape),
        "synchronized_matrix_freq": sync_freq,
        "sampling_interval_seconds_estimate": sampling_seconds,
        "timestamp_range": {
            "start_utc": (cleaned["timestamp_utc"].min().isoformat()
                          if not cleaned.empty else None),
            "end_utc": (cleaned["timestamp_utc"].max().isoformat()
                        if not cleaned.empty else None),
        },
        "exclusions_applied": {
            "value_filter": clean_diag,
            "max_sensor_missing_rate": max_sensor_missing_rate,
            "min_observations_per_day": 60,
        },
        "timestamp_strategy": ts_diag["timestamp_strategy"],
        "notes": [],
    }
    if not valid_sensors:
        summary["notes"].append(
            "No sensors passed the missing-rate filter; matrix is empty.")
    if not valid_days:
        summary["notes"].append("No valid days after filtering.")

    return PreprocessingResult(long_df=cleaned, matrix=matrix_filtered,
                               summary=summary)


def save_outputs(result: PreprocessingResult) -> None:
    """Persist long-form data, sensor matrix and preprocessing summary."""
    paths.ensure_dirs()
    long_df = result.long_df.copy()
    long_df["timestamp_utc"] = long_df["timestamp_utc"].dt.tz_convert("UTC")
    # Parquet does not accept python date objects directly via pyarrow always;
    # cast to string for safety.
    out_long = long_df.copy()
    out_long["date"] = out_long["date"].astype(str)
    out_long["timestamp_local"] = out_long["timestamp_local"].astype(str)
    out_long["station_id"] = out_long["station_id"].astype(str)
    out_long.to_parquet(paths.SYNCHRONIZED_PARQUET, index=False)
    out_long.to_csv(paths.SYNCHRONIZED_CSV, index=False)

    matrix = result.matrix.copy()
    matrix.index = matrix.index.astype(str)
    matrix.to_parquet(paths.SENSOR_MATRIX_PARQUET)

    import json
    with open(paths.PREPROCESSING_SUMMARY_JSON, "w", encoding="utf-8") as fh:
        json.dump(result.summary, fh, indent=2, ensure_ascii=False,
                  default=str)
    logger.info("Preprocessing outputs written.")
