"""Preliminary per-sensor informativeness around sunrise.

For each sensor and each valid day, the pre- and post-sunrise means are
estimated within a configurable window. The Gaussian per-sensor divergence

    D_i = (mu_1,i - mu_0,i)^2 / (2 * sigma_i^2)

is then computed by averaging mu_0, mu_1, and sigma^2 across days.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths

logger = logging.getLogger(__name__)


def _ensure_utc(ts: pd.Series) -> pd.Series:
    if ts.dt.tz is None:
        return ts.dt.tz_localize("UTC")
    return ts.dt.tz_convert("UTC")


def compute_sensor_informativeness(
        long_df: pd.DataFrame,
        sunrise_records: list[dict],
        pre_window_minutes: int = 120,
        post_window_minutes: int = 120,
        ) -> dict:
    """Compute per-sensor informativeness around sunrise.

    Args:
        long_df: Long-format dataframe with columns ``timestamp_utc``,
            ``date``, ``station_id``, ``value``.
        sunrise_records: List of records with ``date`` (ISO) and
            ``sunrise_utc`` (ISO).
        pre_window_minutes: Pre-sunrise window length in minutes.
        post_window_minutes: Post-sunrise window length in minutes.

    Returns:
        Dictionary with ``sensors`` (ranked list) and metadata.
    """
    if long_df.empty or not sunrise_records:
        return {
            "sensors": [],
            "pre_window_minutes": pre_window_minutes,
            "post_window_minutes": post_window_minutes,
            "notes": ["Insufficient data for informativeness estimation."],
        }

    df = long_df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)

    # Index sunrise UTC times by date.
    sunrise_by_date: dict[str, datetime] = {}
    for r in sunrise_records:
        sunrise_by_date[r["date"]] = pd.Timestamp(r["sunrise_utc"]).tz_convert(
            "UTC").to_pydatetime()

    pre_td = timedelta(minutes=pre_window_minutes)
    post_td = timedelta(minutes=post_window_minutes)

    per_sensor: dict[str, dict] = {}
    for station_id, sub in df.groupby("station_id"):
        pre_means: list[float] = []
        post_means: list[float] = []
        all_pre_vals: list[np.ndarray] = []
        all_post_vals: list[np.ndarray] = []
        valid_days = 0
        n_expected_days = 0

        for d, daygrp in sub.groupby("date"):
            if d not in sunrise_by_date:
                continue
            n_expected_days += 1
            sr = sunrise_by_date[d]
            pre_mask = (
                (daygrp["timestamp_utc"] >= sr - pre_td)
                & (daygrp["timestamp_utc"] < sr)
            )
            post_mask = (
                (daygrp["timestamp_utc"] >= sr)
                & (daygrp["timestamp_utc"] < sr + post_td)
            )
            pre_vals = daygrp.loc[pre_mask, "value"].to_numpy(dtype=float)
            post_vals = daygrp.loc[post_mask, "value"].to_numpy(dtype=float)
            pre_vals = pre_vals[np.isfinite(pre_vals)]
            post_vals = post_vals[np.isfinite(post_vals)]
            if len(pre_vals) >= 3 and len(post_vals) >= 3:
                pre_means.append(float(np.mean(pre_vals)))
                post_means.append(float(np.mean(post_vals)))
                all_pre_vals.append(pre_vals)
                all_post_vals.append(post_vals)
                valid_days += 1

        if valid_days == 0:
            per_sensor[str(station_id)] = {
                "sensor_id": str(station_id),
                "n_valid_days": 0,
                "mu_0": None, "mu_1": None,
                "sigma2": None, "sigma": None, "D_i": None,
                "missing_rate": 1.0,
                "mu_0_global": None, "mu_1_global": None,
                "sigma2_global": None, "sigma_global": None,
                "D_i_global": None,
                "parameter_estimation_method": (
                    "insufficient samples; sensor excluded from "
                    "global parameter estimation."
                ),
            }
            continue
        mu_0 = float(np.mean(pre_means))
        mu_1 = float(np.mean(post_means))
        # Pooled variance from concatenated within-window deviations.
        pre_concat = np.concatenate(all_pre_vals)
        post_concat = np.concatenate(all_post_vals)
        pre_dev = pre_concat - np.mean(pre_concat)
        post_dev = post_concat - np.mean(post_concat)
        pooled = np.concatenate([pre_dev, post_dev])
        sigma2 = float(np.var(pooled, ddof=1)) if len(pooled) > 1 else float("nan")
        if sigma2 and sigma2 > 0 and np.isfinite(sigma2):
            D_i = float((mu_1 - mu_0) ** 2 / (2.0 * sigma2))
        else:
            D_i = None
        missing_rate = (
            1.0 - valid_days / n_expected_days if n_expected_days else 1.0
        )
        per_sensor[str(station_id)] = {
            "sensor_id": str(station_id),
            "n_valid_days": int(valid_days),
            "mu_0": mu_0,
            "mu_1": mu_1,
            "sigma2": sigma2,
            "sigma": (
                float(np.sqrt(sigma2))
                if (sigma2 is not None and np.isfinite(sigma2) and sigma2 > 0)
                else None
            ),
            "D_i": D_i,
            "missing_rate": float(missing_rate),
            # Explicit aliases used by the global-parameter detector.
            "mu_0_global": mu_0,
            "mu_1_global": mu_1,
            "sigma2_global": sigma2,
            "sigma_global": (
                float(np.sqrt(sigma2))
                if (sigma2 is not None and np.isfinite(sigma2) and sigma2 > 0)
                else None
            ),
            "D_i_global": D_i,
            "parameter_estimation_method": (
                "per-day pre/post-sunrise window means averaged across "
                "valid days; pooled within-window variance."
            ),
        }

    sensors_sorted = sorted(
        per_sensor.values(),
        key=lambda r: (r["D_i"] if r["D_i"] is not None else -1.0),
        reverse=True,
    )
    for rank, rec in enumerate(sensors_sorted, start=1):
        rec["rank_by_D"] = rank

    return {
        "sensors": sensors_sorted,
        "pre_window_minutes": pre_window_minutes,
        "post_window_minutes": post_window_minutes,
        "n_sensors": len(sensors_sorted),
        "parameter_estimation_method": (
            "For every sensor, mu_0 and mu_1 are the cross-day averages "
            "of the per-day means computed in the pre- and post-sunrise "
            "windows; sigma^2 is the pooled within-window variance "
            "across all valid days. The resulting (mu_0, mu_1, sigma^2) "
            "are treated as fixed global empirical parameters of each "
            "sensor and are reused unchanged across all detection days."
        ),
        "notes": [],
    }


def write_informativeness(payload: dict,
                          out_path: Path = paths.SENSOR_INFORMATIVENESS_JSON
                          ) -> Path:
    """Write the sensor informativeness JSON file."""
    paths.ensure_dirs()
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    logger.info("Sensor informativeness written to %s", out_path)
    return out_path


def load_global_sensor_parameters(
        path: Path = paths.SENSOR_INFORMATIVENESS_JSON,
        ) -> dict[str, dict]:
    """Load fixed global per-sensor Gaussian parameters.

    Reads the informativeness JSON and returns a mapping
    ``{sensor_id: {mu_0, mu_1, sigma2, sigma, D_i}}`` containing only
    sensors whose parameters are finite and whose variance is strictly
    positive. These parameters are intended to be used unchanged for
    every test day by the global-parameter detector.

    Args:
        path: Path to ``sensor_informativeness.json``.

    Returns:
        Mapping from sensor id (string) to its global Gaussian
        parameters.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Sensor informativeness JSON missing: {path}. "
            "Run 'rank-sensors' first."
        )
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    out: dict[str, dict] = {}
    for rec in payload.get("sensors", []):
        sid = str(rec.get("sensor_id"))
        mu_0 = rec.get("mu_0_global", rec.get("mu_0"))
        mu_1 = rec.get("mu_1_global", rec.get("mu_1"))
        sigma2 = rec.get("sigma2_global", rec.get("sigma2"))
        d_i = rec.get("D_i_global", rec.get("D_i"))
        if mu_0 is None or mu_1 is None or sigma2 is None:
            continue
        if not (np.isfinite(mu_0) and np.isfinite(mu_1)
                and np.isfinite(sigma2) and sigma2 > 0):
            continue
        if d_i is None or not np.isfinite(d_i):
            d_i = float((mu_1 - mu_0) ** 2 / (2.0 * sigma2))
        out[sid] = {
            "mu_0": float(mu_0),
            "mu_1": float(mu_1),
            "sigma2": float(sigma2),
            "sigma": float(np.sqrt(sigma2)),
            "D_i": float(d_i),
        }
    return out
