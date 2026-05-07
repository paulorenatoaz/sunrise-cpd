"""Detector implementations for the Sunrise CPD experiment.

The system is a multi-sensor wireless network. The budget regime
controls how many sensors the sampling policy activates, but the
detector itself is uniformly multi-sensor: at every timestamp it
computes a per-sensor standardized evidence ``z_{i,t}`` and aggregates
it across the active subset before updating a one-sided Page CUSUM
recursion.

Aggregation choice
------------------

For each timestamp ``t`` and each active sensor ``i in S`` we compute

    z_{i,t} = (x_{i,t} - mu_{0,i}) / sigma_{0,i}

where ``mu_{0,i}`` and ``sigma_{0,i}`` are the baseline mean and
standard deviation of sensor ``i`` estimated from the pre-sunrise
segment of the window. We then form the cross-sensor average

    Z_t = (1 / |S_t|) * sum_{i in S_t} z_{i,t}

where ``S_t`` is the subset of active sensors with a finite reading at
time ``t``. The standard upward Page CUSUM is then applied to ``Z_t``:

    S_t = max(0, S_{t-1} + Z_t - k)
    tau = min { t : S_t >= h }

The averaging form keeps the threshold ``h`` on the same scale across
budget regimes (low / medium / high). For ``|S| = 1`` it reduces
exactly to the single-sensor case, which is essential for backward
compatibility with the original low-budget experiment.

Timestamps where every active sensor reports NaN are skipped without
resetting the statistic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class DetectorConfig:
    """Configuration for :func:`cusum_detect_multi_sensor`."""

    threshold: float = 5.0
    drift_k: float = 0.5
    min_baseline_samples: int = 5
    fallback_sigma: float = 1.0
    name: str = "one_sided_page_cusum_multi_sensor"
    aggregation: str = "mean_z"


@dataclass
class DetectionResult:
    """Outcome of running the detector on a single daily window."""

    detected_time: datetime | None
    statistic_at_detection: float | None
    n_observations: int
    baselines: dict = field(default_factory=dict)  # sensor_id -> stats
    notes: list[str] = field(default_factory=list)


def _baseline_stats(values: np.ndarray, fallback_sigma: float
                    ) -> tuple[float, float]:
    """Return ``(mu, sigma)`` for a 1-D array, with safe fallbacks."""
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return 0.0, fallback_sigma
    mu = float(np.mean(finite))
    sigma = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    if not np.isfinite(sigma) or sigma <= 1e-6:
        sigma = fallback_sigma
    return mu, sigma


def cusum_detect_multi_sensor(
        timestamps: Sequence[pd.Timestamp],
        values_by_sensor: dict[str, np.ndarray],
        sunrise_time: pd.Timestamp,
        config: DetectorConfig,
        ) -> DetectionResult:
    """Run the multi-sensor one-sided CUSUM detector on a daily window.

    Args:
        timestamps: Common timestamp grid (UTC, ascending) for the
            window. Length ``T``.
        values_by_sensor: Mapping ``sensor_id -> array of length T``
            holding the sensor's readings on the same grid. NaN denotes
            a missing observation.
        sunrise_time: Astronomical sunrise time in UTC; the pre-sunrise
            samples of the window are used to estimate per-sensor
            baselines.
        config: Detector configuration.

    Returns:
        :class:`DetectionResult`.
    """
    notes: list[str] = []
    if not values_by_sensor:
        return DetectionResult(None, None, 0, {}, ["No sensors selected."])
    if len(timestamps) == 0:
        return DetectionResult(None, None, 0, {},
                               ["No observations in window."])

    ts = pd.DatetimeIndex(pd.to_datetime(list(timestamps), utc=True))
    sunrise = pd.Timestamp(sunrise_time).tz_convert("UTC")
    pre_mask = np.asarray(ts < sunrise)

    baselines: dict[str, dict] = {}
    z_per_sensor: dict[str, np.ndarray] = {}
    usable_sensors: list[str] = []
    for sid, vals in values_by_sensor.items():
        arr = np.asarray(vals, dtype=float)
        if arr.shape[0] != len(ts):
            raise ValueError(
                f"Sensor {sid} array length {arr.shape[0]} does not "
                f"match timestamp grid length {len(ts)}."
            )
        pre_vals = arr[pre_mask]
        n_pre_finite = int(np.isfinite(pre_vals).sum())
        if n_pre_finite < config.min_baseline_samples:
            notes.append(
                f"Sensor {sid}: insufficient pre-sunrise samples "
                f"({n_pre_finite} < {config.min_baseline_samples}); "
                "excluded from this day's aggregate."
            )
            baselines[sid] = {"mu": None, "sigma": None,
                              "n_pre_finite": n_pre_finite, "used": False}
            continue
        mu, sigma = _baseline_stats(pre_vals, config.fallback_sigma)
        baselines[sid] = {"mu": mu, "sigma": sigma,
                          "n_pre_finite": n_pre_finite, "used": True}
        z_per_sensor[sid] = (arr - mu) / sigma
        usable_sensors.append(sid)

    if not usable_sensors:
        return DetectionResult(None, None, len(ts), baselines,
                               notes + ["No sensors had usable baselines."])

    z_matrix = np.vstack([z_per_sensor[sid] for sid in usable_sensors])
    finite_mask = np.isfinite(z_matrix)
    counts = finite_mask.sum(axis=0)
    sums = np.where(finite_mask, z_matrix, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        z_agg = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    s = 0.0
    detected_idx: int | None = None
    detection_stat: float | None = None
    for i, z in enumerate(z_agg):
        if not np.isfinite(z):
            continue
        s = max(0.0, s + float(z) - config.drift_k)
        if s >= config.threshold:
            detected_idx = i
            detection_stat = s
            break

    detected_time = (
        ts[detected_idx].to_pydatetime() if detected_idx is not None else None
    )
    return DetectionResult(detected_time, detection_stat, len(ts),
                           baselines, notes)


def cusum_detect_window(timestamps: Sequence[pd.Timestamp],
                        values: Sequence[float],
                        sunrise_time: datetime,
                        config: DetectorConfig
                        ) -> DetectionResult:
    """Single-sensor convenience wrapper kept for backward compatibility."""
    return cusum_detect_multi_sensor(
        timestamps=list(timestamps),
        values_by_sensor={"_single": np.asarray(values, dtype=float)},
        sunrise_time=pd.Timestamp(sunrise_time),
        config=config,
    )


# ---------------------------------------------------------------------------
# Global-parameter Gaussian LLR CUSUM (main detector)
# ---------------------------------------------------------------------------

@dataclass
class GlobalLLRConfig:
    """Configuration for :func:`cusum_detect_multi_sensor_global_params`.

    The detector uses fixed global empirical Gaussian parameters per
    sensor (estimated once across the whole dataset, see
    :func:`src.informativeness.load_global_sensor_parameters`) and does
    not estimate any same-day baseline.

    The per-observation evidence is the Gaussian log-likelihood ratio
    between the post-change and pre-change densities,

        llr_{i,t} = ((x_{i,t} - mu_0,i)^2 - (x_{i,t} - mu_1,i)^2)
                    / (2 * sigma_i^2),

    aggregated by mean over the active sensors that report a finite
    reading at time ``t``. The aggregated evidence drives a one-sided
    CUSUM with no drift term:

        S_t = max(0, S_{t-1} + L_t),
        tau = min { t : S_t >= h }.
    """

    threshold: float = 5.0
    name: str = "global_gaussian_llr_cusum_multi_sensor"
    aggregation: str = "mean_llr"
    evidence_type: str = "gaussian_log_likelihood_ratio"


def cusum_detect_multi_sensor_global_params(
        timestamps: Sequence[pd.Timestamp],
        values_by_sensor: dict[str, np.ndarray],
        sensor_params: dict[str, dict],
        config: GlobalLLRConfig,
        ) -> DetectionResult:
    """Run the global-parameter Gaussian LLR CUSUM detector.

    Args:
        timestamps: Common timestamp grid (UTC, ascending) for the
            window. Length ``T``.
        values_by_sensor: Mapping ``sensor_id -> array of length T``
            holding each sensor's readings on the grid; NaN denotes
            missing observations.
        sensor_params: Mapping ``sensor_id -> {mu_0, mu_1, sigma2, ...}``
            of fixed global empirical parameters. Sensors absent from
            this mapping are skipped.
        config: Detector configuration.
    """
    notes: list[str] = []
    if not values_by_sensor:
        return DetectionResult(None, None, 0, {}, ["No sensors selected."])
    if len(timestamps) == 0:
        return DetectionResult(None, None, 0, {},
                               ["No observations in window."])

    ts = pd.DatetimeIndex(pd.to_datetime(list(timestamps), utc=True))

    used_params: dict[str, dict] = {}
    llr_per_sensor: dict[str, np.ndarray] = {}
    for sid, vals in values_by_sensor.items():
        arr = np.asarray(vals, dtype=float)
        if arr.shape[0] != len(ts):
            raise ValueError(
                f"Sensor {sid} array length {arr.shape[0]} does not "
                f"match timestamp grid length {len(ts)}."
            )
        params = sensor_params.get(str(sid))
        if params is None:
            notes.append(
                f"Sensor {sid}: no global parameters available; "
                "excluded from this day's aggregate."
            )
            used_params[str(sid)] = {"used": False, "reason": "no_params"}
            continue
        mu_0 = float(params["mu_0"])
        mu_1 = float(params["mu_1"])
        sigma2 = float(params["sigma2"])
        if not (np.isfinite(mu_0) and np.isfinite(mu_1)
                and np.isfinite(sigma2) and sigma2 > 0):
            notes.append(
                f"Sensor {sid}: invalid global parameters; skipped."
            )
            used_params[str(sid)] = {"used": False, "reason": "invalid_params"}
            continue
        llr = ((arr - mu_0) ** 2 - (arr - mu_1) ** 2) / (2.0 * sigma2)
        llr_per_sensor[str(sid)] = llr
        used_params[str(sid)] = {
            "used": True,
            "mu_0": mu_0,
            "mu_1": mu_1,
            "sigma2": sigma2,
            "D_i": params.get("D_i"),
        }

    if not llr_per_sensor:
        return DetectionResult(None, None, len(ts), used_params,
                               notes + ["No sensors had global parameters."])

    llr_matrix = np.vstack([llr_per_sensor[sid]
                            for sid in llr_per_sensor])
    finite_mask = np.isfinite(llr_matrix)
    counts = finite_mask.sum(axis=0)
    sums = np.where(finite_mask, llr_matrix, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        l_agg = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    s = 0.0
    detected_idx: int | None = None
    detection_stat: float | None = None
    for i, l_t in enumerate(l_agg):
        if not np.isfinite(l_t):
            continue
        s = max(0.0, s + float(l_t))
        if s >= config.threshold:
            detected_idx = i
            detection_stat = s
            break

    detected_time = (
        ts[detected_idx].to_pydatetime() if detected_idx is not None else None
    )
    return DetectionResult(detected_time, detection_stat, len(ts),
                           used_params, notes)

