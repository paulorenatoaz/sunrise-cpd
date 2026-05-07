"""HTML reports for the sunrise change-point detection experiment.

This module renders one HTML report per budget scenario plus an
optional combined comparison report. It does not recompute the
experiments — it only reads the JSON outputs produced by
:mod:`src.experiments` and :mod:`src.comparison`.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import paths
from .detector import DetectorConfig
from .experiments import (
    HIGH_BUDGET_RESULTS_JSON,
    LOW_BUDGET_RESULTS_JSON,
    MEDIUM_BUDGET_RESULTS_JSON,
)

logger = logging.getLogger(__name__)


# Output paths (one report per scenario; legacy alias kept).
EXPERIMENT_REPORT_HTML = paths.REPORTS_DIR / "sunrise_experiment_report.html"
LOW_BUDGET_REPORT_HTML = paths.REPORTS_DIR / "sunrise_low_budget_report.html"
MEDIUM_BUDGET_REPORT_HTML = (
    paths.REPORTS_DIR / "sunrise_medium_budget_report.html"
)
HIGH_BUDGET_REPORT_HTML = paths.REPORTS_DIR / "sunrise_high_budget_report.html"

_SCENARIO_RESULTS_JSON = {
    "low": LOW_BUDGET_RESULTS_JSON,
    "medium": MEDIUM_BUDGET_RESULTS_JSON,
    "high": HIGH_BUDGET_RESULTS_JSON,
}
_SCENARIO_REPORT_HTML = {
    "low": LOW_BUDGET_REPORT_HTML,
    "medium": MEDIUM_BUDGET_REPORT_HTML,
    "high": HIGH_BUDGET_REPORT_HTML,
}

UNIFIED_MODEL_STATEMENT = (
    "The available system is always the same multi-sensor network. The "
    "budget regime changes only the subset of sensors selected by the "
    "sampling policy."
)
GOAL_STATEMENT = (
    "The goal is not to exactly reproduce the theoretical Gaussian model, "
    "but to test whether the same resource-constrained change-point "
    "detection logic appears in real sensor network data."
)
_SCENARIO_HEADLINE = {
    "low": (
        "Low budget. The same multi-sensor model is used. The only "
        "difference is the budget policy, which allows activating a "
        "single sensor — the top-1 by D_i."
    ),
    "medium": (
        "The medium-budget regime uses the same multi-sensor model as "
        "the low- and high-budget regimes. The only difference is the "
        "budget policy, which allows activating a subset of k sensors. "
        "In this run, k = {k} and the selected sensors are the top {k} "
        "sensors according to D_i."
    ),
    "high": (
        "The high-budget regime uses the same multi-sensor model as the "
        "other regimes. The only difference is the budget policy, which "
        "allows activating all valid sensors."
    ),
}


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
       Helvetica, Arial, sans-serif; max-width: 1080px; margin: 2em auto;
       padding: 0 1em; color: #222; line-height: 1.5; }
h1 { border-bottom: 2px solid #333; padding-bottom: .3em; }
h2 { border-bottom: 1px solid #aaa; padding-bottom: .2em; margin-top: 2em; }
table { border-collapse: collapse; margin: 1em 0; width: 100%; }
th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left;
         font-size: 0.92em; vertical-align: top; }
th { background: #f4f4f4; }
.statement { background: #eef6ff; border-left: 4px solid #2c6cb0;
             padding: 1em; margin: 1.5em 0; }
.status-on_time { color: #1b5e20; font-weight: 600; }
.status-detected { color: #0d47a1; font-weight: 600; }
.status-false_alarm { color: #b8651f; font-weight: 600; }
.status-missed_detection { color: #7a1f1f; font-weight: 600; }
code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }
.meta { color: #666; font-size: 0.9em; }
img { max-width: 100%; border: 1px solid #ddd; padding: 4px; background: #fff; }
"""


def _fmt(v) -> str:
    if v is None:
        return "<em>n/a</em>"
    if isinstance(v, float):
        return f"{v:.3f}"
    return escape(str(v))


def _table(headers: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_fmt(c)}</td>" for c in r) + "</tr>"
        for r in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _section(title: str, body_html: str) -> str:
    return f"<section><h2>{escape(title)}</h2>{body_html}</section>"


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Example detection plot
# ---------------------------------------------------------------------------

def _build_example_plot(results: dict) -> Path | None:
    """Render an example detection plot for the first detected day."""
    if not paths.SYNCHRONIZED_PARQUET.exists():
        return None
    selected = [str(s) for s in results.get("selected_sensors", [])]
    if not selected:
        return None
    per_day = results.get("per_day_results", [])
    target = next(
        (p for p in per_day if p["detected_change_point_utc"] is not None
         and not p["missed_detection"]),
        None,
    )
    if target is None and per_day:
        target = per_day[0]
    if target is None:
        return None

    long_df = pd.read_parquet(paths.SYNCHRONIZED_PARQUET)
    long_df["timestamp_utc"] = pd.to_datetime(long_df["timestamp_utc"], utc=True)
    long_df["station_id"] = long_df["station_id"].astype(str)
    sensor_df = long_df[long_df["station_id"].isin(selected)].copy()

    sunrise = pd.Timestamp(target["true_change_point_utc"]).tz_convert("UTC")
    pre_min = results["analysis_window"]["pre_window_minutes"]
    post_min = results["analysis_window"]["post_window_minutes"]
    win_start = sunrise - pd.Timedelta(minutes=pre_min)
    win_end = sunrise + pd.Timedelta(minutes=post_min)
    sub = sensor_df[(sensor_df["timestamp_utc"] >= win_start)
                    & (sensor_df["timestamp_utc"] <= win_end)]
    if sub.empty:
        return None
    grid = pd.DatetimeIndex(sorted(sub["timestamp_utc"].unique()), tz="UTC")

    cfg = DetectorConfig(
        threshold=results["threshold"],
        drift_k=results["detector_parameters"]["drift_k"],
    )

    z_per_sensor: dict[str, np.ndarray] = {}
    sensor_series: dict[str, pd.Series] = {}
    for sid in selected:
        s = sub.loc[sub["station_id"] == sid, ["timestamp_utc", "value"]]
        if s.empty:
            sensor_series[sid] = pd.Series(np.nan, index=grid)
            continue
        series = (s.groupby("timestamp_utc")["value"].mean()
                   .astype(float).sort_index().reindex(grid))
        sensor_series[sid] = series
        arr = series.to_numpy(dtype=float)
        pre_mask = np.asarray(grid < sunrise)
        pre = arr[pre_mask]
        pre = pre[np.isfinite(pre)]
        if len(pre) < cfg.min_baseline_samples:
            continue
        mu = float(np.mean(pre))
        sigma = (float(np.std(pre, ddof=1)) if len(pre) > 1
                 else cfg.fallback_sigma)
        if sigma <= 1e-6:
            sigma = cfg.fallback_sigma
        z_per_sensor[sid] = (arr - mu) / sigma
    if not z_per_sensor:
        return None

    z_matrix = np.vstack(list(z_per_sensor.values()))
    finite = np.isfinite(z_matrix)
    counts = finite.sum(axis=0)
    sums = np.where(finite, z_matrix, 0.0).sum(axis=0)
    z_agg = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    s = 0.0
    stat = []
    for z in z_agg:
        if np.isfinite(z):
            s = max(0.0, s + float(z) - cfg.drift_k)
        stat.append(s)

    paths.ensure_dirs()
    sensor_tag = "-".join(selected)
    asset_path = (paths.ASSETS_DIR
                  / f"example_detection_{results['budget_regime']}"
                    f"_{sensor_tag}_{target['date']}.png")
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for sid, series in sensor_series.items():
        axes[0].plot(series.index, series.values, label=f"Sensor {sid}")
    axes[0].axvline(sunrise, color="#1b5e20", linestyle="--", label="Sunrise")
    if target["detected_change_point_utc"]:
        axes[0].axvline(pd.Timestamp(target["detected_change_point_utc"]),
                        color="#b8651f", linestyle=":", label="Detection")
    axes[0].set_ylabel("Solar radiation [W/m^2]")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set_title(
        f"Example day {target['date']} — "
        f"{results['budget_regime']} budget, sensors {selected}"
    )

    axes[1].plot(grid, stat, color="#7a1f1f", label="Aggregated CUSUM S_t")
    axes[1].axhline(cfg.threshold, color="black", linestyle="--",
                    label=f"Threshold h={cfg.threshold}")
    axes[1].axvline(sunrise, color="#1b5e20", linestyle="--")
    if target["detected_change_point_utc"]:
        axes[1].axvline(pd.Timestamp(target["detected_change_point_utc"]),
                        color="#b8651f", linestyle=":")
    axes[1].set_ylabel("S_t")
    axes[1].set_xlabel("Time (UTC)")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(asset_path, dpi=120)
    plt.close(fig)
    return asset_path


# ---------------------------------------------------------------------------
# Per-scenario report
# ---------------------------------------------------------------------------

def render_scenario_report(scenario: str,
                           out_path: Path | None = None) -> Path:
    """Render the HTML report for a single budget scenario."""
    scenario = scenario.lower()
    if scenario not in _SCENARIO_RESULTS_JSON:
        raise ValueError(f"Unknown scenario: {scenario!r}")
    json_path = _SCENARIO_RESULTS_JSON[scenario]
    if out_path is None:
        out_path = _SCENARIO_REPORT_HTML[scenario]

    results = _load(json_path)
    inv = _load(paths.DATASET_INVENTORY_JSON)
    var = _load(paths.VARIABLE_SELECTION_JSON)
    prep = _load(paths.PREPROCESSING_SUMMARY_JSON)
    info = _load(paths.SENSOR_INFORMATIVENESS_JSON)
    gt = _load(paths.SUNRISE_GROUND_TRUTH_JSON)

    if results is None:
        raise FileNotFoundError(
            f"Results JSON missing for scenario {scenario!r}: {json_path}"
        )

    headline = _SCENARIO_HEADLINE[scenario].format(k=results.get("k"))
    sections: list[str] = []

    sections.append(_section(
        "1. Sunrise CPD — Connection to the Paper",
        "<p>Real-data validation of the resource-constrained multi-sensor "
        "change-point detection framework introduced in <em>Covert "
        "Change-Point Detection with Resource-Constrained Multi-Sensor "
        "Sampling</em>. The system model is always a multi-sensor "
        "network with <code>M</code> available sensors; the detector "
        "selects an active subset <code>S(t)</code> under sampling and "
        "communication constraints. The budget regime controls the "
        "cardinality of <code>S(t)</code>. The true change point "
        "<code>nu_d</code> is the astronomical sunrise on day "
        "<code>d</code>; the detector returns a stopping time "
        "<code>tau_d</code> from solar-radiation observations. "
        "Per-sensor informativeness is summarized by the Gaussian "
        "divergence <code>D_i = (mu_1 - mu_0)^2 / (2 sigma^2)</code>.</p>"
        f"<div class='statement'>{escape(UNIFIED_MODEL_STATEMENT)}</div>"
        f"<div class='statement'>{escape(GOAL_STATEMENT)}</div>"
    ))

    sections.append(_section(
        "2. Dataset Summary",
        _table(["Field", "Value"], [
            ["Dataset", (inv or {}).get("dataset_name")],
            ["Source", (inv or {}).get("source_url")],
            ["Deployment", (inv or {}).get("selected_deployment")],
            ["Variable", (var or {}).get("selected_variable")],
            ["Units", (var or {}).get("units")],
            ["Valid sensors (full set)",
             ", ".join((prep or {}).get("valid_sensors") or []) or "n/a"],
            ["Valid days", (prep or {}).get("n_valid_days")],
            ["Sampling interval (s)",
             (prep or {}).get("sampling_interval_seconds_estimate")],
        ])
    ))

    if gt:
        sections.append(_section(
            "3. Ground Truth",
            f"<p>Sunrise times computed with "
            f"<code>{escape(gt.get('method', ''))}</code> at latitude "
            f"<code>{gt.get('latitude')}</code>, longitude "
            f"<code>{gt.get('longitude')}</code> (timezone "
            f"<code>{escape(gt.get('timezone', ''))}</code>). "
            f"The ground-truth file contains "
            f"{len(gt.get('records', []))} daily records. Detection "
            "delays are computed in UTC; "
            "<code>signed_delay = tau_d - nu_d</code>.</p>"
        ))

    sections.append(_section(
        "4. Multi-Sensor Model and Budget Regimes",
        f"<div class='statement'>{escape(headline)}</div>"
        "<p>The unified experiment always loads the full set of valid "
        "sensors. A budget policy then chooses the active subset:</p>"
        "<ul>"
        "<li><strong>Low budget</strong>: top-1 sensor by D_i.</li>"
        "<li><strong>Medium budget</strong>: top-k sensors by D_i.</li>"
        "<li><strong>High budget</strong>: every valid sensor.</li>"
        "</ul>"
        "<p>Costs <code>C_i</code> and <code>T_i</code> are not "
        "available, so unit cost is assumed; this is equivalent to "
        "ranking by <code>D_i / (C_i + T_i)</code> with "
        "<code>C_i + T_i = 1</code>.</p>"
    ))

    ranking_rows = [
        [r["rank"], r["sensor_id"], r.get("D_i")]
        for r in results.get("sensor_ranking", [])
    ]
    sections.append(_section(
        "5. Budget Policy and Selected Sensors",
        _table(["Field", "Value"], [
            ["Budget regime", results["budget_regime"]],
            ["Description", results["budget_description"]],
            ["Selection policy", results["selection_policy"]],
            ["Unit cost assumption", results["unit_cost_assumption"]],
            ["All valid sensors",
             ", ".join(results.get("all_valid_sensors", []))],
            ["Selected sensors",
             ", ".join(results.get("selected_sensors", []))],
            ["Number of active sensors",
             len(results.get("selected_sensors", []))],
            ["Selection reason", results["selection_reason"]],
            ["Detector input mode", results["detector_input_mode"]],
        ])
        + "<h3>Sensor ranking by D_i</h3>"
        + _table(["Rank", "Sensor ID", "D_i"], ranking_rows)
    ))

    # Per-sensor empirical statistics for the selected sensors.
    if info is not None:
        info_by_id = {str(s.get("sensor_id")): s
                      for s in info.get("sensors", [])}
        rows = []
        for sid in results.get("selected_sensors", []):
            s = info_by_id.get(str(sid)) or {}
            rows.append([sid, s.get("D_i"), s.get("mu_0"), s.get("mu_1"),
                         s.get("sigma2"), s.get("n_valid_days"),
                         s.get("missing_rate")])
        sections.append(_section(
            "6. Selected Sensors — Per-Sensor Empirical Statistics",
            _table(["Sensor", "D_i", "mu_0", "mu_1", "sigma^2",
                    "Valid days", "Missing rate"], rows)
        ))

    params = results["detector_parameters"]
    sections.append(_section(
        "7. Detector",
        "<p>One-sided multi-sensor Page CUSUM. At each timestamp the "
        "detector standardizes every active sensor against its own "
        "pre-sunrise baseline and aggregates by averaging the finite "
        "z-scores:</p>"
        "<pre><code>"
        "z_{i,t} = (x_{i,t} - mu_{0,i}) / sigma_{0,i}\n"
        "Z_t    = mean_{i in S_t} z_{i,t}\n"
        "S_t    = max(0, S_{t-1} + Z_t - k)\n"
        "tau    = min { t : S_t &gt;= h }"
        "</code></pre>"
        f"<p>{escape(results['detector_aggregation_description'])}</p>"
        + _table(["Parameter", "Value"], [
            ["Detector name", results["detector_name"]],
            ["Aggregation", results["aggregation"]],
            ["Drift k", params["drift_k"]],
            ["Threshold h", results["threshold"]],
            ["Min baseline samples", params["min_baseline_samples"]],
            ["Fallback sigma", params["fallback_sigma"]],
            ["Pre window (min)",
             results["analysis_window"]["pre_window_minutes"]],
            ["Post window (min)",
             results["analysis_window"]["post_window_minutes"]],
            ["Sync freq",
             results["analysis_window"].get("sync_freq")],
            ["Tolerance (min)", results["tolerance_minutes"]],
        ])
    ))

    agg = results["aggregate_metrics"]
    sections.append(_section(
        "8. Aggregate Metrics",
        _table(["Metric", "Value"], [[k, v] for k, v in agg.items()])
    ))

    try:
        plot_path = _build_example_plot(results)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not render example plot: %s", exc)
        plot_path = None
    if plot_path is not None:
        rel = plot_path.relative_to(paths.REPORTS_DIR)
        sections.append(_section(
            "9. Example Detection",
            f"<img src='{escape(str(rel))}' "
            "alt='Example detection plot' />"
        ))

    head = ("<tr>" + "".join(f"<th>{h}</th>" for h in [
        "Date", "Sunrise (local)", "Detected (local)",
        "Signed delay [min]", "|Error| [min]", "Status",
        "False alarm", "Missed", "Active sensors", "Used", "Grid pts",
    ]) + "</tr>")
    body_rows = []
    for p in results["per_day_results"]:
        body_rows.append(
            "<tr>"
            f"<td>{escape(p['date'])}</td>"
            f"<td>{escape(str(p['true_change_point_local']))}</td>"
            f"<td>{escape(str(p['detected_change_point_local']))}</td>"
            f"<td>{_fmt(p['signed_delay_minutes'])}</td>"
            f"<td>{_fmt(p['absolute_error_minutes'])}</td>"
            f"<td><span class='status-{p['status']}'>"
            f"{escape(p['status'])}</span></td>"
            f"<td>{p['false_alarm']}</td>"
            f"<td>{p['missed_detection']}</td>"
            f"<td>{escape(','.join(p.get('active_sensors', [])))}</td>"
            f"<td>{escape(','.join(p.get('used_sensors', [])))}</td>"
            f"<td>{p['grid_points']}</td>"
            "</tr>"
        )
    table = (f"<table><thead>{head}</thead><tbody>"
             + "".join(body_rows) + "</tbody></table>")
    sections.append(_section("10. Per-Day Detection Results", table))

    sections.append(_section(
        "11. Experimental Scope",
        "<ul>"
        "<li>The empirical evaluation uses solar radiation from the "
        "SensorScope Grand-St-Bernard deployment.</li>"
        "<li>Astronomical sunrise is used as external ground truth for "
        "the change point.</li>"
        "<li>Detector parameters (drift k, threshold h, tolerance, "
        "analysis window, synchronization frequency, and missing-data "
        "handling) are held fixed across budget regimes to isolate the "
        "effect of the sensor budget.</li>"
        "<li>Sensing and communication costs C_i, T_i are not available "
        "in the dataset; unit costs are assumed, so ranking by "
        "D_i / (C_i + T_i) reduces to ranking by D_i.</li>"
        "<li>The evaluation addresses the non-adversarial "
        "resource-constrained setting; the covert adversarial model is "
        "not analyzed.</li>"
        "<li>Sunset detection is not considered.</li>"
        "</ul>"
    ))

    title = f"Sunrise CPD — {results['budget_regime'].title()} Budget Report"
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{escape(title)}</title>"
        f"<style>{CSS}</style></head><body>"
        f"<h1>{escape(title)}</h1>"
        f"<p class='meta'>Generated: "
        f"{datetime.now(timezone.utc).isoformat()}</p>"
        + "".join(sections)
        + "</body></html>"
    )
    paths.ensure_dirs()
    out_path.write_text(html, encoding="utf-8")
    logger.info("%s report written to %s", scenario, out_path)
    return out_path


def render_experiment_report(out_path: Path = EXPERIMENT_REPORT_HTML) -> Path:
    """Backward-compatible alias: render the low-budget report.

    Also writes the canonical low-budget report at
    :data:`LOW_BUDGET_REPORT_HTML` for consistency with the new naming.
    """
    render_scenario_report("low", out_path=LOW_BUDGET_REPORT_HTML)
    return render_scenario_report("low", out_path=out_path)
