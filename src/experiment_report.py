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
        "Low budget. The dynamic two-budget policy enforces "
        "sensing_budget B = 1.0 and transmission_budget C = 0.0, so "
        "at most one sensor (the local/reference sensor) is selected "
        "at each timestamp; no cooperative sensors are allowed."
    ),
    "medium": (
        "Medium budget. The dynamic two-budget policy enforces "
        "sensing_budget B = 3.0 and transmission_budget C = 2.0, so "
        "up to three sensors are selected at each timestamp: one "
        "local/reference sensor plus up to two cooperative sensors."
    ),
    "high": (
        "High budget. The dynamic two-budget policy uses "
        "sensing_budget B equal to the number of valid sensors and "
        "transmission_budget C equal to that number minus one, so "
        "all available sensors may be selected at every timestamp."
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
.status-late_detection { color: #b8651f; font-weight: 600; }
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
    """Render an example detection plot for the first detected day.

    The plot shows the sensors most frequently selected across the run
    (up to five) for context, and the aggregated CUSUM trajectory.
    """
    if not paths.SYNCHRONIZED_PARQUET.exists():
        return None
    most_freq = (results.get("selected_sensors_summary", {})
                 .get("most_frequently_selected_sensors", []))
    selected = [str(r["sensor_id"]) for r in most_freq[:5]]
    if not selected:
        selected = [str(s) for s in results.get("full_candidate_sensors", [])][:5]
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

    cfg_threshold = float(results["threshold"])

    # Load fixed global Gaussian parameters per sensor.
    from .informativeness import load_global_sensor_parameters
    try:
        global_params = load_global_sensor_parameters(
            paths.SENSOR_INFORMATIVENESS_JSON)
    except FileNotFoundError:
        global_params = {}

    llr_per_sensor: dict[str, np.ndarray] = {}
    sensor_series: dict[str, pd.Series] = {}
    for sid in selected:
        s = sub.loc[sub["station_id"] == sid, ["timestamp_utc", "value"]]
        if s.empty:
            sensor_series[sid] = pd.Series(np.nan, index=grid)
            continue
        series = (s.groupby("timestamp_utc")["value"].mean()
                   .astype(float).sort_index().reindex(grid))
        sensor_series[sid] = series
        params = global_params.get(str(sid))
        if params is None:
            continue
        arr = series.to_numpy(dtype=float)
        mu_0 = params["mu_0"]
        mu_1 = params["mu_1"]
        sigma2 = params["sigma2"]
        llr_per_sensor[sid] = ((arr - mu_0) ** 2 - (arr - mu_1) ** 2) / (
            2.0 * sigma2)
    if not llr_per_sensor:
        return None

    llr_matrix = np.vstack(list(llr_per_sensor.values()))
    finite = np.isfinite(llr_matrix)
    counts = finite.sum(axis=0)
    sums = np.where(finite, llr_matrix, 0.0).sum(axis=0)
    l_agg = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    s = 0.0
    stat = []
    for l_t in l_agg:
        if np.isfinite(l_t):
            s = max(0.0, s + float(l_t))
        stat.append(s)

    paths.ensure_dirs()
    sensor_tag = "-".join(selected)
    asset_path = (paths.ASSETS_DIR
                  / f"example_detection_{results['budget_regime']}"
                    f"_top_{sensor_tag}_{target['date']}.png")
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
        f"{results['budget_regime']} budget, "
        f"B={results.get('sensing_budget')}, "
        f"C={results.get('transmission_budget')}, "
        f"top sensors {selected}"
    )

    axes[1].plot(grid, stat, color="#7a1f1f", label="Aggregated CUSUM S_t")
    axes[1].axhline(cfg_threshold, color="black", linestyle="--",
                    label=f"Threshold h={cfg_threshold}")
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

    headline = _SCENARIO_HEADLINE[scenario]
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
        "4. Multi-Sensor Model and Two-Budget Regimes",
        f"<div class='statement'>{escape(headline)}</div>"
        "<p>The unified experiment always loads the full set of valid "
        "sensors. The dynamic two-budget policy then selects the "
        "active subset <code>S(t)</code> at every timestamp under "
        "<em>two separate</em> resource constraints (mirroring the "
        "paper):</p>"
        "<pre><code>"
        "sum_{i in S(t)} C_i &lt;= sensing_budget B\n"
        "sum_{i in S(t)} T_i &lt;= transmission_budget C"
        "</code></pre>"
        "<ul>"
        "<li><strong>Low budget</strong>: <code>B = 1.0</code>, "
        "<code>C = 0.0</code> — at most one sensor per timestamp "
        "(local/reference only); no cooperative sensors.</li>"
        "<li><strong>Medium budget</strong>: <code>B = 3.0</code>, "
        "<code>C = 2.0</code> — up to one local/reference sensor "
        "plus up to two cooperative sensors per timestamp.</li>"
        "<li><strong>High budget</strong>: <code>B = |valid|</code>, "
        "<code>C = |valid| - 1</code> — all available sensors may be "
        "selected at every timestamp.</li>"
        "</ul>"
        "<h3>Local/reference and cooperative sensors</h3>"
        "<p>At each timestamp the first selected sensor is treated as "
        "the <em>local/reference sensor</em> and pays an effective "
        "transmission cost of <code>0</code> because it does not need "
        "to transmit to itself. Each additional selected sensor is a "
        "<em>cooperative/remote sensor</em> and pays its full nominal "
        "transmission cost. The local/reference sensor is not fixed: "
        "it is whichever available sensor has the highest <code>D_i</code> "
        "at that timestamp.</p>"
        "<h3>Cost model (homogeneous synthetic)</h3>"
        "<p>Sensing and transmission costs <code>C_i</code>, "
        "<code>T_i</code> are not measured in the dataset; this report "
        "applies the explicit homogeneous synthetic assumption "
        "<code>C_i = 1.0</code>, <code>T_i = 1.0</code> stored in "
        "<code>output/json/sensor_costs.json</code>. The local/reference "
        "sensor's effective transmission cost <code>T_i_eff = 0</code> "
        "is applied only dynamically inside the per-timestamp selection "
        "step. Under homogeneous costs, ranking by "
        "<code>D_i</code> is equivalent to ranking by information per "
        "cost; the implementation is structured so that heterogeneous "
        "costs can be added later without changing the policy code.</p>"
    ))

    ranking_rows = [
        [r.get("rank"), r.get("sensor_id"), r.get("D_i"),
         r.get("sensing_cost"), r.get("transmission_cost"),
         r.get("score")]
        for r in results.get("sensor_ranking", [])
    ]
    sel_summary = results.get("selected_sensors_summary", {}) or {}
    most_freq = sel_summary.get("most_frequently_selected_sensors", [])
    most_freq_local = sel_summary.get(
        "most_frequently_selected_local_sensors", [])
    most_freq_coop = sel_summary.get(
        "most_frequently_selected_cooperative_sensors", [])

    def _fmt_freq(items):
        if not items:
            return "<em>n/a</em>"
        return ", ".join(
            f"{r['sensor_id']} (n={r['selection_count']})" for r in items
        )

    cost_rows = [
        [c.get("sensor_id"), c.get("sensing_cost"),
         c.get("transmission_cost"), c.get("cost_source")]
        for c in results.get("sensor_costs", [])
    ]
    sections.append(_section(
        "5. Dynamic Two-Budget Policy",
        _table(["Field", "Value"], [
            ["Budget regime", results["budget_regime"]],
            ["Description", results["budget_description"]],
            ["Sensing budget B", results.get("sensing_budget")],
            ["Transmission budget C", results.get("transmission_budget")],
            ["Budget constraints",
             " ; ".join(results.get("budget_constraints", []))],
            ["Cost model", results.get("cost_model")],
            ["Homogeneous cost assumption",
             results.get("homogeneous_cost_assumption")],
            ["Dynamic selection", results.get("dynamic_selection")],
            ["Selection policy", results["selection_policy"]],
            ["Score",
             (results.get("selection_policy_config") or {}).get("score_name")],
            ["Candidate sensors",
             ", ".join(results.get("full_candidate_sensors", []))],
            ["Sensors ever selected",
             ", ".join(sel_summary.get("sensors_ever_selected", []))],
            ["Most frequently selected (any role)",
             _fmt_freq(most_freq)],
            ["Most frequently selected as local/reference",
             _fmt_freq(most_freq_local)],
            ["Most frequently selected as cooperative",
             _fmt_freq(most_freq_coop)],
            ["Detector input mode", results.get("detector_input_mode")],
        ])
        + "<h3>Sensor cost model (nominal)</h3>"
        + _table(["Sensor ID", "C_i", "T_i", "Source"], cost_rows)
        + "<p class='meta'>The local/reference sensor's effective "
          "transmission cost is 0 only inside the per-timestamp "
          "selection step; the table above shows the nominal stored "
          "costs.</p>"
        + "<h3>Sensor ranking by D_i</h3>"
        + _table(["Rank", "Sensor ID", "D_i", "C_i", "T_i", "Score"],
                 ranking_rows)
    ))

    # Per-sensor empirical statistics for all candidate sensors.
    if info is not None:
        info_by_id = {str(s.get("sensor_id")): s
                      for s in info.get("sensors", [])}
        rows = []
        for sid in results.get("full_candidate_sensors", []):
            s = info_by_id.get(str(sid)) or {}
            rows.append([sid, s.get("D_i"), s.get("mu_0"), s.get("mu_1"),
                         s.get("sigma2"), s.get("n_valid_days"),
                         s.get("missing_rate")])
        sections.append(_section(
            "6. Candidate Sensors — Per-Sensor Empirical Statistics",
            _table(["Sensor", "D_i", "mu_0", "mu_1", "sigma^2",
                    "Valid days", "Missing rate"], rows)
        ))

    sections.append(_section(
        "7. Detector",
        "<p>The main detector is a multi-sensor one-sided CUSUM that "
        "uses fixed global empirical Gaussian parameters per sensor. "
        "These parameters "
        "(<code>mu_{0,i}</code>, <code>mu_{1,i}</code>, "
        "<code>sigma_i^2</code>) are estimated once from the SensorScope "
        "sunrise dataset and stored in "
        "<code>output/json/sensor_informativeness.json</code>; the same "
        "parameters are reused unchanged for every test day. The "
        "astronomical sunrise of each day is used only to extract the "
        "evaluation window and to score detections; it is not used to "
        "estimate any detector parameter in this mode. The per-sensor "
        "Gaussian divergence "
        "<code>D_i = (mu_{1,i} - mu_{0,i})^2 / (2 sigma_i^2)</code> is "
        "computed from the same global parameters used by the detector. "
        "At each timestamp, the per-sensor Gaussian log-likelihood ratio "
        "is averaged over the active sensors that report a finite "
        "reading and accumulated by a one-sided CUSUM:</p>"
        "<pre><code>"
        "llr_{i,t} = ((x_{i,t} - mu_{0,i})^2 - "
        "(x_{i,t} - mu_{1,i})^2) / (2 sigma_i^2)\n"
        "L_t       = mean_{i in S_t} llr_{i,t}\n"
        "S_t       = max(0, S_{t-1} + L_t)\n"
        "tau       = min { t : S_t &gt;= h }"
        "</code></pre>"
        f"<p>{escape(results['detector_aggregation_description'])}</p>"
        + _table(["Parameter", "Value"], [
            ["Detector mode", results.get("detector_mode")],
            ["Detector name", results["detector_name"]],
            ["Evidence type", results.get("evidence_type")],
            ["Aggregation", results["aggregation"]],
            ["Parameter source", results.get("parameter_source")],
            ["Global parameter file",
             results.get("global_parameter_file") or "n/a"],
            ["Threshold h", results["threshold"]],
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
        "Most freq. local", "Sensors ever selected",
        "Avg avail.", "Avg |S(t)|",
        "Avg sensing", "Avg transmission", "Grid pts",
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
            f"<td>{escape(str(p.get('most_frequent_local_sensor') or ''))}"
            f" (n={p.get('most_frequent_local_sensor_count', 0)})</td>"
            f"<td>{escape(','.join(p.get('sensors_ever_selected', [])))}</td>"
            f"<td>{_fmt(p.get('average_available_sensors_per_timestamp'))}</td>"
            f"<td>{_fmt(p.get('average_active_sensors_per_timestamp'))}</td>"
            f"<td>{_fmt(p.get('average_sensing_cost_used'))}</td>"
            f"<td>{_fmt(p.get('average_transmission_cost_used'))}</td>"
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
        "<li>Sensing and transmission costs C_i, T_i are not measured "
        "in the dataset; the explicit homogeneous synthetic assumption "
        "C_i = 1.0, T_i = 1.0 is recorded in "
        "<code>output/json/sensor_costs.json</code>. The local/reference "
        "sensor's effective transmission cost is 0 only inside the "
        "per-timestamp selection step. Under homogeneous costs, "
        "ranking by D_i is equivalent to ranking by information per "
        "cost.</li>"
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
