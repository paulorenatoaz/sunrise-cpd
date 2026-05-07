"""Combined comparison of the three budget regimes."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from . import paths
from .experiments import (
    HIGH_BUDGET_RESULTS_JSON,
    LOW_BUDGET_RESULTS_JSON,
    MEDIUM_BUDGET_RESULTS_JSON,
)

logger = logging.getLogger(__name__)


COMPARISON_JSON = paths.JSON_DIR / "sunrise_budget_comparison.json"
COMPARISON_HTML = paths.REPORTS_DIR / "sunrise_budget_comparison_report.html"

_SCENARIOS = [
    ("low", LOW_BUDGET_RESULTS_JSON),
    ("medium", MEDIUM_BUDGET_RESULTS_JSON),
    ("high", HIGH_BUDGET_RESULTS_JSON),
]

_METRIC_FIELDS = [
    ("detected_days_count", "Detected days"),
    ("missed_detection_count", "Missed detections"),
    ("false_alarm_count", "False alarms"),
    ("mean_signed_delay_minutes", "Mean signed delay [min]"),
    ("median_signed_delay_minutes", "Median signed delay [min]"),
    ("mean_absolute_error_minutes", "Mean |error| [min]"),
    ("median_absolute_error_minutes", "Median |error| [min]"),
    ("std_signed_delay_minutes", "Std signed delay [min]"),
    ("min_signed_delay_minutes", "Min signed delay [min]"),
    ("max_signed_delay_minutes", "Max signed delay [min]"),
    ("mean_active_sensors_per_timestamp", "Mean |S(t)|"),
    ("mean_budget_used_per_timestamp", "Mean budget used"),
]


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _summary_row(scenario: str, payload: dict) -> dict:
    agg = payload.get("aggregate_metrics", {})
    sel_summary = payload.get("selected_sensors_summary", {}) or {}
    ever = sel_summary.get("sensors_ever_selected", []) or []
    most_freq = sel_summary.get("most_frequently_selected_sensors", []) or []
    row = {
        "scenario_name": payload.get("scenario_name"),
        "budget_regime": scenario,
        "budget_value": payload.get("budget_value"),
        "cost_model": payload.get("cost_model"),
        "dynamic_selection": payload.get("dynamic_selection"),
        "selection_policy": payload.get("selection_policy"),
        "full_candidate_sensors": list(
            payload.get("full_candidate_sensors", [])),
        "sensors_ever_selected": list(ever),
        "most_frequently_selected_sensors": [
            f"{r['sensor_id']} (n={r['selection_count']})"
            for r in most_freq[:5]
        ],
    }
    for key, _ in _METRIC_FIELDS:
        row[key] = agg.get(key)
    return row


def _interpret(rows: list[dict]) -> dict:
    """Compare scenarios and produce a non-prescriptive narrative."""
    by_regime = {r["budget_regime"]: r for r in rows}
    seq = ["low", "medium", "high"]
    available = [s for s in seq if s in by_regime]

    def metric(s, key):
        return by_regime[s].get(key)

    findings: list[str] = []

    # Detected-days trend.
    det = [(s, metric(s, "detected_days_count")) for s in available]
    if all(v is not None for _, v in det):
        values = [v for _, v in det]
        if len(set(values)) == 1:
            det_trend = "flat"
            findings.append(
                f"All regimes detect the same number of days "
                f"({values[0]})."
            )
        elif values == sorted(values):
            det_trend = "monotonic_improvement"
            findings.append(
                "Detected-days count grows monotonically with the budget."
            )
        elif values == sorted(values, reverse=True):
            det_trend = "monotonic_degradation"
            findings.append(
                "Detected-days count decreases monotonically as the "
                "budget grows."
            )
        else:
            det_trend = "mixed"
            findings.append(
                "Detected-days count is non-monotonic across regimes."
            )
    else:
        det_trend = "unknown"

    # False-alarm trend.
    fa = [(s, metric(s, "false_alarm_count")) for s in available]
    if all(v is not None for _, v in fa):
        fa_values = [v for _, v in fa]
        if len(set(fa_values)) == 1:
            fa_trend = "flat"
        elif fa_values == sorted(fa_values):
            fa_trend = "monotonic_increase"
            findings.append(
                "False-alarm count rises monotonically with the budget."
            )
        elif fa_values == sorted(fa_values, reverse=True):
            fa_trend = "monotonic_decrease"
            findings.append(
                "False-alarm count drops monotonically as the budget grows."
            )
        else:
            fa_trend = "mixed"
    else:
        fa_trend = "unknown"

    # Mean absolute error trend (smaller is better).
    mae = [(s, metric(s, "mean_absolute_error_minutes")) for s in available]
    if all(v is not None for _, v in mae):
        values = [v for _, v in mae]
        if values == sorted(values):
            mae_trend = "monotonic_degradation"
        elif values == sorted(values, reverse=True):
            mae_trend = "monotonic_improvement"
            findings.append(
                "Mean absolute timing error decreases monotonically with "
                "the budget — larger budgets are closer to the true "
                "sunrise."
            )
        else:
            mae_trend = "mixed"
    else:
        mae_trend = "unknown"

    # Composite verdict (non-prescriptive language).
    verdict = "mixed"
    if (det_trend == "monotonic_improvement"
            and mae_trend in {"monotonic_improvement", "flat"}):
        verdict = "monotonic_improvement"
    elif (det_trend == "flat" and mae_trend == "flat"
          and fa_trend == "flat"):
        verdict = "no_improvement"
    elif (det_trend in {"monotonic_degradation"}
          or mae_trend in {"monotonic_degradation"}):
        verdict = "degradation"
    elif det_trend == "monotonic_improvement" or mae_trend == "monotonic_improvement":
        verdict = "partial_improvement"
    elif det_trend in {"flat"} and mae_trend in {"flat"}:
        verdict = "no_improvement"
    else:
        verdict = "mixed"

    return {
        "detected_days_trend": det_trend,
        "false_alarm_trend": fa_trend,
        "mean_absolute_error_trend": mae_trend,
        "verdict": verdict,
        "findings": findings,
        "candidate_factors": [
            "redundant sensors among the top of the D_i ranking",
            "noisy or unreliable individual sensors",
            "missing observations on some days",
            "averaging z-scores reduces variance but also dilutes a single "
            "strong signal",
            "fixed CUSUM threshold (h) shared across regimes",
            "the top-ranked sensor already captures most of the signal",
            "weather variability (clouds, fog) on specific days",
            "different sensor availability patterns across the deployment",
        ],
    }


def build_comparison_payload() -> dict:
    """Load all scenario JSONs and assemble the comparison payload."""
    scenarios: list[dict] = []
    raw_payloads: dict[str, dict] = {}
    for scenario, path in _SCENARIOS:
        payload = _load(path)
        if payload is None:
            logger.warning("Scenario %r results missing at %s; skipping.",
                           scenario, path)
            continue
        raw_payloads[scenario] = payload
        scenarios.append(_summary_row(scenario, payload))

    interpretation = _interpret(scenarios) if scenarios else {}
    detector_signature = None
    if raw_payloads:
        ref = next(iter(raw_payloads.values()))
        detector_signature = {
            "detector_mode": ref.get("detector_mode"),
            "detector_name": ref.get("detector_name"),
            "evidence_type": ref.get("evidence_type"),
            "parameter_source": ref.get("parameter_source"),
            "global_parameter_file": ref.get("global_parameter_file"),
            "detector_parameters": ref.get("detector_parameters"),
            "threshold": ref.get("threshold"),
            "tolerance_minutes": ref.get("tolerance_minutes"),
            "analysis_window": ref.get("analysis_window"),
            "aggregation": ref.get("aggregation"),
            "consistent_across_scenarios": all(
                p.get("detector_name") == ref.get("detector_name")
                and p.get("detector_mode") == ref.get("detector_mode")
                and p.get("threshold") == ref.get("threshold")
                and p.get("detector_parameters") == ref.get("detector_parameters")
                and p.get("analysis_window") == ref.get("analysis_window")
                and p.get("tolerance_minutes") == ref.get("tolerance_minutes")
                for p in raw_payloads.values()
            ),
        }
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenarios": scenarios,
        "detector_signature": detector_signature,
        "interpretation": interpretation,
    }


def write_comparison_payload(payload: dict,
                             out_path: Path = COMPARISON_JSON) -> Path:
    paths.ensure_dirs()
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
    logger.info("Comparison JSON written to %s", out_path)
    return out_path


def _table(headers: list[str], rows: list[list]) -> str:
    def cell(v):
        if v is None:
            return "<em>n/a</em>"
        if isinstance(v, float):
            return f"{v:.3f}"
        if isinstance(v, list):
            return escape(", ".join(str(x) for x in v))
        return escape(str(v))

    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell(c)}</td>" for c in r) + "</tr>"
        for r in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


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
code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }
.meta { color: #666; font-size: 0.9em; }
"""


def render_comparison_report(out_path: Path = COMPARISON_HTML,
                             json_path: Path = COMPARISON_JSON) -> Path:
    """Render the combined comparison HTML report from the JSON payload."""
    payload = _load(json_path)
    if payload is None:
        raise FileNotFoundError(
            f"Comparison JSON missing: {json_path}. Run "
            "`generate-budget-comparison-report` after the per-scenario "
            "experiments."
        )

    scenarios = payload.get("scenarios", [])
    sig = payload.get("detector_signature") or {}
    interp = payload.get("interpretation") or {}

    sections: list[str] = []
    sections.append(
        f"<section><h2>1. Sunrise CPD — Combined Comparison</h2>"
        f"<div class='statement'>"
        "The available system is always the same multi-sensor network. "
        "The budget regime changes only the subset of sensors selected "
        "by the sampling policy."
        f"</div>"
        f"<div class='statement'>"
        "The goal is not to exactly reproduce the theoretical Gaussian "
        "model, but to test whether the same resource-constrained "
        "change-point detection logic appears in real sensor network "
        "data."
        f"</div>"
        "</section>"
    )

    # Detector signature.
    sig_rows = [
        ["Detector mode", sig.get("detector_mode")],
        ["Detector name", sig.get("detector_name")],
        ["Evidence type", sig.get("evidence_type")],
        ["Aggregation", sig.get("aggregation")],
        ["Parameter source", sig.get("parameter_source")],
        ["Global parameter file", sig.get("global_parameter_file") or "n/a"],
        ["Threshold h", sig.get("threshold")],
        ["Tolerance (min)", sig.get("tolerance_minutes")],
        ["Pre window (min)",
         (sig.get("analysis_window") or {}).get("pre_window_minutes")],
        ["Post window (min)",
         (sig.get("analysis_window") or {}).get("post_window_minutes")],
        ["Sync freq",
         (sig.get("analysis_window") or {}).get("sync_freq")],
        ["Consistent across scenarios",
         sig.get("consistent_across_scenarios")],
    ]
    sections.append(
        "<section><h2>2. Detector — Identical Across Scenarios</h2>"
        + _table(["Field", "Value"], sig_rows)
        + "</section>"
    )

    # Comparison table.
    headers = [
        "Scenario", "Budget regime", "Budget B", "Cost model",
        "Candidate sensors", "Sensors ever selected",
        "Most frequently selected",
    ] + [label for _, label in _METRIC_FIELDS]
    rows = []
    for s in scenarios:
        row = [
            s.get("scenario_name"),
            s.get("budget_regime"),
            s.get("budget_value"),
            s.get("cost_model"),
            s.get("full_candidate_sensors"),
            s.get("sensors_ever_selected"),
            s.get("most_frequently_selected_sensors"),
        ]
        for key, _ in _METRIC_FIELDS:
            row.append(s.get(key))
        rows.append(row)
    sections.append(
        "<section><h2>3. Side-by-Side Metrics</h2>"
        + _table(headers, rows)
        + "</section>"
    )

    # Interpretation.
    findings_html = ""
    if interp.get("findings"):
        findings_html = "<ul>" + "".join(
            f"<li>{escape(f)}</li>" for f in interp["findings"]
        ) + "</ul>"
    factors_html = "<ul>" + "".join(
        f"<li>{escape(f)}</li>" for f in interp.get("candidate_factors", [])
    ) + "</ul>"

    sections.append(
        "<section><h2>4. Does More Budget Improve Detection?</h2>"
        f"<p><strong>Verdict:</strong> "
        f"<code>{escape(str(interp.get('verdict', 'unknown')))}</code></p>"
        + _table(["Trend", "Direction"], [
            ["Detected days", interp.get("detected_days_trend")],
            ["False alarms", interp.get("false_alarm_trend")],
            ["Mean absolute error",
             interp.get("mean_absolute_error_trend")],
        ])
        + "<h3>Findings</h3>" + (findings_html or "<p>No findings.</p>")
        + "<h3>Possible reasons (non-prescriptive)</h3>" + factors_html
        + "</section>"
    )

    sections.append(
        "<section><h2>5. Experimental Scope</h2>"
        "<ul>"
        "<li>The empirical evaluation uses solar radiation from the "
        "SensorScope Grand-St-Bernard deployment.</li>"
        "<li>Astronomical sunrise is used as external ground truth.</li>"
        "<li>Detector parameters and the analysis window are held fixed "
        "across budget regimes to isolate the effect of the sensor "
        "budget.</li>"
        "<li>Sensing and communication costs C_i, T_i are not measured "
        "in the dataset; the explicit unit-cost assumption "
        "C_i = 1.0, T_i = 0.0 is recorded in "
        "<code>output/json/sensor_costs.json</code>.</li>"
        "<li>The aggregation rule (mean of finite per-sensor LLRs over "
        "the dynamic subset S(t)) treats all active sensors symmetrically; "
        "weighted aggregation by D_i is not part of the present "
        "configuration.</li>"
        "<li>The evaluation addresses the non-adversarial "
        "resource-constrained setting; the covert adversarial model is "
        "not analyzed.</li>"
        "<li>Sunset detection is not considered.</li>"
        "</ul></section>"
    )

    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Sunrise CPD — Budget Comparison</title>"
        f"<style>{CSS}</style></head><body>"
        "<h1>Sunrise CPD — Budget Comparison Report</h1>"
        f"<p class='meta'>Generated: "
        f"{escape(str(payload.get('generated_at_utc')))}</p>"
        + "".join(sections)
        + "</body></html>"
    )
    paths.ensure_dirs()
    out_path.write_text(html, encoding="utf-8")
    logger.info("Comparison report written to %s", out_path)
    return out_path
