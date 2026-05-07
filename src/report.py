"""Generate the dataset HTML report from JSON outputs."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from . import paths

logger = logging.getLogger(__name__)


REPORT_STATEMENT = (
    "This dataset report characterizes the SensorScope deployment used "
    "in the empirical evaluation of a sunrise-based change-point "
    "detection experiment, summarizing its temporal coverage, spatial "
    "layout, and light-related observations."
)


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _fmt(v: Any) -> str:
    if v is None:
        return "<em>n/a</em>"
    if isinstance(v, float):
        return f"{v:.4f}"
    return escape(str(v))


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_fmt(c)}</td>" for c in r) + "</tr>"
        for r in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _section(title: str, body_html: str) -> str:
    return f"<section><h2>{escape(title)}</h2>{body_html}</section>"


# ---------------------------------------------------------------------------
# Suitability assessment
# ---------------------------------------------------------------------------

def assess_suitability(inv: dict | None,
                       var: dict | None,
                       loc: dict | None,
                       gt: dict | None,
                       prep: dict | None,
                       info: dict | None,
                       k_medium: int = 3,
                       n_high: int = 5,
                       min_valid_days: int = 5
                       ) -> dict:
    """Apply the suitability checklist defined in TASKS.md section 2.8."""
    checks: list[dict] = []

    light_ok = bool(var and var.get("selected_variable"))
    checks.append({
        "name": "light_variable_available",
        "passed": light_ok,
        "detail": (var or {}).get("selected_variable") or "missing",
    })

    ts_ok = bool(prep and prep.get("timestamp_strategy"))
    checks.append({
        "name": "timestamps_parseable",
        "passed": ts_ok,
        "detail": (prep or {}).get("timestamp_strategy") or "unknown",
    })

    tz_ok = bool(loc and loc.get("timezone"))
    checks.append({
        "name": "timezone_resolved",
        "passed": tz_ok,
        "detail": (loc or {}).get("timezone") or "missing",
    })

    coords_ok = bool(loc and loc.get("latitude") is not None
                     and loc.get("longitude") is not None)
    checks.append({
        "name": "coordinates_known",
        "passed": coords_ok,
        "detail": (
            f"lat={loc.get('latitude')}, lon={loc.get('longitude')}"
            if loc else "missing"
        ),
    })

    sr_ok = bool(gt and gt.get("records"))
    checks.append({
        "name": "sunrise_ground_truth_computable",
        "passed": sr_ok,
        "detail": f"{len(gt['records']) if sr_ok else 0} dates",
    })

    n_valid_sensors = (prep or {}).get("n_sensors_after_filtering") or 0
    n_valid_days = (prep or {}).get("n_valid_days") or 0
    checks.append({
        "name": "low_budget_one_sensor",
        "passed": n_valid_sensors >= 1,
        "detail": f"{n_valid_sensors} valid sensors",
    })
    checks.append({
        "name": f"medium_budget_k_{k_medium}_sensors",
        "passed": n_valid_sensors >= k_medium,
        "detail": f"{n_valid_sensors} valid sensors",
    })
    checks.append({
        "name": f"high_budget_at_least_{n_high}_sensors",
        "passed": n_valid_sensors >= n_high,
        "detail": f"{n_valid_sensors} valid sensors",
    })
    checks.append({
        "name": f"enough_valid_days_>={min_valid_days}",
        "passed": n_valid_days >= min_valid_days,
        "detail": f"{n_valid_days} valid days",
    })

    # Missing-data sanity check: at least one sensor with informativeness data.
    info_ok = bool(info and any(
        s.get("D_i") is not None for s in info.get("sensors", [])))
    checks.append({
        "name": "sunrise_windows_usable",
        "passed": info_ok,
        "detail": (
            f"{sum(1 for s in info.get('sensors', []) if s.get('D_i') is not None)}"
            " sensors have D_i" if info else "informativeness not computed"
        ),
    })

    n_failed = sum(1 for c in checks if not c["passed"])
    core_failed = any(
        not c["passed"] for c in checks
        if c["name"] in ("light_variable_available", "timestamps_parseable",
                         "timezone_resolved", "coordinates_known",
                         "sunrise_ground_truth_computable",
                         "low_budget_one_sensor")
    )
    if core_failed:
        status = "not_suitable"
    elif n_failed == 0:
        status = "suitable"
    elif n_failed <= 3:
        status = "partially_suitable"
    else:
        status = "undetermined"

    explanation = (
        f"{len(checks) - n_failed}/{len(checks)} suitability checks passed."
    )
    risks: list[str] = []
    if n_valid_sensors < n_high:
        risks.append(
            f"Fewer than {n_high} sensors available for the high-budget regime.")
    if n_valid_days < min_valid_days:
        risks.append(
            f"Fewer than {min_valid_days} valid days for stable estimates.")
    if not info_ok:
        risks.append("Sensor informativeness could not be estimated.")

    if status == "suitable":
        next_step = "Dataset meets the criteria for the experimental configuration."
    elif status == "partially_suitable":
        next_step = (
            "Dataset is usable; the listed methodological notes apply to "
            "the empirical evaluation."
        )
    elif status == "not_suitable":
        next_step = (
            "Dataset does not meet the criteria required by the "
            "experimental configuration."
        )
    else:
        next_step = "Suitability could not be determined from the available outputs."

    return {
        "status": status,
        "explanation": explanation,
        "checks": checks,
        "risks": risks,
        "next_step": next_step,
    }


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

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
.status { display: inline-block; padding: 4px 12px; border-radius: 4px;
          font-weight: 600; }
.status.suitable { background: #d3f9d8; color: #1b5e20; }
.status.partially_suitable { background: #fff3bf; color: #8d6708; }
.status.not_suitable { background: #ffd6d6; color: #7a1f1f; }
.status.undetermined { background: #e0e0e0; color: #333; }
.pass { color: #1b5e20; font-weight: 600; }
.fail { color: #7a1f1f; font-weight: 600; }
code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }
.meta { color: #666; font-size: 0.9em; }
"""


def render_report(out_path: Path = paths.DATASET_REPORT_HTML) -> Path:
    """Render the dataset HTML report from the JSON outputs."""
    inv = _load(paths.DATASET_INVENTORY_JSON)
    var = _load(paths.VARIABLE_SELECTION_JSON)
    loc = _load(paths.LOCATION_METADATA_JSON)
    gt = _load(paths.SUNRISE_GROUND_TRUTH_JSON)
    prep = _load(paths.PREPROCESSING_SUMMARY_JSON)
    info = _load(paths.SENSOR_INFORMATIVENESS_JSON)

    suitability = assess_suitability(inv, var, loc, gt, prep, info)

    sections: list[str] = []

    # 1. Dataset Overview
    n_inspected_files = sum(
        d.get("file_count", 0) for d in (inv or {}).get("deployments", [])
    )
    n_candidate_files = sum(
        d.get("data_file_count", 0) for d in (inv or {}).get("deployments", [])
    )
    rows = [
        ["Dataset", (inv or {}).get("dataset_name")],
        ["Source", (inv or {}).get("source_url")],
        ["Citation", (inv or {}).get("citation")],
        ["Selected deployment", (inv or {}).get("selected_deployment")],
        ["Local raw data path", (inv or {}).get("local_raw_path")],
        ["GSB meteo dir", (inv or {}).get("gsb_meteo_dir")],
        ["Timestamp range",
         (prep or {}).get("timestamp_range")],
        ["Files inspected", n_inspected_files],
        ["Candidate data files", n_candidate_files],
    ]
    sections.append(_section("1. Dataset Overview", _table(["Field", "Value"], rows)))

    # 2. Deployment and Location
    if loc:
        rows = [[k, v] for k, v in loc.items()]
        sections.append(_section("2. Deployment and Location",
                                 _table(["Field", "Value"], rows)))
    else:
        sections.append(_section(
            "2. Deployment and Location",
            "<p><em>location_metadata.json not available.</em></p>"))

    # 3. Available Variables
    if var:
        rows = [
            ["Selected variable", var.get("selected_variable")],
            ["Units", var.get("units")],
            ["Reason", var.get("reason")],
            ["Candidate variables",
             ", ".join(var.get("candidate_variables") or []) or "n/a"],
            ["Sensors with this variable", var.get("n_sensors_with_variable")],
            ["Valid observations", var.get("n_valid_observations")],
            ["Notes",
             "; ".join(var.get("notes") or []) or "n/a"],
        ]
        sections.append(_section("3. Available Variables",
                                 _table(["Field", "Value"], rows)))
    else:
        sections.append(_section("3. Available Variables",
                                 "<p><em>variable_selection.json not available.</em></p>"))

    # 4. Timestamp & Sampling Structure
    if prep:
        rows = [
            ["Timestamp strategy", prep.get("timestamp_strategy")],
            ["Timezone", prep.get("deployment_timezone")],
            ["Sampling interval estimate (s)",
             prep.get("sampling_interval_seconds_estimate")],
            ["Synchronized matrix freq",
             prep.get("synchronized_matrix_freq")],
            ["Synchronized matrix shape",
             prep.get("synchronized_matrix_shape")],
            ["Timestamp range start (UTC)",
             (prep.get("timestamp_range") or {}).get("start_utc")],
            ["Timestamp range end (UTC)",
             (prep.get("timestamp_range") or {}).get("end_utc")],
        ]
        sections.append(_section("4. Timestamp and Sampling Structure",
                                 _table(["Field", "Value"], rows)))

    # 5. Sensor Availability
    if prep:
        miss = prep.get("missing_rate_by_sensor") or {}
        miss_rows = sorted(miss.items(), key=lambda kv: kv[1])
        miss_table = _table(
            ["Sensor", "Missing rate"],
            [[s, m] for s, m in miss_rows],
        )
        rows = [
            ["Sensors before filtering", prep.get("n_sensors_before_filtering")],
            ["Sensors after filtering", prep.get("n_sensors_after_filtering")],
            ["Valid sensors",
             ", ".join(prep.get("valid_sensors") or []) or "n/a"],
        ]
        sections.append(_section(
            "5. Sensor Availability",
            _table(["Field", "Value"], rows) + "<h3>Missing-data rate by sensor</h3>" + miss_table))

    # 6. Valid Days
    if prep:
        valid_days = prep.get("valid_days") or []
        rows = [
            ["Days before filtering", prep.get("n_days_before_filtering")],
            ["Valid days", prep.get("n_valid_days")],
            ["First valid day", valid_days[0] if valid_days else None],
            ["Last valid day", valid_days[-1] if valid_days else None],
        ]
        sections.append(_section("6. Valid Days",
                                 _table(["Field", "Value"], rows)))

    # 7. Sunrise Ground Truth
    if gt:
        records = gt.get("records") or []
        sample_rows = [
            [r["date"], r["sunrise_local"], r["sunrise_utc"]]
            for r in records[:10]
        ]
        head = _table(
            ["Date", "Sunrise (local)", "Sunrise (UTC)"],
            sample_rows,
        )
        meta_rows = [
            ["Method", gt.get("method")],
            ["Latitude", gt.get("latitude")],
            ["Longitude", gt.get("longitude")],
            ["Timezone", gt.get("timezone")],
            ["Number of dates", gt.get("n_dates")],
            ["Date range", gt.get("date_range")],
        ]
        sections.append(_section(
            "7. Sunrise Ground Truth",
            _table(["Field", "Value"], meta_rows)
            + "<h3>Sample sunrise times</h3>" + head))

    # 8. Preprocessing Summary
    if prep:
        rows = [
            ["Raw rows", prep.get("raw_row_count")],
            ["Cleaned rows", prep.get("cleaned_row_count")],
            ["Selected variable", prep.get("selected_variable")],
            ["Exclusions applied", json.dumps(prep.get("exclusions_applied"))],
            ["Notes", "; ".join(prep.get("notes") or []) or "n/a"],
        ]
        sections.append(_section("8. Preprocessing Summary",
                                 _table(["Field", "Value"], rows)))

    # 9. Sensor Informativeness Preview
    if info:
        sensors = info.get("sensors") or []
        rows = [
            [s.get("rank_by_D"), s.get("sensor_id"), s.get("n_valid_days"),
             s.get("mu_0"), s.get("mu_1"), s.get("sigma2"), s.get("D_i"),
             s.get("missing_rate")]
            for s in sensors[:25]
        ]
        sections.append(_section(
            "9. Sensor Informativeness Preview",
            f"<p>Pre-window: {info.get('pre_window_minutes')} min, "
            f"post-window: {info.get('post_window_minutes')} min.</p>"
            + _table(
                ["Rank", "Sensor", "Valid days", "mu_0", "mu_1",
                 "sigma^2", "D_i", "Missing rate"],
                rows)))
    else:
        sections.append(_section(
            "9. Sensor Informativeness Preview",
            "<p><em>sensor_informativeness.json not available.</em></p>"))

    # 10. Suitability assessment.
    status = suitability["status"]
    # The check table contains pre-built spans; escape() would double-encode,
    # so render the table manually.
    raw_check_table = (
        "<table><thead><tr><th>Check</th><th>Result</th><th>Detail</th>"
        "</tr></thead><tbody>"
        + "".join(
            "<tr><td>" + escape(c["name"]) + "</td>"
            + "<td><span class='" + ("pass" if c["passed"] else "fail")
            + "'>" + ("PASS" if c["passed"] else "FAIL") + "</span></td>"
            + "<td>" + escape(str(c["detail"])) + "</td></tr>"
            for c in suitability["checks"]
        )
        + "</tbody></table>"
    )
    body = (
        f"<p>Final status: "
        f"<span class='status {status}'>{escape(status)}</span></p>"
        f"<p>{escape(suitability['explanation'])}</p>"
        + raw_check_table
        + "<h3>Methodological Notes</h3><ul>"
        + ("".join(f"<li>{escape(r)}</li>" for r in suitability['risks'])
           or "<li>None identified.</li>")
        + "</ul>"
    )
    sections.append(_section("10. Dataset Suitability Assessment", body))

    # Final HTML.
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>SensorScope Dataset Report — Sunrise CPD</title>"
        f"<style>{CSS}</style></head><body>"
        "<h1>SensorScope Dataset Report</h1>"
        f"<p class='meta'>Generated: {datetime.now(timezone.utc).isoformat()}</p>"
        f"<div class='statement'>{escape(REPORT_STATEMENT)}</div>"
        + "".join(sections)
        + "</body></html>"
    )

    paths.ensure_dirs()
    out_path.write_text(html, encoding="utf-8")
    # Also persist the suitability JSON next to the other JSON outputs.
    with open(paths.JSON_DIR / "dataset_suitability.json", "w",
              encoding="utf-8") as fh:
        json.dump(suitability, fh, indent=2, ensure_ascii=False)
    logger.info("Dataset report written to %s", out_path)
    return out_path
