"""CLI entry point for the Sunrise CPD data-acquisition / preprocessing phase.

Subcommands:

* ``acquire-data``: download SensorScope, extract sub-archives, write
  dataset inventory.
* ``inspect-dataset``: parse column definitions, locate stations, write
  variable selection JSON.
* ``preprocess-data``: load the Grand St. Bernard meteo data and produce
  cleaned long/wide tables and a preprocessing summary.
* ``build-ground-truth``: compute sunrise (and sunset) times for the
  valid days from the preprocessing summary.
* ``rank-sensors``: estimate per-sensor ``D_i`` around sunrise.
* ``generate-dataset-report``: render ``output/reports/dataset_report.html``.
* ``run-all``: run the full data pipeline end-to-end.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import paths
from .acquisition import (
    acquire,
    build_inventory,
    get_gsb_meteo_dir,
    get_gsb_location_dir,
    write_inventory,
)
from .ground_truth import build_ground_truth_for_dates, write_ground_truth
from .informativeness import (
    compute_sensor_informativeness,
    write_informativeness,
)
from .inspection import (
    inspect_deployment,
    select_best_light_variable,
)
from .location import build_location_metadata, write_location_metadata
from .preprocessing import preprocess, save_outputs
from .report import render_report
from .experiments import (
    LOW_BUDGET_RESULTS_JSON,
    _REGIME_OUTPUTS,
    run_experiment,
    write_results,
)
from .experiment_report import (
    EXPERIMENT_REPORT_HTML,
    LOW_BUDGET_REPORT_HTML,
    MEDIUM_BUDGET_REPORT_HTML,
    HIGH_BUDGET_REPORT_HTML,
    render_experiment_report,
    render_scenario_report,
)
from .comparison import (
    COMPARISON_HTML,
    COMPARISON_JSON,
    build_comparison_payload,
    render_comparison_report,
    write_comparison_payload,
)


LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"

_SCENARIO_HTML_PATH = {
    "low": LOW_BUDGET_REPORT_HTML,
    "medium": MEDIUM_BUDGET_REPORT_HTML,
    "high": HIGH_BUDGET_REPORT_HTML,
}


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FORMAT,
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Helpers shared across subcommands
# ---------------------------------------------------------------------------

EXTERNAL_COLUMN_FILE = (
    paths.EXTERNAL_DIR / "stbernard_meteo_columns.txt"
)


def _inspect_gsb():
    meteo_dir = get_gsb_meteo_dir()
    if meteo_dir is None:
        raise RuntimeError(
            "Grand St. Bernard meteo directory not found. Run "
            "'acquire-data' first."
        )
    fallback = [paths.EXTERNAL_DIR]
    return meteo_dir, inspect_deployment(
        meteo_dir,
        deployment_name=paths.GSB_DEPLOYMENT_NAME,
        fallback_metadata_dirs=fallback,
    )


def _write_variable_selection(insp, n_sensors: int,
                              n_observations: int | None = None) -> dict:
    selected, reason = select_best_light_variable(insp)
    units = None
    if selected:
        # Try to extract bracketed units from the column-definition entry.
        for cd in insp.column_definitions:
            if cd.name == selected:
                if "[" in cd.name and "]" in cd.name:
                    units = cd.name.split("[", 1)[1].rstrip("]").strip()
                # Inspect the stored raw line for a unit hint.
                line = cd.raw_line
                if "[" in line and "]" in line:
                    units = line.split("[", 1)[1].split("]", 1)[0].strip()
                break
    payload = {
        "selected_variable": selected,
        "reason": reason,
        "candidate_variables": [n for n, _ in insp.candidate_light_columns],
        "candidate_scores": [
            {"name": n, "score": s} for n, s in insp.candidate_light_columns
        ],
        "units": units,
        "deployment": insp.deployment_name,
        "n_sensors_with_variable": n_sensors,
        "n_valid_observations": n_observations,
        "notes": insp.notes,
    }
    paths.ensure_dirs()
    with open(paths.VARIABLE_SELECTION_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return payload


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------

def cmd_acquire_data(args: argparse.Namespace) -> int:
    print("[acquire-data] Downloading and organizing the SensorScope dataset…")
    inventory = acquire(force_download=args.force_download,
                        force_extract=args.force_extract)
    print(f"[acquire-data] Inventory written to {paths.DATASET_INVENTORY_JSON}")
    print(f"[acquire-data] Selected deployment: "
          f"{inventory.get('selected_deployment')}")
    print(f"[acquire-data] GSB meteo dir: {inventory.get('gsb_meteo_dir')}")
    return 0


def cmd_inspect_dataset(args: argparse.Namespace) -> int:
    print("[inspect-dataset] Inspecting Grand St. Bernard deployment…")
    meteo_dir, insp = _inspect_gsb()
    print(f"[inspect-dataset] Stations detected: {len(insp.station_files)}")
    print(f"[inspect-dataset] Column-definition file: "
          f"{insp.column_definition_file}")
    print(f"[inspect-dataset] Light candidates: "
          f"{insp.candidate_light_columns}")
    payload = _write_variable_selection(insp, n_sensors=len(insp.station_files))
    print(f"[inspect-dataset] Selected variable: "
          f"{payload['selected_variable']}")
    print(f"[inspect-dataset] Variable selection written to "
          f"{paths.VARIABLE_SELECTION_JSON}")

    # Refresh the inventory so it reflects post-extraction state.
    write_inventory(build_inventory())
    return 0


def cmd_preprocess_data(args: argparse.Namespace) -> int:
    print("[preprocess-data] Loading and synchronizing meteo data…")
    meteo_dir, insp = _inspect_gsb()
    selected, _ = select_best_light_variable(insp)
    if not selected:
        print("[preprocess-data] No light-related variable available; "
              "writing empty preprocessing summary.", file=sys.stderr)
        empty = {
            "selected_variable": None,
            "deployment_name": insp.deployment_name,
            "deployment_timezone": paths.GSB_TIMEZONE,
            "raw_row_count": 0, "cleaned_row_count": 0,
            "n_sensors_before_filtering": 0,
            "n_sensors_after_filtering": 0,
            "n_days_before_filtering": 0, "n_valid_days": 0,
            "valid_days": [], "valid_sensors": [],
            "missing_rate_by_sensor": {},
            "synchronized_matrix_shape": [0, 0],
            "synchronized_matrix_freq": args.sync_freq,
            "sampling_interval_seconds_estimate": None,
            "timestamp_range": {"start_utc": None, "end_utc": None},
            "exclusions_applied": {},
            "timestamp_strategy": None,
            "notes": ["No light-related variable selected."],
        }
        with open(paths.PREPROCESSING_SUMMARY_JSON, "w", encoding="utf-8") as fh:
            json.dump(empty, fh, indent=2, ensure_ascii=False)
        return 1

    result = preprocess(
        insp,
        variable_name=selected,
        deployment_timezone=paths.GSB_TIMEZONE,
        sync_freq=args.sync_freq,
        max_sensor_missing_rate=args.max_missing_rate,
    )
    save_outputs(result)
    print(f"[preprocess-data] Cleaned rows: {result.summary['cleaned_row_count']}")
    print(f"[preprocess-data] Valid sensors: "
          f"{result.summary['n_sensors_after_filtering']}")
    print(f"[preprocess-data] Valid days: {result.summary['n_valid_days']}")
    print(f"[preprocess-data] Synchronized matrix shape: "
          f"{result.summary['synchronized_matrix_shape']}")

    # Update variable selection with observation counts.
    _write_variable_selection(
        insp,
        n_sensors=result.summary["n_sensors_after_filtering"],
        n_observations=result.summary["cleaned_row_count"],
    )

    # Always (re)write location metadata.
    loc_dir = get_gsb_location_dir()
    coord_file = None
    if loc_dir is not None:
        cand = loc_dir / "station_gsb_XY.txt"
        if cand.exists():
            coord_file = cand
    write_location_metadata(build_location_metadata(coord_file))
    return 0


def cmd_build_ground_truth(args: argparse.Namespace) -> int:
    print("[build-ground-truth] Computing sunrise times…")
    if not paths.PREPROCESSING_SUMMARY_JSON.exists():
        print("[build-ground-truth] preprocessing_summary.json missing; "
              "run 'preprocess-data' first.", file=sys.stderr)
        return 1
    with open(paths.PREPROCESSING_SUMMARY_JSON, "r", encoding="utf-8") as fh:
        prep = json.load(fh)
    valid_days = prep.get("valid_days") or []
    loc_payload = build_location_metadata()
    write_location_metadata(loc_payload)
    payload = build_ground_truth_for_dates(
        valid_days,
        latitude=loc_payload["latitude"],
        longitude=loc_payload["longitude"],
        timezone_name=loc_payload["timezone"],
    )
    write_ground_truth(payload)
    print(f"[build-ground-truth] Sunrise records: "
          f"{len(payload.get('records', []))}")
    return 0


def cmd_rank_sensors(args: argparse.Namespace) -> int:
    print("[rank-sensors] Estimating per-sensor informativeness…")
    if not paths.SYNCHRONIZED_PARQUET.exists():
        print("[rank-sensors] processed parquet missing; run preprocess first.",
              file=sys.stderr)
        return 1
    if not paths.SUNRISE_GROUND_TRUTH_JSON.exists():
        print("[rank-sensors] sunrise_ground_truth.json missing; "
              "run build-ground-truth first.", file=sys.stderr)
        return 1
    import pandas as pd
    long_df = pd.read_parquet(paths.SYNCHRONIZED_PARQUET)
    long_df["timestamp_utc"] = pd.to_datetime(long_df["timestamp_utc"],
                                              utc=True)
    with open(paths.SUNRISE_GROUND_TRUTH_JSON, "r", encoding="utf-8") as fh:
        gt = json.load(fh)
    payload = compute_sensor_informativeness(
        long_df, gt.get("records", []),
        pre_window_minutes=args.pre_window,
        post_window_minutes=args.post_window,
    )
    write_informativeness(payload)
    print(f"[rank-sensors] Sensors ranked: {payload.get('n_sensors')}")
    return 0


def cmd_generate_report(args: argparse.Namespace) -> int:
    print("[generate-dataset-report] Rendering HTML report…")
    out = render_report()
    print(f"[generate-dataset-report] Report written to {out}")
    return 0


def cmd_run_sunrise_experiment(args: argparse.Namespace) -> int:
    scenario = args.scenario.lower()
    if scenario not in ("low", "medium", "high"):
        print(f"[run-sunrise-experiment] Unknown scenario '{scenario}'.",
              file=sys.stderr)
        return 2
    print(f"[run-sunrise-experiment] Unified multi-sensor experiment")
    print(f"[run-sunrise-experiment] Budget regime: {scenario}")
    print(f"[run-sunrise-experiment] Threshold h={args.threshold}, "
          f"drift k={args.drift_k}, tolerance={args.tolerance}min, "
          f"window=[-{args.pre_window}min, +{args.post_window}min]")
    payload = run_experiment(
        regime=scenario,
        k=args.k,
        pre_window_minutes=args.pre_window,
        post_window_minutes=args.post_window,
        tolerance_minutes=args.tolerance,
        threshold=args.threshold,
        drift_k=args.drift_k,
    )
    out_path = write_results(payload)
    agg = payload["aggregate_metrics"]
    print(f"[run-sunrise-experiment] All valid sensors: "
          f"{payload['all_valid_sensors']}")
    print(f"[run-sunrise-experiment] Selection policy: "
          f"{payload['selection_policy']}")
    print(f"[run-sunrise-experiment] Selected sensors: "
          f"{payload['selected_sensors']}")
    print(f"[run-sunrise-experiment] Selection reason: "
          f"{payload['selection_reason']}")
    print(f"[run-sunrise-experiment] Days processed: {payload['number_of_days']}")
    print(f"[run-sunrise-experiment] Detected: {agg['detected_days_count']}, "
          f"missed: {agg['missed_detection_count']}, "
          f"false alarms: {agg['false_alarm_count']}")
    print(f"[run-sunrise-experiment] Mean signed delay: "
          f"{agg['mean_signed_delay_minutes']} min, "
          f"median: {agg['median_signed_delay_minutes']} min")
    print(f"[run-sunrise-experiment] JSON: {out_path}")
    print(f"[run-sunrise-experiment] HTML report (per-scenario): "
          f"{_SCENARIO_HTML_PATH[scenario]}")
    return 0


def cmd_generate_experiment_report(args: argparse.Namespace) -> int:
    scenario = (getattr(args, "scenario", None) or "low").lower()
    print(f"[generate-report] Rendering '{scenario}' scenario report…")
    out = render_scenario_report(scenario)
    print(f"[generate-report] Report written to {out}")
    if scenario == "low":
        legacy = render_experiment_report()
        print(f"[generate-report] Legacy alias written to {legacy}")
    return 0


def cmd_generate_budget_comparison_report(args: argparse.Namespace) -> int:
    print("[generate-budget-comparison-report] Building comparison payload…")
    payload = build_comparison_payload()
    json_out = write_comparison_payload(payload)
    html_out = render_comparison_report(json_path=json_out)
    print(f"[generate-budget-comparison-report] Scenarios compared: "
          f"{[s['budget_regime'] for s in payload['scenarios']]}")
    print(f"[generate-budget-comparison-report] Verdict: "
          f"{payload['interpretation'].get('verdict')}")
    print(f"[generate-budget-comparison-report] JSON: {json_out}")
    print(f"[generate-budget-comparison-report] HTML: {html_out}")
    return 0


def cmd_run_all(args: argparse.Namespace) -> int:
    rc = cmd_acquire_data(args)
    if rc:
        return rc
    rc = cmd_inspect_dataset(args)
    if rc:
        return rc
    rc = cmd_preprocess_data(args)
    if rc:
        return rc
    rc = cmd_build_ground_truth(args)
    if rc:
        return rc
    cmd_rank_sensors(args)  # Allowed to be partial.
    return cmd_generate_report(args)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Sunrise CPD — data acquisition & preprocessing CLI.",
    )
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Enable debug-level logging.")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("acquire-data", help="Download and organize SensorScope.")
    sp.add_argument("--force-download", action="store_true")
    sp.add_argument("--force-extract", action="store_true")
    sp.set_defaults(func=cmd_acquire_data)

    sp = sub.add_parser("inspect-dataset", help="Inspect deployment files.")
    sp.add_argument("--force-download", action="store_true")
    sp.add_argument("--force-extract", action="store_true")
    sp.set_defaults(func=cmd_inspect_dataset)

    sp = sub.add_parser("preprocess-data", help="Preprocess the GSB deployment.")
    sp.add_argument("--sync-freq", default="2min",
                    help="Synchronization cadence (pandas offset string).")
    sp.add_argument("--max-missing-rate", type=float, default=0.5)
    sp.set_defaults(func=cmd_preprocess_data)

    sp = sub.add_parser("build-ground-truth", help="Compute sunrise times.")
    sp.set_defaults(func=cmd_build_ground_truth)

    sp = sub.add_parser("rank-sensors",
                        help="Compute preliminary per-sensor informativeness.")
    sp.add_argument("--pre-window", type=int, default=120,
                    help="Pre-sunrise window in minutes.")
    sp.add_argument("--post-window", type=int, default=120,
                    help="Post-sunrise window in minutes.")
    sp.set_defaults(func=cmd_rank_sensors)

    sp = sub.add_parser("generate-dataset-report",
                        help="Render the dataset HTML report.")
    sp.set_defaults(func=cmd_generate_report)

    sp = sub.add_parser("run-sunrise-experiment",
                        help="Run a sunrise CPD budget regime.")
    sp.add_argument("--scenario", required=True,
                    choices=["low", "medium", "high"],
                    help="Budget regime.")
    sp.add_argument("--k", type=int, default=3,
                    help="Subset size for the medium-budget regime "
                         "(default: 3).")
    sp.add_argument("--threshold", type=float, default=5.0,
                    help="CUSUM detection threshold h (default: 5.0).")
    sp.add_argument("--drift-k", dest="drift_k", type=float, default=0.5,
                    help="CUSUM drift parameter k (default: 0.5).")
    sp.add_argument("--pre-window", type=int, default=120,
                    help="Pre-sunrise window in minutes (default: 120).")
    sp.add_argument("--post-window", type=int, default=120,
                    help="Post-sunrise window in minutes (default: 120).")
    sp.add_argument("--tolerance", type=int, default=15,
                    help="Tolerance window in minutes (default: 15).")
    sp.set_defaults(func=cmd_run_sunrise_experiment)

    sp = sub.add_parser("generate-report",
                        help="Render a sunrise experiment HTML report.")
    sp.add_argument("--scenario", choices=["low", "medium", "high"],
                    default="low",
                    help="Budget scenario to render (default: low).")
    sp.set_defaults(func=cmd_generate_experiment_report)

    sp = sub.add_parser("generate-budget-comparison-report",
                        help="Render the combined budget-comparison "
                             "report from per-scenario JSON outputs.")
    sp.set_defaults(func=cmd_generate_budget_comparison_report)

    sp = sub.add_parser("run-all", help="Run the full data pipeline.")
    sp.add_argument("--force-download", action="store_true")
    sp.add_argument("--force-extract", action="store_true")
    sp.add_argument("--sync-freq", default="2min")
    sp.add_argument("--max-missing-rate", type=float, default=0.5)
    sp.add_argument("--pre-window", type=int, default=120)
    sp.add_argument("--post-window", type=int, default=120)
    sp.set_defaults(func=cmd_run_all)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
