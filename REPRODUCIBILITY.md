# Reproducibility

This document describes how to reproduce the empirical results and HTML
reports of the Sunrise CPD project.

## Dataset

* Source: SensorScope environmental wireless sensor network (EPFL).
* Record: <https://zenodo.org/records/2654726>
* Deployment used in this experiment: **Grand-St-Bernard**.
* Observed variable: **Solar Radiation [W/m²]**.

The raw archive (`Sensorscope.zip`) and its extracted contents are
**not** tracked by Git. The expected local layout after acquisition is:

```
data/
├── external/
│   └── stbernard_meteo_columns.txt
├── raw/
│   └── sensorscope/
│       ├── Sensorscope.zip
│       └── extracted/
│           └── Sensorscope/stbernard/...
└── processed/
    ├── synchronized_sensor_data.parquet
    ├── synchronized_sensor_data.csv
    └── sensor_matrix.parquet
```

## Environment

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.12.

## Pipeline

Run the full pipeline:

```
python -m src.cli run-all
```

Or run stage by stage:

```
python -m src.cli acquire-data
python -m src.cli inspect-dataset
python -m src.cli preprocess-data
python -m src.cli build-ground-truth
python -m src.cli rank-sensors
python -m src.cli generate-dataset-report
python -m src.cli run-sunrise-experiment --scenario low
python -m src.cli run-sunrise-experiment --scenario medium --k 3
python -m src.cli run-sunrise-experiment --scenario high
python -m src.cli generate-report --scenario low
python -m src.cli generate-report --scenario medium
python -m src.cli generate-report --scenario high
python -m src.cli generate-budget-comparison-report
```

## Expected outputs

JSON summaries (under `output/json/`):

* `dataset_inventory.json`
* `variable_selection.json`
* `location_metadata.json`
* `preprocessing_summary.json`
* `sunrise_ground_truth.json`
* `sensor_informativeness.json`
* `dataset_suitability.json`
* `sunrise_low_budget_results.json`
* `sunrise_medium_budget_results.json`
* `sunrise_high_budget_results.json`
* `sunrise_budget_comparison.json`

HTML reports (under `output/reports/`):

* `dataset_report.html`
* `sunrise_low_budget_report.html`
* `sunrise_medium_budget_report.html`
* `sunrise_high_budget_report.html`
* `sunrise_budget_comparison_report.html`

Example detection plots are written to `output/reports/assets/`.
