# Sunrise CPD

**Resource-Constrained Multi-Sensor Change-Point Detection on Sunrise Data**

This repository contains an empirical evaluation of a resource-constrained
multi-sensor change-point detection (CPD) framework. The framework is
applied to a real-world wireless sensor network and uses **astronomical
sunrise** as external ground truth for the change point.

## Connection to the paper

This work accompanies the paper

> *Covert Change-Point Detection with Resource-Constrained Multi-Sensor
> Sampling.*

The paper formulates the multi-sensor CPD problem under sensing and
communication constraints and analyzes the trade-off between sampling
budget and detection performance. This repository provides a
real-data instantiation of the **non-adversarial, resource-constrained**
setting of that framework.

## Dataset

* **Source:** SensorScope environmental wireless sensor network (EPFL).
* **Record:** <https://zenodo.org/records/2654726>
* **Deployment:** Grand-St-Bernard.
* **Observed variable:** Solar Radiation [W/m²].

The SensorScope archive is not redistributed in this repository. See
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for acquisition instructions.

## Experimental setting

* **Change point.** For each calendar day `d`, the true change point
  `nu_d` is the astronomical sunrise time at the deployment coordinates,
  computed with the `astral` library.
* **Observation.** Per-sensor, per-minute solar radiation, synchronized
  on a common time grid.
* **Per-sensor informativeness.** Gaussian divergence
  `D_i = (mu_1 - mu_0)^2 / (2 * sigma^2)`, where `mu_0` and `mu_1` are
  the pre- and post-sunrise means and `sigma^2` the pooled variance.
* **Detector.** Multi-sensor one-sided CUSUM that accumulates the
  Gaussian log-likelihood ratio
  `llr_{i,t} = ((x_{i,t} - mu_{0,i})^2 - (x_{i,t} - mu_{1,i})^2)
  / (2 * sigma_i^2)` averaged over the active sensors that report a
  finite reading, with no drift term. Per-sensor parameters
  `mu_{0,i}`, `mu_{1,i}`, `sigma_i^2` are fixed global empirical
  estimates obtained once per sensor from the dataset and stored in
  `output/json/sensor_informativeness.json`; they are reused unchanged
  for every test day. The astronomical sunrise of each day is used
  only to extract the evaluation window and to score detections, not
  to estimate detector parameters. Threshold `h` is held fixed across
  budget regimes. A diagnostic mode
  (`detector_mode="daily_baseline_zscore"`) reproduces the legacy
  same-day pre-sunrise z-score CUSUM.
* **Costs.** Sensing and communication costs `C_i`, `T_i` are not
  measured in the dataset. The pipeline applies an explicit unit-cost
  assumption (`C_i = 1.0`, `T_i = 0.0`, `total cost 1.0`) recorded in
  [output/json/sensor_costs.json](output/json/sensor_costs.json), so
  the information-per-cost score reduces to `D_i`.

## Budget regimes

The same multi-sensor system is used in all regimes. Each regime is
defined by an explicit numeric budget `B` over the per-sensor cost
`C_i + T_i`. At every timestamp `t`, the dynamic budget policy in
`src/budget.py` selects an active subset
`S(t) ⊆ {1, …, M}` greedily by `D_i / (C_i + T_i)` under the
constraint `Σ_{i ∈ S(t)} (C_i + T_i) ≤ B`. Selection is therefore
*per-timestamp*, not a fixed sensor subset; the active sensor(s) may
change over time depending on availability.

| Regime  | Budget `B`         | Typical `\|S(t)\|` (unit cost) |
|---------|--------------------|--------------------------------|
| Low     | `B = 1.0`          | up to 1 active sensor          |
| Medium  | `B = 3.0`          | up to 3 active sensors         |
| High    | `B = \|valid\|`    | all available sensors          |

## Summary of empirical results

Aggregated over 43 valid days, with `tolerance = 15 min` and threshold
`h = 5.0`, under the unit-cost assumption:

| Regime  | `B`  | Detected | Missed | False alarms |
|---------|------|----------|--------|--------------|
| Low     | 1.0  | 42       | 1      | 0            |
| Medium  | 3.0  | 42       | 1      | 0            |
| High    | 9.0  | 42       | 1      | 0            |

The dynamic policy distributes the low-budget activations over the
sensors that are actually available at each timestamp, rather than
locking onto a single sensor for the whole run; the union of sensors
ever selected at low budget covers the full valid sensor set across
the 43 days. See
[output/reports/sunrise_budget_comparison_report.html](output/reports/sunrise_budget_comparison_report.html)
for the full comparison.

## Repository structure

```
sunrise-cpd/
├── src/                         # Pipeline source code
│   ├── acquisition.py           # SensorScope download and inventory
│   ├── inspection.py            # Variable / station inspection
│   ├── preprocessing.py         # Synchronization and cleaning
│   ├── ground_truth.py          # Sunrise times via astral
│   ├── informativeness.py       # Per-sensor D_i estimation
│   ├── budget.py                # Budget-aware sensor selection
│   ├── detector.py              # Multi-sensor Gaussian LLR CUSUM
│   ├── experiments.py           # Per-regime experiment runner
│   ├── experiment_report.py     # Per-scenario HTML report
│   ├── comparison.py            # Cross-scenario comparison report
│   ├── report.py                # Dataset HTML report
│   ├── location.py              # Deployment coordinates
│   ├── paths.py                 # Centralized paths
│   └── cli.py                   # Command-line entry point
├── output/
│   ├── json/                    # JSON summaries
│   └── reports/                 # HTML reports and figure assets
├── data/                        # Local data (not tracked, see .gitignore)
├── requirements.txt
├── README.md
├── REPRODUCIBILITY.md
├── CITATION.cff
├── LICENSE
└── tasks.md                     # Project task notes
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.12.

## Obtaining the dataset

The raw SensorScope archive and its extracted contents are **not**
included in this repository. They are downloaded automatically by the
acquisition step:

```bash
python -m src.cli acquire-data
```

This populates `data/raw/sensorscope/` from
<https://zenodo.org/records/2654726>.

## Running the pipeline

```bash
python -m src.cli run-all
```

Or stage by stage; see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Regenerating reports

After the pipeline has been executed at least once and the JSON outputs
exist:

```bash
python -m src.cli generate-dataset-report
python -m src.cli generate-report --scenario low
python -m src.cli generate-report --scenario medium
python -m src.cli generate-report --scenario high
python -m src.cli generate-budget-comparison-report
```

The resulting HTML files are written under `output/reports/`.

## Citation

If you use this code or its results in academic work, please cite the
companion paper as well as this repository (see [CITATION.cff](CITATION.cff)).

For the dataset:

> SensorScope: EPFL environmental wireless sensor network.
> Zenodo record 2654726. <https://zenodo.org/records/2654726>

## License

Released under the [MIT License](LICENSE).
