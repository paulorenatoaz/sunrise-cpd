````markdown
# sunrise-cpd.md

# Real-Data Validation of Resource-Constrained Multi-Sensor Change-Point Detection

## 1. Project Overview

This project validates the first three resource-constrained multi-sensor change-point detection regimes described in the paper:

**Covert Change-Point Detection with Resource-Constrained Multi-Sensor Sampling**

The project uses real environmental sensor network data to study whether a detector can identify a real physical change point under different sensor-budget regimes.

The real-world change event is:

**sunrise**

For each valid day in the dataset:

- the true change point is the astronomically computed sunrise time;
- the detector observes light-related environmental sensor signals;
- the detected change point is the time when the detector raises an alarm;
- the detection error is computed by comparing the detected time with the true sunrise time.

The first version of the project covers three budget regimes of the same
multi-sensor model. The regimes differ only in the value of the available
budget; they are not separate models and they do not correspond to fixed
sensor subsets selected once at the start of the experiment. At each
timestamp `t`, the sampling policy selects an active subset
`S(t) ⊆ {1, ..., M}` of available sensors subject to the regime budget,
and the detector is computed only from the sensors in `S(t)`.

1. Low-budget regime: dynamic budget-constrained selection with a small
   budget. Under the unit-cost approximation used in the first version,
   the policy may activate at most one available sensor per timestamp.
2. Medium-budget regime: dynamic budget-constrained selection with an
   intermediate budget, allowing a limited subset of available sensors
   per timestamp.
3. High-budget regime: dynamic budget-constrained selection with enough
   budget to activate all available sensors at each timestamp.

The covert/adversarial part of the paper is outside the scope of this
first implementation.

---

## 2. Theoretical Reference

The project is based on the local PDF paper:

**Covert Change-Point Detection with Resource-Constrained Multi-Sensor Sampling**

The paper defines a multi-sensor quickest change-point detection model.

For each sensor \(i\), before the change:

\[
X_i(t) \sim \mathcal{N}(\mu_{0,i}, \sigma_i^2)
\]

After the change:

\[
X_i(t) \sim \mathcal{N}(\mu_{1,i}, \sigma_i^2)
\]

The true change point is:

\[
\nu
\]

The detector declares a change at stopping time:

\[
\tau
\]

The average detection delay is:

\[
ADD = \mathbb{E}[(\tau - \nu)^+]
\]

The Gaussian per-sensor divergence is:

\[
D_i =
\frac{(\mu_{1,i} - \mu_{0,i})^2}{2\sigma_i^2}
\]

When sensing and communication costs are available, sensors are ranked by:

\[
\frac{D_i}{C_i + T_i}
\]

### 2.1 Dynamic Budget-Constrained Sampling

At each timestamp `t`, the sampling policy selects an active subset:

\[
S(t) \subseteq \{1, \ldots, M\}
\]

subject to a budget constraint of the form:

\[
\sum_{i \in S(t)} (C_i + T_i) \;\leq\; B
\]

where:

* `C_i` is the sensing/acquisition cost of sensor `i`;
* `T_i` is the communication/transmission cost of sensor `i`;
* `B` is the budget available in the current regime.

The per-sensor selection score is information per cost:

\[
\text{score}_i \;=\; \frac{D_i}{C_i + T_i}
\]

When real costs are unavailable, the project must use an explicit
unit-cost model that is documented in the configuration and in the
result JSON, rather than implicitly dropping the cost terms. Under unit
costs, the budget reduces to a cardinality constraint, but selection
must still be performed dynamically at each timestamp using the current
availability of each sensor. Selecting a fixed subset once per scenario
is only acceptable as a preliminary diagnostic approximation and must
not be used as the final budget implementation.

---

## 3. Dataset

The intended dataset is the SensorScope / EPFL environmental monitoring dataset.

The preferred deployment is:

**Grand St. Bernard Pass**

Expected characteristics from prior documentation:

- deployment period: approximately 2007-09-13 to 2007-10-26;
- approximately 23 stations;
- high-mountain environmental monitoring deployment;
- environmental variables may include temperature, humidity, wind, rain, and solar-radiation-related measurements.

The first project phase must focus on data acquisition and inspection.

The project must verify:

- available deployments;
- available files;
- available variables;
- timestamp format;
- timezone information;
- station metadata;
- deployment coordinates;
- number of stations;
- sampling interval;
- missing-data structure;
- whether a light-related variable exists.

Preferred variables, in order:

1. solar radiation;
2. incoming solar radiation;
3. radiation;
4. light;
5. luminosity;
6. another equivalent sunlight-related variable.

Temperature may be considered only as a fallback or secondary variable, because the thermal response to sunrise is indirect and may have systematic delay.

---

## 4. Project Phases

The project is divided into four main phases:

1. Data acquisition.
2. Preprocessing.
3. Model and experiment implementation.
4. HTML reporting.

All experiment results must be saved as JSON.

HTML reports must be generated from the JSON results.

---

# Phase 1 — Data Acquisition

## 1.1 Goals

Acquire and organize the SensorScope dataset locally.

The acquisition phase must produce a reproducible local data structure.

## 1.2 Expected Data Directory

Suggested structure:

```text
data/
  raw/
    sensorscope/
      ...
  external/
    ...
  processed/
    ...
output/
  json/
  reports/
````

## 1.3 Acquisition Outputs

The acquisition phase must produce:

```text
output/json/dataset_inventory.json
```

This JSON file must include:

* dataset name;
* source URL or citation information;
* local raw data path;
* detected deployments;
* detected files;
* file sizes;
* available metadata files;
* candidate data files;
* acquisition timestamp;
* notes about missing or unavailable files.

## 1.4 Dataset Inventory

The inventory must document:

* all readable files;
* file extensions;
* likely delimiter;
* number of rows, if feasible;
* detected columns;
* candidate timestamp columns;
* candidate station/sensor ID columns;
* candidate environmental variables.

The acquisition phase is complete when the dataset can be located, listed, and inspected.

---

# Phase 2 — Preprocessing

## 2.1 Goals

Transform the raw SensorScope data into a clean, synchronized time-series dataset suitable for sunrise change-point detection.

## 2.2 Required Preprocessing Steps

The preprocessing phase must:

1. Load raw data files.
2. Identify timestamp columns.
3. Parse timestamps.
4. Normalize timestamps to a consistent timezone.
5. Identify station or sensor IDs.
6. Identify the light-related variable.
7. Filter invalid or impossible measurements.
8. Handle missing values.
9. Synchronize sensor readings into a time-indexed matrix.
10. Select valid days.
11. Select valid sensors.

## 2.3 Light-Related Variable Selection

The selected variable must be documented in JSON.

The project must save:

```text
output/json/variable_selection.json
```

This file must include:

* selected variable name;
* candidate variable names;
* reason for selection;
* units, if available;
* deployment used;
* number of sensors with this variable;
* number of valid observations;
* notes about data quality.

## 2.4 Coordinates and Timezone

The project must identify the deployment coordinates.

If station-level coordinates exist, they should be stored.

If only deployment-level coordinates are available, use the deployment coordinates for all stations.

The project must save:

```text
output/json/location_metadata.json
```

This file must include:

* deployment name;
* latitude;
* longitude;
* altitude, if available;
* timezone;
* coordinate source;
* whether coordinates are station-level or deployment-level.

## 2.5 Sunrise Ground Truth

For each valid date, compute sunrise time using:

* date;
* latitude;
* longitude;
* timezone.

The project should also compute sunset for completeness, but the first experiment uses sunrise only.

The project must save:

```text
output/json/sunrise_ground_truth.json
```

This file must include one record per date:

* date;
* sunrise local time;
* sunrise UTC time;
* sunset local time;
* sunset UTC time;
* latitude;
* longitude;
* timezone;
* source/method used for astronomical calculation.

## 2.6 Processed Dataset Output

The preprocessing phase must save a processed time-series dataset.

Suggested outputs:

```text
data/processed/synchronized_sensor_data.parquet
data/processed/synchronized_sensor_data.csv
```

The processed dataset must include:

* timestamp;
* date;
* station or sensor ID;
* selected light-related variable;
* cleaned value;
* timezone-normalized timestamp.

A pivoted matrix format may also be saved:

```text
data/processed/sensor_matrix.parquet
```

where:

* rows are timestamps;
* columns are sensor IDs;
* values are the selected light-related variable.

## 2.7 Preprocessing Summary

The preprocessing phase must save:

```text
output/json/preprocessing_summary.json
```

This file must include:

* raw row count;
* cleaned row count;
* number of sensors before filtering;
* number of sensors after filtering;
* number of days before filtering;
* number of valid days;
* missing-data percentage by sensor;
* selected variable;
* timestamp range;
* sampling interval estimate;
* exclusions applied.


````markdown
## 2.8 Dataset HTML Report

After data acquisition and preprocessing, the project must generate a dedicated dataset report.

This report is used to decide whether the SensorScope dataset is suitable for the sunrise change-point detection experiment before implementing or interpreting the full detection model.

The report must be generated from the JSON outputs produced during acquisition and preprocessing. It must not recompute the full preprocessing pipeline.

Expected output:

```text
output/reports/dataset_report.html
````

The report must summarize all relevant information needed to assess dataset suitability.

### Required Input JSON Files

The dataset report should read, when available:

```text
output/json/dataset_inventory.json
output/json/variable_selection.json
output/json/location_metadata.json
output/json/sunrise_ground_truth.json
output/json/preprocessing_summary.json
output/json/sensor_informativeness.json
```

If some files are not available yet, the report should still be generated with clear warnings.

### Required Report Sections

The dataset report must include:

1. **Dataset Overview**

   * dataset name;
   * source/citation;
   * selected deployment;
   * raw data path;
   * timestamp range;
   * number of files inspected;
   * number of candidate data files.

2. **Deployment and Location**

   * deployment name;
   * latitude;
   * longitude;
   * altitude, if available;
   * timezone;
   * whether coordinates are station-level or deployment-level;
   * coordinate source.

3. **Available Variables**

   * list of detected variables;
   * candidate light-related variables;
   * selected variable;
   * variable units, if available;
   * reason for selecting the variable.

4. **Timestamp and Sampling Structure**

   * timestamp format;
   * timezone handling;
   * estimated sampling interval;
   * timestamp coverage;
   * irregularities or gaps.

5. **Sensor Availability**

   * number of stations/sensors detected;
   * number of valid sensors after filtering;
   * observations per sensor;
   * missing-data rate by sensor;
   * sensors excluded and reason for exclusion.

6. **Valid Days**

   * number of total days in the deployment;
   * number of days with usable sunrise windows;
   * number of invalid days;
   * reason for invalid days, when available.

7. **Sunrise Ground Truth**

   * sunrise calculation method;
   * latitude/longitude used;
   * timezone used;
   * date range;
   * example sunrise times;
   * whether sunrise times were converted to UTC.

8. **Preprocessing Summary**

   * raw row count;
   * cleaned row count;
   * filtering rules;
   * missing-data handling;
   * synchronized matrix shape;
   * selected analysis window around sunrise.

9. **Sensor Informativeness Preview**

   * preliminary global `D_i_global` ranking, if available;
   * top candidate sensors by `D_i_global / (C_i + T_i)`;
   * basic pre/post sunrise statistics;
   * whether enough sensors appear informative to support the low-,
     medium-, and high-budget regimes under the configured budget
     values.

10. **Dataset Suitability Assessment**

    * final suitability status:

      * `suitable`;
      * `partially_suitable`;
      * `not_suitable`;
      * `undetermined`;
    * explanation of the assessment;
    * main risks;
    * recommended next step.

### Suitability Criteria

The report should explicitly evaluate whether the dataset satisfies the minimum requirements for the experiment:

* a light-related variable is available;
* timestamps are parseable;
* timezone can be resolved;
* deployment coordinates are known or reasonably approximated;
* sunrise ground truth can be computed;
* at least one valid sensor exists for the low-budget scenario;
* enough valid sensors exist to make the medium- and high-budget
  scenarios meaningful under the configured cost model and budget
  values;
* there are enough valid days to compute meaningful metrics;
* missing data does not make sunrise windows unusable.

### Recommended Suitability Logic

Use the following qualitative logic:

```text
suitable:
  All key requirements are satisfied.

partially_suitable:
  The experiment is possible, but there are relevant limitations
  such as few valid days, few valid sensors, or noisy light data.

not_suitable:
  A core requirement is missing, such as no light-related variable,
  no usable timestamps, or no location metadata.

undetermined:
  More inspection is needed before deciding.
```

### Report Style

The report must be written in academic English.

It should be concise, readable, and diagnostic. The goal is not only to describe the dataset, but to help decide whether the experiment should proceed.

The report should include tables where appropriate.

Suggested tables:

* available variables table;
* sensor availability table;
* missing-data summary table;
* valid days summary table;
* sunrise ground truth sample table;
* top sensors by preliminary `D_i_global`;
* suitability checklist.

### Important Statement

The report must include the following statement:

> This dataset report is intended to verify whether the selected SensorScope deployment contains enough temporal, spatial, and light-related information to support a sunrise-based change-point detection experiment.

### Output Requirement

The dataset report generation must be part of the preprocessing phase.

At the end of preprocessing, the following file must exist:

```text
output/reports/dataset_report.html
```


---

# Phase 3 — Model and Experiment Implementation

## 3.1 Core Experimental Setup

For each valid day (d):

* true change point:

[
\nu_d = sunrise(d)
]

* analysis window:

[
[\nu_d - w_{pre}, \nu_d + w_{post}]
]

Default:

* (w_{pre} = 2) hours;
* (w_{post} = 2) hours.

The detector must produce:

[
\tau_d
]

the detected change point.

The daily detection error is:

[
e_d = \tau_d - \nu_d
]

## 3.2 Global Empirical Gaussian Parameters

The main detector uses fixed global empirical Gaussian parameters per
sensor. These parameters are estimated once per sensor from the
SensorScope dataset and are then reused unchanged across all daily
detections and across all budget regimes. They are the empirical
analogue of the fixed Gaussian parameters assumed in the paper.

For each sensor `i`, estimate from the dataset:

* `mu_{0,i}_global`: pre-change mean (pre-sunrise window, pooled across
  valid days);
* `mu_{1,i}_global`: post-change mean (post-sunrise window, pooled
  across valid days);
* `sigma_i^2_global`: pooled within-window variance.

The per-sensor Gaussian divergence is then:

\[
D_{i,\text{global}} \;=\;
\frac{(\mu_{1,i}_{\text{global}} - \mu_{0,i}_{\text{global}})^2}
     {2 \, \sigma_{i,\text{global}}^2}
\]

The project must save:

```text
output/json/sensor_informativeness.json
```

This file must include, per sensor:

* sensor ID;
* number of valid days used;
* `mu_0_global`;
* `mu_1_global`;
* `sigma2_global`;
* `D_i_global`;
* missing-data rate;
* rank by `D_i_global`;
* parameter estimation method.

The detector must not estimate same-day pre-sunrise baselines in the
main experiment. A same-day pre-sunrise baseline detector is acceptable
only as a diagnostic mode and must be clearly labelled as such.

## 3.3 Cost Model

Because the SensorScope dataset does not provide real sensing or
transmission costs, the project must use an explicit default unit-cost
model rather than implicitly dropping the cost terms.

Default convention (required):

```text
C_i           = 1.0
T_i           = 0.0
total_cost_i  = C_i + T_i = 1.0
```

Default budget values per regime:

* low-budget regime:    `B = 1.0`;
* medium-budget regime: `B = 3.0`;
* high-budget regime:   `B = number of valid sensors`.

The cost model and the budget value of each regime are an empirical
approximation caused by the lack of real cost data, but the
implementation must still expose explicit `C_i`, `T_i`, `total_cost_i`
and `B` fields in configuration and in the result JSON.

## 3.4 Dynamic Budget Policy

At each timestamp `t`, the sampling policy must:

1. identify the set of available sensors with a finite reading at `t`;
2. retrieve each available sensor's score:

   \[
   \text{score}_i \;=\; \frac{D_{i,\text{global}}}{C_i + T_i}
   \]

3. greedily select sensors in decreasing order of `score_i` while:

   \[
   \sum_{i \in S(t)} (C_i + T_i) \;\leq\; B_{\text{regime}}
   \]

4. output the active subset `S(t)`;
5. compute the detector evidence at `t` using only sensors in `S(t)`.

The policy must store, per timestamp or in summarized per-day form,
the selection metadata:

* available sensors;
* selected sensors `S(t)`;
* rejected sensors;
* selection scores;
* total budget used;
* budget value of the regime.

## 3.5 Detection Statistic

The main detector is a multi-sensor one-sided CUSUM that accumulates
the per-sensor Gaussian log-likelihood ratio computed from the global
parameters of Section 3.2:

\[
\ell_{i,t} \;=\;
\frac{(x_{i,t} - \mu_{0,i})^2 - (x_{i,t} - \mu_{1,i})^2}
     {2 \, \sigma_i^2}
\]

\[
L_t \;=\; \frac{1}{|S(t)|}
         \sum_{i \in S(t)} \ell_{i,t}
\]

\[
S_t \;=\; \max(0,\, S_{t-1} + L_t),
\qquad
\tau \;=\; \min\{ t : S_t \geq h \}
\]

The aggregation at each timestamp uses only the sensors in the
dynamically selected subset `S(t)`. The threshold `h` is fixed,
documented in JSON, and held constant across all budget regimes. A
same-day pre-sunrise baseline z-score CUSUM may be retained as a
diagnostic mode (`detector_mode = "daily_baseline_zscore"`); the main
mode is `detector_mode = "global_gaussian_llr"`.

## 3.6 Scenario 1 — Low Budget

Low budget uses dynamic budget-constrained selection with `B = 1.0`
under the unit-cost model. The full candidate sensor set is loaded for
this scenario; the active subset `S(t)` is computed at each timestamp
and may vary over time depending on availability and per-sensor scores.
Under unit costs and `B = 1.0`, the policy selects at most one
available sensor per timestamp; the identity of that sensor may change
across timestamps.

Output file:

```text
output/json/sunrise_low_budget_results.json
```

## 3.7 Scenario 2 — Medium Budget

Medium budget uses dynamic budget-constrained selection with `B = 3.0`
under the unit-cost model. The full candidate sensor set is loaded for
this scenario; the active subset `S(t)` is computed at each timestamp
and may vary over time. Under unit costs and `B = 3.0`, the policy
selects up to three available sensors per timestamp.

Output file:

```text
output/json/sunrise_medium_budget_results.json
```

## 3.8 Scenario 3 — High Budget

High budget uses dynamic budget-constrained selection with `B` equal to
the number of valid sensors under the unit-cost model. The full
candidate sensor set is loaded for this scenario; at each timestamp the
policy can activate all currently available sensors.

A valid sensor must satisfy:

* enough observations in the sunrise analysis windows;
* acceptable missing-data rate;
* stable timestamps;
* selected light-related variable available.

Output file:

```text
output/json/sunrise_high_budget_results.json
```

## 3.9 Scenario Result JSON Structure

Each scenario result JSON must include the following top-level fields:

* `scenario_name`;
* `budget_regime`;
* `budget_value`;
* `cost_model` (per-sensor `C_i`, `T_i`, `total_cost_i`);
* `unit_cost_assumption`;
* `dynamic_selection` set to `true`;
* `selection_policy` description;
* `full_candidate_sensors`;
* `sensor_ranking` by `D_i_global / (C_i + T_i)`;
* `detector_mode`;
* `global_parameter_source` (path to `sensor_informativeness.json`);
* `aggregate_metrics`;
* `per_day_results`.

Any `selected_sensors` field, if present, must be clearly described as
a summary (for example, the union of `S(t)` over the experiment) and
not as a fixed scenario-defining subset.

Each per-day record must include:

* `date`;
* `true_change_point` (sunrise);
* `detected_change_point`;
* `signed_delay_minutes`;
* `absolute_error_minutes`;
* `detection_status`;
* `false_alarm` flag;
* `missed_detection` flag;
* `candidate_sensors`;
* `sensors_ever_selected` (union of `S(t)` over the day);
* `sensor_selection_counts` (per-sensor activation count);
* `average_active_sensors_per_timestamp`;
* `average_budget_used`;
* `timestamps_without_selection`;
* `grid_points`.

## 3.10 Metrics

For each scenario, compute:

* valid days count;
* detected days count;
* missed detection count;
* false alarm count;
* mean delay;
* median delay;
* mean absolute error;
* median absolute error;
* standard deviation of delay;
* minimum delay;
* maximum delay.

Definitions:

* true change point: (\nu_d);
* detected change point: (\tau_d);
* delay: (\tau_d - \nu_d);
* false alarm: (\tau_d < \nu_d - tolerance);
* missed detection: no alarm inside the analysis window;
* correct/on-time detection: (|\tau_d - \nu_d| \leq tolerance).

Default tolerance:

```text
15 minutes
```

## 3.11 Combined Experiment Summary

After running the three scenarios, save:

```text
output/json/sunrise_experiment_summary.json
```

This file must include:

* dataset metadata;
* variable used;
* deployment;
* location;
* number of valid days;
* number of valid sensors;
* scenario comparison table;
* low-budget metrics;
* medium-budget metrics;
* high-budget metrics;
* paths to detailed result JSON files.

---

# Phase 4 — HTML Reporting

## 4.1 Reporting Principle

All reports must be generated from JSON outputs.

The HTML report must not recompute the experiment.

The report generator must read JSON files and render the final report.

## 4.2 Required HTML Reports

The project must generate the following reports:

Dataset report:

```text
output/reports/dataset_report.html
```

Per-scenario reports (one per budget regime):

```text
output/reports/sunrise_low_budget_report.html
output/reports/sunrise_medium_budget_report.html
output/reports/sunrise_high_budget_report.html
```

Combined cross-scenario comparison report:

```text
output/reports/sunrise_budget_comparison_report.html
```

## 4.3 Per-Scenario Report Contents

Each per-scenario report must include:

1. project title;
2. connection to the paper;
3. dataset summary;
4. ground truth definition (sunrise);
5. global Gaussian parameter summary
   (`mu_{0,i}`, `mu_{1,i}`, `sigma_i^2`, `D_i_global` per sensor);
6. dynamic budget policy description, including the form of `S(t)` and
   the budget constraint;
7. cost model (`C_i`, `T_i`, `total_cost_i`, unit-cost assumption);
8. budget value `B` of the regime;
9. selection frequency summary
   (sensors most frequently activated, average size of `S(t)`,
   timestamps without selection);
10. detector description (global Gaussian LLR CUSUM aggregated over
    `S(t)`);
11. detector parameters (threshold, aggregation, evidence type);
12. aggregate metrics;
13. per-day results;
14. experimental scope.

## 4.4 Combined Comparison Report

The combined report must compare the low-, medium-, and high-budget
regimes and must explicitly answer:

> How does detection performance change as the budget increases?

The comparison must report, for each regime:

* aggregate detection metrics (detected, missed, false alarms);
* mean and median signed delay;
* mean and median absolute error;
* average size of `S(t)` per timestamp;
* sensors most frequently selected;
* average budget used.

## 4.5 Required Report Discussion

Each report must explain:

* what `nu` means in the real-data experiment;
* what `tau` means in the real-data experiment;
* how sunrise is used as external ground truth;
* how the global per-sensor Gaussian parameters are estimated from real
  data and reused unchanged across days and regimes;
* how the dynamic budget policy turns a single multi-sensor model into
  the low, medium, and high regimes through different values of `B`;
* why this is a real-data validation rather than an exact reproduction
  of the Gaussian theory.

The report must include the statement:

> The goal is not to exactly reproduce the theoretical Gaussian model, but to test whether the same resource-constrained change-point detection logic appears in real sensor network data.

## 4.6 Optional Figures

The reports may include:

* sensor ranking bar plot by `D_i_global`;
* delay comparison bar plot across regimes;
* example time series around sunrise;
* example detection statistic curve;
* sensor activation frequency plot per regime;
* missing-data summary plot.

Figure files should be stored under:

```text
output/reports/assets/
```

---

# Phase 5 — CLI

## 5.1 CLI Goal

Create a command-line interface for reproducible project execution.

## 5.2 Suggested Commands

```bash
python -m src.cli acquire-data
python -m src.cli inspect-dataset
python -m src.cli preprocess-data
python -m src.cli build-ground-truth
python -m src.cli rank-sensors
python -m src.cli run-sunrise-experiment --scenario low
python -m src.cli run-sunrise-experiment --scenario medium --k 3
python -m src.cli run-sunrise-experiment --scenario high
python -m src.cli run-sunrise-experiment --scenario all
python -m src.cli generate-report
```

The exact module path may be adapted to the project structure.

## 5.3 CLI Requirements

The CLI must:

* print clear progress messages;
* show progress for long-running commands;
* save all outputs to predictable paths;
* fail with clear error messages;
* avoid silent failures;
* expose key parameters as arguments.

Configurable parameters should include:

* dataset path;
* deployment;
* variable name;
* timezone;
* latitude;
* longitude;
* pre-window size;
* post-window size;
* budget value `B` per regime;
* cost model (`C_i`, `T_i`);
* detector mode
  (`global_gaussian_llr` for the main detector,
   `daily_baseline_zscore` for the diagnostic detector);
* detection threshold `h`;
* tolerance window.

---

# Phase 6 — First Executable Test

## 6.1 First Test Scope

The first executable test should focus on:

**Scenario 1 — Low Budget**

The test must verify that the project can:

1. load the processed data;
2. load the sunrise ground truth;
3. load the global per-sensor Gaussian parameters
   (`mu_{0,i}`, `mu_{1,i}`, `sigma_i^2`, `D_i_global`);
4. run the dynamic budget policy with `B = 1.0` under the unit-cost
   model, computing `S(t)` at each timestamp;
5. run the global Gaussian LLR CUSUM detector for at least one valid
   day, aggregating only over the sensors in `S(t)`;
6. produce `nu_d`, `tau_d`, and the signed delay;
7. save JSON results with the fields specified in Section 3.9;
8. generate the corresponding HTML report.

## 6.2 First Test Output

The first test must produce:

```text
output/json/sunrise_low_budget_results.json
output/reports/sunrise_low_budget_report.html
```

The JSON must contain at least one valid day result with the dynamic
selection metadata. The HTML report must include the low-budget
scenario sections defined in Section 4.3.

---

# Coding Standards

Use English for:

* code;
* comments;
* docstrings;
* CLI messages;
* JSON keys;
* reports;
* filenames.

Use Google-style docstrings for new functions.

Prefer modular code:

* data acquisition;
* data inspection;
* preprocessing;
* ground truth generation;
* sensor ranking;
* detection;
* metrics;
* reporting;
* CLI.

Avoid hardcoded paths when possible.

Prefer configuration files or CLI arguments.

Save machine-readable results as JSON.

Generate HTML reports from JSON.

Do not overwrite unrelated outputs.

Do not implement adversarial/covert scenarios in the first version.

Do not implement sunset detection in the first version.

Focus only on sunrise detection under the first three budget regimes.

```
```

