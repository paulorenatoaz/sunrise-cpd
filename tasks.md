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

The first version of the project covers only three regimes:

1. Low-budget regime: one sensor.
2. Medium-budget regime: a small subset of sensors.
3. High-budget regime: all valid sensors.

The covert/adversarial part of the paper is outside the scope of this first implementation.

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

For the first version, if real sensing and communication costs are unavailable, use unit cost:

\[
C_i + T_i = 1
\]

Therefore, the first implementation ranks sensors directly by \(D_i\).

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

   * preliminary (D_i) ranking, if available;
   * top candidate sensors;
   * basic pre/post sunrise statistics;
   * whether enough sensors appear informative for low-, medium-, and high-budget experiments.

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
* at least `k` valid sensors exist for the medium-budget scenario;
* enough valid sensors exist for the high-budget scenario;
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
* top sensors by preliminary (D_i);
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

## 3.2 Sensor Informativeness

For each sensor (i), estimate:

[
\mu_{0,i}
]

from the pre-sunrise window, and:

[
\mu_{1,i}
]

from the post-sunrise window.

Estimate:

[
\sigma_i^2
]

from the local window or a pooled pre/post estimate.

Compute:

[
D_i =
\frac{(\mu_{1,i} - \mu_{0,i})^2}{2\sigma_i^2}
]

The project must save:

```text
output/json/sensor_informativeness.json
```

This file must include:

* sensor ID;
* number of valid days;
* mean pre-sunrise value;
* mean post-sunrise value;
* estimated variance;
* estimated (D_i);
* missing-data rate;
* rank by (D_i).

## 3.3 Detection Statistic

The first implementation should use a simple reproducible CUSUM-like detector or cumulative standardized change score.

For a single sensor, the detector may use a standardized score based on pre/post estimates.

For multiple sensors, combine evidence additively across selected sensors.

The detector must be deterministic for fixed inputs and configuration.

The threshold may be:

* fixed;
* configured;
* or calibrated from pre-sunrise behavior.

The threshold strategy must be documented in JSON.

## 3.4 Scenario 1 — Low Budget

Low budget uses one sensor.

Sensor selection:

* select the single best valid sensor according to (D_i).

Output file:

```text
output/json/sunrise_low_budget_results.json
```

This file must include:

* scenario name;
* selected sensor;
* selected sensor (D_i);
* configuration;
* per-day results;
* aggregate metrics.

Per-day results must include:

* date;
* true change point (\nu_d);
* detected change point (\tau_d);
* delay in minutes;
* absolute error in minutes;
* detection status;
* false alarm flag;
* missed detection flag.

## 3.5 Scenario 2 — Medium Budget

Medium budget uses a small subset of sensors.

Default:

* select top (k = 3) sensors by (D_i).

Output file:

```text
output/json/sunrise_medium_budget_results.json
```

This file must include:

* scenario name;
* selected sensors;
* selected sensors' (D_i);
* value of (k);
* configuration;
* per-day results;
* aggregate metrics.

## 3.6 Scenario 3 — High Budget

High budget uses all valid sensors.

A valid sensor must satisfy:

* enough observations in the sunrise analysis windows;
* acceptable missing-data rate;
* stable timestamps;
* selected light-related variable available.

Output file:

```text
output/json/sunrise_high_budget_results.json
```

This file must include:

* scenario name;
* number of valid sensors;
* list of valid sensors;
* configuration;
* per-day results;
* aggregate metrics.

## 3.7 Metrics

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

## 3.8 Combined Experiment Summary

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

## 4.2 Main HTML Report

Generate:

```text
output/reports/sunrise_experiment_report.html
```

The report must include:

1. Title and project summary.
2. Theoretical connection to the paper.
3. Dataset description.
4. Data acquisition summary.
5. Preprocessing summary.
6. Selected deployment and location.
7. Sunrise ground truth construction.
8. Selected light-related variable.
9. Sensor informativeness ranking.
10. Description of the three budget regimes.
11. Low-budget results.
12. Medium-budget results.
13. High-budget results.
14. Scenario comparison.
15. Example daily detection plots.
16. Limitations.
17. Next steps.

## 4.3 Required Report Discussion

The report must explain:

* what (\nu) means in the real-data experiment;
* what (\tau) means in the real-data experiment;
* how sunrise is used as external ground truth;
* how (D_i) is estimated from real data;
* how low, medium, and high budget regimes map to sensor subsets;
* why this is a real-data validation rather than an exact reproduction of the Gaussian theory.

The report must include the statement:

> The goal is not to exactly reproduce the theoretical Gaussian model, but to test whether the same resource-constrained change-point detection logic appears in real sensor network data.

## 4.4 Optional Figures

The report may include:

* sensor ranking bar plot by (D_i);
* delay comparison bar plot across scenarios;
* example time series around sunrise;
* example detection statistic curve;
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
* medium-budget (k);
* detection threshold;
* tolerance window.

---

# Phase 6 — First Executable Test

## 6.1 First Test Scope

The first executable test should focus on:

**Scenario 1 — Low Budget**

The test must verify that the project can:

1. load the processed data;
2. load sunrise ground truth;
3. select the best sensor by (D_i);
4. run the detector for at least one valid day;
5. produce (\nu_d), (\tau_d), and delay;
6. save JSON results;
7. generate an HTML report.

## 6.2 First Test Output

The first test must produce:

```text
output/json/sunrise_low_budget_results.json
output/reports/sunrise_experiment_report.html
```

The JSON must contain at least one valid day result.

The HTML report must include the low-budget result section.

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

