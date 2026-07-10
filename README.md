# Denver Motor Vehicle Accident Severity Prediction

An applied machine-learning project by Lily Holmes, Summer 2026.

This project uses public Denver traffic-crash records to model whether a crash involved a serious injury or fatality. The goal is not to present a production-ready public-safety system. The goal is to build a reproducible, end-to-end data science workflow around a real, messy, imbalanced dataset and evaluate the resulting classifiers honestly.

The project includes raw data extraction, cleaning, feature engineering, a geospatial speed-limit join, model comparison, threshold selection, and validation reporting.

## Project summary

The modeling target is:

```text
injured = seriously_injured > 0 OR fatalities > 0
```

This is a rare-event classification problem. In the current validation split, serious-injury/fatal crashes make up about 2.3% of records. Because of that class imbalance, accuracy alone is not useful: a dummy classifier that predicts no injury for every crash is about 97.7% accurate while finding 0% of injury cases.

The project therefore emphasizes recall, precision, PR AUC, ROC AUC, and confusion matrices rather than accuracy alone.

## Pipeline

```text
raw geodatabase
-> SQL extract
-> cleaning
-> feature engineering
-> geospatial speed-limit join
-> modeling dataset
-> model comparison outputs
```

Main stack:

* Python
* pandas
* GeoPandas
* scikit-learn
* matplotlib

Current model families:

* dummy baseline
* balanced logistic regression
* decision tree
* naive Bayes

## Evaluation design

The split is temporal:

```text
Train:      records through 2023
Validation: 2024 records
Final test: 2025 and later records
```

Model and threshold selection are performed on the validation split. The final test split is held out and should only be used after the modeling protocol is frozen.

## Frozen validation-selected protocol

Before final test evaluation, the current selected protocol is:

```text
Data:
- Use Denver crash records from the raw traffic geodatabase.
- Keep police districts 1–6.
- Join street speed-limit features from Denver street centerlines.
- Target = seriously_injured > 0 OR fatalities > 0.

Split:
- Train: records through 2023.
- Validation: 2024 records.
- Final test: 2025 and later records.

Selected model:
- Balanced logistic regression.

Selected threshold:
- 0.48.

Primary validation priority:
- Recall, interpreted alongside precision and the confusion matrix.
```

After the final test set is evaluated, model type, features, filtering rules, and threshold should not be changed based on the test result. If those decisions change, the test set becomes another validation set.

## Final test evaluation

The final test script is separate from the main validation pipeline so it can be run deliberately after the protocol above is frozen.

When ready, run:

```bash
python -m src.pipeline.run_final_test
```

This fits the selected balanced logistic regression model on the combined train and validation periods, then evaluates once on the held-out 2025+ final test split.

Final test outputs are written to:

```text
output/metrics/final_test/
├── final_test_model_metrics.csv
├── final_test_pr_curve.png
├── final_test_roc_curve.png
└── final_test_confusion_matrix.png
```

## Current validation results

The latest finalist comparison is saved in `output/metrics/best_models/`.

| Model | Threshold | Recall | Precision | Accuracy | F1 | ROC AUC | PR AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| dummy_no_injury | 0.50 | 0.000 | 0.000 | 0.977 | 0.000 | 0.500 | 0.023 |
| logistic_balanced | 0.48 | 0.704 | 0.079 | 0.804 | 0.142 | 0.841 | 0.266 |
| tuned_tree | 0.50 | 0.836 | 0.038 | 0.518 | 0.074 | 0.737 | 0.173 |
| naive_bayes | 0.10 | 0.504 | 0.072 | 0.840 | 0.126 | 0.746 | 0.147 |

The balanced logistic regression is the selected candidate because it provides the strongest validation PR AUC and a more usable recall/precision tradeoff than the tuned tree. The tuned tree reaches higher recall on validation but does so with much lower precision and accuracy.

For the selected logistic model at threshold `0.48`, the validation confusion matrix is:

```text
True negatives:  13,038
False positives:  3,124
False negatives:    112
True positives:     267
```

This result should be interpreted as a screening or risk-ranking exercise, not a standalone decision system. The model can identify many serious-injury/fatal crashes, but false positives remain substantial.

## Data

The project uses open data from the City and County of Denver.

### Traffic Accidents

[Traffic Accidents](https://opendata-geospatialdenver.hub.arcgis.com/datasets/db00bd99ea534d8987e0913a191ebe19_325/explore?location=39.759262%2C-104.902794%2C10)

This source provides crash-level records, location fields, crash context, vehicle/action fields, and injury/fatality indicators.

### Street Centerlines

[Street Centerlines](https://opendata-geospatialdenver.hub.arcgis.com/datasets/street-centerlines/explore?location=39.778461%2C-104.843897%2C10)

This source is used to join speed-limit information onto crash records using spatial proximity.

Raw and interim data are not committed to the repository. The repository includes the SQL query needed to recreate the modeling extract and small sample files for reference.

## Repository structure

```text
.
├── data/
│   ├── raw/                  # Local raw database and road files, ignored by Git
│   ├── interim/              # Local pipeline intermediates, ignored by Git
│   ├── sample/               # Small sample files committed for reference
│   └── processed/            # Local processed outputs, ignored by Git
├── output/
│   └── metrics/
│       ├── individual/        # Per-family diagnostics, local/generated
│       └── best_models/       # Finalist validation comparison outputs
├── sql/
│   └── build_dataset.sql      # Query used to extract raw modeling data
├── src/
│   ├── pipeline/              # End-to-end pipeline stages
│   ├── cleaning_and_features/ # Cleaning, feature engineering, and geospatial helpers
│   ├── modeling/              # Model builders, split helpers, metrics, and plots
│   ├── individual_models/     # Per-family model runners
│   ├── audit.py               # Dataset inspection helpers
│   ├── paths.py               # Shared project paths and input validation
│   └── run_pipeline.py        # Main pipeline entry point
├── requirements.txt
├── LICENSE
└── README.md
```

## Reproducibility

Raw data files are not committed to this repository. To run the full pipeline, download the source data and place the files here:

```text
data/raw/traffic.geodatabase
data/raw/streetcenterlines.geojson
```

Clone the repository:

```bash
git clone https://github.com/lily-harper/injury_risk_classifier
cd injury_risk_classifier
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the main pipeline:

```bash
python -m src.run_pipeline
```

To also regenerate individual candidate-model diagnostics before the finalist comparison, run:

```bash
python -m src.run_pipeline --run-individual-models
```

The pipeline checks for required raw files before running. Generated intermediate data are written under `data/interim/` and `data/processed/`. Model outputs are written under `output/metrics/`.

## Important limitations

This project is predictive, not causal. It does not estimate the causal effect of roadway design, speed limits, driver behavior, police district, or any other feature.

The model is not intended for legal, legislative, enforcement, insurance, or emergency-response decisions. It is an applied data science portfolio project that demonstrates reproducible workflow design, feature engineering, threshold-aware classification, and honest evaluation under class imbalance.

Known limitations include:

* serious-injury/fatal crashes are rare, which keeps precision low;
* crash reports may reflect reporting and data-entry patterns;
* geographic and district features may encode structural or operational differences that require careful interpretation;
* final test results are not yet reported in this README;
* additional validation would be required before any operational use.

## Project context

This project revisits a Denver traffic-crash dataset previously used in a STAT5000 project. This version focuses on Python, reproducible pipelines, classification modeling, validation metrics, and project organization.

Related earlier project:

[Denver Car Accident Analysis](https://github.com/lily-harper/denver_car_accident_analysis/tree/main)

## AI assistance disclosure

OpenAI tools, including ChatGPT and Codex, were used during development for code organization, debugging, and documentation support.

All modeling choices, interpretations, limitations, and final project decisions are my responsibility.
