# Denver Motor Vehicle Accident Severity Prediction

a work in progress by Lily Holmes / Summer 2026

**FOR LEARNING and PRACTICING**

---

This project uses Denver motor vehicle accident records to explore whether crash-level information can help predict whether an incident involves medical harm or requires emergency medical response.

> The goal is to practice a full applied data science workflow: querying raw data, cleaning features, defining a modeling target, training baseline classifiers, evaluating model performance, and documenting limitations.

## Status

This project is moving from notebook exploration toward a reproducible Python pipeline.

Current pieces:

* SQL query for building the raw modeling dataset
* Pipeline scripts for raw data extraction and basic cleaning
* Reusable cleaning, feature engineering, geospatial, and modeling helpers
* Notebooks for exploration, feature checks, and baseline modeling

## Pipeline

```text
raw database -> SQL extract -> clean -> feature engineering -> model -> evaluate
```

Main stack: `pandas`, `scikit-learn`, `matplotlib`

Current baseline models: dummy classifiers, logistic regression, naive Bayes, and decision trees.

## Data

The project uses open source data from the City of Denver:

### Motor vehicle incident

[Traffic Accidents (Offenses)](https://opendata-geospatialdenver.hub.arcgis.com/datasets/db00bd99ea534d8987e0913a191ebe19_325/explore?location=39.759262%2C-104.902794%2C10)

Raw and interim data are local-only. The repository includes the SQL needed to recreate the modeling extract and a small processed sample for reference.

### Streetcenter lines

I use this dataset to join speed limits onto the crash data with coordinate pairs from both datasets. 

[Street Centerlines](https://opendata-geospatialdenver.hub.arcgis.com/datasets/street-centerlines/explore?location=39.778461%2C-104.843897%2C10)

## Repo Structure

```text
.
├── data/
│   ├── raw/                  # Local raw database and road files, ignored by Git
│   ├── interim/              # Local pipeline intermediates, ignored by Git
│   ├── sample/               # Sample raw and clean data
│   └── processed/            # Final processed outputs and sample data
├── output/
│   └── metrics/              # Local model metrics and validation plots, ignored by Git
│       ├── individual/        # Per-family diagnostics
│       └── best_models/       # Finalist model comparison outputs
├── sql/
│   └── build_dataset.sql     # Query used to extract raw modeling data
├── src/
│   ├── pipeline/
│   │   ├── build_data.py      # Builds the raw CSV extract from the database
│   │   ├── clean_data.py      # Cleans the raw extract for feature building
│   │   ├── features_data.py   # Creates features from clean data
│   │   ├── geo_features.py    # Adds speed limit features
│   │   └── run_model_comparisons.py # Compares finalist models
│   ├── cleaning_and_features/
│   │   ├── clean.py           # Cleaning helpers and type conversion
│   │   ├── features.py        # Feature engineering helpers
│   │   ├── geo.py             # Geospatial joins and speed-limit features
│   │   └── maps.py            # Text binning maps
│   ├── modeling/
│   │   ├── feature_sets.py    # Model feature lists
│   │   ├── preprocessors.py   # sklearn preprocessors by model type
│   │   ├── split.py           # Temporal train/validation/test split helpers
│   │   ├── models.py          # Model and pipeline builders
│   │   ├── evaluate.py        # Classification metric helpers
│   │   ├── metric_vis.py      # ROC, precision-recall, and threshold plots
│   │   ├── tune_tree.py       # Decision-tree hyperparameter search
│   │   └── ind_model_functions.py # Shared individual-model run helper
│   ├── individual_models/
│   │   ├── run_dummy_models.py          # Dummy baseline family
│   │   ├── run_logistic_regression.py   # Logistic regression family
│   │   ├── run_naive_bayes.py           # Naive Bayes family
│   │   ├── run_decision_tree.py         # Decision tree family
│   │   └── run_every_model.py           # Runs all individual model families
│   ├── audit.py               # Dataset inspection helpers
│   ├── run_pipeline.py        # Full data pipeline runner
│   └── paths.py               # Shared project paths
├── requirements.txt
├── .gitignore
└── README.md
```

All reusable logic and pipeline steps live in `src/`. Notebooks are local development artifacts and are ignored in version history.

`run_every_model.py` runs each model family separately and saves detailed per-family diagnostics, such as metrics by split, ROC and precision-recall plots, confusion matrices, and threshold outputs where relevant. `run_model_comparisons.py` is narrower: it compares the selected finalist models across families and saves the overall comparison metrics and validation curves.

## Reprodicibility 

(im thinking of storing the specific data in cloud so someone can run it, pull it from my s3 bucket without having to downlaod it? integrate it in the script) 

for now 
Obtain data from links above. 

place Accident data in data/raw/"traffic.geodatabase"
place Street data in data/raw/"streetcenterlines.geojson" 

Clone this repository:

```bash
git clone https://github.com/lily-harper/injury_risk_classifier
cd injury-risk-classifier
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run:

```bash
PYTHONPATH=. python3 src.run_pipeline
```

## Notes on...

#### Iterations 

I used this data source for a final project in *Statistical Methods & Applications I (STAT5000, fall 2025)*. In this, used R to build confidence intervals, tested hypothesis, and completed a report paper. No genAI/agenticAI was used in that iteration. 

I wanted to revisit this data after learning about classification methods as several of the questions dodged around the main one.

So, in summer 2026 I worked on the second iteration, which in itself, is a first iteration of something else. While the data source is the same, the methods, tools, and workflow are much different. The main goal of this project was to complete a fully reproducible project using classification methods with a clean modularized workflow.

I revisted the STAT5000 project to store it a GitHub repo, which will allow anyone to see the difference in premise between this project.  
[Denver Car Accident Analysis](https://github.com/lily-harper/denver_car_accident_analysis/tree/main)

Further iterations could inclde more features engineered, the implementation of more advanved ML algorithms, and potentially a deployment.  

#### Goals

* Deliver a reproducible pipeline
* Provide an executive summary
* Learn stuff

#### AgenticAI / GenAI use

I worked in tandem with OpenAI's tools, ChatGPT and CODEX to modify my code and add more code to improve the repo's organization. 

**I take full ownership for any choices made and conclusions achieved.**

#### Use / high level purpose 

This project was mainly done for the purpose of learning (methods, interpretations, and working outside notebooks). 

* No conclusions or methods are causal 
* This is not intended for any legal or legistlative use 
* This is not prescritive. 

Please wear your seatbelt and follow road regulations.   
