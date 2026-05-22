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

Main stack: `pandas`, `geopandas`, `scikit-learn`, `matplotlib`, `SQLite`

Current baseline models: dummy classifiers, logistic regression, naive Bayes, and decision trees.

## Data

The project uses open source motor vehicle incident data from the City of Denver:

https://opendata-geospatialdenver.hub.arcgis.com/datasets/db00bd99ea534d8987e0913a191ebe19_325/explore?location=39.759262%2C-104.902794%2C10

Raw and interim data are local-only. The repository includes the SQL needed to recreate the modeling extract and a small processed sample for reference.

## Repo Structure

```text
.
├── data/
│   ├── raw/                  # Local raw database and road files, ignored by Git
│   ├── interim/              # Local pipeline intermediates, ignored by Git
│   └── processed/            # Final processed outputs and sample data
├── notebooks/
│   ├── 01_database_exploration.ipynb
│   ├── 02_cleaning.ipynb
│   ├── 03_EDA.ipynb
│   ├── 04_features.ipynb
│   └── 05_baseline.ipynb
├── sql/
│   └── build_dataset.sql     # Query used to extract raw modeling data
├── src/
│   ├── pipeline/
│   │   ├── build_data.py      # Builds the raw CSV extract from the database
│   │   └── clean_data.py      # Cleans the raw extract for feature building
│   ├── cleaning_and_features/
│   │   ├── clean.py           # Cleaning helpers and type conversion
│   │   ├── features.py        # Feature engineering helpers
│   │   ├── geo.py             # Geospatial joins and speed-limit features
│   │   └── maps.py            # Text binning maps
│   ├── modeling/
│   │   ├── feature_sets.py    # Model feature lists
│   │   ├── preprocessing.py   # sklearn preprocessors
│   │   └── metrics.py         # Evaluation helpers
│   ├── audit.py               # Dataset inspection helpers
│   └── paths.py               # Shared project paths
├── requirements.txt
├── .gitignore
└── README.md
```

## Goals

* Deliver a reproducible pipeline
* Provide an executive summary
* Learn stuff
