# Denver Motor Vehicle Accident Severity Prediction

a work in progress by Lily Holmes / Summer 2026

**FOR LEARNING and PRACTICING** 

---

This project uses Denver motor vehicle accident records to explore whether crash-level information can help predict whether an incident involves medical harm or requires emergency medical response.

> The goal is to practice a full applied data science workflow: querying raw data, cleaning features, defining a modeling target, training baseline classifiers, evaluating model performance, and documenting limitations.

## Pipeline 

```
data -> query (current) -> clean -> model -> evaluate -> communicate 
```

Stack: `sklearn`, `matplotlib`, `pandas`, `SQLlite`

planned models: logistic regression, naive bayes, decision trees 

## Data

The project uses open source motor vehicle indident data from the City of Denver

https://opendata-geospatialdenver.hub.arcgis.com/datasets/db00bd99ea534d8987e0913a191ebe19_325/explore?location=39.759262%2C-104.902794%2C10 

This repo will include SQL scripts to query the modeling data, sample processed dataset. 

## Repo structure
```text
.
├── data/
│   ├── processed/        
│   └── raw/              # Raw data, ignored by Git
├── notebooks/           
│   └── 01_database_exploration.ipynb                                           
├── requirements.txt      
├── .gitignore
└── README.md
```

## Goals:

* Deliver a reproducible pipeline 
* Provide executive summary 
* learn stuff   