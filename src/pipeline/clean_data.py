"""
run after build_data.py

This takes the csv created in the build_data.py script and 
returns a clean dataframe ready for feature building  
"""

import pandas as pd 

from pathlib import Path
from src.paths import DATA_DIR, PROJECT_ROOT
from src.cleaning_and_features import clean 

IN_DATA = DATA_DIR / "interim" / "raw_data.csv"
OUT_DATA = DATA_DIR / "interim" / "cleaner_data.parquet"

def import_data(raw_data_path: Path) -> pd.DataFrame:
    if not raw_data_path.exists():
        raise FileNotFoundError(f"File not found: {raw_data_path}")
    
    return pd.read_csv(raw_data_path) 

def clean_basic(raw_csv: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic cleaning steps.

    This function:
    - removes duplicate rows
    - cleans column names
    - cleans text values
    - creates datetime variables
    - creates the injury outcome

    It does not handle all missing values.
    """
    df = raw_csv.drop_duplicates().copy()

    df = clean.clean_column_names(df)
    df = clean.clean_text_columns(df, [col for col in clean.TEXT_COLUMNS if col in df.columns])

    df = clean.split_datetime(df, "first_occurrence_date")
    df = clean.create_outcome(df)

    return df 

def main():
    df_raw = import_data(IN_DATA)

    df = clean_basic(df_raw)
    df = clean.bin_text_columns(df)

    df = df.rename(columns={
    "geo_lon": "lon",
    "geo_lat": "lat",
    "tu1_driver_humancontribfactor_binned":"tu1_human_fac_binned",
    "tu2_driver_humancontribfactor_binned":"tu2_human_fac_binned"
    })
    df = clean.convert_column_types(
        df,
        bool_cols=["injured"],
        float_cols=["lat", "lon"],
            category_cols=[
                "tu1_vehicle_type_binned",
                "tu2_vehicle_type_binned",
                "tu1_human_fac_binned",
                "tu2_human_fac_binned",
                "tu1_driver_action_binned",
                "tu2_driver_action_binned",
                "light_condition",
                "road_condition",
    ],
        string_cols=["incident_address", "top_traffic_accident_offense",],
    )

    cols_to_drop = [col for col in clean.DROP_AFTER_CLEANING if col in df.columns]
    missing_drop_cols = sorted(set(clean.DROP_AFTER_CLEANING) - set(df.columns))

    print("Dropping columns:", cols_to_drop)
    print("Drop columns not found:", missing_drop_cols)

    df = df.drop(columns=cols_to_drop)

    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DATA, index = False)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Saved {df.shape[0]} rows and {df.shape[1]} columns to {OUT_DATA}")

if __name__== "__main__":
    main()