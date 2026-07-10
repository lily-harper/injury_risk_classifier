"""
run after build_data.py

This takes the csv created in the build_data.py script and 
returns a clean dataframe ready for feature building  
"""

import pandas as pd 

from pathlib import Path
from src.paths import import_data, PROJECT_ROOT, RAW_PATH, CLEANED_DATA_PATH
from src.cleaning_and_features import clean 
from src.cleaning_and_features import maps


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

def clean_travel_direction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for unit in ["tu1", "tu2"]:
        raw_col = f"{unit}_travel_direction"
        clean_col = f"{unit}_direction"

        df[clean_col] = (
            df[raw_col]
            .astype("string")
            .str.strip()
            .str.lower()
            .map(maps.DIRECTION_MAP)
            .fillna("unknown")
        )

    return df


def clean_and_filter_district(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["district_id"] = (
        df["district_id"]
        .astype("string")
        .str.strip()
    )

    valid_districts = {"1", "2", "3", "4", "5", "6"}
    valid_district = df["district_id"].isin(valid_districts)

    return df.loc[valid_district].copy()


def text_condition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and replace road/light condition text columns.
    """
    df = df.copy()

    df = clean.text_binning(
        df=df,
        columns=["road_description"],
        mapping=maps.ROAD_DESCRIP_MAP,
        default_value="other",
        unknown_value="unknown",
    )

    df = df.drop(columns=["road_description"], errors="ignore")

    df = clean.text_binning(
        df=df,
        columns=["light_condition"],
        mapping=maps.LIGHT_MAP,
        default_value="other",
        unknown_value="unknown",
        new_suffix="_binned",
    )

    df = clean.text_binning(
        df=df,
        columns=["road_condition"],
        mapping=maps.ROAD_CONDITION_MAP,
        default_value="other",
        unknown_value="unknown",
        new_suffix="_binned",
    )

    return df

def main():
    df_raw = import_data(RAW_PATH)
    df = clean_basic(df_raw)
    df = clean_and_filter_district(df)

    df = text_condition(df)

    df = clean.bin_text_columns(df)
    df = clean_travel_direction(df)

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
                "tu1_direction",
                "tu2_direction",
                "district_id",
    ],
        string_cols=["incident_address", "top_traffic_accident_offense",],
    )

    cols_to_drop = [col for col in clean.DROP_AFTER_CLEANING if col in df.columns]
    missing_drop_cols = sorted(set(clean.DROP_AFTER_CLEANING) - set(df.columns))

    print("Dropping columns:", cols_to_drop)
    print("Drop columns not found:", missing_drop_cols)

    df = df.drop(columns=cols_to_drop)

    CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEANED_DATA_PATH, index = False)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Saved {df.shape[0]} rows and {df.shape[1]} columns to {CLEANED_DATA_PATH}")

if __name__== "__main__":
    main()
