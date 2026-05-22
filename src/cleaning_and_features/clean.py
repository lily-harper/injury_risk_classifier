import pandas as pd

from src.cleaning_and_features.maps import (VEHICLE_TYPE_MAP,
                      HUMAN_CONTRIB_MAP,
                      DRIVER_ACTION_MAP)

# pipeline functions 

def create_outcome(df):
    df = df.copy()
    df['injured'] = (df["seriously_injured"] > 0) | (df["fatalities"] > 0)
    df['injured'] = df['injured'].astype(int)

    return df 

# dates 

def split_datetime(df, old_date_col):
    df = df.copy()

    dt = (pd.to_datetime(
        df[old_date_col],
        unit="D",
        origin="julian",
        errors="coerce"
    ))

    df["datetime"] = dt
    df["date"] = dt.dt.date
    df["time"] = dt.dt.strftime("%H:%M:%S")

    return df

# columns 

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    standardize column names
    """
    df = df.copy()

    df.columns = (
        df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
    )

    return df 

def clean_text_columns(df, columns):
    df = df.copy()

    for col in columns:
        df[col] = (
            df[col]
            .astype("string")
            .str.lower()
            .str.strip()
        )
    
    return df 

# Binning mesy text 
vehicle_type_cols = [
    "tu1_vehicle_type",
    "tu2_vehicle_type"]

human_contrib_cols = [
    "tu1_driver_humancontribfactor",
    "tu2_driver_humancontribfactor"]

driver_action_cols = [
    "tu1_driver_action",
    "tu2_driver_action"]

UNKNOWN_VALS = {"", "nan", "none", "<na>"}

def text_binning(df, columns, mapping, default_value = "other",
                 new_suffix = "_binned", unknown_value = "unknown") -> pd.DataFrame:

    for col in columns:
        normalized = df[col].astype("string").str.strip().str.lower()
        new_col = f"{col}{new_suffix}"

        df[new_col] = normalized.map(mapping)
        df[new_col] = df[new_col].fillna(default_value)

        df.loc[
            normalized.isna() | normalized.isin(UNKNOWN_VALS), new_col
        ] = unknown_value 
    
    return df 

def bin_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in the clean (but missing valued) dataframe
    returns succinct categories for vehicle type, 
    driver action, etc"""
    df = df.copy()

    df = text_binning(
        df = df,
        columns = vehicle_type_cols,
        mapping= VEHICLE_TYPE_MAP,
        default_value="other",
        unknown_value="unknown")

    df = text_binning(
        df=df,
        columns=human_contrib_cols,
        mapping=HUMAN_CONTRIB_MAP,
        default_value="other",
        unknown_value="unknown",
    )

    df = text_binning(
        df=df,
        columns=driver_action_cols,
        mapping=DRIVER_ACTION_MAP,
        default_value="other_or_invalid",
        unknown_value="other_or_invalid",
    )

    return df 

# ensuring types are what i want 

def convert_column_types(
    df,
    int_cols=None,
    float_cols=None,
    bool_cols=None,
    category_cols=None,
    string_cols=None,
    errors="coerce"):
    
    int_cols = int_cols or []
    float_cols = float_cols or []
    bool_cols = bool_cols or []
    category_cols = category_cols or []
    string_cols = string_cols or []

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors=errors).astype("Int64")

    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors=errors)

    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype("boolean")

    for col in category_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")

    return df

# cleaning the direction column 



TEXT_COLUMNS = [
    "TU1_TRAVEL_DIRECTION",
    "TU2_TRAVEL_DIRECTION",
    "TU1_VEHICLE_TYPE",
    "TU2_VEHICLE_TYPE",
    "TU1_DRIVER_ACTION",
    "TU2_DRIVER_ACTION",
    "TU1_DRIVER_HUMANCONTRIBFACTOR",
    "TU2_DRIVER_HUMANCONTRIBFACTOR",
    "TU1_PEDESTRIAN_ACTION",
    "TU2_PEDESTRIAN_ACTION",
    "ROAD_DESCRIPTION",
    "ROAD_CONDITION",
    "LIGHT_CONDITION",
    "tu1_travel_direction",
    "tu2_travel_direction",
    "road_description",
    "road_condition",
    "light_condition"
]

DROP_AFTER_CLEANING = [
    "fatality_mode_1",
    "fatality_mode_2",
    "seriously_injured_mode_1",
    "seriously_injured_mode_2",
    "tu1_driver_action",
    "tu2_driver_action",
    "tu1_vehicle_type",
    "tu2_vehicle_type",
    "tu1_driver_humancontribfactor",
    "tu2_driver_humancontribfactor",
    "tu1_pedestrian_action",
    "tu2_pedestrian_action",
    "seriously_injured",
    "fatalities",
    "first_occurrence_date",
    "tu1_travel_direction",
    "tu2_travel_direction"
]

# exploring / evaluating in notebooks 

def explore(df: pd.DataFrame, missing: bool, threshold:int = None):
    """
    data quality report
    returns preliminary insights about the data

    args:
        df: the dataframe 
        missing: boolean, if true, returns the missing report  
    """

    print("inspecting data...\n")

    df = df.copy()

    print(f"There are {df.shape[0]} rows and {df.shape[1]} columns")

    dup_count = df.duplicated().sum()
    print(f"There {dup_count} duplicate rows\n")

    if missing == True:
        # If you want a report of missing values per column

        missing_count = df.isna().sum()
        missing_percent = (missing_count / len(df)) * 100

        miss_report = pd.DataFrame({
            "data_types": df.dtypes,
            "missing_count": missing_count,
            "missing_percent": missing_percent.round(2)
        })

        miss_report["high missing"] = miss_report["missing_percent"] > threshold

    return df.head(), miss_report

def create_types_tables(
        df, 
        first_col,
        second_col,
        first_label = "first",
        second_label = "second"
):
    df = df.copy()

    types = pd.DataFrame({
        first_label: df[first_col].value_counts(dropna=False),
        second_label: df[second_col].value_counts(dropna=False)
    }).fillna(0).astype(int)

    types["total"] = types[first_label] + types[second_label]

    types = types.sort_values("total", ascending = False)

    return types

def accident_summary_wrt(df, col):
    summary = (
        df.groupby(df[col])["injured"]
        .agg(["count", "mean"])
        .sort_values("mean", ascending=False)
    )

    return summary