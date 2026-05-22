import pandas as pd 
import numpy as np

from src.cleaning_and_features import maps

def add_vehicle_size_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["tu1_vehicle_size"] = (
        df["tu1_vehicle_type_binned"]
        .map(maps.VEHICLE_SIZE_MAP)
        .astype("Int64")
    )

    df["tu2_vehicle_size"] = (
        df["tu2_vehicle_type_binned"]
        .map(maps.VEHICLE_SIZE_MAP)
        .astype("Int64")
    )

    df["vehicle_type_match"] = np.select(
        [
            df["tu1_vehicle_type_binned"].eq("unknown") |
            df["tu2_vehicle_type_binned"].eq("unknown"),

            df["tu1_vehicle_type_binned"].eq(df["tu2_vehicle_type_binned"]),
        ],
        [
            "unknown",
            "same",
        ],
        default="different",
    )

    df["vehicle_size_diff"] = df["tu1_vehicle_size"] - df["tu2_vehicle_size"]
    df["abs_vehicle_size_diff"] = df["vehicle_size_diff"].abs()

    df["vehicle_size_relation"] = np.select(
        [
            df["tu1_vehicle_size"].isna() | df["tu2_vehicle_size"].isna(),
            
            df["tu1_vehicle_size"]
            .gt(df["tu2_vehicle_size"])
            .fillna(False),

            df["tu1_vehicle_size"]
            .lt(df["tu2_vehicle_size"])
            .fillna(False),
        ],
        [
            "unknown",
            "vehicle_1_larger",
            "vehicle_2_larger",
        ],
        default="same_size",
    )

    return df

def create_time_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["weekend"] = df["day_of_week"].isin([5, 6])
    df["weekday"] = ~df["weekend"]

    df["month"] = df["datetime"].dt.month
    
    df["hour"] = df["datetime"].dt.hour
    df["is_night"] = (df["hour"] >= 20) | (df["hour"] <= 5)

    df["morning_rush"] = (
        df["weekday"] &
        (df["hour"] >= 7) &
        (df["hour"] < 9)
    )

    df["evening_rush"] = (
        df["weekday"] &
        (df["hour"] >= 16) &
        (df["hour"] < 19)
    )

    return df

def presence(df):
    df = df.copy()

    df["same_driver_action"] = (
    df["tu1_driver_action_binned"] == df["tu2_driver_action_binned"]
    )

    df["any_unknown_vehicle"] = (
        (df["tu1_vehicle_type_binned"] == "unknown") |
        (df["tu2_vehicle_type_binned"] == "unknown")
    )

    df["any_aggressive_or_careless"] = (
        (df["tu1_driver_action_binned"] == "aggressive_or_careless") |
        (df["tu2_driver_action_binned"] == "aggressive_or_careless")
    )

    df["any_failure_to_yield"] = (
        (df["tu1_driver_action_binned"] == "failure_to_yield") |
        (df["tu2_driver_action_binned"] == "failure_to_yield")
    )

    df["any_unknown_human_factor"] = (
        (df["tu1_human_factor_binned"] == "unknown") |
        (df["tu2_human_factor_binned"] == "unknown")
    )

    return df 

def vehicle_size_actions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    size_missing = df["tu1_vehicle_size"].isna() | df["tu2_vehicle_size"].isna()

    tu1_smaller = df["tu1_vehicle_size"].lt(df["tu2_vehicle_size"]).fillna(False)
    tu2_smaller = df["tu2_vehicle_size"].lt(df["tu1_vehicle_size"]).fillna(False)
    same_size = df["tu1_vehicle_size"].eq(df["tu2_vehicle_size"]).fillna(False)

    df["smaller_vehicle_action"] = np.select(
        [
            size_missing,
            tu1_smaller,
            tu2_smaller,
            same_size,
        ],
        [
            "unknown",
            df["tu1_driver_action_binned"],
            df["tu2_driver_action_binned"],
            "same_size",
        ],
        default="unknown",
    )

    df["larger_vehicle_action"] = np.select(
        [
            size_missing,
            tu2_smaller,  # if TU2 is smaller, TU1 is larger
            tu1_smaller,  # if TU1 is smaller, TU2 is larger
            same_size,
        ],
        [
            "unknown",
            df["tu1_driver_action_binned"],
            df["tu2_driver_action_binned"],
            "same_size",
        ],
        default="unknown",
    )

    return df

def create_highway_indicator(
    df: pd.DataFrame,
    address_col: str = "incident_address"
) -> pd.DataFrame:
    df = df.copy()

    df["incident_address_clean"] = (
        df[address_col]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    highway_pattern = (
        r"\bi[-\s]?25\b|"
        r"\bi[-\s]?70\b|"
        r"\bi[-\s]?225\b|"
        r"\bi25\b|"
        r"\bi70\b|"
        r"\bi225\b|"
        r"\bhwy\b|"
        r"\bhwynb\b|"
        r"\bhwysb\b|"
        r"\bhwyeb\b|"
        r"\bhwywb\b|"
        r"\bhighway\b|"
        r"\bpena blvd\b"
    )

    df["is_highway"] = df["incident_address_clean"].str.contains(
        highway_pattern,
        regex=True,
        na=False
    )

    return df

def direction_conflict_type(
        df: pd.DataFrame,
        col1: str,
        col2: str,
        new_column: str = "direction_conflict"
):
    df = df.copy()

    angle1 = df[col1].map(maps.DIRECTION_ANGLE_MAP) 
    angle2 = df[col2].map(maps.DIRECTION_ANGLE_MAP)

    diff = (angle1 - angle2).abs()
    angle_diff = np.minimum(diff, 360 - diff)

    df[new_column] = np.select(
        [
            angle1.isna() | angle2.isna(),
            angle_diff.eq(0).fillna(False),
            angle_diff.eq(180).fillna(False),
            angle_diff.eq(90).fillna(False),
            angle_diff.isin([45, 135]).fillna(False),
        ],
        [
            "unknown",
            "same_direction",
            "opposite_direction",
            "crossing_or_perpendicular",
            "angled_conflict",
        ],
        default="unknown",
    )

    return df
