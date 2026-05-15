import pandas as pd 
import numpy as np

def vehicle_match(row):
    v1 = row["tu1_vehicle_type_binned"]
    v2 = row["tu2_vehicle_type_binned"]

    if v1 == "unknown" or v2 == "unknown":
        return "unknown"
    elif v1 == v2:
        return "same"
    else:
        return "different"

vehicle_size_map = {
    "motorcycle": 1,
    "bicycle": 1,
    "passenger_car_or_van": 2,
    "suv": 3,
    "pickup_or_utility_van": 3,
    "bus": 4,
    "truck_or_heavy_vehicle": 5,
    "unknown": None
}

def vehicle_sizes(df):
    df = df.copy()

    df["tu1_vehicle_size"] = df["tu1_vehicle_type_binned"].map(vehicle_size_map)
    df["tu2_vehicle_size"] = df["tu2_vehicle_type_binned"].map(vehicle_size_map)

    df["vehicle_size_diff"] = df["tu1_vehicle_size"] - df["tu2_vehicle_size"]
    df["abs_vehicle_size_diff"] = df["vehicle_size_diff"].abs()

    return df 

def size_relation(df):
    df["vehicle_size_relation"] = np.select(
        [
            df["tu1_vehicle_size"].isna() | df["tu2_vehicle_size"].isna(),
            df["tu1_vehicle_size"] > df["tu2_vehicle_size"],
            df["tu1_vehicle_size"] < df["tu2_vehicle_size"],
            df["tu1_vehicle_size"] == df["tu2_vehicle_size"]
        ],
        [
            "unknown",
            "vehicle_1_larger",
            "vehicle_2_larger",
            "same_size"
        ],
        default="unknown"
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

    df["smaller_vehicle_action"] = np.select(
    [
        df["tu1_vehicle_size"].isna() | df["tu2_vehicle_size"].isna(),
        df["tu1_vehicle_size"] < df["tu2_vehicle_size"],
        df["tu2_vehicle_size"] < df["tu1_vehicle_size"],
        df["tu1_vehicle_size"] == df["tu2_vehicle_size"]
    ],
    [
        "unknown",
        df["tu1_driver_action_binned"],
        df["tu2_driver_action_binned"],
        "same_size"
    ],
    default="unknown"
    )
    
    df["larger_vehicle_action"] = np.select(
    [
        df["tu1_vehicle_size"].isna() | df["tu2_vehicle_size"].isna(),
        df["tu1_vehicle_size"] > df["tu2_vehicle_size"],
        df["tu2_vehicle_size"] > df["tu1_vehicle_size"],
        df["tu1_vehicle_size"] == df["tu2_vehicle_size"]
    ],
    [
        "unknown",
        df["tu1_driver_action_binned"],
        df["tu2_driver_action_binned"],
        "same_size"
    ],
    default="unknown"
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