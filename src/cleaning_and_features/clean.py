import pandas as pd

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

def clean_text_colummns(df, columns):
    df = df.copy()

    for col in columns:
        df[col] = (
            df[col]
            .astype("string")
            .str.lower()
            .str.strip()
        )
    
    return df 

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

# Binning mesy text 

def bin_driver_action(x):
    if pd.isna(x) or str(x).strip() == "":
        return "unknown"

    x = str(x).strip().lower()

    no_action = {
        "", "00", "0", "-", "none", "non",
        "no action", "no contributing action"
    }

    unknown = {
        "<na>", "under investigation"
    }

    speed = {
        "speeding",
        "exceed safe/posted speed",
        "too fast for conditions",
        "racing"
    }

    lane_position = {
        "lane violation",
        "improper passing on left",
        "improper passing on right",
        "turned from wrong lane or position",
        "over-correcting/over-steering",
        "signaling violation"
    }

    failure_to_obey = {
        "failed to stop at signal",
        "disregard stop sign",
        "disregarded stop sign",
        "disregarded other device",
        "disregarded other device/sign/markings"
    }

    turning_backing = {
        "improper backing",
        "improper turn",
        "other improper turns"
    }

    aggressive_careless = {
        "careless driving",
        "reckless driving",
        "followed too closely"
    }

    yield_row = {
        "failed to yield row"
    }

    traffic_flow = {
        "impeded traffic"
    }

    other = {
        "other",
        "other contributing action"
    }

    if x in no_action:
        return "no_action"
    elif x in unknown:
        return "unknown"
    elif x in speed:
        return "speed_related"
    elif x in lane_position:
        return "lane_or_position"
    elif x in failure_to_obey:
        return "failure_to_obey"
    elif x in turning_backing:
        return "turning_or_backing"
    elif x in aggressive_careless:
        return "aggressive_or_careless"
    elif x in yield_row:
        return "failure_to_yield"
    elif x in traffic_flow:
        return "traffic_flow"
    elif x in other:
        return "other"
    else:
        return "other"
    

def bin_human_contrib_factor(x):
    """
    Bin messy human contributing factor values into broader modeling categories.
    Designed to be used with .apply() on a single pandas column.
    """

    if pd.isna(x) or str(x).strip() == "":
        return "unknown"

    x = str(x).strip().lower()

    no_apparent = {
        "no apparent",
        "no apparent contributing factor",
        "00",
        "0"
    }

    unknown = {
        "<na>",
        "not observed",
        "under investigation",
        "17",
        "18",
        "06"
    }

    other = {
        "other",
        "other factor"
    }

    aggressive = {
        "aggressive driving",
        "evading law enforcement officer"
    }

    distracted = {
        "distracted-other",
        "distracted/other interior",
        "distracted/other exterior",
        "distracted passenger",
        "distracted radio",
        "distracted/other occupant",
        "distracted/manipulating vehicle control",
        "distracted eating/drinking",
        "distracted/smoking",
        "looked/did not see"
    }

    phone_or_device = {
        "distracted cellphone",
        "manipulating electronic device",
        "talking on phone/holding",
        "talking on phone/hands free"
    }

    impairment = {
        "dui/dwai/duid"
    }

    inexperience_unfamiliar = {
        "driver inexperience",
        "driver unfamiliar with area"
    }

    fatigue_sleep = {
        "driver fatigue",
        "asleep at the wheel",
        "asleep or fatigued"
    }

    medical_ability = {
        "illness/medical",
        "illness",
        "medical",
        "physical disability",
        "age/driver ability"
    }

    emotional = {
        "driver emotionally upset"
    }

    environmental_visibility = {
        "sun glare"
    }

    if x in no_apparent:
        return "no_apparent"
    elif x in unknown:
        return "unknown"
    elif x in other:
        return "other"
    elif x in aggressive:
        return "aggressive_or_evading"
    elif x in distracted:
        return "distracted"
    elif x in phone_or_device:
        return "phone_or_device"
    elif x in impairment:
        return "impairment"
    elif x in inexperience_unfamiliar:
        return "inexperience_or_unfamiliar"
    elif x in fatigue_sleep:
        return "fatigue_or_sleep"
    elif x in medical_ability:
        return "medical_or_ability"
    elif x in emotional:
        return "emotional"
    elif x in environmental_visibility:
        return "environmental_visibility"
    else:
        return "other"

def bin_vehicle_type(x):
    """
    Bin messy vehicle type values into broader modeling categories.
    """

    if x is None or str(x).strip() == "" or str(x).strip().lower() in {"<na>", "nan", "none"}:
        return "unknown"

    x = str(x).strip().lower()

    passenger = {
        "passenger car/van",
        "passenger car/passenger van",
        "passenger car/van with trailer"
    }

    suv = {
        "suv",
        "suv with trailer"
    }

    pickup_or_van = {
        "pickup truck/utility van",
        "pickup truck/utility van with trailier",
        "pickup truck/utility van with trailer"
    }

    heavy_truck = {
        "vehicle over 10000 lbs",
        "medium/heavy trucks gvwr/gcwr 16,001 and over",
        "medium/heavy trucks gvwr/gcwr between 10,001 and 16,000"
    }

    bus = {
        "transit bus",
        "non-school bus",
        "school bus",
        "non-school bus (9 occupants or more including driver) in commerce",
        "school bus (all school buses)"
    }

    vulnerable_road_user = {
        "bicycle",
        "motorized bicycle"
    }

    motorcycle_like = {
        "motorcycle",
        "autocycle"
    }

    rail = {
        "light rail",
        "heavy train"
    }

    equipment_or_special = {
        "working vehicle/equipment",
        "farm equipment",
        "off highway vehicle/atv",
        "snowmobile",
        "low speed vehicle"
    }

    unknown = {
        "hit and run unknown",
        "unknown (hit and run only)",
        "under investigation",
        "unk",
        "0"
    }

    other = {
        "other",
        "other vehicle type (describe in narative)",
        "other vehicle type (describe in narrative)"
    }

    if x in passenger:
        return "passenger_car_or_van"
    elif x in suv:
        return "suv"
    elif x in pickup_or_van:
        return "pickup_or_utility_van"
    elif x in heavy_truck:
        return "heavy_truck"
    elif x in bus:
        return "bus"
    elif x in vulnerable_road_user:
        return "bicycle_or_motorized_bicycle"
    elif x in motorcycle_like:
        return "motorcycle_or_autocycle"
    elif x in rail:
        return "rail"
    elif x in equipment_or_special:
        return "equipment_or_special_vehicle"
    elif x == "motor home":
        return "motor_home"
    elif x in unknown:
        return "unknown"
    elif x in other:
        return "other"
    else:
        return "other"
    
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
        df[col] = pd.to_numeric(df[col], errors=errors).astype("Int64")

    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors=errors)

    for col in bool_cols:
        df[col] = df[col].astype("boolean")
        # nullable boolean type

    for col in category_cols:
        df[col] = df[col].astype("category")

    for col in string_cols:
        df[col] = df[col].astype("string")

    return df