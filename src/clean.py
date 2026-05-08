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

# dates 

def split_datetime(df, old_date):
    df = df.copy()

    dt = pd.to_datetime(old_date, unit="D", origin="julian")

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
