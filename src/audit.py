import pandas as pd
from collections import Counter

def quick_audit(df, target="injured"):
    print("shape:", df.shape)
    print("\ncolumns:", len(df.columns))

    if target in df.columns:
        print("\ntarget counts:")
        print(df[target].value_counts(dropna=False))

        print("\ntarget rates:")
        print(df[target].value_counts(normalize=True, dropna=False))

    print("\nmissing top 10:")
    print(df.isna().mean().sort_values(ascending=False).head(10))

def check_rows_per_year(
    df: pd.DataFrame,
    datetime_col: str = "datetime",
    sort: bool = True
) -> pd.DataFrame:
    """
    Counts rows per year from an existing datetime column.
    """
    out = df.copy()

    out[datetime_col] = pd.to_datetime(out[datetime_col], errors="coerce")

    year_counts = (
        out.assign(year=out[datetime_col].dt.year)
           .groupby("year", dropna=False)
           .size()
           .reset_index(name="n_rows")
    )

    year_counts["pct_rows"] = year_counts["n_rows"] / len(out)

    if sort:
        year_counts = year_counts.sort_values("year")

    return year_counts

def audit_bins(df, group):
    # These are the raw values from your actual dataframe
    raw_values = (
        df["tu1_vehicle_type"]
        .astype("string")
        .str.strip()
        .str.lower()
        .dropna()
        .unique()
    )

    # Flatten all mapped raw values
    mapped_values = [
        raw_value
        for raw_values in group.values()
        for raw_value in raw_values
    ]

    counts = Counter(mapped_values)

    duplicates = [value for value, count in counts.items() if count > 1]
    missing = [value for value in raw_values if value not in counts]
    extra = [value for value in counts if value not in raw_values]

    print("Duplicates:", duplicates)
    print("Missing:", missing)
    print("Extra:", extra)