import pandas as pd

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