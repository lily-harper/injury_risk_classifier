from pathlib import Path
import sqlite3
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# CHANGE THE TRAFFIC.GEODATABSE TO YOUR DOWNLOADED FILE NAME FROM THE WEBSITE 
DB_PATH = PROJECT_ROOT / "data" / "raw" / "traffic.geodatabase"
SQL_PATH = PROJECT_ROOT / "sql" / "build_dataset.sql"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "raw_modeling_data.csv"


def read_sql_file(sql_path: Path) -> str:
    """Read a SQL query from a .sql file."""
    with open(sql_path, "r", encoding="utf-8") as file:
        return file.read()


def run_query(db_path: Path, query: str) -> pd.DataFrame:
    """Run a SQL query against a SQLite-compatible database."""
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)


def main() -> None:
    """Build the raw modeling dataset from the source database."""
    query = read_sql_file(SQL_PATH)
    df = run_query(DB_PATH, query)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"Saved {df.shape[0]} rows and {df.shape[1]} columns to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()