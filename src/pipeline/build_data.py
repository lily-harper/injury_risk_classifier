import pandas as pd
from pathlib import Path
import sqlite3

from src.paths import (PROJECT_ROOT, DB_PATH, 
                       RAW_DATA_DIR, SQL_PATH)

RAW_PATH = PROJECT_ROOT / "data" / "interim" / "raw_data.csv"

def read_sql_file(sql_path: Path) -> str:
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found at: {sql_path}")
    
    with open(sql_path, "r", encoding = "utf-8") as file:
        return file.read()
    
def run_query(df_path: Path, query: str) -> pd.DataFrame:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database file not found at: {DB_PATH}"
        )

    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(query, conn)

def main() -> None:
    query = read_sql_file(SQL_PATH)
    df = run_query(DB_PATH, query)

    RAW_PATH.parent.mkdir(parents=True, exist_ok= True)
    df.to_csv(RAW_PATH, index = False, encoding="utf-8")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Saved {df.shape[0]} rows and {df.shape[1]} columns to {RAW_PATH}")

if __name__== "__main__":
    main()