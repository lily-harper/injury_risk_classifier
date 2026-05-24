import sqlite3
from pathlib import Path

import pandas as pd


def list_tables(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table';
            """,
            conn,
        )


def table_schema(db_path: Path, table_name: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            f'PRAGMA table_info("{table_name}");',
            conn,
        )


def count_rows(db_path: Path, table_name: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            f'SELECT COUNT(*) AS n_rows FROM "{table_name}";',
            conn,
        )


def value_counts(db_path: Path, table_name: str, column: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            f"""
            SELECT "{column}", COUNT(*) AS n
            FROM "{table_name}"
            GROUP BY "{column}"
            ORDER BY n DESC;
            """,
            conn,
        )