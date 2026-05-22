from pathlib import Path

def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "sql").exists() and (parent / "data").exists():
            return parent

    raise FileNotFoundError(
        "Could not find project root. Expected to find both 'sql/' and 'data/' folders."
    )

PROJECT_ROOT = find_project_root(Path(__file__).resolve())

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIRECTORY = DATA_DIR / "sample"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SRC_DIR = PROJECT_ROOT / "src"

DB_PATH = PROJECT_ROOT / RAW_DATA_DIR / "traffic.geodatabase"
SQL_PATH = PROJECT_ROOT / "sql" / "build_dataset.sql"

COOR_PATH = RAW_DATA_DIR / "streetcenterlines.geojson"

RAW_PATH = INTERIM_DATA_DIR / "raw_data.csv"
CLEANED_DATA_PATH = INTERIM_DATA_DIR / "cleaner_data.parquet"
MODELING_DATA = PROCESSED_DATA_DIR / "modeling_data.parquet"

SAMPLE_DATA_PATH = SAMPLE_DATA_DIRECTORY / "sample_cleaned_data.csv"
SAMPLE_DATA_PATH_RAW = SAMPLE_DATA_DIRECTORY / "sample_raw_data.csv"