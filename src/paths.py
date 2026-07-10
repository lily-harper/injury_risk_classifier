from pathlib import Path
import pandas as pd

def import_data(raw_data_path: Path, input = "csv") -> pd.DataFrame:
    if input == "parquet":
        if not raw_data_path.exists():
            raise FileNotFoundError(f"File not found: {raw_data_path}")
        return pd.read_parquet(raw_data_path) 
    
    elif input == "csv":
        if not raw_data_path.exists():
            raise FileNotFoundError(f"File not found: {raw_data_path}")
        return pd.read_csv(raw_data_path) 

def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "sql").exists() and (parent / "data").exists():
            return parent

    raise FileNotFoundError(
        "Could not find project root. Expected to find both 'sql/' and 'data/' folders."
    )

def save_data(data, path: Path, output = "csv"):
    if output == "csv":
        path.parent.mkdir(parents = True, exist_ok=True)
        data.to_csv(path, index = False)
    elif output == "parquet":
        path.parent.mkdir(parents = True, exist_ok= True)
        data.to_parquet(path, index = False)

    print(f"Data saved to {path} as a {output}")

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

DB_PATH = RAW_DATA_DIR / "traffic.geodatabase"
SQL_PATH = PROJECT_ROOT / "sql" / "build_dataset.sql"

COOR_PATH = RAW_DATA_DIR / "streetcenterlines.geojson"

RAW_PATH = INTERIM_DATA_DIR / "raw_data.csv"
CLEANED_DATA_PATH = INTERIM_DATA_DIR / "cleaner_data.parquet"
BASE_FEATURES = PROCESSED_DATA_DIR / "base_feat_data.parquet"
ALL_DATA = PROCESSED_DATA_DIR / "all_columns.parquet"
MODELING_DATA = PROCESSED_DATA_DIR / "modeling_data.parquet"

SAMPLE_DATA_PATH = SAMPLE_DATA_DIRECTORY / "sample_cleaned_data.csv"
SAMPLE_DATA_PATH_RAW = SAMPLE_DATA_DIRECTORY / "sample_raw_data.csv"

OUTPUT_DIR = PROJECT_ROOT / "output"
METRICS_DIR = OUTPUT_DIR / "metrics" / "individual"
BEST_MODELS_METRICS_DIR = OUTPUT_DIR / "metrics" / "best_models"

REQUIRED_RAW_FILES = {
    "traffic accident geodatabase": DB_PATH,
    "street centerlines GeoJSON": COOR_PATH,
}


def validate_raw_inputs() -> None:
    missing = [
        f"- {description}: {path}"
        for description, path in REQUIRED_RAW_FILES.items()
        if not path.exists()
    ]

    if missing:
        missing_files = "\n".join(missing)
        raise FileNotFoundError(
            "Missing required local raw data files:\n"
            f"{missing_files}\n\n"
            "Download the Denver Traffic Accidents geodatabase and Street "
            "Centerlines GeoJSON, then place them in data/raw/ before running "
            "the full pipeline."
        )
