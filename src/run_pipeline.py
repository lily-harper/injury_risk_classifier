from src.pipeline import build_data
from src.pipeline import clean_data
from src.pipeline import features_data
from src.pipeline import geo_features
from src.pipeline import run_model_comparisons

def main():
    print("Step 1: building raw data")
    build_data.main()

    print("Step 2: Cleaning data...")
    clean_data.main()

    print("Step 3: Making features...")
    features_data.main()

    print("Step 4: Making more features (speed limit)")
    geo_features.main()

    print("Step 5: Running models...")
    run_model_comparisons.main()

    print("Pipeline complete")

if __name__ == "__main__":
    main()