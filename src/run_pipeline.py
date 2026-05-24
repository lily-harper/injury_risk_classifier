from src.pipeline import build_data
from src.pipeline import clean_data
from src.pipeline import features_data
from src.pipeline import geo_features

def main():
    print("Step 1: building raw data")
    build_data.main()

    print("Step 2: Cleaning data...")
    clean_data.main()

    print("Step 3: Making features...")
    features_data.main()

    print("Step 4: Making more features (speed limit)")
    geo_features.main()

    print("Pipeline complete")

if __name__ == "__main__":
    main()