import argparse

from src.individual_models import run_every_model
from src.pipeline import build_data
from src.pipeline import clean_data
from src.pipeline import features_data
from src.pipeline import geo_features
from src.pipeline import run_model_comparisons
from src.paths import validate_raw_inputs


def main(run_individual_models=False):
    validate_raw_inputs()

    print("Step 1: building raw data")
    build_data.main()

    print("Step 2: Cleaning data...")
    clean_data.main()

    print("Step 3: Making features...")
    features_data.main()

    print("Step 4: Making more features (speed limit)")
    geo_features.main()

    if run_individual_models:
        print("Step 5: Running individual model candidates...")
        run_every_model.main(run_comparison=False)

        print("Step 6: Running finalist model comparison...")
        run_model_comparisons.main()
    else:
        print("Step 5: Running finalist model comparison...")
        run_model_comparisons.main()

    print("Pipeline complete")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the injury-risk data pipeline."
    )
    parser.add_argument(
        "--run-individual-models",
        action="store_true",
        help=(
            "Also run individual dummy, logistic regression, decision tree, "
            "and naive Bayes candidate diagnostics before the finalist comparison."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(run_individual_models=args.run_individual_models)
