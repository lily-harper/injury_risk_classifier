from pathlib import Path

from src.modeling.ind_model_functions import run_model_family
from src.modeling.models import build_dummy_model
from src.paths import METRICS_DIR

OUTPUT_DUMMY = METRICS_DIR / "dummy_models"

def main():
    models = {
        "dummy_all_false": build_dummy_model(0),
        "dummy_all_true": build_dummy_model(1)
    }

    run_model_family(
        models=models,
        family="dummy",
        output_dir=OUTPUT_DUMMY,
        save_threshold=False
    )

    print("Saving dummy model metrics")

if __name__ == "__main__":
    main()
