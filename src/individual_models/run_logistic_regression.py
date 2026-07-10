from src.modeling.models import build_logistic_model
from src.modeling.ind_model_functions import run_model_family
from src.paths import METRICS_DIR

OUTPUT_LOGREG = METRICS_DIR / "logreg_models"
OUTPUT_LOGREG.mkdir(parents=True, exist_ok=True)

from pathlib import Path

from src.paths import METRICS_DIR


def main():
    models = {
        "logistic_unbalanced": build_logistic_model(class_weight=None),
        "logistic_balanced": build_logistic_model(class_weight="balanced"),}

    run_model_family(
        models=models,
        family="logistic",
        output_dir=OUTPUT_LOGREG,
        save_threshold=True,
        model_thresholds={"logistic_balanced": 0.48},
    )

    print("Saving logistic model metrics")

if __name__ == "__main__":
    main()
