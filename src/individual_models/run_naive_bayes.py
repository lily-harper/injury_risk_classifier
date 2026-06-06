from src.modeling.models import build_naive_bayes
from src.modeling.ind_model_functions import run_model_family
from src.paths import METRICS_DIR

OUTPUT_NB = METRICS_DIR / "naive_bayes"
OUTPUT_NB.mkdir(parents=True, exist_ok=True)

from pathlib import Path

from src.modeling.ind_model_functions import run_model_family
from src.paths import METRICS_DIR


def main():
    models = {
        "naive_bayes": build_naive_bayes(),}

    run_model_family(
        models=models,
        family="naive_bayes",
        output_dir=OUTPUT_NB,
        save_threshold=True
    )

    print("Saving NB metrics")

if __name__ == "__main__":
    main()