from src.modeling.models import (build_logistic_model, 
                                 build_dummy_model,
                                 build_naive_bayes,
                                 build_tree_model)

from src.modeling.ind_model_functions import run_model_family
from src.paths import METRICS_DIR
from pathlib import Path

import json

OUTPUT_TREE = METRICS_DIR / "decision_tree"
BEST_TREE_PARAMS = OUTPUT_TREE / "best_tree_params.json"

OUTPUT_ALL_DIR = Path("output/metrics/best_models")
OUTPUT_ALL_DIR.mkdir(parents=True, exist_ok=True)

def main():
    with BEST_TREE_PARAMS.open() as f:
        tuned_payload = json.load(f)
    
    tuned_tree_params = tuned_payload["tree_params"]
    models = {
        "dummy_no_injury": build_dummy_model(0),
        "logistic_balanced": build_logistic_model(class_weight="balanced"),
        "tuned_tree": build_tree_model(**tuned_tree_params),
        "naive_bayes": build_naive_bayes(),
    }
    
    run_model_family(
        models=models,
        family="best_models",
        output_dir=OUTPUT_ALL_DIR,
        save_threshold=False
    )

    print("Saving all model metrics")

if __name__ == "__main__":
    main()
