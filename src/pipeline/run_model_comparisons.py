from src.modeling.models import (build_logistic_model, 
                                 build_dummy_model,
                                 build_naive_bayes,
                                 build_tree_model)

from src.modeling.ind_model_functions import run_model_family
from src.modeling.feature_sets import MODEL_FEATURES
from src.modeling.split import split_config, temporal_split
from src.modeling.tune_tree import tune_tree
from src.paths import BEST_MODELS_METRICS_DIR, METRICS_DIR, MODELING_DATA

import json
import pandas as pd

OUTPUT_TREE = METRICS_DIR / "decision_tree"
BEST_TREE_PARAMS = OUTPUT_TREE / "best_tree_params.json"

OUTPUT_ALL_DIR = BEST_MODELS_METRICS_DIR
OUTPUT_ALL_DIR.mkdir(parents=True, exist_ok=True)


def tree_params_need_tuning() -> bool:
    if not BEST_TREE_PARAMS.exists():
        return True

    with BEST_TREE_PARAMS.open() as f:
        tuned_payload = json.load(f)

    return tuned_payload.get("split") != split_config()


def ensure_current_tree_params():
    if not tree_params_need_tuning():
        return

    df = pd.read_parquet(MODELING_DATA)
    X_train, y_train, X_val, y_val, _, _ = temporal_split(
        df,
        features=MODEL_FEATURES,
        target_col="injured",
        date_col="date",
    )

    tune_tree(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        output_path=BEST_TREE_PARAMS,
    )


def main():
    ensure_current_tree_params()

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
        save_threshold=False,
        model_thresholds={
            "logistic_balanced": 0.48,
            "naive_bayes": 0.10,
        },
    )

    print("Saving all model metrics")

if __name__ == "__main__":
    main()
