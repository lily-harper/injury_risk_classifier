from src.modeling.models import build_tree_model
from src.modeling.ind_model_functions import run_model_family
from src.modeling.feature_sets import MODEL_FEATURES
from src.modeling.split import split_config, temporal_split
from src.modeling.tune_tree import tune_tree
from src.paths import METRICS_DIR, MODELING_DATA

import json
import pandas as pd

OUTPUT_TREE = METRICS_DIR / "decision_tree"
BEST_TREE_PARAMS = OUTPUT_TREE / "best_tree_params.json"

OUTPUT_TREE.mkdir(parents=True, exist_ok=True)


def tree_params_need_tuning() -> bool:
    if not BEST_TREE_PARAMS.exists():
        return True

    with BEST_TREE_PARAMS.open() as f:
        tuned_payload = json.load(f)

    return tuned_payload.get("split") != split_config()


def main():
    if tree_params_need_tuning():
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

    with BEST_TREE_PARAMS.open() as f:
        tuned_payload = json.load(f)
    
    tuned_tree_params = tuned_payload["tree_params"]

    models = {
    "base_tree": build_tree_model(),
    "balanced_tree": build_tree_model(class_weight="balanced"),
    "shallow_balanced_tree": build_tree_model(
        max_depth=4,
        min_samples_leaf=100,
        min_samples_split=100,
        class_weight="balanced",
    ),
    "tuned_tree": build_tree_model(**tuned_tree_params),
}

    run_model_family(
        models=models,
        family="tree",
        output_dir=OUTPUT_TREE,
        save_threshold=False
    )

    print("Saving tree model metrics")

if __name__ == "__main__":
    main()
