from sklearn.model_selection import PredefinedSplit, GridSearchCV
from src.modeling.models import build_tree_model

import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.modeling.split import split_config, temporal_split
from src.modeling.feature_sets import MODEL_FEATURES
from src.paths import MODELING_DATA


def tune_tree(X_train, X_val, y_train, y_val, output_path=None, scoring="recall"):
    if output_path is None:
        from src.paths import METRICS_DIR
        output_path = METRICS_DIR / "decision_tree" / "best_tree_params.json"

    tree_pipe = build_tree_model(class_weight=None)

    param_grid = {
        "tree__criterion": ["gini", "entropy"],
        "tree__max_depth": [3, 5, 6, 8, 10, None],
        "tree__min_samples_leaf": [10, 25, 50, 100, 200],
        "tree__min_samples_split": [25, 50, 100, 200],
        "tree__class_weight": [None, "balanced"],
        } 

    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)

    test_fold = np.concatenate([
        np.full(len(X_train), -1),  # -1 means always train
        np.zeros(len(X_val))        # 0 means validation fold
        ])

    if len(test_fold) != len(X_train_val):
        raise ValueError(
            "PredefinedSplit length does not match combined training data. "
            "Check tune_tree argument order: X_train, X_val, y_train, y_val."
        )

    predefined_split = PredefinedSplit(test_fold)
    
    grid = GridSearchCV(
        estimator=tree_pipe,
        param_grid=param_grid,
        scoring=scoring,
        cv=predefined_split,
        n_jobs=-1,
        refit=True,
        verbose=1
        )
    
    grid.fit(X_train_val, y_train_val)

    best_params = grid.best_params_
    tree_params = {
        key.replace("tree__", ""): value
        for key, value in best_params.items()
        if key.startswith("tree__")
    }

    output = {
        "scoring": scoring,
        "split": split_config(),
        "best_score": float(grid.best_score_),
        "best_params": best_params,
        "tree_params": tree_params,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)

    return output

def main():
    df = pd.read_parquet(MODELING_DATA)
    X_train, y_train, X_val, y_val, _, _ = temporal_split(
        df,
        features=MODEL_FEATURES,
        target_col="injured",
        date_col="date")
    params =  tune_tree(X_train, X_val, y_train, y_val)
    print("Getting best tree parameters")

if __name__ == "__main__":
    main()
