from src.modeling.models import (build_dummy_model,
                                 build_logistic_model,
                                 build_base_tree_model)
from src.modeling.split import temporal_split

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.paths import MODELING_DATA
from src.modeling.feature_sets import MODEL_FEATURES
from src.modeling.evaluate import evaluate_model
from src.modeling.metric_vis import plot_precision_recall, plot_roc_curves
from src.paths import METRICS_DIR

from src.modeling.models import positive_class_proba

OUTPUT_ALL_DIR = METRICS_DIR / "all_models"
OUTPUT_ALL_DIR.mkdir(parents=True, exist_ok=True)

def run_models():
    df = pd.read_parquet(MODELING_DATA)
    X_train, y_train, X_val, y_val, X_test, y_test = temporal_split(
        df,
        features=MODEL_FEATURES,
        target_col="injured",
        date_col="date")

    MODELS = {
        "dummy_all_false": build_dummy_model(0),
        "logistic_unbalanced": build_logistic_model(class_weight=None),
        "logistic_balanced": build_logistic_model(class_weight="balanced"),
        "base_decision_tree": build_base_tree_model(class_weight="balanced"),
        }

    all_metrics = []

    fitted_models = {}
    val_probas = {}

    for model_name, model in MODELS.items():
        model.fit(X_train, y_train)

        fitted_models[model_name] = model
        val_probas[model_name] = positive_class_proba(model, X_val, positive_label=1)

        metrics = evaluate_model(
            model=model,
            model_name=model_name,
            X_test=X_val,
            y_test=y_val,
        )
        all_metrics.append(metrics)

    comparison_df = pd.DataFrame(all_metrics)

    comparison_df.to_csv(OUTPUT_ALL_DIR / "model_comparison.csv", index=False)

    return val_probas, y_val

def vis_models(val_probas, y_val):
    OUTPUT_ALL_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plot_roc_curves(
        model_preds=val_probas, 
        y_true=y_val,
        title="Validation ROC Curves by Model",
        save_path= OUTPUT_ALL_DIR / "all_roc_curves.png",
    )
    plt.close(fig)

    fig, ax = plot_precision_recall(
        model_preds=val_probas,
        y_true = y_val,
        save_path= OUTPUT_ALL_DIR / "all_precision_recall_curves.png"
    )
    plt.close(fig)

def main():
    val_probas, y_val = run_models()
    vis_models(val_probas, y_val)

if __name__ == "__main__":
    main()
