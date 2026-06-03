from src.modeling.models import build_dummy_model
from src.modeling.split import temporal_split
from src.modeling.evaluate import evaluate_model

import pandas as pd 
import matplotlib.pyplot as plt
from src.modeling.metric_vis import (
    plot_roc_curves,
    plot_precision_recall,
    threshold_metrics,
    plot_threshold_metrics,
)

from src.paths import MODELING_DATA
from src.modeling.feature_sets import MODEL_FEATURES
from src.modeling.models import positive_class_proba
from src.paths import METRICS_DIR
from sklearn.metrics import ConfusionMatrixDisplay

OUTPUT_DUMMY = METRICS_DIR / "dummy_models"
OUTPUT_DUMMY.mkdir(parents=True, exist_ok=True)

def dummy():
    df = pd.read_parquet(MODELING_DATA)
    X_train, y_train, X_val, y_val, _, _ = temporal_split(
        df,
        features=MODEL_FEATURES,
        target_col="injured",
        date_col="date")

    MODELS = {
        "dummy_all_false": build_dummy_model(0),
        "dummy_all_true": build_dummy_model(1),}

    all_metrics = []

    fitted_models = {}
    val_probas = {}

    for model_name, model in MODELS.items():
        model.fit(X_train, y_train)

        fitted_models[model_name] = model
        val_probas[model_name] = positive_class_proba(model, X_val, positive_label=1)

        metrics, cm = evaluate_model(
            model=model,
            model_name=model_name,
            X_test=X_val,
            y_test=y_val,
            return_confusion=True
        )
        all_metrics.append(metrics)

    comparison_df = pd.DataFrame(all_metrics)


    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                  display_labels=["No injury", "Injury"])

    fig, ax = plt.subplots(figsize=(6,5)) 
    disp.plot(ax=ax, values_format="d", colorbar=False)

    ax.set_title(f"{model_name} Confusion Matrix")

    fig.savefig(
        OUTPUT_DUMMY / f"{model_name}_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
        )
    
    comparison_df.to_csv(OUTPUT_DUMMY / "dummy_model_metrics.csv", index=False)

    return val_probas, y_val

def vis_models(val_probas, y_val):
    OUTPUT_DUMMY.mkdir(parents=True, exist_ok=True)

    fig, ax = plot_roc_curves(
        model_preds=val_probas, 
        y_true=y_val,
        title="Validation ROC Curves by Model",
        save_path= OUTPUT_DUMMY / "dummy_roc_curves.png",
    )
    plt.close(fig)

    fig, ax = plot_precision_recall(
        model_preds=val_probas,
        y_true = y_val,
        save_path= OUTPUT_DUMMY / "dummy_pr_curves.png"
    )
    plt.close(fig)


def main():
    val_probas, y_val = dummy()
    vis_models(val_probas, y_val)

if __name__ == "__main__":
    main()