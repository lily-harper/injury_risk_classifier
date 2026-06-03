from src.modeling.models import build_logistic_model, positive_class_proba
from src.modeling.split import temporal_split
from src.modeling.evaluate import evaluate_model

from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd 
import matplotlib.pyplot as plt

from src.paths import MODELING_DATA
from src.modeling.feature_sets import MODEL_FEATURES
from src.paths import METRICS_DIR
from src.modeling.metric_vis import (
    plot_roc_curves,
    plot_precision_recall,
    threshold_metrics,
    plot_threshold_metrics,
)
OUTPUT_LOGREG = METRICS_DIR / "logreg_models"
OUTPUT_LOGREG.mkdir(parents=True, exist_ok=True)

def logreg():
    df = pd.read_parquet(MODELING_DATA)
    X_train, y_train, X_val, y_val, _, _ = temporal_split(
        df,
        features=MODEL_FEATURES,
        target_col="injured",
        date_col="date")

    MODELS = {
        "logistic_unbalanced": build_logistic_model(class_weight=None),
        "logistic_balanced": build_logistic_model(class_weight="balanced"),}

    all_metrics = []

    fitted_models = {}
    val_probas = {}

    for model_name, model in MODELS.items():
        model.fit(X_train, y_train)

        fitted_models[model_name] = model
        val_probas[model_name] = positive_class_proba(model, X_val, positive_label=1)

        metrics, cm= evaluate_model(
            model=model,
            model_name=model_name,
            X_test=X_val,
            y_test=y_val,
            return_confusion = True,
        )
        all_metrics.append(metrics)

    comparison_df = pd.DataFrame(all_metrics)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                  display_labels=["No injury", "Injury"])

    fig, ax = plt.subplots(figsize=(6,5)) 
    disp.plot(ax=ax, values_format="d", colorbar=False)

    ax.set_title(f"{model_name} Confusion Matrix")

    fig.savefig(
        OUTPUT_LOGREG / f"{model_name}_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
        )
    
    comparison_df.to_csv(OUTPUT_LOGREG / "logreg_model_metrics.csv", index=False)

    
    return val_probas, y_val

def vis_models(val_probas, y_val):
    OUTPUT_LOGREG.mkdir(parents=True, exist_ok=True)

    fig, ax = plot_roc_curves(
        model_preds=val_probas, 
        y_true=y_val,
        title="Validation ROC Curves by Model",
        save_path= OUTPUT_LOGREG / "logreg_roc_curves.png",
    )
    plt.close(fig)

    fig, ax = plot_precision_recall(
        model_preds=val_probas,
        y_true = y_val,
        save_path= OUTPUT_LOGREG / "logreg_pr_curves.png"
    )
    plt.close(fig)

    for model_name, y_proba in val_probas.items():
        threshold_df = threshold_metrics(
            y_true = y_val,
            y_proba=y_proba
        )

        threshold_df.to_csv(
            OUTPUT_LOGREG / f"{model_name}_threshold_metrics.csv",
            index=False,
        )

        fig, ax = plot_threshold_metrics(
            threshold_df,
            save_path=OUTPUT_LOGREG / f"{model_name}_threshold_metrics.png",
        )
        plt.close(fig)


def main():
    val_probas, y_val = logreg()
    vis_models(val_probas, y_val)

if __name__ == "__main__":
    main()