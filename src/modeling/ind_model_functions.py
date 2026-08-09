from pathlib import Path
import math
from src.modeling.evaluate import evaluate_model
from src.modeling.split import temporal_split
import pandas as pd
from src.modeling.models import positive_class_proba
from sklearn.metrics import ConfusionMatrixDisplay

from src.modeling.metric_vis import (
    display_model_name,
    plot_roc_curves,
    plot_precision_recall,
    threshold_metrics,
    plot_threshold_metrics,
)

import matplotlib.pyplot as plt


def run_model_family(
    models: dict,
    family: str,
    output_dir: Path,
    save_threshold=True,
    model_thresholds=None,
):
    """Fit models on training data and evaluate only train/validation splits.

    The final test split is returned by temporal_split for later use, but is
    intentionally not evaluated here.
    """
    from src.paths import MODELING_DATA
    from src.modeling.feature_sets import MODEL_FEATURES
    
    output_dir.mkdir(parents=True, exist_ok = True)

    df = pd.read_parquet(MODELING_DATA)

    X_train, y_train, X_val, y_val, _X_test, _y_test = temporal_split(
        df,
        features=MODEL_FEATURES,
        target_col="injured",
        date_col="date")
    
    eval_sets = {
        "train": (X_train, y_train),
        "validation": (X_val, y_val),
    }

    all_metrics = []
    fitted_models = {}
    validation_probas = {}
    validation_confusion_matrices = {}
    model_thresholds = model_thresholds or {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[model_name] = model

        is_probabalistic = hasattr(model, "predict_proba")
        threshold = model_thresholds.get(model_name, 0.5)

        for split_name, (X_eval, y_eval) in eval_sets.items():
            metrics, cm = evaluate_model(
                model=model,
                model_name=model_name,
                X_test=X_eval,
                y_test=y_eval,
                threshold=threshold,
                return_confusion=True
            )

            metrics["split"] = split_name
            all_metrics.append(metrics) 

            if split_name == "validation":
                validation_confusion_matrices[model_name] = {
                    "matrix": cm,
                    "threshold": threshold,
                    "recall": metrics["recall"],
                }

        if is_probabalistic:
            y_val_proba = positive_class_proba(
                model, X_val, positive_label=1
            )

            validation_probas[model_name] = y_val_proba

        if is_probabalistic and save_threshold:
            threshold_df = threshold_metrics(
                y_true=y_val,
                y_proba=y_val_proba,
            )

            threshold_df.to_csv(
                output_dir / f"{model_name}_threshold_metrics_validation.csv",
                index=False,
            )

            fig, ax = plot_threshold_metrics(
                threshold_df,
                save_path=output_dir / f"{model_name}_threshold_metrics_validation.png",
                model_name=model_name
            )
            plt.close(fig)

    metrics_df = pd.DataFrame(all_metrics)
    first_cols = ["model", "split", "recall"]
    metrics_df = metrics_df[
        first_cols + [col for col in metrics_df.columns if col not in first_cols]
    ]

    metrics_df.to_csv(
        output_dir / f"{family}_model_metrics_by_split.csv",
        index=False,
    )

    if validation_confusion_matrices:
        n_models = len(validation_confusion_matrices)
        n_cols = min(2, n_models)
        n_rows = math.ceil(n_models / n_cols)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(6 * n_cols, 5 * n_rows),
            squeeze=False,
        )

        flat_axes = axes.ravel()

        for ax, (model_name, details) in zip(flat_axes, validation_confusion_matrices.items()):
            display_name = display_model_name(model_name)
            disp = ConfusionMatrixDisplay(
                confusion_matrix=details["matrix"],
                display_labels=["No injury", "Injury"],
            )
            disp.plot(ax=ax, values_format="d", colorbar=False)
            ax.set_title(
                f"{display_name} Validation\n"
                f"threshold = {details['threshold']:.2f}, "
                f"recall = {details['recall']:.3f}"
            )

        for ax in flat_axes[n_models:]:
            ax.axis("off")

        fig.tight_layout()
        fig.savefig(
            output_dir / f"{family}_confusion_matrices_validation.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    if validation_probas:
        family_display_name = family.replace("_", " ").title()
        fig, ax = plot_roc_curves(
            model_preds=validation_probas,
            y_true=y_val,
            title=f"{family_display_name} Validation ROC Curves",
            save_path=output_dir / f"{family}_roc_curves_validation.png",
        )
        plt.close(fig)

        fig, ax = plot_precision_recall(
            model_preds=validation_probas,
            y_true=y_val,
            skip_model_names={"dummy_no_injury"},
            title=f"{family_display_name} Validation Precision–Recall Curves",
            save_path=output_dir / f"{family}_pr_curves_validation.png",
        )
        plt.close(fig)

    return metrics_df, validation_probas, fitted_models
                
