from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

from src.modeling.evaluate import evaluate_model
from src.modeling.feature_sets import MODEL_FEATURES
from src.modeling.metric_vis import plot_precision_recall, plot_roc_curves
from src.modeling.models import build_logistic_model, positive_class_proba
from src.modeling.split import temporal_split
from src.paths import FINAL_TEST_METRICS_DIR, MODELING_DATA


FINAL_MODEL_NAME = "logistic_balanced"
FINAL_THRESHOLD = 0.48


def main():
    FINAL_TEST_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(MODELING_DATA)
    X_train, y_train, X_val, y_val, X_test, y_test = temporal_split(
        df,
        features=MODEL_FEATURES,
        target_col="injured",
        date_col="date",
    )

    X_train_final = pd.concat([X_train, X_val], axis=0)
    y_train_final = pd.concat([y_train, y_val], axis=0)

    model = build_logistic_model(class_weight="balanced")
    model.fit(X_train_final, y_train_final)

    metrics, confusion = evaluate_model(
        model=model,
        model_name=FINAL_MODEL_NAME,
        X_test=X_test,
        y_test=y_test,
        threshold=FINAL_THRESHOLD,
        return_confusion=True,
    )

    metrics["split"] = "final_test"
    metrics["train_period"] = "through_2024"
    metrics["test_period"] = "2025_and_later"

    metrics_df = pd.DataFrame([metrics])
    first_cols = ["model", "split", "recall"]
    metrics_df = metrics_df[
        first_cols + [col for col in metrics_df.columns if col not in first_cols]
    ]
    metrics_df.to_csv(
        FINAL_TEST_METRICS_DIR / "final_test_model_metrics.csv",
        index=False,
    )

    y_test_proba = positive_class_proba(
        model,
        X_test,
        positive_label=1,
    )

    fig, ax = plot_precision_recall(
        model_preds={FINAL_MODEL_NAME: y_test_proba},
        y_true=y_test,
        save_path=FINAL_TEST_METRICS_DIR / "final_test_pr_curve.png",
    )
    plt.close(fig)

    fig, ax = plot_roc_curves(
        model_preds={FINAL_MODEL_NAME: y_test_proba},
        y_true=y_test,
        title="Final Test ROC Curve",
        save_path=FINAL_TEST_METRICS_DIR / "final_test_roc_curve.png",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion,
        display_labels=["No injury", "Injury"],
    )
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(
        f"{FINAL_MODEL_NAME} Final Test\n"
        f"threshold = {FINAL_THRESHOLD:.2f}, recall = {metrics['recall']:.3f}"
    )
    fig.tight_layout()
    fig.savefig(
        FINAL_TEST_METRICS_DIR / "final_test_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved final test metrics to {FINAL_TEST_METRICS_DIR}")


if __name__ == "__main__":
    main()
