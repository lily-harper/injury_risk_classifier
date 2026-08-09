import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    roc_auc_score
)

MODEL_DISPLAY_NAMES = {
    "dummy_no_injury": "No-Injury Dummy Baseline",
    "logistic_balanced": "Balanced Logistic Regression",
    "tuned_tree": "Tuned Decision Tree",
    "naive_bayes": "Naive Bayes",
}


def display_model_name(model_name):
    return MODEL_DISPLAY_NAMES.get(model_name, model_name.replace("_", " ").title())

def plot_precision_recall(
        model_preds,
        y_true,
        save_path=None,
        skip_model_names=None,
        show_baseline=True,
        title="Precision–Recall Curve",
):
    fig, ax = plt.subplots(figsize = (8, 6))
    skip_model_names = set(skip_model_names or [])

    if show_baseline:
        prevalence = np.mean(y_true)
        ax.axhline(
            prevalence,
            color="gray",
            linestyle="--",
            linewidth=2,
            label=f"Baseline prevalence = {prevalence:.3f}",
        )

    for model_name, y_proba in model_preds.items():
        if model_name in skip_model_names:
            continue

        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        display_name = display_model_name(model_name)

        ax.plot(
            recall,
            precision,
            linewidth = 2,
            label=f"{display_name} AP = {avg_precision:.3f}"
        )

    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    ax.grid(True, alpha = .25)
    ax.legend(loc ="upper right")

    if save_path is not None:
        fig.savefig(save_path, dpi = 300, bbox_inches = "tight")
    
    return fig, ax

def plot_roc_curves(
        model_preds,
        y_true,
        title = "ROC Curve",
        save_path = None
):
    fig, ax = plt.subplots(figsize = (8, 6))

    for model_name, y_proba in model_preds.items():
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        display_name = display_model_name(model_name)

        ax.plot(
            fpr, tpr, linewidth = 2,
            linestyle = "--",
            label = f"{display_name} - AUC {auc:.3f}"
        )
    
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate / Recall")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_threshold_metrics(threshold_df, save_path = None, model_name: str = None,):
    metric_cols = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 score"
    }

    fig, ax = plt.subplots(figsize = (10,6))

    for col, label in metric_cols.items():
        ax.plot(
            threshold_df["threshold"],
            threshold_df[col],
            label=label,
            linewidth = 2,
        )

    ax.set_title(f"{model_name} Classification Metrics across Probability Thresholds")
    ax.set_xlabel("Probability Threshold")
    ax.set_ylabel("Metric Score")
    ax.set_ylim(0, 1)
    ax.legend(title = "Metric", loc = "lower left")
    ax.grid(True, alpha = .25)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi = 300, bbox_inches = "tight")

    return fig, ax

def threshold_metrics(y_true, y_proba, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)

    rows = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        rows.append({
            "threshold": threshold,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp
        })

    return pd.DataFrame(rows)

def model_thresholds(model, X_test, y_test):

    y_test_proba = model.predict_proba(X_test)[:, 1]

    threshold_df = threshold_metrics(y_test, y_test_proba)

    return threshold_df
