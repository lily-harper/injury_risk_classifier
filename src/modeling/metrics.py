from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

def classification_metrics(
    model_name,
    y_true,
    y_pred,
    y_prob=None,
    return_confusion=False
):
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)

    if return_confusion:
        con_mat = confusion_matrix(y_true, y_pred, labels=[0, 1])
        return metrics, con_mat

    return metrics

def evaluate_model(
    model,
    model_name,
    X_test,
    y_test,
    return_confusion=False
):
    y_pred = model.predict(X_test)

    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]

    return classification_metrics(
        model_name=model_name,
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        return_confusion=return_confusion
    )