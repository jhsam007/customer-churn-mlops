from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from src.decision import apply_threshold
from src.config import evaluation_config


def evaluate_classification(y_true, y_proba):
    """
    Evaluate classification model using configurable threshold.
    """
    threshold = evaluation_config.decision_threshold
    y_pred = apply_threshold(y_proba, threshold)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }

    cm = confusion_matrix(y_true, y_pred)

    return metrics, cm