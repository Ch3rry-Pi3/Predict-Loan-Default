# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------

from typing import Dict
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

# -------------------------------------------------------------------
# Metrics helper
# -------------------------------------------------------------------

def metrics_dict(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute common binary classification metrics from predicted probabilities.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels {0, 1}.
    y_prob : np.ndarray
        Predicted probabilities or scores for the positive class.
    threshold : float, default=0.5
        Decision threshold used to convert probabilities into hard labels.

    Returns
    -------
    dict
        Dictionary with AUC, accuracy, precision, recall, and F1.
    """

    # Convert probabilities to hard predictions
    y_pred = (y_prob >= threshold).astype(int)

    # Compute metrics
    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    return {
        "auc": float(auc),
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }