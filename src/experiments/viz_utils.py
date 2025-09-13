# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------

import io, os
from pathlib import Path
from tensorflow import keras
import numpy as np
import matplotlib
os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------------------------------------------
# Save artifacts helpers
# -------------------------------------------------------------------

def save_keras_model_summary(model: "keras.Model", out_path: str) -> None:
    """
    Save a textual Keras model summary to a file.

    Parameters
    ----------
    model : keras.Model
        The compiled/fitted Keras model.
    out_path : str
        Path for the output text file.
    """

    # Capture the summary into a string buffer
    buffer = io.StringIO()

    # Run model.summary(), redirecting each line into the buffer
    model.summary(print_fn=lambda x: buffer.write(x + "\n"))

    # Extract the full summary text from the buffer
    text = buffer.getvalue()

    # Ensure output directory exists
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Write to disk
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

def save_classification_report_text(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        out_path: str
    ) -> None:
    """
    Save sklearn classification_report to a text file.

    Parameters
    ----------
    y_true : np.ndarry
        Ground truth binary labels.
    y_pred : np. ndarry
        Predicted hard labels (0/1).
    out_path : str
        Path for the output text file.
    """
    
    # Generate the classification report string
    report = classification_report(y_true, y_pred, digits=2)

    # Ensure output direcotry exists
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Write report to disk
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

def save_confusion_matrix_png(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        out_path: str, 
        labels: list[str] | None = None
) -> None:
    """
    Plot and save a confusion matrix as a PNG.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_pred : np.ndarray
        Predicted hard labels (0/1).
    out_path : str
        Path to the PNG file to write.
    labels : list of str or None, optional
        Class labels for axes; defaults to ["0", "1"].
    """

    cm = confusion_matrix(y_true, y_pred)
    labels = labels or ["0", "1"]

    # Ensure output directory exists
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Heatmap
    fig, ax = plt.subplots(dpi=150)
    im = ax.imshow(cm)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix"        
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
