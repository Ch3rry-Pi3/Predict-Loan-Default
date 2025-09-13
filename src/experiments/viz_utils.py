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
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

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
        y_true,
        y_pred,
        out_path,
        labels=None,
        cmap="Blues",
        normalize=None,
        values_format=None,
):
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
    cmap : str, default = "Blues"
        Matplotlib colourmap.
    normalize : {"true", "pred", "all"}, optional
        If given, confusion matrix will be normalised.
    values_format : str, optional
        Format of cell annotations (e.g., "d", ".2f", ".0%")
    """

    # Compute CM 
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)

    # Pick sensible default formatting
    if values_format is None:
        values_format = ".0%" if normalize else "d"

    # Build the display 
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=cmap, values_format=values_format)

    plt.title("Confusion Matrix")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
