# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import RANDOM_STATE, FEATURES_PATH, TARGET_COL

# -------------------------------------------------------------------
# Data load helper
# -------------------------------------------------------------------

# def load_data(input_path: str = "data/features.csv") -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
def load_data(input_path: str = FEATURES_PATH) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load the features CSV, validate target, and create a stratified train/test split.

    Parameters
    ----------
    input_path : str, default="data/features.csv"
        Path to the CSV file containing features and the binary target column.

    Returns
    -------
    (X_train, X_test, y_train, y_test) : tuple
        Train/test split where 'X_*' are DataFrames and 'y_*' are NumPy arrays (int32).

    Raises
    ------
    AssertionError
        If the target column is missing or contains non-binary labels.
    """

    # Load dataset
    df = pd.read_csv(input_path)

    # Ensure target present
    assert TARGET_COL in df.columns, f"Expected column '{TARGET_COL}' in the dataset."

    # Split features/target
    X = df.drop(columns=[TARGET_COL])
    # Ensure strict binary 0/1 int labels and pass numpy array to sklearn
    y = df[TARGET_COL].astype("int32").to_numpy(copy=False)

    # Validate labels are {0, 1}
    labels = set(np.unique(y))
    bad = labels - {0, 1}
    assert not bad, f"Unexpected labels in y: {bad}. Expected binary 0/1."

    # Stratified split for class balance
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    return X_tr, X_te, y_tr, y_te