# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------

import io, os, random
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, make_scorer
from sklearn.metrics import classification_report, confusion_matrix

import mlflow, mlflow.sklearn, mlflow.keras
from mlflow.tracking import MlflowClient

# Must be set BEFORE importing tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scikeras.wrappers import KerasClassifier

import json
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Configurations (readable from environment; with safe defaults)
# -------------------------------------------------------------------

# Load .env variables (if present)
load_dotenv()

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "5901"))
N_RUNS_PER_MODEL = int(os.getenv("N_RUNS_PER_MODEL", "1"))
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5555")
TARGET_COL = os.getenv("TARGET_COL", "loan_status")
EXPERIMENT_KERAS = os.getenv("EXPERIMENT_KERAS", "loan_default_keras_v2")
FEATURES_PATH = os.getenv("FEATURES_PATH", "data/features.csv")

# -------------------------------------------------------------------
# MLflow helpers
# -------------------------------------------------------------------

def ensure_experiment(name: str) -> str:
    """
    Ensure an MLflow experiment exists and return its experiment ID.

    Parameters
    ----------
    name : str
        The name of the MLflow experiment.

    Returns
    -------
    str
        The experiment ID corresponding to 'name'. Prints the artifact
        location to confirm where files will land.
    """

    # Point MLflow at tracking server/URI
    mlflow.set_tracking_uri(TRACKING_URI)

    # Create client and check for existing experiment
    client = MlflowClient()
    exp = client.get_experiment_by_name(name)

    # Create if missing, otherwise reuse existing
    if exp is None:
        exp_id = client.create_experiment(name)
        print(f"[MLflow] Created experiment '{name}' (id={exp_id})")
        exp = client.get_experiment(exp_id)  # fetch to print artifact root
    else:
        exp_id = exp.experiment_id
        print(f"[MLflow] Using experiment '{exp.name}' (id={exp_id})")

    # Show artifact base location
    print(f"[MLflow] artifact_location: {exp.artifact_location}")
    return exp_id


def _reset_active_run() -> None:
    """
    End any MLflow run that is accidentally still open.

    Notes
    -----
    Defensive helper to avoid "active run" errors when nesting or re-running.
    """

    # End any run that's accidentally still open
    if mlflow.active_run() is not None:
        mlflow.end_run()


# -------------------------------------------------------------------
# Data / metrics helpers
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test


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

def save_keras_model_summary(model: keras.Model, out_path: str) -> None:
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
        yticks=np.arange(cm.shape[0])
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------
# Model builder / scorer
# -------------------------------------------------------------------

def build_keras_model(
    hidden_layers: int = 2,
    hidden_units: int = 128,
    dropout: float = 0.2,
    lr: float = 1e-3,
    input_dim: int | None = None,
    **kwargs: Any,
) -> keras.Model:
    """
    Factory for a binary-classification MLP in Keras (for use with SciKeras).

    Parameters
    ----------
    hidden_layers : int, default=2
        Number of hidden Dense layers.
    hidden_units : int, default=128
        Units per hidden layer.
    dropout : float, default=0.2
        Dropout probability applied after each hidden layer (0 disables).
    lr : float, default=1e-3
        Learning rate for the Adam optimiser.
    input_dim : int or None, default=None
        Number of input features. If None, attempts to infer from SciKeras meta.
    **kwargs : Any
        Extra arguments tolerated for SciKeras (e.g., classes, n_features_in_, meta).

    Returns
    -------
    keras.Model
        A compiled Keras model with a sigmoid output for binary classification.
    """

    # Infer input_dim if not provided
    if input_dim is None:
        input_dim = kwargs.get("n_features_in_") or kwargs.get("meta", {}).get("n_features_in_")

    # Define a simple MLP
    model = keras.Sequential([layers.Input(shape=(input_dim,))])
    for _ in range(hidden_layers):
        model.add(layers.Dense(hidden_units, activation="relu"))
        if dropout and dropout > 0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation="sigmoid"))

    # Compile with AUC metric
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model


# def _safe_auc(estimator, X, y) -> float:
#     """
#     Robust AUC scorer for estimators that may lack 'predict_proba'.

#     Parameters
#     ----------
#     estimator : Any
#         Fitted estimator supporting 'predict_proba', 'decision_function', or 'predict'.
#     X : array-like
#         Feature matrix.
#     y : array-like
#         True binary labels.

#     Returns
#     -------
#     float
#         ROC-AUC computed from the best-available score output.
#     """

#     try:
#         # Prefer predict_proba when available
#         proba = estimator.predict_proba(X)
#         if proba.ndim == 2 and proba.shape[1] > 1:
#             y_score = proba[:, 1]
#         else:
#             y_score = np.ravel(proba)
#     except Exception:
#         try:
#             # Fall back to decision_function
#             y_score = estimator.decision_function(X)
#         except Exception:
#             # Last resort: use predictions as scores
#             y_score = np.ravel(estimator.predict(X))
#     return roc_auc_score(y, y_score)


# AUC_SCORER = make_scorer(_safe_auc, greater_is_better=True)

# -------------------------------------------------------------------
# Keras (Neural Network) Trial
# -------------------------------------------------------------------

def run_keras_trial(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    trial_idx: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Run a single randomized-search trial for a Keras MLP and log to MLflow.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Train/test feature matrices.
    y_train, y_test : np.ndarray
        Train/test binary labels.
    trial_idx : int
        Index of the current trial (used for run naming).
    rng : np.random.Generator
        Random generator to seed SciKeras and searches reproducibly.

    Returns
    -------
    dict
        Best params and cross-validated AUC for this trial.
    """

    # Early stopping with validation split inside each fit
    es = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", mode="max", patience=5, restore_best_weights=True
    )

    # Wrap Keras model for sklearn interface
    clf = KerasClassifier(
        model=build_keras_model,
        input_dim=X_train.shape[1],
        verbose=1,
        callbacks=[es],
        classes=[0, 1],                 
        classifier=True,
        fit__validation_split=0.2,
    )

    # Define random search space
    dist = {
        "model__hidden_layers": [1, 2, 3],
        "model__hidden_units": [64, 128, 256],
        "model__dropout": [0.0, 0.2, 0.5],
        "model__lr": [1e-3, 3e-4, 1e-4],
        "batch_size": [32, 64, 128],
        "epochs": [25, 40, 60],
        "random_state": [int(rng.integers(0, 1_000_000))],  # SciKeras seeds per fit
    }

    # Use small CV to reduce runtime; shuffle for robustness
    search_random_state = int(rng.integers(0, 1_000_000))
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)

    with mlflow.start_run(run_name=f"KerasMLP_trial_{trial_idx + 1}", nested=True):
        # Tag run for discoverability
        mlflow.set_tags({"model": "KerasMLP", "trial": trial_idx + 1})

        # Log Keras training curves; log model manually later
        mlflow.keras.autolog(log_models=False)

        # Randomized search
        search = RandomizedSearchCV(
            estimator=clf,
            param_distributions=dist,
            n_iter=1,
            scoring="accuracy",     
            cv=cv,
            random_state=search_random_state,
            n_jobs=-1,              
            verbose=1,
            refit=True,
        )

        # Fit on training data
        search.fit(X_train, y_train)

        # Retrieve best estimator
        best = search.best_estimator_

        # Predict probabilities (handle both (n,1) and (n,) shapes)
        try:
            proba = best.predict_proba(X_test)
        except Exception:
            # Fallback: use predictions as scores (less ideal, but safe)
            proba = best.predict(X_test)

        # Extract positive-class scores
        y_prob = proba[:, 1] if (proba.ndim == 2 and proba.shape[1] > 1) else np.ravel(proba)

        # Compute test metrics
        m = metrics_dict(y_test, y_prob)

        # Log params/metrics
        mlflow.log_params({"search": "RandomizedSearchCV", **search.best_params_})
        mlflow.log_metrics({"cv_auc": float(search.best_score_), **m})

        # Log underlying Keras model (Keras v3 single-file format)
        model_file = f"keras_trial_{trial_idx+1}.keras"
        # Ensure output directory exists (artifact path handled by MLflow)
        best.model_.save(model_file)
        mlflow.log_artifact(model_file, artifact_path="model")
        os.remove(model_file)

        return {"params": search.best_params_, "cv_auc": float(search.best_score_)}


# -------------------------------------------------------------------
# Orchestrator
# -------------------------------------------------------------------

def run_keras_experiment() -> None:
    """
    Run the Keras MLP experiment: multiple trials + final refit and logging.

    Notes
    -----
    - Ensures the MLflow experiment exists and is active.
    - Loads/validates data once; reuses across trials.
    - Runs 'N_RUNS_PER_MODEL' randomized-search trials (nested runs).
    - Rebuilds and refits the best configuration on the full training set.
    - Logs final test metrics and best model artifact.
    """

    # MLflow: ensure experiment exists and is selected
    mlflow.set_tracking_uri(TRACKING_URI)
    exp_id = ensure_experiment(EXPERIMENT_KERAS)
    mlflow.set_experiment(EXPERIMENT_KERAS)  
    _reset_active_run()

    # Load dataset once (train/test)
    X_train, X_test, y_train, y_test = load_data()

    # RNG for reproducibility across trials
    rng = np.random.default_rng(RANDOM_STATE)

    # ---------------------------------------------------------------
    # Keras (Neural Network) Experiment
    # ---------------------------------------------------------------
    with mlflow.start_run(run_name="KerasMLP_group") as parent_run:
        # Show parent context / artifact base
        print("Parent run_id:", parent_run.info.run_id)
        print("Artifact base (parent):", mlflow.get_artifact_uri())

        # Track best trial by CV AUC
        best = None

        # Loop over randomized trials
        for i in range(N_RUNS_PER_MODEL):
            # Run a trial and collect results
            res = run_keras_trial(X_train, X_test, y_train, y_test, i, rng)

            # Keep best by cv_auc
            if best is None or res["cv_auc"] > best["cv_auc"]:
                best = res

        # -----------------------------------------------------------
        # Rebuild best Keras and refit on full training (keep ES)
        # -----------------------------------------------------------

        # Seed all frameworks for deterministic behaviour
        np.random.seed(RANDOM_STATE); random.seed(RANDOM_STATE); tf.random.set_seed(RANDOM_STATE)

        # Early stopping for the final fit
        es = keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=5, restore_best_weights=True
        )

        # Unpack best configuration from search
        best_cfg = best["params"]

        # Recreate classifier with best hyperparameters
        best_keras = KerasClassifier(
            model=build_keras_model,
            input_dim=X_train.shape[1],
            verbose=1,
            callbacks=[es],
            classes=[0, 1],
            classifier=True,
            fit__validation_split=0.2,
            **best_cfg,
        )

        # Fit on training set (with internal validation split)
        best_keras.fit(X_train, y_train)

        # As above, prefer probabilities but fall back safely
        try:
            proba = best_keras.predict_proba(X_test)
        except Exception:
            proba = best_keras.predict(X_test)
        y_prob = proba[:, 1] if (proba.ndim == 2 and proba.shape[1] > 1) else np.ravel(proba)

        # Convert probabilities to hard labels (default to threshold 0.5)
        y_pred = (y_prob >= 0.5).astype(int)

        # Compute and log test metrics
        m = metrics_dict(y_test, y_prob)
        mlflow.set_tags({"model": "KerasMLP", "best": "true"})
        mlflow.log_params({f"best_{k}": v for k, v in best_cfg.items()})
        mlflow.log_metrics({f"best_test_{k}": v for k, v in m.items()})

        # -----------------------------------------------------------
        # Save & log extra artefacts (reports)
        # -----------------------------------------------------------

        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Model summary (text)
        summary_path = reports_dir / "keras_best_model_summary.txt"
        save_keras_model_summary(best_keras.model_, str(summary_path))
        mlflow.log_artifact(str(summary_path), artifact_path="reports")

        # Classification report (text)
        clf_report_path = reports_dir / "keras_classification_report.txt"
        save_classification_report_text(y_test, y_pred, str(clf_report_path))
        mlflow.log_artifact(str(clf_report_path), artifact_path="reports")

        # Confusion Matrix (PNG)
        cm_path = reports_dir / "confusion_matrix.png"
        save_confusion_matrix_png(y_test, y_pred, str(cm_path), labels=["0", "1"])
        mlflow.log_artifact(str(cm_path), artifact_path="reports")

        # -----------------------------------------------------------
        # Save the final Keras model and log as artifact
        # -----------------------------------------------------------

        best_file = "keras_best_model.keras"
        # Ensure output directory exists (artifact path handled by MLflow)
        best_keras.model_.save(best_file)
        mlflow.log_artifact(best_file, artifact_path="best_model")
        os.remove(best_file)

    # Final confirmation message
    print(f"✅ Done. Logged {N_RUNS_PER_MODEL} trial run(s) + 1 parent run to MLflow at {TRACKING_URI}.")


# -------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    # Seed all frameworks for reproducibility when run as a script
    np.random.seed(RANDOM_STATE); random.seed(RANDOM_STATE); tf.random.set_seed(RANDOM_STATE)
    run_keras_experiment()
