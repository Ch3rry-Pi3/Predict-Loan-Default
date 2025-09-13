# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
from typing import Dict, Any
import mlflow, mlflow.keras
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from scikeras.wrappers import KerasClassifier
from tensorflow import keras
from .config import N_FOLDS, RANDOM_STATE
from .metrics_utils import metrics_dict
from .models_keras import build_keras_model

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
        random_state=int(rng.integers(0, 1_000_000)),
    )

    # Define random search space
    dist = {
        "model__hidden_layers": [1, 2, 3],
        "model__hidden_units": [64, 128, 256],
        "model__dropout": [0.0, 0.2, 0.5],
        "model__lr": [1e-3, 3e-4, 1e-4],
        "batch_size": [32, 64, 128],
        "epochs": [25, 40, 60],
    }

    # Use small CV to reduce runtime; shuffle for robustness
    search_random_state = int(rng.integers(0, 1_000_000))
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

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
        mlflow.log_metrics({"cv_accuracy": float(search.best_score_), **m})

        # Log underlying Keras model (Keras v3 single-file format)
        model_file = f"keras_trial_{trial_idx+1}.keras"
        # Ensure output directory exists (artifact path handled by MLflow)
        best.model_.save(model_file)
        mlflow.log_artifact(model_file, artifact_path="model")
        os.remove(model_file)

        return {"params": search.best_params_, "cv_accuracy": float(search.best_score_)}