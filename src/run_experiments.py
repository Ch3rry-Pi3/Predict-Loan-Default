import os, random
from pathlib import Path

import numpy as np, pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import mlflow, mlflow.sklearn, mlflow.xgboost, mlflow.keras

os.environ["TP_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scikeras.wrappers import KerasClassifier

# -------------------------------------------------------------------
# Configurations
# -------------------------------------------------------------------

RANDOM_STATE = 5901
N_RUNS_PER_MODEL = 5
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5555")
TARGET_COL = "loan_status"

EXPERIMENT_RF = "loan_default_rf"
EXPERIMENT_XGB = "loan_default_xgb"
EXPERIMENT_KERAS = "loan_default_keras"

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def load_data(input_path: str = "data/features.csv"):
    """Load features from data/features.csv and split train/test"""
    df = pd.read_csv(input_path)

    assert TARGET_COL in df.columns

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test

def metrics_dict(y_true, y_prob, threshold=0.5):
    """Return common binary classification metrics from probabilities."""
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    
    return {
        "auc": float(auc), 
        "accuracy": float(acc), 
        "precision": float(p), 
        "recall": float(r), 
        "f1": float(f1)
    }


def build_keras_model(hidden_layers=2, hidden_units=128, dropout=0.2, lr=1e-3, input_dim=None):
    """Factory function KerasClassifier"""
    model = keras.Sequential([layers.Input(shape=(input_dim,))])

    for _ in range(hidden_layers):
        model.add(layers.Dense(hidden_units, activation="relu"))
        if dropout and dropout > 0:
            model.add(layers.Dropout(dropout))
    
    model.add(layers.Dense(1, activation="sigmoid"))
    
    model.compile(
        optimiser=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentrophy",
        metrics=[keras.metrics.AUC(name="auc")],
    )

    return model

# -------------------------------------------------------------------
# Keras (Neural Network) Trial
# -------------------------------------------------------------------

def run_keras_trial(X_train, X_test, y_train, y_test, trial_idx, rng):
    # Early stopping with validation split inside each fit
    es = keras.callbacks.EarlyStopping(
        monitor="val_auc", 
        mode="max", 
        patience=5, 
        restore_best_weights=True
    )

    clf = KerasClassifier(
        model=build_keras_model,
        input_dim=X_train.shape[1],
        verbose=0,
        callbacks=[es],
        fit__validation_split=0.2,
    )

    dist = {
        "model__hidden_layers": [1, 2, 3],
        "model__hidden_units": [64, 128, 256],
        "model__dropout": [0.0, 0.2, 0.5],
        "model__lr": [1e-3, 3e-4, 1e-4],
        "batch_size": [32, 64, 128],
        "epochs": [25, 50, 100],
        "random_state": [int(rng.integers(0, 1_000_000))],
    }

    with mlflow.start_run(run_name=f"KerasMLP_trial_{trial_idx + 1}", nested=True):
        mlflow.set_tags({"model": "KerasMLP", "trial": trial_idx + 1})
        mlflow.keras.autolog(log_models=False)

        search_random_state = int(rng.integers(0, 1_000_000))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        search = RandomizedSearchCV(
            estimator=clf,
            param_distributions=dist,
            n_iter=1,
            scoring="roc_auc",
            cv=cv,
            random_state=search_random_state,
            n_jobs=-1,
            verbose=0,
            refit=True,
            return_train_score=True,
        )

        search.fit(X_train, y_train)

        best = search.best_estimator_

        proba = best.predict_proba(X_test)
        y_prob = proba[:, 1] if (proba.ndim == 2 and proba.shape[1] > 3) else proba.ravel()
        m = metrics_dict(y_test, y_prob)

        return {"params": search.best_params_, "cv_auc": float(search.best_score_)}

if __name__ == "__main__":
    # MLflow
    mlflow.set_tracking_uri(TRACKING_URI)

    # Data once
    X_train, X_test, y_train, y_test = load_data()
    rng = np.random.default_rng(RANDOM_STATE)

    # ---------------------------------------------------------------
    # Keras (Neural Network) Trial
    # ---------------------------------------------------------------

    


    print("OK")