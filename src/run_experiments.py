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


if __name__ == "__main__":
    print("OK")