import os, random
from pathlib import Path

import numpy as np, pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, make_scorer
from sklearn.ensemble import RandomForestClassifier  # reserved for later

from xgboost import XGBClassifier  # reserved for later

import mlflow, mlflow.sklearn, mlflow.xgboost, mlflow.keras
from mlflow.tracking import MlflowClient

# Must be set BEFORE importing tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scikeras.wrappers import KerasClassifier


# -------------------------------------------------------------------
# Configurations
# -------------------------------------------------------------------

RANDOM_STATE = 5901
N_RUNS_PER_MODEL = 1
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5555")
TARGET_COL = "loan_status"

EXPERIMENT_RF = "loan_default_rf"
EXPERIMENT_XGB = "loan_default_xgb"
EXPERIMENT_KERAS = "loan_default_keras_v2"  # <- new experiment name (uses proxy settings)

# Reduce TF thread contention on Windows (helps stability with sklearn CV)
try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass


# -------------------------------------------------------------------
# MLflow helpers
# -------------------------------------------------------------------

def ensure_experiment(name: str) -> str:
    """
    Ensure an MLflow experiment exists and return its experiment_id.
    Prints the artifact root to confirm where files will land.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    exp = client.get_experiment_by_name(name)
    if exp is None:
        exp_id = client.create_experiment(name)
        print(f"[MLflow] Created experiment '{name}' (id={exp_id})")
        # Fetch to print the artifact root after creation
        exp = client.get_experiment(exp_id)
    else:
        exp_id = exp.experiment_id
        print(f"[MLflow] Using experiment '{exp.name}' (id={exp_id})")
    print(f"[MLflow] artifact_location: {exp.artifact_location}")
    return exp_id


def _reset_active_run():
    # End any run that's accidentally still open
    if mlflow.active_run() is not None:
        mlflow.end_run()


# -------------------------------------------------------------------
# Data / metrics helpers
# -------------------------------------------------------------------

def load_data(input_path: str = "data/features.csv"):
    """Load features from data/features.csv and split train/test."""
    df = pd.read_csv(input_path)

    assert TARGET_COL in df.columns, f"Expected column '{TARGET_COL}' in the dataset."

    X = df.drop(columns=[TARGET_COL])
    # Ensure strict binary 0/1 int labels and pass numpy array to sklearn
    y = df[TARGET_COL].astype("int32").to_numpy(copy=False)

    labels = set(np.unique(y))
    bad = labels - {0, 1}
    assert not bad, f"Unexpected labels in y: {bad}. Expected binary 0/1."

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test


def metrics_dict(y_true, y_prob, threshold=0.5):
    """Return common binary classification metrics from probabilities."""
    y_pred = (y_prob >= threshold).astype(int)
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


# -------------------------------------------------------------------
# Model builder / scorer
# -------------------------------------------------------------------

def build_keras_model(
    hidden_layers=2,
    hidden_units=128,
    dropout=0.2,
    lr=1e-3,
    input_dim=None,
    **kwargs,
):
    """
    Factory for SciKeras KerasClassifier. Accept **kwargs to tolerate
    SciKeras-injected params (e.g., classes, n_features_in_, meta).
    """
    # Infer input_dim if not provided
    if input_dim is None:
        input_dim = kwargs.get("n_features_in_") or kwargs.get("meta", {}).get("n_features_in_")

    model = keras.Sequential([layers.Input(shape=(input_dim,))])
    for _ in range(hidden_layers):
        model.add(layers.Dense(hidden_units, activation="relu"))
        if dropout and dropout > 0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


# --- AUC scorer that prefers predict_proba but gracefully falls back to predict
def _safe_auc(estimator, X, y):
    try:
        proba = estimator.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] > 1:
            y_score = proba[:, 1]
        else:
            y_score = np.ravel(proba)
    except Exception:
        try:
            y_score = estimator.decision_function(X)  # if available
        except Exception:
            y_score = np.ravel(estimator.predict(X))  # fallback
    return roc_auc_score(y, y_score)


AUC_SCORER = make_scorer(_safe_auc, greater_is_better=True)


# -------------------------------------------------------------------
# Keras (Neural Network) Trial
# -------------------------------------------------------------------

def run_keras_trial(X_train, X_test, y_train, y_test, trial_idx, rng):
    # Early stopping with val split inside each fit
    es = keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=5, restore_best_weights=True
    )

    clf = KerasClassifier(
        model=build_keras_model,
        input_dim=X_train.shape[1],
        verbose=1,
        callbacks=[es],
        classes=[0, 1],                 # make classification explicit
        classifier=True,
        # pass-through to model.fit(...)
        fit__validation_split=0.2,
    )

    dist = {
        "model__hidden_layers": [1, 2, 3],
        "model__hidden_units": [64, 128, 256],
        "model__dropout": [0.0, 0.2, 0.5],
        "model__lr": [1e-3, 3e-4, 1e-4],
        "batch_size": [32, 64, 128],
        "epochs": [25, 40, 60],
        "random_state": [int(rng.integers(0, 1_000_000))],  # SciKeras seeds per fit
    }
    search_random_state = int(rng.integers(0, 1_000_000))
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)

    with mlflow.start_run(run_name=f"KerasMLP_trial_{trial_idx + 1}", nested=True):
        mlflow.set_tags({"model": "KerasMLP", "trial": trial_idx + 1})
        mlflow.keras.autolog(log_models=False)  # log curves; we'll log model manually

        search = RandomizedSearchCV(
            estimator=clf,
            param_distributions=dist,
            n_iter=1,
            scoring=_safe_auc,     # custom scorer avoids sklearn forcing predict_proba signature
            cv=cv,
            random_state=search_random_state,
            n_jobs=1,              # avoid TF + multiprocessing hangs on Windows
            verbose=1,
            refit=True,
        )
        search.fit(X_train, y_train)

        best = search.best_estimator_

        # Predict proba (handle both (n,1) and (n,) shapes)
        try:
            proba = best.predict_proba(X_test)
        except Exception:
            # Fallback: use predictions as scores (less ideal, but safe)
            proba = best.predict(X_test)

        y_prob = proba[:, 1] if (proba.ndim == 2 and proba.shape[1] > 1) else np.ravel(proba)
        m = metrics_dict(y_test, y_prob)

        mlflow.log_params({"search": "RandomizedSearchCV", **search.best_params_})
        mlflow.log_metrics({"cv_auc": float(search.best_score_), **m})

        # Log underlying Keras model
        model_file = f"keras_trial_{trial_idx+1}.keras"
        best.model_.save(model_file)                 # Keras v3 format (single file)
        mlflow.log_artifact(model_file, artifact_path="model")
        os.remove(model_file)

        return {"params": search.best_params_, "cv_auc": float(search.best_score_)}


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    # MLflow: ensure experiment exists and is selected
    mlflow.set_tracking_uri(TRACKING_URI)
    exp_id = ensure_experiment(EXPERIMENT_KERAS)
    mlflow.set_experiment(EXPERIMENT_KERAS)  # safe even if already created
    _reset_active_run()

    # Data once
    X_train, X_test, y_train, y_test = load_data()
    rng = np.random.default_rng(RANDOM_STATE)

    # ---------------------------------------------------------------
    # Keras (Neural Network) Experiment
    # ---------------------------------------------------------------
    with mlflow.start_run(run_name="KerasMLP_group") as parent_run:
        print("Parent run_id:", parent_run.info.run_id)
        print("Artifact base (parent):", mlflow.get_artifact_uri())

        best = None
        for i in range(N_RUNS_PER_MODEL):
            res = run_keras_trial(X_train, X_test, y_train, y_test, i, rng)
            if best is None or res["cv_auc"] > best["cv_auc"]:
                best = res

        # Rebuild best Keras and refit on full training (keep ES + val_split)
        np.random.seed(RANDOM_STATE); random.seed(RANDOM_STATE); tf.random.set_seed(RANDOM_STATE)
        es = keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=5, restore_best_weights=True
        )
        best_cfg = best["params"]
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
        best_keras.fit(X_train, y_train)

        # As above, prefer proba but fall back
        try:
            proba = best_keras.predict_proba(X_test)
        except Exception:
            proba = best_keras.predict(X_test)
        y_prob = proba[:, 1] if (proba.ndim == 2 and proba.shape[1] > 1) else np.ravel(proba)

        m = metrics_dict(y_test, y_prob)

        mlflow.set_tags({"model": "KerasMLP", "best": "true"})
        mlflow.log_params({f"best_{k}": v for k, v in best_cfg.items()})
        mlflow.log_metrics({f"best_test_{k}": v for k, v in m.items()})
        best_file = "keras_best_model.keras"
        best_keras.model_.save(best_file)
        mlflow.log_artifact(best_file, artifact_path="best_model")
        os.remove(best_file)

    print(
        f"Done. Logged {N_RUNS_PER_MODEL} trial runs + 1 parent run to MLflow at {TRACKING_URI}."
    )


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE); random.seed(RANDOM_STATE); tf.random.set_seed(RANDOM_STATE)
    main()
