# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------

import os, random
from pathlib import Path
import numpy as np
import mlflow, mlflow.keras
from scikeras.wrappers import KerasClassifier
import tensorflow as tf
from tensorflow import keras
from .config import RANDOM_STATE, TRACKING_URI, EXPERIMENT_KERAS, N_RUNS_PER_MODEL
from .mlflow_utils import ensure_experiment, reset_active_run
from .data_utils import load_data
from .metrics_utils import metrics_dict
from .trial_keras import run_keras_trial
from .models_keras import build_keras_model
from .viz_utils import save_keras_model_summary, save_classification_report_text, save_confusion_matrix_png
    

# -------------------------------------------------------------------
# Orchestrator (Keras)
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
    reset_active_run()

    # Load dataset once (train/test)
    X_train, X_test, y_train, y_test = load_data()

    # RNG for reproducibility across trials
    rng = np.random.default_rng(RANDOM_STATE)

    # ---------------------------------------------------------------
    # Keras (Neural Network) Experiment (parent run)
    # ---------------------------------------------------------------

    with mlflow.start_run(run_name="KerasMLP_group") as parent_run:
        # Show parent context / artifact base
        print("Parent run_id:", parent_run.info.run_id)
        print("Artifact base (parent):", mlflow.get_artifact_uri())

        # Trials -> Select best by cv_accurary
        best = None
        for i in range(N_RUNS_PER_MODEL):
            # Run a trial and collect results
            res = run_keras_trial(X_train, X_test, y_train, y_test, i, rng)

            # Keep best by cv_accuracy
            if best is None or res["cv_accuracy"] > best["cv_accuracy"]:
                best = res

        # -----------------------------------------------------------
        # Rebuild best Keras and refit on full training (keep ES)
        # -----------------------------------------------------------

        # Seed all frameworks for deterministic behaviour
        np.random.seed(RANDOM_STATE); random.seed(RANDOM_STATE); tf.random.set_seed(RANDOM_STATE)

        # Early stopping for the final fit
        es = keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=5, restore_best_weights=True
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
        # Save & log extra artefacts for Keras model
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
        cm_path = reports_dir / "keras_confusion_matrix.png"
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
    print(f"✅ Keras: Logged {N_RUNS_PER_MODEL} trial run(s) + 1 parent run to MLflow at {TRACKING_URI}.")