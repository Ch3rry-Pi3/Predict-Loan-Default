import os, random, joblib
from pathlib import Path
import numpy as np
import mlflow, mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from .config import RANDOM_STATE, TRACKING_URI, EXPERIMENT_RF, N_RUNS_PER_MODEL
from .mlflow_utils import ensure_experiment, reset_active_run
from .data_utils import load_data
from .metrics_utils import metrics_dict
from .trial_rf import run_rf_trial
from .viz_utils import save_classification_report_text, save_confusion_matrix_png

def run_rf_experiment() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    ensure_experiment(EXPERIMENT_RF)
    mlflow.set_experiment(EXPERIMENT_RF)
    reset_active_run()

    X_train, X_test, y_train, y_test = load_data()
    rng = np.random.default_rng(RANDOM_STATE)

    with mlflow.start_run(run_name="RandomForest_group") as parent_run:
        print("Parent run_id:", parent_run.info.run_id)
        print("Artifact base (parent):", mlflow.get_artifact_uri())

        # Trials -> select best by cv_accuracy
        best = None
        for i in range(N_RUNS_PER_MODEL):
            res = run_rf_trial(X_train, X_test, y_train, y_test, i, rng)
            if best is None or res["cv_accuracy"] > best["cv_accuracy"]:
                best = res
        
        # Final refit on full training set
        np.random.seed(RANDOM_STATE); random.seed(RANDOM_STATE)
        best_params = best["params"]
        best_rf = RandomForestClassifier(
            n_jobs=-1,
            random_state=RANDOM_STATE,
            **best_params,
        )

        mlflow.set_tags({"model": "RandomForest"m "best": "true"})
        mlflow.sklearn.autolog(log_models=False)
        best_rf.fit(X_train, y_train)

        # Probabilities -> metrics
        try:
            y_prob = best_rf.predict_proba(X_test)[:, 1]
        except Exception:
            proba = best_rf.predict(X_test)
            y_prob = proba[:, 1] if getattr(proba, "ndim", 1) > 1 else np.ravel()
        
        y_pred = (y_prob >= 0.5).astype(int)

        m = metrics_dict(y_test, y_prob)