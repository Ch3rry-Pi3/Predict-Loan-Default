# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------

import os, random, joblib
from pathlib import Path
import numpy as np
import mlflow, mlflow.xgboost
from xgboost import XGBClassifier
from .config import RANDOM_STATE, TRACKING_URI, EXPERIMENT_XGB, N_RUNS_PER_MODEL
from .mlflow_utils import ensure_experiment, reset_active_run
from .data_utils import load_data
from .metrics_utils import metrics_dict
from .trial_xgb import run_xgb_trial
from .viz_utils import save_classification_report_text, save_confusion_matrix_png


# -------------------------------------------------------------------
# Orchestrator (XGBoost)
# -------------------------------------------------------------------

def run_xgb_experiment() -> None:
    """
    Run the XGBoost experiment: multiple trials + final refit and logging.

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
    exp_id = ensure_experiment(EXPERIMENT_XGB)
    mlflow.set_experiment(EXPERIMENT_XGB)  
    reset_active_run()

    # Load dataset once (train/test)
    X_train, X_test, y_train, y_test = load_data()

    # RNG for reproducibility across trials
    rng = np.random.default_rng(RANDOM_STATE)

    # ---------------------------------------------------------------
    # XGBoost Experiment (parent run)
    # ---------------------------------------------------------------

    with mlflow.start_run(run_name="XGBoost_group") as parent_run:
        # Show parent context / artifact base
        print("Parent run_id:", parent_run.info.run_id)
        print("Artifact base (parent):", mlflow.get_artifact_uri())

        # Trials -> Select best by cv_accurary
        best = None
        for i in range(N_RUNS_PER_MODEL):
            # Run a trial and collect results
            res = run_xgb_trial(X_train, X_test, y_train, y_test, i, rng)

            # Keep best by cv_accuracy
            if best is None or res["cv_accuracy"] > best["cv_accuracy"]:
                best = res
        
        # Seed frameworks
        np.random.seed(RANDOM_STATE); random.seed(RANDOM_STATE)

        # Refit best configuration on full training set with early stopping
        best_params = best["params"]
        best_xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="error",
            tree_method="hist",
            enable_categorical=False,
            random_state=RANDOM_STATE,
            **best_params,
        )

        mlflow.set_tags({"model": "XGBoost", "best": "true"})
        mlflow.xgboost.autolog(log_models=False)
        best_xgb.fit(X_train, y_train)

        # As above, prefer probabilities but fall back safely
        try:
            proba = best_xgb.predict_proba(X_test)
        except Exception:
            proba = best_xgb.predict(X_test)
        y_prob = proba[:, 1] if (proba.ndim == 2 and proba.shape[1] > 1) else np.ravel(proba)

        y_pred = (y_prob >= 0.5).astype(int)
        m = metrics_dict(y_test, y_prob)

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics({f"best_test_{k}": v for k, v in m.items()})

        # -----------------------------------------------------------
        # Save the final XGBoost model and log as artifact
        # -----------------------------------------------------------

        reports_dir = Path("reports"); reports_dir.mkdir(parents=True, exist_ok=True)
        save_classification_report_text(y_test, y_pred, str(reports_dir / "xgb_classification_report.txt"))
        save_confusion_matrix_png(y_test, y_pred, str(reports_dir / "xgb_confusion_matrix.png"), labels=["0","1"])

        joblib.dump(best_xgb, "xgb_best_model.joblib")
        mlflow.log_artifact("xgb_best_model.joblib", artifact_path="best_model")
        os.remove("xgb_best_model.joblib")

    # Final confirmation message
    print(f"✅ XGBoost: Logged {N_RUNS_PER_MODEL} trial run(s) + 1 parent run to MLflow at {TRACKING_URI}.")
