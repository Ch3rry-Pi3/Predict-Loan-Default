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

        mlflow.set_tags({"model": "RandomForest", "best": "true"})
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

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics({f"best_test_{k}": v for k, v in m.items()})

        # Reports
        reports = Path("reports"); reports.mkdir(parents=True, exist_ok=True)
        save_classification_report_text(y_test, y_pred, str(reports / "rf_classification_report.txt"))
        save_confusion_matrix_png(y_test, y_pred, str(reports / "rf_confusion_matrix.png"), labels=["0", "1"])

        # Feature importances (CSV)
        try:
            import pandas as pd
            fi = pd.DataFrame(
                {
                    "feature": X_train.columns,
                    "importance": best_rf.feature_importances_,
                }
            ).sort_values(by="importance", ascending=False)
            fi_path = reports / "rf_feature_importances.csv"
            fi.to_csv(fi_path, index=False)
            mlflow.log_artifact(str(fi_path), artifact_path="reports")

        except Exception:
            pass

        # Save model artifact
        joblib.dump(best_rf, "rf_best_model.joblib")
        mlflow.log_artifact("rf_best_model.joblib", artifact_path="best_model")
        os.remove("rf_best_model.joblib")

    print(f"✅ RandomForest: Logged {N_RUNS_PER_MODEL} trial run(s) + 1 parent run to MLflow at {TRACKING_URI}.")