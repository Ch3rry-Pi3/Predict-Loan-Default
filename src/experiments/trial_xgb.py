# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------

import numpy as np
import pandas as pd
from typing import Dict, Any
import mlflow, mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from .config import N_FOLDS, RANDOM_STATE
from .metrics_utils import metrics_dict

# -------------------------------------------------------------------
# XGBoost Trial
# -------------------------------------------------------------------

def run_xgb_trial(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
        trial_idx: int,
        rng: np.random.Generator,
) -> Dict[str, Any]:
    """Run a single randomised-search trial for XGBoost and log to MLflow"""
    pass
    # Base esitmator
    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="error",
        tree_method="hist",
        enable_categorical=False,
        random_state=int(rng.integers(0, 1_000_000))
    )

    # Define random search space
    dist = {
        "n_estimators": [200, 300, 500],
        "max_depth": [3, 4, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1, 5],
        "reg_lambda": [0.5, 1.0, 2.0],
        "reg_alpha": [0.0, 0.1],
    }
    
    # Use small CV to reduce runtime; shuffle for robustness
    search_random_state = int(rng.integers(0, 1_000_000))
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    with mlflow.start_run(run_name=f"XGB_trial_{trial_idx + 1}", nested=True):

        # Tag run for discoverability
        mlflow.set_tags({"model": "XGBoost", "trial": trial_idx + 1})

        # Log XGBoost training curves; log model manually later
        mlflow.xgboost.autolog(log_models=False) 

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
    
        # Early stopping via eval_set
        search.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose = False,
        )

        best = search.best_estimator_
        y_prob = best.predict_proba(X_test)[:, 1]
        m = metrics_dict(y_test, y_prob)

        mlflow.log_params({"search": "RandomizedSearchCV", **search.best_params_})
        mlflow.log_metrics({"cv_accuracy": float(search.best_score_), **m})

        return {"params": search.best_params_, "cv_accuracy": float(search.best_score_)}