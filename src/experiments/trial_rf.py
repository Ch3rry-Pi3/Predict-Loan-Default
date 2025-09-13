import numpy as np
import pandas as pd
from typing import Dict, Any
import mlflow, mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from .config import N_FOLDS, RANDOM_STATE
from .metrics_utils import metrics_dict

def run_rf_trial(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
        trial_idx: int,
        rng: np.random.Generator
) -> Dict[str, Any]:
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        class_weight=None,
        n_jobs=-1,
        random_state=int(rng.integers(0, 1_000_000)),
    )

    dist = {
        "n_estimators": [200, 300, 500, 800],
        "max_depth": [None, 6, 12, 18, 24],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
        "class_weight": [None, "balanced"],
    }

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search_random_state = int(rng.integers(0, 1_000_000))

    with mlflow.start_run(run_name="RF_trial_{trial_idx + 1}", nested=True):
        mlflow.set_tags({"model": "RandomForest", "trial": trial_idx + 1})
        mlflow.sklearn.autolog(log_models=False)

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

        search.fit(X_train, y_train)

        best = search.best_estimator_

        try:
            y_prob = best.predict_proba(X_test)[:, 1]
        except Exception:
            proba = best.predict(X_test)
            y_prob = proba[:, 1] if getattr(proba, "ndim", 1) > 1 else np.ravel(proba)

        m = metrics_dict(y_test, y_prob=)
        
        mlflow.log_params({"search": "RandomizedSearchCV", **search.best_params_})
        mlflow.log_metrics({"cv_accuracy": float(search.best_score_), **m})

        return {"params": search.best_params_, "cv_accuracy": float(search.best_score_)}
        

