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