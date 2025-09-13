# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------

if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    # project root = two levels up from this file
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import os, random
import numpy as np
import tensorflow as tf
from src.experiments.config import RANDOM_STATE, RUN_RF, RUN_XGB, RUN_KERAS
from src.experiments.experiment_rf import run_rf_experiment
from src.experiments.experiment_xgb import run_xgb_experiment
from src.experiments.experiment_keras import run_keras_experiment

# Must be set BEFORE importing pyplot anywhere else
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLBACKEND", "Agg")

# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------

def run_cli():
    np.random.seed(RANDOM_STATE); random.seed(RANDOM_STATE); tf.random.set_seed(RANDOM_STATE)

    if RUN_RF:
        run_rf_experiment()

    # if RUN_XGB:
    #     run_xgb_experiment()

    # if RUN_KERAS:
    #     run_keras_experiment()

if __name__ == "__main__":
    run_cli()
