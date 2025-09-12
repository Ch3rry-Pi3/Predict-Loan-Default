# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------

import os, random
import numpy as np
import tensorflow as tf
from .config import RANDOM_STATE, RUN_XGB, RUN_KERAS
from .experiment_xgb import run_xgb_experiment
from .experiment_keras import run_keras_experiment

# Must be set BEFORE importing pyplot anywhere else
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLBACKEND", "Agg")

# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------

def main():
    np.random.seed(RANDOM_STATE); random.seed(RANDOM_STATE); tf.random.set_seed(RANDOM_STATE)
    if RUN_XGB:
        run_xgb_experiment()
    if RUN_KERAS:
        run_keras_experiment()

if __name__ == "__main__":
    main()
