# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------

import os
from dotenv import load_dotenv

# -------------------------------------------------------------------
# Environmental variables
# -------------------------------------------------------------------

# Load .env early
load_dotenv()

# Repro
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "5901"))

# Search/CV
N_RUNS_PER_MODEL = int(os.getenv("N_RUNS_PER_MODEL", "1"))
N_FOLDS = int(os.getenv("N_FOLDS", "2"))

# Data
FEATURES_PATH = os.getenv("FEATURES_PATH", "data/features.csv")
TARGET_COL = os.getenv("TARGET_COL", "loan_status")

# MLflow
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5555")
EXPERIMENT_RF = os.getenv("EXPERIMENT_RF", "loan_default_rf_v1")
EXPERIMENT_XGB = os.getenv("EXPERIMENT_XGB", "loan_default_xgb_v1")
EXPERIMENT_KERAS = os.getenv("EXPERIMENT_KERAS", "loan_default_keras_v1")

# Toggles
RUN_RF = os.getenv("RUN_RF", "1") == "1"
RUN_XGB = os.getenv("RUN_XGB", "1") == "1"
RUN_KERAS = os.getenv("RUN_KERAS", "1") == "1"

# Devices
XGB_DEVICE = os.getenv("XGB_DEVICE", "cpu")