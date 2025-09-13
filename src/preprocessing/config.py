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

# Paths (used as defaults inside each module)
RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "data/bronze/raw.csv")
CLEAN_DATA_PATH = os.getenv("CLEAN_DATA_PATH", "data/silver/clean.csv")
FEATURES_PATH = os.getenv("FEATURES_PATH", "data/bronze/features.csv")

# Toggles to run steps from the CLI
RUN_IMPORT = os.getenv("RUN_IMPORT", "1") == "1"
RUN_CLEAN = os.getenv("RUN_CLEAN", "1") == "1"
RUN_FEATURE = os.getenv("RUN_FEATURE", "1") == "1"