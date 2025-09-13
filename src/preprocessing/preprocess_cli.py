# -------------------------------------------------------------------
# Imports & path guard (allow running this file directly)
# -------------------------------------------------------------------

if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    
    # Project root = two levels up from this file
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import os
from pathlib import Path
from src.preprocessing.config import RUN_IMPORT, RUN_CLEAN, RUN_FEATURE
from src.preprocessing.import_data import import_data
from src.preprocessing.clean_data import clean_data
from src.preprocessing.feature_engineering import feature_engineering

# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

def run_preprocessing():

    # Run stages based on .env toggles (defaults: all on)
    if RUN_IMPORT:
        import_data()

    if RUN_CLEAN:
        clean_data()
    
    if RUN_FEATURE:
        feature_engineering()

if __name__ == "__main__":
    run_preprocessing()