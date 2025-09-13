"""
Pipeline Orchestrator
---------------------

This script provides a single entry point to run the entire pipeline:

1. Proprocessing stage
    - Import raw data from soruce
    - Clean the raw data
    - Apply feature engineering

2. Experiments stage
 - Train/evaluate ML models
 - Track results in MLflow

Usage
-----
Run the script directly:
    python src/app.py

Notes
-----
- Each stage is configurable via '.env'
- Preprocessing behaviour is controlled in 'src/preprocessing/preprocess_cli.py
- Experiment behaviour is controlled in 'src/experiments/exp_cli.py
"""

from src.preprocessing.preprocess_cli import run_preprocessing
from src.experiments.exp_cli import run_experiments

def main():
    """
    Execute the full end-to-end pipeline:
        1. Preprocess data (import -> clean -> feature engineering)
        2. Run experiments (train/evaluate ML models)

    Returns
    -------
    None
    """
    
    # Run preprocessing stages
    run_preprocessing()

    # Run ML experiment stages
    run_experiments()

if __name__ == "__main__":
    main()