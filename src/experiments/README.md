# **Experiments - Model Training & Evaluation**

This folder (`src/experiments/`) contains the **model training, evaluation, and experiment tracking** stage for the Loan Default Prediction project. It integrates with **MLflow** to log metrics, parameters, and artifacts for each run.

## **Modules**

- **`exp_cli.py`**
  CLI entry point for running experiments (Random Forest, XGBoost, Keras MLP), based on toggles in `.env`.

- **`experiment_rf.py`**
  Random Forest experiment pipeline: hyperparameter search, cross-validation, metrics logging.

- **`experiment_xgb.py`**
  XGBoost experiment pipeline: hyperparameter search, cross-validation, metrics logging.

- **`experiment_keras.py`**
  Keras MLP experiment pipeline: model definition, search with SciKeras, training, evaluation.

- **`trial_rf.py`, `trial_xgb.py`, `trial_keras.py`**
  Define how individual training/evaluation trials are executed, including hyperparameter search and CV strategy.

- **`metrics_utils.py`**
  Helper functions for computing and logging evaluation metrics (accuracy, precision, recall, AUC, etc.).

- **`viz_utils.py`**
  Utilities for generating visual artifacts such as confusion matrices and classification reports.

- **`mlflow_utils.py`**
  MLflow helpers: create experiments, manage active runs, log metadata.

- **`models_keras.py`**
  Defines reusable Keras model architectures (MLPs) for experimentation.

- **`data_utils.py`**
  Data handling functions for splitting datasets, ensuring reproducibility, and preparing folds.

- **`config.py`**
  Centralises experiment configuration and `.env` toggles (e.g., which models to run, random seeds, tracking URI).

## **Usage**

Run all experiments (based on `.env` toggles):

```bash
python -m src.experiments.exp_cli
```

Run a specific model experiment directly:

```bash
python -m src.experiments.experiment_rf
python -m src.experiments.experiment_xgb
python -m src.experiments.experiment_keras
```

## MLflow Integration**

- Tracks parameters, metrics, confusion matrices, and model artifacts.
- UI available at: [http://localhost:5555](http://localhost:5555)
- Parent runs group mulitple trials for easier comparison.
