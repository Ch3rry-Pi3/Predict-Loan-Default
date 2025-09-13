# **Source Code** - `src/`

This folder contains the core source code for the Loan Default Prediction project. It is split into two main subpackages:
- `preprocessing/` - Implements the Medallion Architecture pipeline:
  - Import raw data -> Clean -> Feature engineering.
  - Produces Bronze, Silver, and Gold datasets.

- `experiments/` - Implements ML experiments:
  - Random Forest, XGBoost, and Keras MLP models.
  - Training, evaluation, and MLflow integration.

The `__init__.py` file ensures this folder is treated as a Python package, enabling clean imports across the project.