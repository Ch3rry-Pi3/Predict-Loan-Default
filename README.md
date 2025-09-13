# **Loan Default Prediction – End-to-End ML Pipeline**

This project implements a complete machine learning workflow for **Loan Default Prediction**, powered by data from [OpenML](https://www.openml.org/).
The pipeline follows the **Medallion Architecture** (Bronze → Silver → Gold), includes preprocessing, three ML model families, and experiment tracking with **MLflow**.

## **Project Structure**

```
mlops-loan-default/
├── data/
│   ├── bronze/                  # Raw ingested data
│   ├── silver/                  # Cleaned, analysis-ready data
│   └── gold/                    # Engineered feature dataset
├── reports/                     # Generated reports (e.g., confusion matrices, summaries)
├── src/
│   ├── preprocessing/           # Preprocessing stage (Bronze → Silver → Gold)
│   │   ├── import_data.py
│   │   ├── clean_data.py
│   │   ├── feature_engineering.py
│   │   └── preprocess_cli.py
│   └── experiments/             # Model training & evaluation
│       ├── experiment_rf.py
│       ├── experiment_xgb.py
│       ├── experiment_keras.py
│       └── exp_cli.py
├── .env                         # Environment variables (paths, toggles, configs)
├── app.py                       # Orchestrator (Preprocessing + Experiments)
├── docker-compose.yml           # MLflow tracking server (UI at localhost:5555)
├── pyproject.toml               # Dependencies managed by uv
└── README.md
```

> Note: `.venv/` and other transient folders (e.g., `__pycache__/`) are ignored from version control.

## **What’s Included**

1. **Dataset (Bronze Layer)**

   * Sourced from [OpenML](https://www.openml.org/).
   * Loan default dataset (12 variables including demographics, loan details, credit history).
   * Saved locally as `data/bronze/raw.csv`.

2. **Preprocessing (Medallion Architecture)**

   * **Bronze → Silver**: Enforce schema, drop duplicates, handle missing values.
   * **Silver → Gold**: One-hot encoding of categoricals, z-score standardisation of numeric features.
   * Config-driven with `.env` for all paths.

3. **Experiments (Model Training)**

   * Random Forest
   * XGBoost
   * Keras MLP (SciKeras wrapper)
   * Each experiment runs with cross-validation, multiple trials, and logs results to MLflow.

4. **Experiment Tracking with MLflow**

   * Params, metrics, confusion matrices, classification reports, and models are logged.
   * Accessible at `http://localhost:5555`.


## **Setup Instructions**

1. **Install [uv](https://github.com/astral-sh/uv)** (Python package/dependency manager).

    ```
    pip install uv
    ```

2. **Set up the project environment**

   ```bash
   cd predict-loan-default
   uv venv
   source .venv/bin/activate   # on Linux/Mac
   .venv\Scripts\activate      # on Windows
   uv sync
   ```

3. **Launch MLflow (via Docker Compose)**

   ```bash
   docker compose up -d
   ```

   MLflow UI available at: [http://localhost:5555](http://localhost:5555)

4. **Run the full pipeline (Preprocessing + Experiments)**

   ```bash
   python app.py
   ```

5. **Shut down MLflow services after use**

   ```bash
   docker compose down
   ```

6. **Deactivate virtual environment**
    ```
    deactivate
    ```

## **Usage Notes**

* Preprocessing modules can also be run independently:

  ```bash
  python -m src.preprocessing.import_data
  python -m src.preprocessing.clean_data
  python -m src.preprocessing.feature_engineering
  ```
* Experiments can be run individually:

  ```bash
  python -m src.experiments.experiment_rf
  python -m src.experiments.experiment_xgb
  python -m src.experiments.experiment_keras
  ```
