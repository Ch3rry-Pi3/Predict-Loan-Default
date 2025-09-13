# **Loan Default Prediction – End-to-End ML Pipeline**

This project implements a complete machine learning workflow for **Loan Default Prediction**, powered by data from [OpenML](https://www.openml.org/d/45021).
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
