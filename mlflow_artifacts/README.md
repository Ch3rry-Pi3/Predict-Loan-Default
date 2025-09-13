# **MLflow Artifacts - Experiment Tracking**

The `mlflow_artifacts/` folder (located in the project root) stores all artifacts produced during model experiments. Artifacts are automatically managed by **MLflow** and linked to experiment runs recorded in the tracking database (`mlflow_data/mlflow.db`).

## **General Structure**

```
mlflow_artifacts/
├── <experiment_id>/
│   ├── <run_id>/artifacts/
│   │   ├── model/                  # Saved model object (e.g., Keras, XGB, RF)
│   │   ├── reports/                # Reports (classification reports, confusion matrices, etc.)
│   │   ├── estimator.html          # Model summary in HTML
│   │   ├── model_summary.txt       # Text-based model summary
│   │   └── cv_results.csv          # Hyperparameter search results
│   └── ...
└── ...
```

## **How It Works**
- Each **experiment** has a unique `experiment_id` folder.
- Each **run** within that experiment creates a subfolder identified by a unique `run_id`.
- Inside each run's `artifacts/` folder:
  - **Model objects** are saved (e.g., `.keras`, `.pkl`, `.json`).
  - **Reports and metrics** are stored for reproducibility and later review.
  - **Visuals** (confusion matrices, plots) are logged alongside text reports.

## **Notes**
- Artifacts are directly accessible from the MLflow UI at [http://localhost:5555](http://localhost:5555)
- Best models are often logged in a `best_model/` subfolder for clarity.
- These outputs should be considered **generated artifacts** and do not need manual editing.
- By default, they are stored locally but can be configured to use cloud storage via MLflow settings.