# **Preprocessing - Bronze -> Silver -> Gold**

SThis folder (`src/preprocessing/`) contains the **data preprocessing pipeline** for the Loan Default Prediction project. It follows the **Medallion Architecture**:

- **Bronze**: Raw ingested dataset (from OpenML).
- **Silver**: Cleaned, schema-enforced dataset.
- **Gold**: Feature-engineered dataset ready for ML models.

## **Modules**

- **`import_data.py`**
  Downloads the loan default dataset from OpenML and saves it to the **Bronze layer** (`data/bronze/raw.csv`).

- **`clean_data.py`**
  Applies schema enforcement, removes duplicates, handles missing values, and saves to the **Silver layer** (`data/silver/clean.csv`)

- **`feature_engineering.py`**
  Applies simple feature engineering:
  - One-hot encoding of categorical variables,
  - Z-score standardisation of numeric columns.
  Produces the **Gold layer** dataset (`data/gold/features.csv`).

- **`config.py`**
  Centralises environment variables (paths and toggles) loaded from `.env`.

- **`preprocess_cli.py`**
  CLI entry point to run preprocessing stages sequentially (Import -> Clean -> Feature Engineering).

## **Usage**

Run the entire preprocessing pipeline:

```bash
python -m src.preprocessing.preprocess_cli
```

Or run individual stages for development/testing:

```bash
pythin -m src.preprocessing.import_data
pythin -m src.preprocessing.clean_data
pythin -m src.preprocessing.feature_engineering
```