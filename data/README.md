# **Data – Medallion Architecture**

This folder implements the **Medallion Architecture** pattern for data processing.  
It organises datasets into **Bronze**, **Silver**, and **Gold** layers, providing a structured and reproducible data pipeline.

## **Layers**

### **Bronze Layer**
- **File:** `raw.csv`  
- **Purpose:** The raw dataset imported directly from [OpenML](https://www.openml.org/).  
- Contains unprocessed records in their original state (except for minor formatting such as removing ARFF headers).  
- Acts as the **immutable source of truth** for the pipeline.

### **Silver Layer**
- **File:** `clean.csv`  
- **Purpose:** The cleaned dataset after enforcing schema, removing duplicates, and handling missing values.  
- Ensures **consistent datatypes** and a reliable intermediate table for further processing.  
- Suitable for validation and exploratory data analysis.

### **Gold Layer**
- **File:** `features.csv`  
- **Purpose:** The fully processed dataset after **feature engineering**.  
- Includes:
  - One-hot encoded categorical variables,
  - Standardised numerical variables,
  - Ready for machine learning experiments.  
- Acts as the **model-ready feature table** for training and evaluation.

## **Workflow**

1. **Bronze → Silver**
   - Run `clean_data.py`  
   - Cleans raw data and enforces schema.

2. **Silver → Gold**
   - Run `feature_engineering.py`  
   - Produces model-ready features for experiments.

3. **Gold → Experiments**
   - Used by training scripts (`experiment_rf.py`, `experiment_xgb.py`, `experiment_keras.py`) for ML modelling.

## **Why Medallion Architecture?**

- **Separation of concerns** – raw, cleaned, and engineered data are always stored separately.  
- **Reproducibility** – each layer is derived from the previous one in a deterministic way.  
- **Scalability** – easy to extend with new transformations, datasets, or outputs.  
- **Transparency** – each layer provides visibility into how the dataset evolves.
