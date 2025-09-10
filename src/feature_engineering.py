# -------------------------------------------------------------------
# Simple feature engineering module
# -------------------------------------------------------------------

import pandas as pd
from pathlib import Path

def feature_engineering(input_path: str = "data/clean.csv",
                        output_path: str = "data/features.csv") -> None:
    """
    Load the cleaned dataset, apply simple feature engineering, and save to a new CSV.

    Feature steps
    -------------
    1. One-hot encode categorical columns
    2. Standardise numeric columns (optional simple z-score)
    
    Parameters
    ----------
    input_path : str, optional
        Path to the cleaned CSV file (default: 'data/clean.csv').
    output_path : str, optional
        Path where the features CSV will be saved (default: 'data/features.csv').

    Returns
    -------
    None
    """
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = pd.read_csv(input_path)

    # Target column to exclude from transforms
    target_col = "loan_status"

    # Identify categorical vs numeric columns (simple heuristic)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Remove target from numeric transformations if present
    if target_col in num_cols:
        num_cols.remove(target_col)

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    # Standardise numeric columns (z-score)
    for col in num_cols:
        mean, std = df[col].mean(), df[col].std()
        if std > 0:  # avoid divide by zero
            df[col] = (df[col] - mean) / std

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Feature dataset saved to {output_path} (rows={len(df)}, cols={len(df.columns)})")

    # Quick peek
    print(df.head())
    print(df.info())

# -------------------------------------------------------------------
# Run from CLI
# -------------------------------------------------------------------
if __name__ == "__main__":
    feature_engineering()
