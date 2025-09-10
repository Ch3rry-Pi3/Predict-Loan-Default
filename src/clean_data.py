# -------------------------------------------------------------------
# Data cleaning module
# -------------------------------------------------------------------

import pandas as pd
from pathlib import Path

def clean_data(input_path: str = "data/raw.csv", output_path: str = "data/clean.csv") -> None:
    """
    Load the raw dataset, apply simple cleaning, and save to a new CSV.

    Cleaning Steps
    --------------
    1. Drop duplicate rows
    2. Drop rows with any missing values
    3. Reset index

    Parameters
    ----------
    input_path : str, optional
        Path to the raw CSV file (default: 'data/raw.csv')
    output_path : str, optional
        Path where the cleaned CSV will be saved (default: 'data/clean.csv')

    Returns
    -------
    None
    """

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = pd.read_csv(input_path)

    # Calculate rows before cleaning
    rows_before = df.shape[0]
    print(f"Rows before cleaning: {rows_before}")

    # Simple cleaning
    df = df.drop_duplicates()
    df = df.dropna()
    df = df.reset_index(drop=True)

    # Calculate rows after cleaning
    rows_after = df.shape[0]
    print(f"Rows after cleaning: {rows_after}")

    # Calculate difference before and after cleaning
    print(f"Difference in rows = {rows_before} - {rows_after} = {rows_before - rows_after} rows")

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to {output_path}, (rows = {df.shape[0]}, columns = {df.shape[1]})")

# -------------------------------------------------------------------
# Run from CLI
# -------------------------------------------------------------------

if __name__ == "__main__":
    clean_data()