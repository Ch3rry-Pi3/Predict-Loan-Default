# -------------------------------------------------------------------
# Data cleaning module
# -------------------------------------------------------------------

import pandas as pd
from pathlib import Path

def clean_data(input_path: str = "data/raw.csv", output_path: str = "data/clean.csv") -> None:
    """
    Load the raw dataset, enforce simple dtypes, apply basic cleaning, and save to CSV

    Cleaning Steps
    --------------
    1) Enforce dtypes:
       - Integers: person_age, person_income, loan_amnt, loan_status, cb_person_cred_hist_length
       - Floats:   person_emp_length, loan_int_rate, loan_percent_income
       - Categories/strings: person_home_ownership, loan_intent, loan_grade, cb_person_default_on_file
    2) Drop duplicates
    3) Drop rows with any missing values
    4) Reset index

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

    # Define desired schema
    int_cols   = ["person_age", "person_income", "loan_amnt", "loan_status", "cb_person_cred_hist_length"]
    float_cols = ["person_emp_length", "loan_int_rate", "loan_percent_income"]
    cat_cols   = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]

    # Coerce numerics (anything invalid -> NaN)
    for c in int_cols + float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Cast floats
    for c in float_cols:
        if c in df.columns:
            df[c] = df[c].astype("float64")

    # Cast integers
    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].astype("Int64")
    
    # Cast categoricals
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")

    # Calculate rows before cleaning
    rows_before = len(df)
    print(f"Rows before cleaning: {rows_before}")

    # Simple cleaning
    df = df.drop_duplicates()
    df = df.dropna()
    df = df.reset_index(drop=True)

    # Finalise integer columns now NaNs are gone
    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].astype(int)

    # Calculate rows after cleaning
    rows_after = len(df)
    print(f"Rows after cleaning:  {rows_after}")
    print(f"Difference in rows = {rows_before} - {rows_after} = {rows_before - rows_after}")

    # Calculate difference before and after cleaning
    print(f"Difference in rows = {rows_before} - {rows_after} = {rows_before - rows_after} rows")

    # Quick dtype summary
    print("\nDtype summary:")
    print(df.dtypes)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to {output_path}, (rows = {df.shape[0]}, columns = {df.shape[1]})")

# -------------------------------------------------------------------
# Run from CLI
# -------------------------------------------------------------------

if __name__ == "__main__":
    clean_data()