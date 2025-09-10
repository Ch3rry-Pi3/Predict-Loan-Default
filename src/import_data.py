# -------------------------------------------------------------------
# Data import module
# -------------------------------------------------------------------

# Import dependencies
import pandas as pd
from pathlib import Path

# URL for the loan default dataset (OpenML)
DATA_URL = "https://www.openml.org/data/download/22102279/dataset"

def import_data(output_path: str = "data/raw.csv") -> None:
    """
    Download the loan default dataset and save as a CSV file.

    Parameters
    ----------
    output_path : str, optional
        Path where the CSV file will be saved (default: 'data/raw.csv)

    Returns
    -------
    None
    """

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load dataset directly from URL, skipping the ARFF header
    df = pd.read_csv(DATA_URL, skiprows=75)

    # Columns
    cols = [
        "person_age",
        "person_income",
        "person_home_ownership",
        "person_emp_length",
        "loan_intent",
        "loan_grade", 
        "loan_amnt", 
        "loan_int_rate",
        "loan_status", 
        "loan_percent_income",
        "cb_person_default_on_file",
        "cb_person_cred_hist_length"
    ]

    # Add columns
    df.columns = cols

    # Save to CSV
    df.to_csv(output_path, index=False)

    print(f"Dataset saved to {output_path} (rows = {df.shape[0]}, columns = {df.shape[1]})")

# -------------------------------------------------------------------
# Run from CLI
# -------------------------------------------------------------------

if __name__ == "__main__":
    import_data()
