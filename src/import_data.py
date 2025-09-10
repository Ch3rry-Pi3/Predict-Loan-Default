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

# -------------------------------------------------------------------
# Run from CLI
# -------------------------------------------------------------------
if __name__ == "__main__":
    import_data()
