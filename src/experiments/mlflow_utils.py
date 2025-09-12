# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------

from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
from .config import TRACKING_URI

# -------------------------------------------------------------------
# MLflow helpers
# -------------------------------------------------------------------

def ensure_experiment(name: str) -> str:
    """
    Ensure an MLflow experiment exists and return its experiment ID.

    Parameters
    ----------
    name : str
        The name of the MLflow experiment.

    Returns
    -------
    str
        The experiment ID corresponding to 'name'. Prints the artifact
        location to confirm where files will land.
    """

    # Point MLflow at tracking server/URI
    mlflow.set_tracking_uri(TRACKING_URI)

    # Create client and check for existing experiment
    client = MlflowClient()
    exp = client.get_experiment_by_name(name)

    # Create if missing, otherwise reuse existing
    if exp is None:
        exp_id = client.create_experiment(name)
        print(f"[MLflow] Created experiment '{name}' (id={exp_id})")
        exp = client.get_experiment(exp_id)  # fetch to print artifact root
    else:
        exp_id = exp.experiment_id
        print(f"[MLflow] Using experiment '{exp.name}' (id={exp_id})")

    # Show artifact base location
    print(f"[MLflow] artifact_location: {exp.artifact_location}")
    return exp_id


def _reset_active_run() -> None:
    """
    End any MLflow run that is accidentally still open.

    Notes
    -----
    Defensive helper to avoid "active run" errors when nesting or re-running.
    """

    # End any run that's accidentally still open
    if mlflow.active_run() is not None:
        mlflow.end_run()