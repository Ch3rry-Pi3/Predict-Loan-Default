from mlflow.tracking import MlflowClient
import mlflow
from pathlib import Path

# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------
TRACKING_URI = "http://localhost:5555"
EXPERIMENT_NAME = "loan_default_keras_v2"   # use a fresh name so artifacts land correctly

# --------------------------------------------------------------------
# Setup client and experiment
# --------------------------------------------------------------------
mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

exp = client.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    exp_id = client.create_experiment(EXPERIMENT_NAME)
    print(f"Created new experiment '{EXPERIMENT_NAME}' with id={exp_id}")
else:
    exp_id = exp.experiment_id
    print(f"Using existing experiment '{exp.name}' (id={exp.experiment_id})")
    print("artifact_location:", exp.artifact_location)

# --------------------------------------------------------------------
# Start a run and log a test artifact
# --------------------------------------------------------------------
Path("hello.txt").write_text("hi")

with mlflow.start_run(experiment_id=exp_id):
    mlflow.log_artifact("hello.txt")
    print("Run artifact URI:", mlflow.get_artifact_uri())
