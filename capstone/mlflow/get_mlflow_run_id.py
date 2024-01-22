from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient


EXPERIMENT_NAME = "credit-card-fraud-detection"
MLFLOW_TRACKING_URI = "http://localhost:5001"
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def get_experiment_id(experiment_name: str) -> int:
    for experiment in client.search_experiments():
        if experiment.name == experiment_name:
            return experiment.experiment_id


def get_run_id(_experiment_id: str):
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.rmse ASC"],
    )
    for run in runs:
        return run.info.run_id


if __name__ == "__main__":
    experiment_id = str(get_experiment_id(EXPERIMENT_NAME))
    print(get_run_id(experiment_id))
