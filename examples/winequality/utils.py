from config import configuration, environment

import mlflow

class Mlflow:
    def __init__(self) -> None:
        self.uri = configuration.MLFLOW_TRACKING_URI

    def set_uri (self):
        mlflow.set_tracking_uri(self.uri)
        print(f"mlflow server: {mlflow.get_tracking_uri()}")

    @staticmethod
    def set_experiment(name: str):
        mlflow.set_experiment(f'{name}_{environment}')

    @staticmethod
    def log_parameters(**params) -> None:
        for key, value in params.items():
             mlflow.log_param(key, value)

    @staticmethod
    def log_model(model: any, path: str, registered_model_name: str):
        mlflow.sklearn.log_model(
            sk_model = model, 
            artifact_path=path,
            registered_model_name = registered_model_name
            )

    def start_run(name: str) -> None:
        mlflow.start_run(run_name=name)

    def end_run(name: str) -> None:
        mlflow.end_run(name)