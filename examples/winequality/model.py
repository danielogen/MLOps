from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from pathlib import Path

import pandas as pd
import numpy as np
import mlflow
import os

BASE_DIR = Path(__file__).resolve().parent
data_path = os.path.join(BASE_DIR, 'data')

class DataExtractor:

    def __init__(self):
        pass

    def fetch_data()->pd.DataFrame:
        print("fetching data....")

        fetch = f'{data_path}/winequality-red.csv'
        csv_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        data = pd.read_csv(fetch, sep=";")
        
        print("fetchin data completed....")
        
        return data

class EvalModel:

    def __init__(self):
        pass

    @staticmethod
    def eval_metrics(actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

class TrainModel:
    def __init__(self):
        pass

    @staticmethod
    def train_model_(data, alpha=0.5, l1_ratio=0.5) -> None:

        train, test = train_test_split(data)

        # The predicted column is "quality" which is a scalar from [3, 9]
        train_x = train.drop(["quality"], axis=1)
        test_x = test.drop(["quality"], axis=1)
        train_y = train[["quality"]]
        test_y = test[["quality"]]


        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)
        rmse, mae, r2 = EvalModel.eval_metrics(test_y, predicted_qualities)

        # Print out metrics
        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

    def train_model(data, mlflow_run_name, alpha=0.9, l1_ratio=0.5):
        # set the experiement name
        mlflow.set_experiment("Wine Quality Prediction 2")
        train, test = train_test_split(data)

        # The predicted column is "quality" which is a scalar from [3, 9]
        train_x = train.drop(["quality"], axis=1)
        test_x = test.drop(["quality"], axis=1)
        train_y = train[["quality"]]
        test_y = test[["quality"]]
        
        with mlflow.start_run(run_name=mlflow_run_name):
            lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
            lr.fit(train_x, train_y)
            predicted_qualities = lr.predict(test_x)
            (rmse, mae, r2) = EvalModel.eval_metrics(test_y, predicted_qualities)

            print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
            print("  RMSE: %s" % rmse)
            print("  MAE: %s" % mae)
            print("  R2: %s" % r2)

            mlflow.log_param("alpha", alpha)
            mlflow.log_param("l1_ratio", l1_ratio)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            mlflow.sklearn.log_model(
                sk_model=lr,
                artifact_path="artifact",
                registered_model_name="myfirstmodel")
