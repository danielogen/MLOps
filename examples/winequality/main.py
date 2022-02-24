from model import DataExtractor, TrainModel
import mlflow

# set tracking url
mlflow.set_tracking_uri('http://127.0.0.1:5000')



if __name__ == "__main__":
    data = DataExtractor.fetch_data()
    #TrainModel.train_model_(data, alpha=0.5, l1_ratio=0.5)

    TrainModel.train_model(mlflow_run_name='elasticNet', data=data, alpha=0.9, l1_ratio=0.7)