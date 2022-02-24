import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
dotenv_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path)


class Config:
    DEBUG = False
    TESTING = False
    CSRF_ENABLED = True

class ProductionConfig(Config):
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
    MLFLOW_TRACKING_USERNAME = os.getenv('MLFLOW_TRACKING_USERNAME')
    MLFLOW_TRACKING_PASSWORD = os.getenv('MLFLOW_TRACKING_PASSWORD')

class TestingConfig(Config):
    DEBUG = True
    TESTING = True
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
    MLFLOW_TRACKING_USERNAME = os.getenv('MLFLOW_TRACKING_USERNAME')
    MLFLOW_TRACKING_PASSWORD = os.getenv('MLFLOW_TRACKING_PASSWORD')

class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = True
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI_DEV')



app_config = {"development": DevelopmentConfig,
              "testing": TestingConfig,
              "production": ProductionConfig,
              "staging": TestingConfig}

environment = os.getenv("ENV")
print("ENVIRONMENT", environment or 'staging')

configuration = app_config.get(environment, TestingConfig)