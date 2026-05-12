from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # MLflow
    MLFLOW_TRACKING_URI: str = "http://mlflow:5000"
    MODEL_NAME: str = "modelserve-model"
    MODEL_STAGE: str = "Production"

    # Feast
    FEAST_REPO_PATH: str = "./feast_repo"

    # API
    APP_NAME: str = "ModelServe"

    class Config:
        env_file = ".env"


settings = Settings()