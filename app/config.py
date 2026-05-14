from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # =========================================================================
    # MLflow
    # =========================================================================
    mlflow_tracking_uri: str = "http://mlflow:5000"
    model_name: str = "modelserve-model"
    model_stage: str = "Production"

    # =========================================================================
    # Feast
    # =========================================================================
    feast_repo_path: str = "./feast_repo"

    # =========================================================================
    # FastAPI
    # =========================================================================
    app_name: str = "ModelServe"
    fastapi_host: str = "0.0.0.0"
    fastapi_port: int = 8000
    log_level: str = "INFO"

    # =========================================================================
    # Pydantic Settings
    # =========================================================================
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )


settings = Settings()