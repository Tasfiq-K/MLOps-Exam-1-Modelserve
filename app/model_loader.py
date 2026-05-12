# ============================================================================
# ModelServe — MLflow Model Loader
# ============================================================================
# TODO: Implement model loading from the MLflow Model Registry.
#
# This module should:
#   - Connect to the MLflow Tracking Server
#   - Load a model by name and stage (e.g., "Production")
#   - Store the loaded model and its version string
#   - Provide a predict() method that runs inference on feature inputs
#   - Handle connection failures gracefully (log errors, don't crash the app)
#
# Key MLflow APIs to use:
#   - mlflow.set_tracking_uri(...)
#   - mlflow.pyfunc.load_model(f"models:/{name}/{stage}")
#   - model.predict(features_dataframe)
#
# The model must be loaded ONCE and reused across requests.
# ============================================================================


# ============================================================================
# ModelServe — MLflow Model Loader
# ============================================================================

import logging
from typing import Optional

import mlflow
import mlflow.pyfunc
import pandas as pd

from mlflow.tracking import MlflowClient

from app.config import settings


# ----------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# ----------------------------------------------------------------------------
# MODEL SERVICE
# ----------------------------------------------------------------------------

class ModelService:
    """
    Loads and serves MLflow models from the MLflow Model Registry.

    Responsibilities:
        - Connect to MLflow tracking server
        - Load model once during application startup
        - Store loaded model + version metadata
        - Provide prediction APIs
        - Handle loading failures gracefully
    """

    def __init__(self):

        self.model = None
        self.model_name = settings.MODEL_NAME
        self.model_stage = settings.MODEL_STAGE

        self.model_version = "unknown"
        self.model_uri = None

        self.client = None

        self._initialize()

    # ------------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------------

    def _initialize(self):
        """
        Initialize MLflow connection and load the model.
        """

        try:

            logger.info(
                "Connecting to MLflow Tracking Server: %s",
                settings.MLFLOW_TRACKING_URI,
            )

            mlflow.set_tracking_uri(
                settings.MLFLOW_TRACKING_URI
            )

            self.client = MlflowClient()

            self.model_uri = (
                f"models:/{self.model_name}/{self.model_stage}"
            )

            logger.info(
                "Loading model from registry: %s",
                self.model_uri,
            )

            # ----------------------------------------------------------------
            # Load model ONCE
            # ----------------------------------------------------------------

            self.model = mlflow.pyfunc.load_model(
                self.model_uri
            )

            # ----------------------------------------------------------------
            # Fetch model version metadata
            # ----------------------------------------------------------------

            latest_versions = self.client.get_latest_versions(
                self.model_name,
                stages=[self.model_stage],
            )

            if latest_versions:
                self.model_version = (
                    latest_versions[0].version
                )

            logger.info(
                "Successfully loaded model '%s' "
                "(stage=%s, version=%s)",
                self.model_name,
                self.model_stage,
                self.model_version,
            )

        except Exception as e:

            logger.exception(
                "Failed to initialize MLflow model: %s",
                str(e),
            )

            # Keep service alive even if model loading fails
            self.model = None

    # ------------------------------------------------------------------------
    # HEALTH CHECK
    # ------------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """
        Indicates whether the model loaded successfully.
        """

        return self.model is not None

    # ------------------------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------------------------

    def predict(self, features: pd.DataFrame):
        """
        Run inference using the loaded model.

        Args:
            features:
                Pandas DataFrame containing feature columns.

        Returns:
            Model predictions.
        """

        if not self.is_ready:
            raise RuntimeError(
                "Model is not loaded"
            )

        return self.model.predict(features)

    # ------------------------------------------------------------------------
    # PREDICT PROBABILITIES
    # ------------------------------------------------------------------------

    def predict_proba(self, features: pd.DataFrame):
        """
        Run probability inference if supported by model.

        Args:
            features:
                Pandas DataFrame containing feature columns.

        Returns:
            Prediction probabilities.
        """

        if not self.is_ready:
            raise RuntimeError(
                "Model is not loaded"
            )

        # unwrap python model if needed
        raw_model = getattr(
            self.model,
            "_model_impl",
            self.model,
        )

        if hasattr(raw_model, "predict_proba"):
            return raw_model.predict_proba(features)

        # fallback for pyfunc models
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(features)

        raise NotImplementedError(
            "Loaded model does not support predict_proba"
        )

    # ------------------------------------------------------------------------
    # MODEL INFO
    # ------------------------------------------------------------------------

    def get_model_info(self) -> dict:
        """
        Returns metadata about the currently loaded model.
        """

        return {
            "model_name": self.model_name,
            "model_stage": self.model_stage,
            "model_version": self.model_version,
            "model_uri": self.model_uri,
            "ready": self.is_ready,
        }