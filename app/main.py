# ============================================================================
# ModelServe — FastAPI Inference Service
# ============================================================================
# TODO: Implement the FastAPI application with the following endpoints:
#
#   GET  /health
#     - Returns: {"status": "healthy", "model_version": "<version>"}
#     - Used by Docker healthchecks and CI deploy verification
#
#   POST /predict
#     - Accepts: {"entity_id": <int>}
#     - Steps:
#       1. Fetch features from Feast online store using entity_id
#       2. Run the model (loaded on startup from MLflow Registry)
#       3. Record Prometheus metrics (request count, duration, errors)
#       4. Return: {"prediction": <int>, "probability": <float>,
#                   "model_version": "<version>", "timestamp": "<iso8601>"}
#
#   GET  /predict/<entity_id>?explain=true
#     - Same as POST /predict but also returns the feature values used
#     - Useful for debugging predictions during the demo
#
#   GET  /metrics
#     - Exposes Prometheus metrics in text format
#     - Must include: prediction_requests_total, prediction_duration_seconds,
#       prediction_errors_total, model_version_info
#
# Key design requirements:
#   - Load the model from MLflow Registry ONCE on startup (not per request)
#   - Fetch features through Feast SDK (not direct Redis queries)
#   - Return structured JSON errors with appropriate HTTP status codes
#   - Use Pydantic models for request/response validation
# ============================================================================


# ============================================================================
# ModelServe — FastAPI Inference Service
# ============================================================================

import logging
import time

from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    HTTPException,
    Query,
)

from fastapi.responses import (
    JSONResponse,
    Response,
)

from prometheus_client import (
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from app.schemas import (
    PredictionRequest,
    PredictionResponse,
    ExplainPredictionResponse,
)

from app.model_loader import ModelService
from app.feature_client import FeatureClient

from app.metrics import (
    prediction_requests_total,
    prediction_duration_seconds,
    prediction_errors_total,
    model_version_info,
)

from app.config import settings


# ----------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# GLOBAL SERVICES
# ----------------------------------------------------------------------------
# These are initialized ONCE during startup.
# This avoids:
#   - repeated MLflow model loading
#   - repeated Feast initialization
#   - unnecessary latency
# ----------------------------------------------------------------------------

model_service = None
feature_client = None


# ----------------------------------------------------------------------------
# FASTAPI LIFESPAN
# ----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    global model_service
    global feature_client

    logger.info("Starting ModelServe application")

    # ------------------------------------------------------------------------
    # Initialize MLflow model service
    # ------------------------------------------------------------------------

    model_service = ModelService()

    # ------------------------------------------------------------------------
    # Initialize Feast feature client
    # ------------------------------------------------------------------------

    feature_client = FeatureClient()

    # ------------------------------------------------------------------------
    # Expose model version metric
    # ------------------------------------------------------------------------

    model_version_info.labels(
        version=model_service.model_version
    ).set(1)

    logger.info(
        "Application startup completed successfully"
    )

    yield

    logger.info("Shutting down ModelServe application")


# ----------------------------------------------------------------------------
# FASTAPI APP
# ----------------------------------------------------------------------------

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    lifespan=lifespan,
)


# ----------------------------------------------------------------------------
# INTERNAL PREDICTION PIPELINE
# ----------------------------------------------------------------------------

def run_prediction(entity_id: int):

    start_time = time.time()

    try:

        prediction_requests_total.inc()

        # --------------------------------------------------------------------
        # Fetch features from Feast online store
        # --------------------------------------------------------------------

        features = feature_client.get_features(entity_id)

        if not features:

            prediction_errors_total.inc()

            raise HTTPException(
                status_code=404,
                detail="Features not found",
            )

        # --------------------------------------------------------------------
        # Convert to DataFrame
        # --------------------------------------------------------------------

        feature_df = (
            feature_client.get_features_dataframe(entity_id)
        )

        # --------------------------------------------------------------------
        # Run inference
        # --------------------------------------------------------------------

        prediction = int(
            model_service.predict(feature_df)[0]
        )

        # --------------------------------------------------------------------
        # Probability prediction
        # --------------------------------------------------------------------

        probability_output = (
            model_service.predict_proba(feature_df)
        )

        probability = float(
            probability_output[0][1]
        )

        # --------------------------------------------------------------------
        # Build response
        # --------------------------------------------------------------------

        result = {
            "prediction": prediction,
            "probability": probability,
            "model_version": str(
                model_service.model_version
            ),
            "timestamp": datetime.utcnow().isoformat(),
            "features": features,
        }

        return result

    except HTTPException:
        raise

    except Exception as e:

        prediction_errors_total.inc()

        logger.exception(
            "Prediction failed for entity_id=%s",
            entity_id,
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    finally:

        prediction_duration_seconds.observe(
            time.time() - start_time
        )


# ----------------------------------------------------------------------------
# HEALTH ENDPOINT
# ----------------------------------------------------------------------------

@app.get("/health")
def health():

    status = (
        "healthy"
        if model_service and model_service.is_ready
        else "unhealthy"
    )

    return {
        "status": status,
        "model_version": (
            model_service.model_version
            if model_service
            else "unknown"
        ),
    }


# ----------------------------------------------------------------------------
# POST /predict
# ----------------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest):

    result = run_prediction(
        entity_id=request.entity_id
    )

    # Exclude feature dump
    result.pop("features", None)

    return result


# ----------------------------------------------------------------------------
# GET /predict/{entity_id}?explain=true
# ----------------------------------------------------------------------------

@app.get(
    "/predict/{entity_id}",
    response_model=ExplainPredictionResponse,
)
def predict_explain(
    entity_id: int,
    explain: bool = Query(default=False),
):

    result = run_prediction(entity_id)

    if explain:
        return result

    result.pop("features", None)

    return result


# ----------------------------------------------------------------------------
# METRICS ENDPOINT
# ----------------------------------------------------------------------------

@app.get("/metrics")
def metrics():

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ----------------------------------------------------------------------------
# GLOBAL EXCEPTION HANDLER
# ----------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(
    request,
    exc,
):

    logger.exception(
        "Unhandled application error: %s",
        str(exc),
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error"
        },
    )