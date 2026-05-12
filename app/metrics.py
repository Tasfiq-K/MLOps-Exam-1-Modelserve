# ============================================================================
# ModelServe — Prometheus Metrics
# ============================================================================
# TODO: Define Prometheus metrics for the inference service.
#
# Required metrics (from exam document):
#   - prediction_requests_total (Counter)
#       Total number of prediction requests received
#
#   - prediction_duration_seconds (Histogram)
#       Time taken to process each prediction (feature fetch + model inference)
#
#   - prediction_errors_total (Counter)
#       Number of failed prediction requests
#
#   - model_version_info (Gauge with a "version" label)
#       Currently served model version — set once on startup
#
#   - feast_online_store_hits_total (Counter)
#       Successful feature lookups from Feast
#
#   - feast_online_store_misses_total (Counter)
#       Failed or empty feature lookups from Feast
#
# Use the prometheus_client library:
#   from prometheus_client import Counter, Histogram, Gauge
#
# To expose metrics at /metrics, use generate_latest() from prometheus_client
# and return it as a Starlette Response with the correct content type.
# ============================================================================


# ============================================================================
# ModelServe — Prometheus Metrics
# ============================================================================

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
)

# ----------------------------------------------------------------------------
# PREDICTION REQUESTS
# ----------------------------------------------------------------------------
# Total number of prediction requests received by the API.
#
# Incremented:
#   - once per prediction request
#
# Useful for:
#   - traffic monitoring
#   - request rate calculations
#   - throughput dashboards
# ----------------------------------------------------------------------------

prediction_requests_total = Counter(
    name="prediction_requests_total",
    documentation="Total number of prediction requests received",
)

# ----------------------------------------------------------------------------
# PREDICTION LATENCY
# ----------------------------------------------------------------------------
# Measures end-to-end prediction latency:
#   - Feast online feature retrieval
#   - Feature preprocessing
#   - Model inference
#
# Prometheus automatically exposes:
#   - bucket counts
#   - sum
#   - count
#
# Useful for:
#   - p50/p95/p99 latency dashboards
#   - SLA/SLO monitoring
# ----------------------------------------------------------------------------

prediction_duration_seconds = Histogram(
    name="prediction_duration_seconds",
    documentation="Prediction request duration in seconds",

    # Custom latency buckets for inference workloads
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
    ),
)

# ----------------------------------------------------------------------------
# PREDICTION ERRORS
# ----------------------------------------------------------------------------
# Counts failed prediction requests.
#
# Incremented on:
#   - feature retrieval failures
#   - model inference failures
#   - validation errors
#   - unexpected exceptions
#
# Useful for:
#   - alerting
#   - error rate dashboards
# ----------------------------------------------------------------------------

prediction_errors_total = Counter(
    name="prediction_errors_total",
    documentation="Total number of failed prediction requests",
)

# ----------------------------------------------------------------------------
# MODEL VERSION INFO
# ----------------------------------------------------------------------------
# Exposes the currently served model version.
#
# This is implemented as a Gauge with labels.
#
# Example exposed metric:
#   model_version_info{version="7"} 1
#
# Useful for:
#   - deployment verification
#   - rollback visibility
#   - Grafana annotations
# ----------------------------------------------------------------------------

model_version_info = Gauge(
    name="model_version_info",
    documentation="Currently loaded MLflow model version",
    labelnames=["version"],
)

# ----------------------------------------------------------------------------
# FEAST ONLINE STORE HITS
# ----------------------------------------------------------------------------
# Counts successful online feature retrievals from Feast.
#
# Incremented when:
#   - features are successfully returned
#
# Useful for:
#   - monitoring feature store health
#   - online serving reliability
# ----------------------------------------------------------------------------

feast_online_store_hits_total = Counter(
    name="feast_online_store_hits_total",
    documentation="Successful Feast online feature retrievals",
)

# ----------------------------------------------------------------------------
# FEAST ONLINE STORE MISSES
# ----------------------------------------------------------------------------
# Counts failed or empty online feature retrievals.
#
# Incremented when:
#   - features missing
#   - entity not found
#   - Feast/Redis unavailable
#
# Useful for:
#   - detecting feature skew
#   - monitoring Redis/Feast failures
# ----------------------------------------------------------------------------

feast_online_store_misses_total = Counter(
    name="feast_online_store_misses_total",
    documentation="Failed or empty Feast online feature retrievals",
)