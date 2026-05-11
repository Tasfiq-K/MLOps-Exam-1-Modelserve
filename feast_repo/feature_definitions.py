# ============================================================================
# ModelServe — Feast Feature Definitions
# ============================================================================
# TODO: Define your Feast entities, data sources, and feature views.
#
# You need to create:
#
#   1. Entity — the credit card number (cc_num) from the dataset
#      - This is the join key for feature lookups
#
#   2. FileSource (or S3 source) — points to your features.parquet file
#      - Must specify the timestamp_field for point-in-time joins
#
#   3. FeatureView — maps the entity to features from the data source
#      - List every feature with its data type (Float64, Int64, String, etc.)
#      - Set a TTL (time-to-live) for feature freshness
#
# The features defined here must match exactly what train.py exports
# to features.parquet and what the FastAPI service requests from Feast.
#
# After defining these, run:
#   cd feast_repo && feast apply
#   python scripts/materialize_features.py
#
# Refer to Feast documentation: https://docs.feast.dev/
# ============================================================================


from datetime import timedelta

from feast import (
    Entity,
    FeatureView,
    Field,
    FileSource,
)

from feast.types import Float64, Int64

# The entity represents the lookup key used during online inference.
# In this fraud dataset, we use the credit card number.

cc_num = Entity(
    name="cc_num",
    join_keys=["cc_num"],
    description="Credit card number entity",
)

# Feast reads features from this parquet file.
# event_timestamp is REQUIRED for point-in-time correctness.

fraud_source = FileSource(
    path="../training/features.parquet",
    timestamp_field="event_timestamp",
)

#  Feature view 
# Defines:
#   - feature schema
#   - online serving behavior
#   - freshness TTL

fraud_features_view = FeatureView(
    name="fraud_features",
    entities=[cc_num],

    ttl=timedelta(days=1),

    schema=[
        Field(name="amt", dtype=Float64),
        Field(name="amt_log", dtype=Float64),
        Field(name="lat", dtype=Float64),
        Field(name="long", dtype=Float64),
        Field(name="city_pop", dtype=Int64),
        Field(name="hour", dtype=Int64),
        Field(name="day", dtype=Int64),
        Field(name="month", dtype=Int64),
        Field(name="merch_lat", dtype=Float64),
        Field(name="merch_long", dtype=Float64),
    ],

    source=fraud_source,

    online=True,

    tags={
        "project": "modelserve",
        "team": "mlops",
    },
)