# ============================================================================
# ModelServe — Feast Feature Client
# ============================================================================
# TODO: Implement feature fetching from the Feast online store.
#
# This module should:
#   - Initialize a Feast FeatureStore client pointing at your feast_repo
#   - Provide a get_features(entity_id) method that:
#     1. Calls store.get_online_features() with the entity key (cc_num)
#     2. Converts the result to a dictionary or DataFrame
#     3. Handles missing features gracefully (log warning, return defaults)
#   - Track hit/miss counts for Prometheus metrics
#
# Key Feast APIs to use:
#   - feast.FeatureStore(repo_path=...)
#   - store.get_online_features(features=[...], entity_rows=[...])
#
# IMPORTANT: Use the Feast SDK — do NOT query Redis directly.
# The TA will check this during the demo.
# ============================================================================

# ============================================================================
# ModelServe — Feast Feature Client
# ============================================================================

import logging
from typing import Dict, Any

import pandas as pd

from feast import FeatureStore

from app.config import settings
from app.metrics import (
    feature_store_hits_total,
    feature_store_misses_total,
)


# ----------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# FEATURE CLIENT
# ----------------------------------------------------------------------------

class FeatureClient:
    """
    Handles online feature retrieval using Feast.

    Responsibilities:
        - Connect to Feast FeatureStore
        - Fetch online features using entity_id (cc_num)
        - Handle missing features gracefully
        - Track hit/miss metrics
    """

    def __init__(self):

        logger.info(
            "Initializing Feast FeatureStore from repo: %s",
            settings.FEAST_REPO_PATH,
        )

        self.store = FeatureStore(
            repo_path=settings.FEAST_REPO_PATH
        )

        # --------------------------------------------------------------------
        # Feature references MUST match:
        #   - Feast FeatureView schema
        #   - training feature columns
        # --------------------------------------------------------------------

        self.feature_refs = [
            "fraud_features:amt",
            "fraud_features:amt_log",
            "fraud_features:lat",
            "fraud_features:long",
            "fraud_features:city_pop",
            "fraud_features:hour",
            "fraud_features:day",
            "fraud_features:month",
            "fraud_features:merch_lat",
            "fraud_features:merch_long",
        ]

        # --------------------------------------------------------------------
        # Default values used when features are missing
        # --------------------------------------------------------------------

        self.default_values = {
            "amt": 0.0,
            "amt_log": 0.0,
            "lat": 0.0,
            "long": 0.0,
            "city_pop": 0,
            "hour": 0,
            "day": 0,
            "month": 0,
            "merch_lat": 0.0,
            "merch_long": 0.0,
        }

    # ------------------------------------------------------------------------
    # GET FEATURES
    # ------------------------------------------------------------------------

    def get_features(
        self,
        entity_id: int,
    ) -> Dict[str, Any]:
        """
        Fetch online features from Feast.

        Args:
            entity_id:
                Credit card number (cc_num)

        Returns:
            Dictionary of feature values
        """

        try:

            logger.info(
                "Fetching features for entity_id=%s",
                entity_id,
            )

            # ----------------------------------------------------------------
            # Feast online feature lookup
            # ----------------------------------------------------------------

            response = self.store.get_online_features(
                features=self.feature_refs,
                entity_rows=[
                    {"cc_num": entity_id}
                ],
            )

            feature_dict = response.to_dict()

            parsed_features = {}

            missing_features = []

            # ----------------------------------------------------------------
            # Convert Feast format into flat dict
            # ----------------------------------------------------------------

            for feature_ref in self.feature_refs:

                feature_name = feature_ref.split(":")[-1]

                values = feature_dict.get(feature_ref)

                value = None

                if values and len(values) > 0:
                    value = values[0]

                # ------------------------------------------------------------
                # Handle missing feature values
                # ------------------------------------------------------------

                if value is None:

                    missing_features.append(feature_name)

                    parsed_features[feature_name] = (
                        self.default_values[feature_name]
                    )

                else:
                    parsed_features[feature_name] = value

            # ----------------------------------------------------------------
            # Metrics
            # ----------------------------------------------------------------

            if missing_features:

                feature_store_misses.inc()

                logger.warning(
                    "Missing features for entity_id=%s: %s",
                    entity_id,
                    missing_features,
                )

            else:

                feature_store_hits.inc()

            return parsed_features

        except Exception as e:

            feature_store_misses.inc()

            logger.exception(
                "Failed to fetch features for entity_id=%s: %s",
                entity_id,
                str(e),
            )

            # ----------------------------------------------------------------
            # Return safe defaults instead of crashing API
            # ----------------------------------------------------------------

            return self.default_values.copy()

    # ------------------------------------------------------------------------
    # DATAFRAME HELPER
    # ------------------------------------------------------------------------

    def get_features_dataframe(
        self,
        entity_id: int,
    ) -> pd.DataFrame:
        """
        Convenience helper for model inference.

        Returns:
            Single-row pandas DataFrame
        """

        features = self.get_features(entity_id)

        return pd.DataFrame([features])