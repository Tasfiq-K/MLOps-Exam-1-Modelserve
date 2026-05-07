# ============================================================================
# ModelServe — Model Training Script
# ============================================================================
# TODO: Implement model training and MLflow registration.
#
# Dataset: https://www.kaggle.com/datasets/kartik2112/fraud-detection
#   - Use fraudTrain.csv (~1.3M rows, 22 features)
#   - Target column: is_fraud
#   - Entity key: cc_num
#   - Use class_weight='balanced' to handle class imbalance
#
# This script should:
#   1. Load fraudTrain.csv with pandas
#   2. Select and engineer features (15-20 features is enough)
#   3. Split into train/test sets (stratified on is_fraud)
#   4. Train a sklearn-compatible model (RandomForest, XGBoost, LightGBM)
#   5. Log to MLflow:
#      - Parameters: model type, hyperparameters, feature list
#      - Metrics: accuracy, precision, recall, f1, roc_auc
#      - The model artifact itself
#   6. Register the model in MLflow Model Registry
#   7. Transition the model version to "Production" stage
#   8. Export features.parquet (feature columns + cc_num + event_timestamp)
#      for Feast ingestion
#   9. Export sample_request.json with a valid entity_id for testing
#
# Prerequisites:
#   - MLflow and Postgres must be running (docker compose up postgres mlflow)
#   - fraudTrain.csv must be downloaded from Kaggle
#
# Usage:
#   python training/train.py
#
# IMPORTANT: This script must be reproducible — running it again should
# register a new model version with comparable metrics.
# Do NOT spend more than one session on model quality.
# A baseline AUC of 0.85+ is sufficient.
# ============================================================================


# ============================================================================
# ModelServe — Model Training Script
# ============================================================================

import os
import json
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from mlflow.tracking import MlflowClient


# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

DATA_PATH = "training/fraudTrain.csv"
MODEL_NAME = "modelserve-model"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "http://localhost:5000"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("modelserve-experiment")


# ----------------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------------

def load_data():
    df = pd.read_csv(DATA_PATH)

    # Convert datetime
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

    # Create event timestamp (for Feast)
    df["event_timestamp"] = df["trans_date_trans_time"]

    return df


# ----------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ----------------------------------------------------------------------------

def feature_engineering(df):
    df = df.copy()

    # Time-based features
    df["hour"] = df["event_timestamp"].dt.hour
    df["day"] = df["event_timestamp"].dt.day
    df["month"] = df["event_timestamp"].dt.month

    # Amount transformations
    df["amt_log"] = np.log1p(df["amt"])

    # Select features (keep it simple)
    feature_cols = [
        "amt",
        "amt_log",
        "lat",
        "long",
        "city_pop",
        "hour",
        "day",
        "month",
        "merch_lat",
        "merch_long",
    ]

    # Fill NA
    df[feature_cols] = df[feature_cols].fillna(0)

    return df, feature_cols


# ----------------------------------------------------------------------------
# 3. TRAIN MODEL
# ----------------------------------------------------------------------------

def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train, y_train)
    return model


# ----------------------------------------------------------------------------
# 4. EVALUATION
# ----------------------------------------------------------------------------

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
    }

    return metrics


# ----------------------------------------------------------------------------
# 5. MAIN PIPELINE
# ----------------------------------------------------------------------------

def main():

    print("Loading data...")
    df = load_data()

    print("Feature engineering...")
    df, feature_cols = feature_engineering(df)

    X = df[feature_cols]
    y = df["is_fraud"]

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    print("Training model...")

    with mlflow.start_run():

        model = train_model(X_train, y_train)

        print("Evaluating...")
        metrics = evaluate(model, X_test, y_test)

        # -------------------------
        # MLflow Logging
        # -------------------------
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("features", feature_cols)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Log model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        print("Metrics:", metrics)

    # ----------------------------------------------------------------------------
    # 6. TRANSITION MODEL TO PRODUCTION
    # ----------------------------------------------------------------------------

    client = MlflowClient()

    # Get latest model version
    latest_versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    latest_version = max(
        latest_versions,
        key=lambda v: int(v.version)
    )

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest_version.version,
        stage="Production",
        archive_existing_versions=True,
    )

    print(f"Model version {latest_version.version} promoted to Production")

    # ----------------------------------------------------------------------------
    # 7. EXPORT FEATURES FOR FEAST
    # ----------------------------------------------------------------------------

    print("Exporting features.parquet...")

    feast_df = df[feature_cols + ["cc_num", "event_timestamp"]].copy()
    feast_df.to_parquet("training/features.parquet", index=False)

    # ----------------------------------------------------------------------------
    # 8. SAMPLE REQUEST
    # ----------------------------------------------------------------------------

    print("Exporting sample_request.json...")

    sample_entity = int(df["cc_num"].iloc[0])

    sample = {
        "entity_id": sample_entity
    }

    with open("training/sample_request.json", "w") as f:
        json.dump(sample, f, indent=2)

    print("Done!")


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()