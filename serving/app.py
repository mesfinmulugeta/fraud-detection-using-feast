"""Online fraud scoring service: Feast online features + the saved model pipeline.

For each incoming transaction, the service reads the customer and terminal features from the Feast online store,
lets Feast compute the on-demand weekend/night flags from the transaction time, adds the amount from the request,
and returns the model's fraud probability.

Run it from the repository root:

    FEAST_REPO_PATH=feature_repo_option1 MODEL_DIR=models/fraud_model_v1 uvicorn serving.app:app --port 8765
"""

import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from feast import FeatureStore

FEAST_REPO_PATH = Path(os.environ.get("FEAST_REPO_PATH", "feature_repo_option1"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models/fraud_model_v1"))

# Loaded once at start-up and shared by all requests
store = FeatureStore(repo_path=str(FEAST_REPO_PATH))
model = joblib.load(MODEL_DIR / "model.joblib")
metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
INPUT_FEATURES = metadata["input_features"]

# The features the model was trained with, taken from its feature service. `transaction_features` is left out:
# its only feature, TX_AMOUNT, comes with the request, since a new transaction is never in the online store.
ONLINE_FEATURES = [
    f"{projection.name}:{feature.name}"
    for projection in store.get_feature_service(metadata["feature_service"]).feature_view_projections
    if projection.name != "transaction_features"
    for feature in projection.features
]


class Transaction(BaseModel):
    # What the payment system knows when the transaction arrives
    TRANSACTION_ID: int
    CUSTOMER_ID: int
    TERMINAL_ID: int
    TX_AMOUNT: float
    TX_DATETIME: datetime


app = FastAPI(title="Fraud scoring service")


@app.get("/health")
def health():
    return metadata


@app.post("/predict")
def predict(transactions: list[Transaction]):
    # The entity keys select the online rows; event_timestamp is the request data of the on-demand feature view
    entity_rows = [
        {"CUSTOMER_ID": t.CUSTOMER_ID, "TERMINAL_ID": t.TERMINAL_ID, "event_timestamp": t.TX_DATETIME}
        for t in transactions
    ]
    online_features = store.get_online_features(features=ONLINE_FEATURES, entity_rows=entity_rows).to_df()
    online_features["TX_AMOUNT"] = [t.TX_AMOUNT for t in transactions]
    model_inputs = online_features[INPUT_FEATURES].astype(float)

    # A customer or terminal missing from the online store (new, or older than the TTL) can't be scored
    complete = model_inputs.notna().all(axis=1)
    probabilities = pd.Series(None, index=model_inputs.index, dtype=object)
    if complete.any():
        probabilities[complete] = model.predict_proba(model_inputs[complete])[:, 1]

    # The inputs are returned with the score so the caller can inspect them; a production service would log them
    model_inputs = model_inputs.astype(object).where(model_inputs.notna(), None)
    return [
        {"TRANSACTION_ID": t.TRANSACTION_ID, "fraud_probability": probability, "features": features}
        for t, probability, features in zip(transactions, probabilities, model_inputs.to_dict(orient="records"))
    ]
