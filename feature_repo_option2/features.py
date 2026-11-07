"""Feast feature definitions for card-fraud detection - Option 2: a single transaction-keyed view.

All stored features are attached to the transaction:

* ``transaction_all_features``       keyed on TRANSACTION_ID (amount, customer spending behaviour and
  terminal risk scores)
* ``transaction_datetime_features``  an on-demand feature view computing the weekend/night flags
  from the transaction timestamp at retrieval time

The windowed customer/terminal features are computed in the notebook (following the Fraud Detection
Handbook, Chapter 3) and written as Parquet files partitioned by day (``date=%Y-%m-%d``).
"""

import os
from datetime import timedelta
from pathlib import Path

import pandas as pd

from feast import Entity, FeatureService, FeatureView, Field, FileSource, RequestSource, ValueType
from feast.data_format import ParquetFormat
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float64, Int64, UnixTimestamp

# Must match the notebook: set FRAUD_DATA_PREFIX to use a location other than <repo root>/data
DATA_PREFIX = Path(
    os.environ.get("FRAUD_DATA_PREFIX", Path(__file__).resolve().parents[1] / "data")
).expanduser().resolve()
TRANSFORMED_DATA_DIR = str(DATA_PREFIX / "simulated-data-transformed")

WINDOW_SIZES_IN_DAYS = [1, 7, 30]
DELAY_PERIOD_IN_DAYS = 7

# ---------------------------------------------------------------------------
# Entity: the key features are attached to
# ---------------------------------------------------------------------------
transaction = Entity(
    name="transaction",
    join_keys=["TRANSACTION_ID"],
    value_type=ValueType.INT64,
    description="A card payment transaction",
)

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------
# Batch source: the hive-partitioned Parquet directory written by the notebook
transformed_transactions_source = FileSource(
    name="transformed_transactions_source",
    path=TRANSFORMED_DATA_DIR,
    file_format=ParquetFormat(),
    timestamp_field="TX_DATETIME",
    description="Transactions enriched with the windowed baseline features, partitioned by date=%Y-%m-%d",
)

# Request source: data that comes with the request (the entity dataframe, or the online request)
transaction_request = RequestSource(
    name="transaction_request",
    schema=[Field(name="event_timestamp", dtype=UnixTimestamp)],
    description="Timestamp of the transaction being scored (the event_timestamp of the entity row / online request)",
)

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------
transaction_all_fields = [Field(name="TX_AMOUNT", dtype=Float64, description="Transaction amount")]
for window_size in WINDOW_SIZES_IN_DAYS:
    transaction_all_fields += [
        Field(
            name=f"CUSTOMER_ID_NB_TX_{window_size}DAY_WINDOW",
            dtype=Float64,
            description=f"Number of customer transactions in the last {window_size} day(s)",
        ),
        Field(
            name=f"CUSTOMER_ID_AVG_AMOUNT_{window_size}DAY_WINDOW",
            dtype=Float64,
            description=f"Average customer transaction amount in the last {window_size} day(s)",
        ),
    ]
for window_size in WINDOW_SIZES_IN_DAYS:
    transaction_all_fields += [
        Field(
            name=f"TERMINAL_ID_NB_TX_{window_size}DAY_WINDOW",
            dtype=Float64,
            description=f"Number of terminal transactions in a {window_size}-day window, "
            f"shifted back by a {DELAY_PERIOD_IN_DAYS}-day delay",
        ),
        Field(
            name=f"TERMINAL_ID_RISK_{window_size}DAY_WINDOW",
            dtype=Float64,
            description=f"Share of fraudulent terminal transactions in a {window_size}-day window, "
            f"shifted back by a {DELAY_PERIOD_IN_DAYS}-day delay",
        ),
    ]

# ---------------------------------------------------------------------------
# Feature view
# ---------------------------------------------------------------------------
transaction_all_features = FeatureView(
    name="transaction_all_features",
    entities=[transaction],
    ttl=timedelta(days=1),
    schema=transaction_all_fields,
    source=transformed_transactions_source,
    online=True,
    tags={"option": "2", "team": "fraud"},
    description="Amount, customer spending behaviour and terminal risk features, keyed on the transaction ID",
)


# ---------------------------------------------------------------------------
# On-demand feature view: transformation defined with a decorator, run at retrieval time
# ---------------------------------------------------------------------------
@on_demand_feature_view(
    sources=[transaction_request],
    schema=[
        Field(name="TX_DURING_WEEKEND", dtype=Int64),
        Field(name="TX_DURING_NIGHT", dtype=Int64),
    ],
    mode="pandas",
    tags={"option": "2", "team": "fraud"},
    description="Date/time features computed from the transaction timestamp: TX_DURING_WEEKEND (1 on weekends) "
    "and TX_DURING_NIGHT (1 if hour <= 6)",
)
def transaction_datetime_features(inputs: pd.DataFrame) -> pd.DataFrame:
    tx_datetime = pd.to_datetime(inputs["event_timestamp"])
    df = pd.DataFrame()
    # 0 is Monday, 6 is Sunday: 1 if weekend, 0 if weekday
    df["TX_DURING_WEEKEND"] = (tx_datetime.dt.weekday >= 5).astype("int64")
    # 1 if hour less than or equal to 6, and 0 otherwise
    df["TX_DURING_NIGHT"] = (tx_datetime.dt.hour <= 6).astype("int64")
    return df


# ---------------------------------------------------------------------------
# Feature service: the set of features consumed by the model
# ---------------------------------------------------------------------------
fraud_model_v2 = FeatureService(
    name="fraud_model_v2",
    features=[transaction_all_features, transaction_datetime_features],
    tags={"option": "2"},
    description="Model inputs from the single transaction-keyed feature view",
)
