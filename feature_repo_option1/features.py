"""Feast feature definitions for card-fraud detection - Option 1: natural entities.

Features are attached to the entity they describe:

* ``transaction_features``        keyed on TRANSACTION_ID (the transaction amount)
* ``customer_spending_features``  keyed on CUSTOMER_ID (spending behaviour over 1/7/30 days)
* ``terminal_risk_features``      keyed on TERMINAL_ID (risk scores over 1/7/30 days, delayed by 7 days)
* ``transaction_datetime_features``  an on-demand feature view computing the weekend/night flags
  from the transaction timestamp at retrieval time

The customer and terminal views also have a push source, so fresh values can be written straight to the
online store (``store.push``) when a new transaction arrives.

The windowed customer/terminal features are computed in the notebook (following the Fraud Detection
Handbook, Chapter 3) and written as Parquet files partitioned by day (``date=%Y-%m-%d``).
"""

from datetime import timedelta

import pandas as pd

from feast import Entity, FeatureService, FeatureView, Field, FileSource, PushSource, RequestSource, ValueType
from feast.data_format import ParquetFormat
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float64, Int64, UnixTimestamp

TRANSFORMED_DATA_DIR = "/Users/mesfinmulugeta/projects/personal/card-fraud/simulated-data-transformed"

WINDOW_SIZES_IN_DAYS = [1, 7, 30]
DELAY_PERIOD_IN_DAYS = 7

# ---------------------------------------------------------------------------
# Entities: the keys features are attached to
# ---------------------------------------------------------------------------
transaction = Entity(
    name="transaction",
    join_keys=["TRANSACTION_ID"],
    value_type=ValueType.INT64,
    description="A card payment transaction",
)

customer = Entity(
    name="customer",
    join_keys=["CUSTOMER_ID"],
    value_type=ValueType.INT64,
    description="A customer (card holder)",
)

terminal = Entity(
    name="terminal",
    join_keys=["TERMINAL_ID"],
    value_type=ValueType.INT64,
    description="A payment terminal (point of sale)",
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

# Push sources: fresh feature values sent with store.push (e.g. by a stream processor that recomputes a
# customer's windows on every transaction). The batch_source is where the same features are read offline,
# for training and materialization.
customer_spending_push_source = PushSource(
    name="customer_spending_push_source",
    batch_source=transformed_transactions_source,
    description="Customer spending features recomputed when a new transaction arrives",
)

terminal_risk_push_source = PushSource(
    name="terminal_risk_push_source",
    batch_source=transformed_transactions_source,
    description="Terminal risk features recomputed when a new transaction arrives",
)

# Request source: data that comes with the request (the entity dataframe, or the online request)
transaction_request = RequestSource(
    name="transaction_request",
    schema=[Field(name="event_timestamp", dtype=UnixTimestamp)],
    description="Timestamp of the transaction being scored (the event_timestamp of the entity row / online request)",
)

# ---------------------------------------------------------------------------
# Feature schemas
# ---------------------------------------------------------------------------
customer_fields = []
for window_size in WINDOW_SIZES_IN_DAYS:
    customer_fields += [
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

terminal_fields = []
for window_size in WINDOW_SIZES_IN_DAYS:
    terminal_fields += [
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
# Feature views
# ---------------------------------------------------------------------------
transaction_features = FeatureView(
    name="transaction_features",
    entities=[transaction],
    ttl=timedelta(days=1),
    schema=[Field(name="TX_AMOUNT", dtype=Float64, description="Transaction amount")],
    source=transformed_transactions_source,
    online=True,
    tags={"option": "1", "team": "fraud"},
    description="Features describing the transaction itself",
)

customer_spending_features = FeatureView(
    name="customer_spending_features",
    entities=[customer],
    ttl=timedelta(days=max(WINDOW_SIZES_IN_DAYS)),
    schema=customer_fields,
    source=customer_spending_push_source,
    online=True,
    tags={"option": "1", "team": "fraud"},
    description="Customer spending behaviour (frequency and monetary value) over 1/7/30-day windows",
)

terminal_risk_features = FeatureView(
    name="terminal_risk_features",
    entities=[terminal],
    ttl=timedelta(days=DELAY_PERIOD_IN_DAYS + max(WINDOW_SIZES_IN_DAYS)),
    schema=terminal_fields,
    source=terminal_risk_push_source,
    online=True,
    tags={"option": "1", "team": "fraud"},
    description="Terminal risk scores over 1/7/30-day windows, delayed by 7 days",
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
    tags={"option": "1", "team": "fraud"},
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
fraud_model_v1 = FeatureService(
    name="fraud_model_v1",
    features=[
        transaction_features,
        transaction_datetime_features,
        customer_spending_features,
        terminal_risk_features,
    ],
    tags={"option": "1"},
    description="Model inputs assembled from transaction, customer and terminal features",
)
