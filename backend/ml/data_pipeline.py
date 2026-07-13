"""
Data ingestion & feature engineering for SupplySense.

Loads the DataCo Smart Supply Chain CSV, cleans it, engineers a small set of
derived features, and exposes target-specific feature sets for the three
CatBoost regressors (sales, delivery time, profit). Categorical columns are
kept as raw strings (no one-hot) since CatBoost consumes them natively via
`cat_features`.
"""
import os
import pandas as pd
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "DataCoSupplyChainDataset.csv")

# Columns that are pure identifiers, PII, or otherwise non-predictive noise.
DROP_COLS = [
    "Customer Email", "Customer Password", "Customer Fname", "Customer Lname",
    "Customer Street", "Product Description", "Product Image",
    "Customer Id", "Order Customer Id", "Order Id", "Order Item Id",
    "Order Item Cardprod Id", "Product Card Id", "Product Category Id",
    "Order Zipcode", "Customer Zipcode",
    "Order Profit Per Order",  # exact duplicate of "Benefit per order"
]

CATEGORICAL_COLS = [
    "Type", "Category Name", "Customer City", "Customer Country",
    "Customer Segment", "Customer State", "Department Name", "Market",
    "Order City", "Order Country", "Order Region", "Order State",
    "Order Status", "Product Name", "Shipping Mode",
]


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Impute the few remaining nulls in fields we keep.
    if "Customer Lname" in df.columns:
        df = df.drop(columns=["Customer Lname"])

    df["order date (DateOrders)"] = pd.to_datetime(df["order date (DateOrders)"])
    df["shipping date (DateOrders)"] = pd.to_datetime(df["shipping date (DateOrders)"])

    df["order_year"] = df["order date (DateOrders)"].dt.year
    df["order_month"] = df["order date (DateOrders)"].dt.month
    df["order_day"] = df["order date (DateOrders)"].dt.day
    df["order_dow"] = df["order date (DateOrders)"].dt.dayofweek

    df = df.drop(columns=["order date (DateOrders)", "shipping date (DateOrders)"])

    for c in CATEGORICAL_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("missing")

    df = df.dropna(subset=["Days for shipping (real)", "Days for shipment (scheduled)",
                            "Sales per customer", "Benefit per order"])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # scheduled - real: positive means delivered earlier than scheduled, negative means late.
    df["lead_time_diff"] = df["Days for shipment (scheduled)"] - df["Days for shipping (real)"]
    # Order urgency: scheduled shipment of 1 day or less is treated as an expedited/urgent order.
    df["order_urgency"] = (df["Days for shipment (scheduled)"] <= 1).astype(int)
    # Order profitability ratio already exists as "Order Item Profit Ratio"; kept as-is for
    # models where it is not derived from the prediction target.
    return df


NUMERIC_BASE = [
    "Days for shipment (scheduled)", "Order Item Discount", "Order Item Discount Rate",
    "Order Item Product Price", "Order Item Quantity", "Product Price",
    "Latitude", "Longitude", "order_year", "order_month", "order_day", "order_dow",
    "order_urgency",
]

CAT_BASE = list(CATEGORICAL_COLS)


def get_feature_sets(df: pd.DataFrame):
    """Return {target_name: (X, y, cat_feature_names)} for the three CatBoost targets,
    with target-specific exclusions to avoid label leakage."""
    sets = {}

    # 1) Sales prediction — realized per-order-item sales after discount.
    sales_num = NUMERIC_BASE + ["lead_time_diff"]
    sales_cat = CAT_BASE + ["Delivery Status"]
    X_sales = df[sales_num + sales_cat].copy()
    y_sales = df["Sales per customer"]
    sets["sales"] = (X_sales, y_sales, sales_cat)

    # 2) Delivery time prediction — must NOT see anything derived from real shipping days
    #    (Delivery Status / Late_delivery_risk / lead_time_diff are all downstream of it).
    delivery_num = [c for c in NUMERIC_BASE if c != "order_urgency"] + ["order_urgency", "Order Item Profit Ratio"]
    delivery_cat = CAT_BASE
    X_delivery = df[delivery_num + delivery_cat].copy()
    y_delivery = df["Days for shipping (real)"]
    sets["delivery"] = (X_delivery, y_delivery, delivery_cat)

    # 3) Profit prediction — exclude Order Item Profit Ratio (derived directly from profit/sales).
    profit_num = NUMERIC_BASE + ["lead_time_diff"]
    profit_cat = CAT_BASE + ["Delivery Status"]
    X_profit = df[profit_num + profit_cat].copy()
    y_profit = df["Benefit per order"]
    sets["profit"] = (X_profit, y_profit, profit_cat)

    return sets


def load_and_prepare(path: str = DATA_PATH):
    df = load_raw(path)
    df = clean(df)
    df = engineer_features(df)
    return df


if __name__ == "__main__":
    df = load_and_prepare()
    print("Shape after cleaning:", df.shape)
    print("Nulls remaining:\n", df.isnull().sum()[df.isnull().sum() > 0])
    sets = get_feature_sets(df)
    for name, (X, y, cats) in sets.items():
        print(name, X.shape, "target mean", float(np.mean(y)))
