"""
preprocess.py
Loads, validates, and prepares churn data for modelling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


REQUIRED_COLUMNS = {
    "customer_id", "tenure_months", "login_frequency_monthly",
    "support_tickets_90d", "features_used", "avg_order_value",
    "days_since_last_login", "monthly_spend", "contract_type", "churned",
}

FEATURE_COLS = [
    "tenure_months", "login_frequency_monthly", "support_tickets_90d",
    "features_used", "avg_order_value", "days_since_last_login",
    "monthly_spend", "contract_type_One Year", "contract_type_Two Year",
    # Month-to-Month is the dropped reference category
]

TARGET_COL = "churned"


def load_and_validate(path: str = "data/churn_data.csv") -> pd.DataFrame:
    """Load CSV and validate schema and data quality."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Run generate_data.py first."
        )

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.isnull().any().any():
        null_counts = df.isnull().sum()
        raise ValueError(f"Null values found:\n{null_counts[null_counts > 0]}")

    if not df["churned"].isin([0, 1]).all():
        raise ValueError("Target column 'churned' must contain only 0 or 1.")

    if (df[["tenure_months", "login_frequency_monthly", "support_tickets_90d",
            "features_used", "days_since_last_login"]] < 0).any().any():
        raise ValueError("Negative values found in columns that must be non-negative.")

    print(f"Loaded {len(df):,} rows. Churn rate: {df[TARGET_COL].mean():.1%}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode contract type (drop Month-to-Month as reference)."""
    df = df.copy()
    dummies = pd.get_dummies(df["contract_type"], prefix="contract_type", drop_first=False)
    # Explicitly drop Month-to-Month to avoid dummy variable trap
    if "contract_type_Month-to-Month" in dummies.columns:
        dummies = dummies.drop(columns=["contract_type_Month-to-Month"])
    df = pd.concat([df.drop(columns=["contract_type"]), dummies], axis=1)
    return df


def split_and_scale(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """
    Split into train/test sets and apply StandardScaler.

    Returns
    -------
    X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=FEATURE_COLS, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=FEATURE_COLS, index=X_test.index
    )

    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Train churn rate: {y_train.mean():.1%} | Test churn rate: {y_test.mean():.1%}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
