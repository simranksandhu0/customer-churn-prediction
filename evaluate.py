"""
evaluate.py
Evaluates the trained model on the hold-out test set and
maps the top 5 churn drivers to retention interventions.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix,
)
from pathlib import Path


RETENTION_MAP = {
    "days_since_last_login":     "Triggered re-engagement email with personalised usage tips",
    "support_tickets_90d":       "Proactive outreach from customer success team",
    "login_frequency_monthly":   "In-app nudge + discount offer at 30-day inactivity mark",
    "features_used":             "Onboarding follow-up to surface underused features",
    "tenure_months":             "Loyalty reward or early renewal incentive",
    "monthly_spend":             "Targeted upsell or bundle offer based on spend tier",
    "avg_order_value":           "Personalised recommendation engine based on order history",
    "contract_type_One Year":    "Offer upgrade path with added benefits",
    "contract_type_Two Year":    "Priority support lane as retention anchor",
}


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Run predictions and print a full classification report.

    Returns
    -------
    dict with accuracy, f1, precision, recall, confusion_matrix
    """
    y_pred = model.predict(X_test)

    acc   = accuracy_score(y_test, y_pred)
    f1    = f1_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred)
    rec   = recall_score(y_test, y_pred)
    cm    = confusion_matrix(y_test, y_pred)

    print("=" * 50)
    print("HOLD-OUT TEST SET RESULTS")
    print("=" * 50)
    print(f"Accuracy  : {acc:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))
    print("Confusion Matrix:")
    print(cm)

    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "confusion_matrix": cm}


def get_feature_importance(model, feature_names: list, top_n: int = 5) -> pd.DataFrame:
    """
    Extract feature importances (works for tree-based models).
    Maps top N drivers to retention interventions.
    """
    if not hasattr(model, "feature_importances_"):
        print("Model does not expose feature_importances_. Skipping importance extraction.")
        return pd.DataFrame()

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    top_features = importance_df.head(top_n).copy()
    top_features["retention_intervention"] = top_features["feature"].map(RETENTION_MAP)

    print("\nTop Churn Drivers & Retention Interventions:")
    print(top_features[["feature", "importance", "retention_intervention"]].to_string(index=False))

    Path("outputs").mkdir(exist_ok=True)
    top_features.to_csv("outputs/churn_drivers_retention_map.csv", index=False)
    print("\nSaved to outputs/churn_drivers_retention_map.csv")

    return top_features
