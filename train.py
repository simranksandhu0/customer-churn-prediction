"""
train.py
Trains Logistic Regression, Random Forest, and Gradient Boosting classifiers.
Handles class imbalance with SMOTE (imbalanced-learn).
Selects the best model by F1-score on the training set (cross-validated).
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score


MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
}


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, seed: int = 42):
    """
    Apply SMOTE to balance the training set.
    SMOTE is from imbalanced-learn, NOT scikit-learn.

    Returns resampled X and y as DataFrames/Series.
    """
    smote = SMOTE(random_state=seed)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=y_train.name)
    print(f"After SMOTE — Class distribution: {y_res.value_counts().to_dict()}")
    return X_res, y_res


def train_and_select(
    X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5
) -> tuple:
    """
    Train all models, evaluate with cross-validated F1-score,
    and return the best model along with all CV results.

    Returns
    -------
    best_model, best_name, cv_results (dict)
    """
    cv_results = {}

    for name, model in MODELS.items():
        scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1
        )
        cv_results[name] = {"mean_f1": scores.mean(), "std_f1": scores.std()}
        print(f"{name}: F1 = {scores.mean():.4f} ± {scores.std():.4f}")

    best_name = max(cv_results, key=lambda k: cv_results[k]["mean_f1"])
    print(f"\nBest model: {best_name} (F1 = {cv_results[best_name]['mean_f1']:.4f})")

    # Refit the best model on the full training set
    best_model = MODELS[best_name]
    best_model.fit(X_train, y_train)

    return best_model, best_name, cv_results
