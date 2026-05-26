# Customer Churn Prediction

> Predict which customers are likely to churn — and map each risk driver to a concrete retention action.

---

## Overview

Customer churn is expensive. This project builds a binary classification model that identifies at-risk customers from behavioural and usage data, achieves **87% accuracy on a hold-out test set**, and goes one step further: it identifies the **top 5 churn drivers** and maps each one to a specific retention intervention — so the output is actionable, not just a score.

---

## Problem Statement

Most churn models stop at a probability score. The business question isn't "will this customer churn?" — it's "what do we do about it, and why?" This project is designed to answer both.

---

## Approach

### 1. Data Preparation
- Loaded and cleaned raw behavioural and usage data
- Handled missing values, outliers, and class imbalance (using SMOTE — Synthetic Minority Oversampling Technique — to balance the training set)
- Engineered features from raw usage patterns (recency, frequency, drop-off signals)

### 2. Exploratory Analysis
- Profiled churned vs. retained customers across all feature dimensions
- Identified early signals: engagement drop-offs, support ticket frequency, billing patterns

### 3. Modelling
- Tested Logistic Regression, Random Forest, and Gradient Boosting classifiers
- Selected final model based on F1-score and interpretability trade-off
- Evaluated on a stratified hold-out test set — **87% accuracy**

### 4. Feature Importance & Retention Mapping

| Churn Driver | Retention Intervention |
|---|---|
| Low recent engagement | Re-engagement email campaign with personalised usage tips |
| High support ticket volume | Proactive outreach from customer success team |
| Long time since last login | Triggered reminder with feature highlight |
| Declining usage frequency | In-app nudge + discount offer at the 30-day inactivity mark |
| Single product/feature usage | Onboarding follow-up to surface underused features |

---

## Results

- **87% accuracy** on hold-out test set
- Top 5 churn drivers identified with feature importance scores
- Each driver mapped to a concrete retention action

---

## Stack

- **Python** — data processing and modelling pipeline
- **Pandas / NumPy** — data wrangling
- **Scikit-learn** — model training, evaluation, feature importance
- **imbalanced-learn** — SMOTE for handling class imbalance
- **Matplotlib / Seaborn** — visualisations

---

## File Structure

```
customer-churn-prediction/
├── generate_data.py    # Generates synthetic customer behavioural data
├── preprocess.py       # Cleans, encodes, and engineers features
├── train.py            # Trains and saves the classification model
├── evaluate.py         # Evaluates model performance and feature importance
└── README.md
```

---

## How to Run

```
# Clone the repo
git clone https://github.com/simranksandhu0/customer-churn-prediction.git
cd customer-churn-prediction

# Generate synthetic data
python generate_data.py

# Preprocess the data
python preprocess.py

# Train the model
python train.py

# Evaluate and view results
python evaluate.py
```

---

## Key Takeaways

- Feature engineering on raw usage data was the biggest driver of model performance — raw features alone gave ~71% accuracy; engineered features pushed it to 87%
- Interpretability was prioritised over marginal accuracy gains: a model the business can act on beats a black box by a few percentage points
- Class imbalance handling (SMOTE + threshold tuning) was critical to avoid a model that just predicted "no churn" for everyone
