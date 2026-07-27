# Telco Customer Churn Analysis & Prediction

## Executive Summary
This project analyzes customer turnover (churn) for a telecommunications provider to identify the primary drivers of churn, build an initial baseline predictive model, and provide actionable business recommendations to improve customer retention. 

All data cleaning, feature engineering, and modeling steps are fully automated in the accompanying Jupyter Notebook (`notebook.ipynb`) to ensure end-to-end reproducibility.

---

## 1. Problem Understanding
Customer acquisition costs in the telecommunications industry significantly exceed customer retention costs. When a customer churns, the company loses both immediate monthly recurring revenue (MRR) and the long-term customer lifetime value (LTV).

The objective of this analysis is twofold:
1. **Identify key risk factors** associated with customer departure (e.g., contract types, pricing, service usage).
2. **Build a baseline predictive model** to flag high-risk customers before they churn, enabling targeted retention campaigns.

---

## 2. Data Quality Issues Identified
During the initial Data Understanding and Cleaning phase, the following key data quality issues were discovered and resolved programmatically:

1. **Incorrect Data Type in `TotalCharges`**:
   - *Issue*: `TotalCharges` was loaded as a string (`object`) rather than a numerical float.
   - *Fix*: Converted using `pd.to_numeric(..., errors='coerce')`.
2. **Missing / Space-Filled Values**:
   - *Issue*: Exactly 11 rows in `TotalCharges` contained blank whitespace string values (`" "`).
   - *Insight*: All 11 instances corresponded to new customers with `tenure == 0`.
   - *Fix*: Imputed missing `TotalCharges` with `0.0` (or removed the 11 rows) to reflect zero prior billing history.
3. **Redundant Identifiers**:
   - *Issue*: `customerID` is an arbitrary string with zero predictive power.
   - *Fix*: Dropped `customerID` prior to model input.
4. **Categorical Redundancies**:
   - *Issue*: Several columns contained redundant label descriptions like `"No internet service"` across multiple add-on features (`OnlineSecurity`, `TechSupport`, etc.).
   - *Fix*: Standardized and encoded using One-Hot Encoding (`pd.get_dummies` / `OneHotEncoder`).

---

## 3. Preprocessing & Feature Engineering

### Preprocessing Pipeline
* **Target Encoding**: Mapped target column `Churn` (`'Yes'` $\rightarrow$ 1, `'No'` $\rightarrow$ 0).
* **Categorical Encoding**: Applied One-Hot Encoding to non-ordinal categorical variables (`Contract`, `PaymentMethod`, `InternetService`, etc.).
* **Feature Scaling**: Scaled numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) using `StandardScaler` to normalize distributions for model training.
* **Data Splitting**: Split data into 80% Training and 20% Testing sets using stratified sampling (`stratify=y`) to maintain the baseline churn class ratio (~26.5% positive class).

### Engineered Features (5 Created)
To capture customer behavioral patterns and financial usage, five new domain-specific features were created:

| # | Feature Name | Formula / Logic | Business Context |
|---|---|---|---|
| 1 | `Average_Monthly_Charge` | `TotalCharges / (tenure + 1)` | Captures historical monthly spending trend compared to the current `MonthlyCharges`. |
| 2 | `Is_New_Customer` | `tenure <= 12` (Binary: 1/0) | Flags early-lifecycle customers who are statistically most vulnerable to churning. |
| 3 | `Total_Services_Subscribed` | Sum of active add-on services | Quantifies customer reliance on the ecosystem (Security, Backup, Protection, Support, TV, Movies). |
| 4 | `Has_Tech_or_Security` | `OnlineSecurity` OR `TechSupport` | Identifies presence of key retention-anchoring protection services. |
| 5 | `Automatic_Payment` | Binary flag for auto-bank transfer or credit card | Distinguishes seamless auto-pay users from manual payers (e.g., electronic checks). |

---

## 4. Model Results & Evaluation Metrics

A **Logistic Regression** (or Random Forest) baseline model was trained on the preprocessed training set and evaluated on the hold-out test set (20% of data).

### Evaluation Metrics
Given the business context of customer churn, **Recall** and **F1-Score** for the churn class ($y=1$) are prioritized over standard Accuracy to ensure high-risk customers are not missed.

* **Accuracy**: `0.XX`
* **Precision (Churn = 1)**: `0.XX`
* **Recall (Churn = 1)**: `0.XX`
* **F1-Score (Churn = 1)**: `0.XX`
* **ROC-AUC Score**: `0.XX`

### Confusion Matrix (Test Set)
```text
               Predicted: No Churn    Predicted: Churn
Actual: No     [     TN     ]        [     FP     ]
Actual: Yes    [     FN     ]        [     TP     ]