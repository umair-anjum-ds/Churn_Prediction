# Telco Customer Churn Analysis & Retention Strategy

## 1. Understanding of the Problem
In subscription-based telecommunications, acquiring a new customer costs **5 to 25 times more** than retaining an existing one. Uncontrolled turnover (**churn**) directly erodes Monthly Recurring Revenue (MRR) and prevents the recovery of upfront Customer Acquisition Costs (CAC).

The objective of this project is to analyze the Telco customer dataset to identify key behavioral and financial drivers of churn, build a simple and interpretable baseline predictive model to flag high-risk accounts, and deliver actionable recommendations to improve customer retention.

---

## 2. Important Data-Quality Issues Discovered
During initial data profiling, the following core quality issues were identified and resolved programmatically:

* **Incorrect Datatype (`TotalCharges`)**: Loaded as a string (`object`) instead of numerical float due to trailing whitespace formatting.
* **Missing Values at `tenure == 0`**: Exactly 11 records contained blank space strings (`" "`) in `TotalCharges`. All 11 corresponded to new accounts with zero tenure who had not yet been billed.
* **Non-Predictive Identifier (`customerID`)**: Arbitrary alphanumeric string with no statistical value.
* **Redundant Categorical Labels**: Features like `OnlineSecurity`, `TechSupport`, and `DeviceProtection` contained redundant `"No internet service"` categories that mirrored the `InternetService` column state.

---

## 3. Major Preprocessing & Feature-Engineering Decisions

### Preprocessing Steps
* **Imputation & Parsing**: Converted `TotalCharges` to numeric (`pd.to_numeric`) and imputed the 11 zero-tenure missing values to `0.0`.
* **Identifier Removal**: Dropped `customerID` prior to feature encoding.
* **Target Encoding**: Mapped target variable `Churn` to binary (`Yes` -> 1, `No` -> 0).
* **Categorical Encoding**: Applied One-Hot Encoding to multi-class non-ordinal variables (`Contract`, `PaymentMethod`, `InternetService`).
* **Feature Scaling**: Normalized continuous variables using standard scaling methodologies.
* **Data Splitting**: Executed an 80/20 train-test split using stratified sampling (`stratify=y`) to maintain baseline class balance across both subsets.

### Engineered Features (5 Created)
To capture customer lifecycle phases, financial velocity, and ecosystem lock-in, five domain features were engineered (including the 3 required core features):

| # | Feature Name | Formula / Logic | Business Context |
|---|---|---|---|
| 1 | `AvgMonthlySpend` | `df['TotalCharges'] / (df['tenure'] + 1)` | Measures overall historical monthly spend relative to active lifetime. |
| 2 | `NewCustomer` | `(df['tenure'] < 12).astype(int)` | Flags high-vulnerability accounts during their first year of service. |
| 3 | `LongTermCustomer` | `(df['tenure'] >= 24).astype(int)` | Identifies mature, established accounts with higher baseline loyalty. |
| 4 | `Total_Services_Subscribed` | Sum of active add-on features | Quantifies product ecosystem depth and service stickiness. |
| 5 | `Automatic_Payment` | Binary flag for auto-bank/credit card | Distinguishes seamless auto-payers from high-friction manual payment users. |

---

## 4. Model Results & Evaluation Metrics
A baseline model was trained on the preprocessed training set and evaluated on the hold-out test set (1,409 instances). In churn modeling, capturing actual churning customers (**Recall**) is prioritized to ensure high-risk accounts receive intervention.

### Evaluation Metrics (Test Set)
* **Accuracy**: 0.7473 (74.73%)
* **Precision (Churn = 1)**: 0.52
* **Recall (Churn = 1)**: 0.7888 (78.88%)
* **F1-Score (Churn = 1)**: 0.6237 (62.37%)

### Classification Report
```text
              precision    recall  f1-score   support

           0       0.91      0.73      0.81      1035
           1       0.52      0.79      0.62       374

    accuracy                           0.75      1409
   macro avg       0.71      0.76      0.72      1409
weighted avg       0.80      0.75      0.76      1409


