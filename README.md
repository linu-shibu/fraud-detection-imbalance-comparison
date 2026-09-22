# Credit Card Fraud Detection: Multi-Model Comparison Under Class Imbalance

A comparative study of machine learning classifiers for credit card fraud
detection, focused on two questions that matter more than raw accuracy in
this domain: **which algorithm generalizes best to realistic, heavily
imbalanced data**, and **which imbalance-correction strategy is worth its
cost**.

## Problem

Fraud detection is a textbook extreme-imbalance classification problem —
in this dataset, legitimate transactions outnumber fraudulent ones **172:1**.
A naive classifier can score >99% accuracy by predicting "not fraud" for
everything, while catching zero actual fraud. This project treats that as
the central challenge to design around, not an afterthought.

## Dataset

~1.3M simulated transactions with cardholder demographics, merchant,
category, amount, timestamp, and geolocation fields, labeled `is_fraud`
(0/1).

## Approach

1. **EDA** — examined class balance, missing values, and time-based fraud
   patterns (hour of day, day of week).
2. **Leakage-safe split** — data was split into train/test **before** any
   imbalance correction, so the test set retained the real 172:1 ratio.
   Evaluating on an artificially balanced test set would have produced
   misleadingly optimistic metrics; this project deliberately avoids that.
3. **Feature engineering** — target encoding for high-cardinality
   categorical fields (merchant, category, job, state, gender), fit on
   training labels only; derived `age` from date of birth and transaction
   timestamp; derived time-based features (`hour`, `day_of_week`,
   `is_late_night`).
4. **Model comparison** — trained and evaluated 5 classifiers (Logistic
   Regression, Random Forest, XGBoost, SVM, MLP) on identical
   train/test splits using precision, recall, F1, ROC-AUC, and PR-AUC —
   PR-AUC and recall were prioritized over accuracy and ROC-AUC, since
   both are far more robust to extreme class imbalance.
5. **Imbalance-method comparison** — took the best model (XGBoost) and
   compared three ways of handling the imbalance: random undersampling,
   class weighting (`scale_pos_weight`), and SMOTE oversampling.

## Results

### Model comparison (undersampled training data)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **XGBoost** | 0.178 | 0.979 | 0.302 | 0.998 | **0.840** |
| Random Forest | 0.128 | 0.950 | 0.225 | 0.993 | 0.804 |
| MLP | 0.059 | 0.903 | 0.111 | 0.970 | 0.516 |
| SVM | 0.047 | 0.846 | 0.089 | 0.936 | 0.338 |
| Logistic Regression | 0.035 | 0.811 | 0.068 | 0.914 | 0.224 |

XGBoost outperformed every other model on every metric, consistent with
gradient-boosted trees' typical strength on structured/tabular data.

### Imbalance-method comparison (XGBoost, full training data)

| Method | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Class Weighted | 0.679 | **0.911** | 0.778 | **0.928** |
| SMOTE | 0.856 | 0.856 | **0.856** | 0.912 |
| Undersampling | 0.178 | 0.979 | 0.302 | 0.840 |

Class weighting and SMOTE both substantially outperformed undersampling —
because both retain the full training set, while undersampling discards
over 98% of legitimate transactions to force balance. Between the two,
class weighting catches more fraud (higher recall), while SMOTE produces
fewer false alarms per fraud caught (higher precision/F1) — the right
choice depends on the relative business cost of a missed fraud versus a
false positive.

### Feature importance

`is_late_night` and transaction `amt` were by far the most predictive
features for XGBoost, ahead of any target-encoded categorical field.

## Key engineering decision worth calling out

Mid-project, comparing XGBoost trained on undersampled data against a test
set encoded using the *full* dataset's label distribution produced a
sharp, suspicious drop in recall (0.98 → 0.16). Root cause: target
encoding values are learned from whichever training labels produced them —
encoding fit on a 50/50 undersampled set assigns very different values to
the same real-world category than encoding fit on the true 172:1
distribution. Evaluating a model against features encoded on a different
label distribution than it was trained on silently breaks the model's
learned thresholds. Fixed by ensuring each model is always evaluated
against a test set encoded consistently with its own training labels.

## Tech stack

Python, pandas, scikit-learn, XGBoost, imbalanced-learn (SMOTE),
matplotlib.

## Repo structure

```
├── Credit_Card_Fraud_Detection.ipynb
└── README.md
```

## How to run

1. Download the dataset: https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset and place it source folder.
2. Open the notebook in Jupyter or Google Colab.
3. Run all cells top to bottom.
