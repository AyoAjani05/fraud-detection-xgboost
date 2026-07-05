# Fraud Detection — End-to-End ML Pipeline

Binary classification pipeline for detecting fraudulent transactions,
built on the public [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection)
(Kaggle competition data from Vesta Corporation).

## Results

| Metric | Value |
|---|---|
| ROC-AUC | **0.972** |
| F1-score (fraud class) | **0.788** |
| Best decision threshold | 0.25–0.30 (tuned, not default 0.5) |

Achieved via a blended cross-validation strategy — a weighted combination
of GroupKFold (grouped by transaction month, to guard against temporal
leakage) and StratifiedKFold (to handle class imbalance) — rather than
a single CV scheme alone.

## Why this approach

Fraud is ~0.5% of transactions, and fraud patterns drift over time as
attackers adapt. Two implications shaped the whole pipeline:

- **Class imbalance** → optimize for Precision-Recall AUC and F1, not
  raw accuracy, which is misleading at this imbalance ratio.
- **Temporal drift** → GroupKFold by month prevents the model from
  "cheating" by learning month-specific patterns that won't generalize
  to future, unseen fraud tactics.

## Repo structure

```
├── 01_EDA.ipynb        Data loading, exploratory analysis, feature
│                        engineering, and full write-up of the modeling
│                        approach (dataset structure, evaluation strategy,
│                        notebook roadmap)
├── 02_modeling.ipynb    Model training, GroupKFold + StratifiedKFold
│                        blend, threshold tuning, Optuna hyperparameter
│                        search
└── README.md
```

## Pipeline overview

1. **EDA** — target distribution, structural missingness (identity
   table only covers ~24% of transactions), feature distributions by
   fraud label
2. **Feature engineering** — missingness indicator flags, device
   metadata parsing, UID reconstruction from card/address/day deltas,
   transaction-amount interaction features
3. **Modeling** — XGBoost classifier, tuned via Optuna (max_depth,
   learning_rate, subsample, colsample_bytree, gamma, reg_alpha/lambda)
4. **Validation** — out-of-fold predictions blended across GroupKFold
   (temporal generalization) and StratifiedKFold (class balance),
   weighted 0.3 / 0.7
5. **Threshold tuning** — precision/recall/F1 swept across thresholds
   0.20–0.55 to find the operating point that best trades off false
   positives (manual review cost) against false negatives (missed
   fraud cost)

## Running it

Requires the IEEE-CIS Fraud Detection dataset (`train_transaction.csv`,
`train_identity.csv`) downloaded from Kaggle into a local `./data/`
directory. Key dependencies: `pandas`, `numpy`, `xgboost`, `scikit-learn`,
`optuna`.

## Notes

This was built and iterated on on Google Colab; notebook paths have
been generalized to local relative paths for portability.
