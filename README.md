[README.md](https://github.com/user-attachments/files/32415573/README.md)
# Customer Churn Prediction (XGBoost)

XGBoost-based model predicting customer churn probability, built for Kaggle's **Playground Series - Season 6, Episode 3** competition.

## Overview

This project was built for Kaggle's Playground Series - Season 6, Episode 3 competition. The goal is to predict customer churn probability using tabular customer data. The model uses XGBoost with the `hist` tree method and memory-optimized preprocessing to handle large datasets efficiently.

- **Competition:** [Playground Series - S6E3](https://www.kaggle.com/competitions/playground-series-s6e3)
- **Task:** Binary classification (churn / no churn)
- **Metric:** ROC-AUC

## Tech Stack

- Python
- pandas / numpy
- scikit-learn
- XGBoost

## Project Structure

```
customer-churn-xgboost/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── train.py
├── submission.csv        # generated after running train.py
└── README.md
```

## Setup

```bash
pip install pandas numpy scikit-learn xgboost
```

Download the competition data from Kaggle and place `train.csv`, `test.csv`, and `sample_submission.csv` inside the `data/` folder.

## Usage

```bash
python train.py
```

The script will:
1. Load and memory-optimize the dataset
2. Encode categorical features
3. Train an XGBoost model with early stopping
4. Print validation ROC-AUC
5. Generate `submission.csv` ready for Kaggle submission

## Results

| Model | Validation AUC |
|-------|-----------------|
| XGBoost (baseline) | TBD |

## Notes

- Large dataset handling: reduced numeric dtypes and `tree_method="hist"` are used to keep memory usage and training time manageable.
- GPU training can be enabled by uncommenting the `device = "cuda"` line in `train.py`.
