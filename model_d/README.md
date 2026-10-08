# Model D — LightGBM Baseline for Traffic Demand

## Objective
This notebook introduces LightGBM as the primary model for traffic demand forecasting. The goal is to build a faster and highly effective baseline using a tree-based ensemble that works especially well with structured tabular data.

## Data Preparation
The workflow resolves the same core issues seen across the project:

- timestamp mismatch between train and test
- median imputation for Temperature
- Unknown replacement for missing categorical values
- cleaning and standardization for model readiness

## Feature Engineering
The model uses several important temporal and spatial features:

- hour
- minute
- time_in_minutes
- is_day_49
- geohash_zone_4
- geohash_zone_5
- location-based demand context

These help the model understand when and where traffic demand is likely to rise.

## Algorithm Used
Model: LGBMRegressor

Why LightGBM was selected:
- very fast on tabular and high-volume data
- handles categorical features efficiently
- captures complex non-linear traffic patterns
- usually gives strong prediction quality with less tuning than deeper ensembles

## Model Configuration
- Train/validation split: 80/20
- n_estimators = 1500
- learning_rate = 0.03
- num_leaves = 63
- max_depth = 8
- subsample = 0.8
- colsample_bytree = 0.8
- objective = regression

## Validation Result
- R² = 0.95963

This is a very strong benchmark and confirms that LightGBM is a very suitable solution for this dataset.

## Outputs
- clean_train.ipynb: full training workflow
- submission_lgb.csv: final LightGBM predictions

## Summary
Model D is the first strong LightGBM-based solution in the project. It gives a very competitive performance while remaining fast, scalable, and easier to work with for tabular traffic demand prediction.
