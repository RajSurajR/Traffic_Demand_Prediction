# Model B — Improved XGBoost with Better Spatial Signals

## Objective
This version improves the baseline by making preprocessing more consistent across train and test sets and by adding richer location-based features. The idea is to reduce mismatch errors and give the model stronger traffic-location signals.

## Data Preparation
The notebook improves the training workflow by:

- unifying train and test preprocessing in one pipeline
- fixing timestamp inconsistencies
- filling missing Temperature values with the median
- replacing missing categorical values with Unknown
- creating an is_train flag to cleanly separate the datasets later

## Feature Engineering
This model adds a stronger feature set:

- hour
- minute
- time_in_minutes
- is_day_49
- geohash_zone_4
- geohash_zone_5
- geohash_avg_demand
- zone4_avg_demand

These features help the model learn both daily variation and area-specific traffic patterns.

## Algorithm Used
Model: XGBRegressor

Why XGBoost was chosen:
- excellent for tabular regression tasks
- handles categorical variables efficiently
- captures non-linear relationships in traffic demand
- performs well with structured spatial and temporal features

## Model Configuration
- Train/validation split: 80/20
- n_estimators = 1200
- learning_rate = 0.03
- max_depth = 7
- subsample = 0.85
- colsample_bytree = 0.85
- tree_method = hist
- enable_categorical = True
- early_stopping_rounds = 50

## Validation Result
- R² ≈ 0.8864

This improvement shows that adding location-aware historical baselines and time features helps the model better predict demand patterns.

## Outputs
- clean_train.ipynb: training workflow
- b_submission_improved.csv: final tuned submission

## Summary
Model B is a major improvement over the baseline because it combines clean preprocessing, temporal features, and geohash-based demand baselines into a more realistic forecasting pipeline.
