# Model C — Spatially Strong XGBoost Model

## Objective
This notebook focuses on making the XGBoost model more aware of location patterns. Instead of looking only at raw features, it introduces demand baselines by geohash and region to learn how traffic changes across nearby areas.

## Data Processing
The notebook keeps the same core preprocessing logic used in the project:

- standardize timestamp values
- fill missing Temperature values with median
- replace missing RoadType and Weather values with Unknown
- convert location and category columns to model-friendly types
- create a day-specific signal for the test-period behavior

## Feature Engineering
The model uses a richer set of engineered features:

- hour
- minute
- time_in_minutes
- is_day_49
- geohash_zone_4
- geohash_zone_5
- geohash_avg_demand
- zone4_avg_demand

This version is especially important because it captures both local and regional traffic structure.

## Algorithm Used
Model: XGBRegressor

Why XGBoost was used:
- best for tabular structured data
- handles categorical variables effectively
- supports native tree-based learning with high predictive power
- works well when spatial and temporal features are engineered properly

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

## Validation Performance
- R² ≈ 0.9610

This is one of the strongest single-model results in the project and shows the value of historical location encoding.

## Outputs
- clean_train.ipynb: notebook with full pipeline
- c_submission.csv: final prediction file

## Summary
Model C is a much stronger XGBoost solution because it mixes time, geography, and location demand history into a single predictive model. It shows that traffic forecasting improves significantly when spatial baselines are included.
