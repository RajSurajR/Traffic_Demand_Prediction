# Model A — Baseline XGBoost Traffic Demand Predictor

## Objective
This notebook builds the first baseline model for traffic demand forecasting. The goal is to clean the raw dataset, create useful time and context features, and train a strong gradient-boosted regressor that can predict demand for unseen locations.

## Dataset Snapshot
- Training rows: 77,299
- Test rows: 41,778
- Target: demand
- Main features: geohash, day, timestamp, RoadType, NumberofLanes, LargeVehicles, Landmarks, Temperature, Weather

## Data Cleaning and Preprocessing
The notebook handles the main data-quality issues:

- Timestamp mismatch fixed between train and test values
- Missing Temperature values filled with the median
- Missing RoadType and Weather values replaced with Unknown
- Categorical fields converted to pandas category dtype
- Time components extracted from timestamp for better modeling

## Feature Engineering
The model uses a compact feature set:

- hour
- minute
- geohash
- RoadType
- Temperature
- LargeVehicles
- Landmarks
- Weather

These features help the model learn daily traffic patterns and location-specific demand changes without excessive complexity.

## Algorithm Used
Model: XGBRegressor

Why XGBoost was used:
- handles tabular data very well
- supports categorical variables natively
- captures non-linear demand patterns effectively
- performs strongly on structured traffic data

## Model Configuration
- Train/validation split: 80/20
- n_estimators = 1000
- learning_rate = 0.05
- tree_method = hist
- enable_categorical = True
- early_stopping_rounds = 50

## Validation Result
The notebook reports a validation R² score of approximately:

- 0.8565

This is a strong baseline and confirms that the cleaned feature set has meaningful predictive power.

## Outputs
- clean_train.ipynb: training notebook
- submission_a.csv: test predictions
- a_xgb_model.pkl: saved trained model

## Summary
Model A is the baseline reference version. It is simple, efficient, and reliable, giving a solid benchmark before more advanced spatial and temporal feature engineering is used in later models.
