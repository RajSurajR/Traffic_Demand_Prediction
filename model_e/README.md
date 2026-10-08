# Model E — 5-Fold LightGBM Ensemble

## Objective
This notebook improves the baseline LightGBM model by using cross-validation and more expressive time features. The goal is to reduce variance and make the model more stable across different train/validation folds.

## Data Preparation
The notebook applies the same preprocessing logic across the competition data:

- fix timestamp mismatch between train and test
- fill missing Temperature values with median
- replace missing category values with Unknown
- keep the feature space consistent for both train and test sets

## Feature Engineering
The model uses a stronger traffic-aware feature set:

- hour
- minute
- time_in_minutes
- time_sin
- time_cos
- geohash_zone_4
- geohash_zone_5
- historical demand averages by geohash and region
- is_day_49

The cyclical time features are important because demand repeats over a daily cycle, and times like 23:59 and 00:01 should be treated as close together.

## Algorithm Used
Model: LGBMRegressor

Why LightGBM was used:
- very fast for structured tabular data
- supports categorical features well
- performs strongly on spatial-temporal regression tasks
- works efficiently in cross-validation settings

## Cross-Validation Setup
- 5-fold cross-validation
- shuffle enabled
- early stopping for stable training
- averaged predictions across folds

## Validation Results
Per-fold R² values:

- Fold 1: 0.96283
- Fold 2: 0.96250
- Fold 3: 0.96449
- Fold 4: 0.95859
- Fold 5: 0.96305

Average cross-validation R²:
- 0.96229

## Outputs
- clean_train_e.ipynb: notebook with full pipeline
- submission_lgb_kfold.csv: final ensemble submission

## Summary
Model E is one of the strongest solutions in the project. It combines cyclical time features, spatial hierarchy, and cross-validation in a robust LightGBM ensemble that generalizes well across folds.


