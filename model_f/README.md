# Model F — Expanded LightGBM Cross-Validation Workflow

## Objective
This notebook increases the robustness of the LightGBM model by using a larger cross-validation setup and averaging predictions across more folds. The goal is to reduce fold-level instability and produce a more reliable final forecast.

## Data Preparation
The preprocessing flow remains consistent with the project:

- normalize timestamp values across train and test
- impute Temperature with median
- fill missing categorical values with Unknown
- maintain a standardized feature pipeline for all rows

## Feature Engineering
The model uses a comprehensive set of features:

- hour
- minute
- time_in_minutes
- time_sin
- time_cos
- geohash_zone_4
- geohash_zone_5
- is_day_49
- historical demand averages by location and region

This gives the model both temporal continuity and spatial awareness, which are essential in traffic forecasting.

## Algorithm Used
Model: LGBMRegressor

Why LightGBM was selected:
- very efficient on tabular data
- handles categorical and mixed features well
- captures complex demand patterns without heavy preprocessing
- performs strongly in repeated validation setups

## Cross-Validation Setup
- 10-fold cross-validation
- early stopping used during training
- predictions averaged across folds for the final submission

## Validation Performance
Per-fold R² scores:

- Fold 1: 0.95841
- Fold 2: 0.96128
- Fold 3: 0.95962
- Fold 4: 0.95815
- Fold 5: 0.95841
- Fold 6: 0.96150
- Fold 7: 0.95650
- Fold 8: 0.95729
- Fold 9: 0.95874
- Fold 10: 0.95999

Average cross-validation R²:
- 0.95899

## Outputs
- clean_train_f.ipynb: training and validation workflow
- submission_lgb_kfold_f.csv: final ensemble file

## Summary
Model F is a more robust and stable LightGBM ensemble. It follows the same sound preprocessing and feature-engineering strategy as earlier versions but adds stronger validation coverage to reduce model volatility.


