# Analysis Notebook — Data Profiling and Preprocessing Review

## Overview
This folder focuses on understanding the traffic demand dataset before model training. The analysis notebooks inspect the raw train and test data, confirm schema consistency, detect missing values, and identify the feature patterns that matter most for forecasting.

## Dataset Summary
The dataset contains:

- Training rows: 77,299
- Test rows: 41,778
- Training columns: 11
- Test columns: 10

Main features include:
- geohash
- day
- timestamp
- demand
- RoadType
- NumberofLanes
- LargeVehicles
- Landmarks
- Temperature
- Weather

## Key Findings
The analysis notebook confirms several important patterns:

- No missing values in geohash, day, timestamp, and demand in the train set.
- Temperature has 2,495 missing values in train and 1,349 in test.
- RoadType has 600 missing values in train and 324 in test.
- Weather has 797 missing values in train and 431 in test.
- Demand ranges from roughly 6.25e-07 to 1.0, with a mean of about 0.09394.
- The training data contains 1,249 unique geohash values; the test data contains 1,190 unique geohash values.
- Most training rows belong to day 48, while a smaller subset belongs to day 49.

## Why This Matters
The analysis stage highlights the most important modeling decisions:

- timestamp formats differ between train and test
- missing values must be handled carefully
- spatial geohash patterns will matter for prediction
- day-wise traffic behavior may shift across time

## Preprocessing Signals Identified
The analysis directly supports the following downstream choices:

- unify timestamp formatting across train and test
- impute Temperature using median
- fill categorical missing values with Unknown
- generate time features such as hour, minute, and time_in_minutes
- create geohash-based spatial features and regional demand baselines

## Files in This Folder
- dataset_info.ipynb: dataset profiling and statistics
- preprocess.ipynb: exploration and preprocessing checks
- README.md: summary of the analysis workflow

## Outcome
This folder provides the data understanding needed before building the actual traffic-demand models. It makes the preprocessing strategy more reliable and helps reduce model drift caused by inconsistent raw data.
