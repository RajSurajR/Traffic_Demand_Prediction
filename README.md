# Flipkart Gridlock Hackathon 2.0 — Traffic Demand Prediction

![Hackathon](https://img.shields.io/badge/Hackathon-Flipkart%20Gridlock%202.0-FF6B35?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python)
![Task](https://img.shields.io/badge/Task-Tabular%20Regression-28A745?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-FFB000?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-00B894?style=for-the-badge)

## Overview

This repository contains my solution for **Flipkart Gridlock Hackathon 2.0**, a machine learning challenge focused on predicting traffic demand from time, location, road, and environmental information.

I approached the problem as a practical tabular ML task: first understand the data, then build a reliable preprocessing pipeline, create useful temporal and spatial features, and iterate through different modeling strategies rather than relying on a single model.

The final experiments reached **around 0.96 R² on local validation**, with the strongest results coming from LightGBM combined with cyclical time features, geohash-based demand signals, and cross-validation.

> **Note:** The dataset provided by the hackathon organizers is not included in this repository. Generated submission/output files are also excluded. Please obtain the dataset from the official competition source and follow its terms of use.

**Competition:** [Flipkart Gridlock Hackathon 2.0](https://gridlock2point0.hackerearth.com/)

---

## Why I Built It This Way

Traffic demand has two things that stood out early in the analysis:

- **Time matters** — demand changes throughout the day.
- **Location matters** — different areas have very different traffic patterns.

So instead of treating the data as a plain regression table, I tried to give the models a better representation of these patterns.

The main ideas were:

1. Clean and standardize the raw data.
2. Extract useful information from timestamps.
3. Represent location at multiple geohash levels.
4. Build historical demand signals for locations and zones.
5. Compare XGBoost and LightGBM.
6. Use cross-validation to check whether improvements were actually stable.
7. Keep the final pipeline reproducible and competition-ready.

---

## Results

| Model | Approach | Validation R² |
|---|---|---:|
| Model A | XGBoost baseline | ~0.8565 |
| Model B | XGBoost + spatial demand features | ~0.8864 |
| Model C | Spatially enhanced XGBoost | ~0.9610 |
| Model D | LightGBM baseline | ~0.9596 |
| Model E | 5-Fold LightGBM + cyclical time features | **~0.9623** |
| Model F | 10-Fold LightGBM ensemble | ~0.9590 |

The best local validation result came from **Model E**, with an average 5-fold R² of approximately **0.9623**.

### Model E fold scores

- Fold 1: 0.96283
- Fold 2: 0.96250
- Fold 3: 0.96449
- Fold 4: 0.95859
- Fold 5: 0.96305

I kept Model F as a separate experiment because more folds do not automatically mean a better score. Its average score was slightly lower, but the experiment was useful for checking stability across different splits.

That comparison was an important part of the project: **the goal was not just to chase one validation score, but to understand which changes actually helped.**

---

## Dataset

The competition dataset contains traffic-demand records associated with geohashes and timestamps.

### Main features

- `geohash` — location identifier
- `day` — day/time bucket
- `timestamp` — measurement time
- `demand` — target variable
- `RoadType` — road category
- `NumberofLanes` — number of lanes
- `LargeVehicles` — traffic profile information
- `Landmarks` — surrounding context
- `Temperature` — environmental information
- `Weather` — weather condition

### Dataset size

- Training rows: **77,299**
- Test rows: **41,778**
- Training columns: **11**
- Test columns: **10**
- Unique geohashes: **1,249 in train / 1,190 in test**

The dataset had missing values in features such as `Temperature`, `RoadType`, and `Weather`, which were handled during preprocessing.

---

## Data Analysis

Before training models, I spent time understanding the structure and quality of the data.

The analysis focused on:

- train/test consistency
- missing-value patterns
- feature distributions
- geohash coverage
- demand distribution
- temporal patterns
- possible sources of leakage
- differences between training and test data

One issue that needed attention was the **different timestamp representation between train and test data**. I normalized the timestamps before generating time-based features.

Other preprocessing decisions included:

- median imputation for numerical temperature values
- `Unknown` handling for missing categorical values
- categorical dtype conversion where appropriate
- timestamp normalization
- consistent feature creation across train and test

More details can be found in [`analysis/`](analysis/).

---

## Feature Engineering

Feature engineering was the most important part of the modeling process.

### 1. Time Features

I extracted:

- `hour`
- `minute`
- `time_in_minutes`
- `time_sin`
- `time_cos`

The cyclical features were particularly useful because traffic does not suddenly become unrelated when the clock moves from 23:59 to 00:00.

### 2. Spatial Features

Geohash information was used at different levels:

- `geohash`
- `geohash_zone_4`
- `geohash_zone_5`

This allowed the models to learn both local and broader regional patterns.

### 3. Historical Demand Features

I also experimented with historical demand statistics such as:

- `geohash_avg_demand`
- `zone4_avg_demand`

These features provide a baseline expectation for traffic demand in a location or region.

The aggregation process was designed with the training workflow in mind to avoid leaking target information into validation data.

### 4. Context Features

The original road and environmental information was retained:

- `Temperature`
- `Weather`
- `RoadType`
- `NumberofLanes`
- `LargeVehicles`
- `Landmarks`

---

## Model Iterations

Rather than jumping directly to a final model, I kept each major experiment separate so I could see what actually improved the result.

### Model A — XGBoost Baseline

**Goal:** Establish a simple and reliable benchmark.

The first model handled timestamp inconsistencies, missing values, basic time features, and trained an XGBoost regressor.

**Validation R²:** ~0.8565

This gave me a baseline to measure later feature engineering against.

---

### Model B — XGBoost + Spatial Features

The next iteration added geohash-level and zone-level demand signals along with more consistent preprocessing.

**Validation R²:** ~0.8864

The improvement confirmed that location-specific demand was an important signal.

---

### Model C — Spatially Enhanced XGBoost

I pushed the spatial representation further by combining geohash information with historical demand features at different geographic levels.

**Validation R²:** ~0.9610

This was a significant jump from the earlier XGBoost models and showed how much feature representation mattered for this problem.

---

### Model D — LightGBM

I then tested LightGBM as an alternative gradient-boosting model.

The pipeline kept the important preprocessing and temporal/spatial features while changing the underlying model.

**Validation R²:** ~0.9596

LightGBM performed very competitively while offering fast training for repeated experiments.

---

### Model E — Fold LightGBM Ensemble

This became the strongest validation result.

Changes included:

- 5-fold cross-validation
- cyclical time features
- geohash and zone demand features
- training a model per fold
- averaging predictions across folds

**Average CV R²:** **~0.9623**

This was the best-performing experiment in the repository.

---

### Model F — 10-Fold LightGBM Ensemble

The final experiment increased the number of folds from 5 to 10.

**Average CV R²:** ~0.9590

Although it did not beat Model E, it was useful as a robustness experiment. It also reinforced a practical lesson from the project: **more training complexity does not necessarily translate into better generalization.**

---

## Project Structure

```text
traffic_demand_prediction/
│
├── analysis/
│   ├── README.md
│   ├── approach.txt
│   ├── dataset_info.ipynb
│   ├── preprocess.ipynb
│   └── analysis_outputs/
│
├── model_a/
│   ├── README.md
│   ├── approach.txt
│   └── clean_train.ipynb
│
├── model_b/
│   ├── README.md
│   ├── approach.txt
│   └── clean_train.ipynb
│
├── model_c/
│   ├── README.md
│   ├── approach.txt
│   └── clean_train.ipynb
│
├── model_d/
│   ├── README.md
│   ├── approach.txt
│   └── clean_train.ipynb
│
├── model_e/
│   ├── README.md
│   ├── approach.txt
│   └── clean_train_e.ipynb
│
├── model_f/
│   ├── README.md
│   ├── approach.txt
│   └── clean_train_f.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```

The repository is organized around the progression of the experiments rather than hiding everything inside one notebook.

---

## What I Learned

A few things stood out while working through the experiments:

- Good feature engineering can matter more than changing the model.
- Traffic demand needs both temporal and spatial context.
- A strong validation setup is important when comparing model iterations.
- Cross-validation is useful, but adding more folds is not automatically an improvement.
- Small inconsistencies in train/test preprocessing can become major problems later.
- Keeping experiments separate makes it much easier to understand why a model improved or got worse.

For me, the biggest takeaway was that **the model is only one part of an ML solution**. Understanding the data and representing the problem correctly had a much larger impact than simply switching algorithms.

---

## Tech Stack

- **Python**
- **Pandas / NumPy**
- **Scikit-learn**
- **XGBoost**
- **LightGBM**
- **Matplotlib / Seaborn**
- **Jupyter Notebook**

---

## Team & My Role

This was completed as a **4-member team** during the hackathon.

Because we were working remotely, each member developed and tested their own approach. We regularly reviewed each other's work, discussed results, and shared ideas for improvements.

**My work in this repository:** I built the complete ML pipeline shown here — from data cleaning and preprocessing to feature engineering, model training, validation, and the different XGBoost and LightGBM experiments. I developed all six model iterations and used the results from each experiment to guide the next one.

I also reviewed other team members' approaches and shared feedback and improvements during the development process.

---

## Final Takeaway

This project was a useful exercise in taking a messy, real-world-style tabular problem and turning it into a structured ML workflow.

The final approach combined:

**data analysis → preprocessing → temporal features → spatial features → gradient boosting → cross-validation → model comparison**

The strongest result came from a **LightGBM ensemble with cyclical time features and geohash-based demand signals**, reaching approximately **0.9623 average validation R²**.

More than the final score, the repository shows the progression of the work — what I tried, what improved the model, what did not, and why I chose the final approach.

---

### Disclaimer

This repository is a personal portfolio representation of my work from the hackathon.

The competition dataset and generated submission/output files are intentionally not included. Please refer to the official competition source for dataset access.
