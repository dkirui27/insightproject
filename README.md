# TrainAware: Work Smarter, Not Harder

A machine learning web application that predicts overtraining risk using wearable fitness data, helping athletes reduce their chance of workout-related injury.

Built during the [Insight Data Science](https://insightfellows.com/) fellowship (Session 20B).

## Overview

Overtraining is a common cause of injury among athletes and fitness enthusiasts. TrainAware combines data from wearable fitness trackers (Apple Health, Welltory) with an XGBoost classifier to predict the probability of overtraining in the coming week. When risk is elevated, the app recommends targeted stretching routines for recovery.

**Key finding:** The Acute-to-Chronic Workload Ratio (ACWR) is a stronger predictor of overtraining than heart rate variability (HRV) alone.

## Project Structure

```
trainaware/
├── README.md
├── .gitignore
├── requirements.txt
├── app/
│   └── trainaware.py                   # Streamlit web application
├── data/
│   ├── raw/                             # Original datasets (Welltory, Apple Health)
│   └── processed/                       # Cleaned, engineered features
├── notebooks/
│   ├── 01_eda_and_wrangling.ipynb       # Data cleaning & feature engineering
│   ├── 02_model_development.ipynb       # Feature validation & stationarity testing
│   ├── 03a_random_forest.ipynb          # Random Forest baseline (ROC-AUC: 0.881)
│   ├── 03b_random_forest_alt.ipynb      # Random Forest with alt class weights
│   ├── 04_xgboost.ipynb                 # XGBoost with GridSearchCV (production model)
│   ├── 05_adaboost.ipynb                # AdaBoost comparison
│   ├── 06_streamlit_data_prep.ipynb     # Data preparation for deployment
│   └── 07_stretching_scraping.ipynb     # Web scraping for recovery exercises
└── docs/
    └── demo_slides.pdf                  # Project presentation slides
```

## Data

The dataset combines multiple sources from the [Welltory](https://welltory.com/) study:

| Source | Description | Records |
|--------|-------------|---------|
| `hrv_measurements.csv` | Heart rate variability (RMSSD, PNN50, BPM) | 1,949 |
| `wearables.csv` | Apple Health (steps, calories, pulse, temperature) | 2,101 |
| `participants.csv` | Demographics (age, gender, height, weight) | 141 |
| `heart_rate.csv` | Detailed heart rate measurements | 413,267 |
| `sleep.csv`, `surveys.csv`, `weather.csv` | Supplementary data | Various |

**Engineered features (28 total):** ACWR, ln(RMSSD), BMI, unit conversions, Z-score normalization, one-hot encoded demographics.

**Train/test split:** Temporal split at 2020-04-17 (1,402 train / 722 test observations).

## Model

| Model | ROC-AUC |
|-------|---------|
| Random Forest (200 trees) | 0.881 |
| **XGBoost (tuned)** | **Production model** |
| AdaBoost | Comparison |

The production XGBoost model was tuned via GridSearchCV:
- Learning rate: 0.2, max depth: 4, min child weight: 5, gamma: 0.1
- Objective: `multi:softprob` (calibrated probability output)
- 20 boosting rounds

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the App

From the project root:

```bash
streamlit run app/trainaware.py
```

The app lets you input metrics from your fitness tracker (calories burned, heart rate, ACWR, etc.) and returns your predicted overtraining risk percentage along with targeted stretching recommendations.

### Explore the Analysis

Notebooks are numbered in order. Start with `01_eda_and_wrangling.ipynb` for the full data pipeline.

## Tech Stack

- **ML:** scikit-learn, XGBoost
- **Data:** pandas, NumPy, SciPy, statsmodels
- **Visualization:** matplotlib, seaborn
- **Web app:** Streamlit
- **Scraping:** BeautifulSoup, requests
