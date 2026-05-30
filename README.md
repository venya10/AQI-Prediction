# AQI Prediction — Indian Cities (2015–2020)

Predicting Air Quality Index (AQI) for Indian cities using machine learning (Random Forest) and a fuzzy logic model (ANFIS in MATLAB), trained on daily pollution records from 2015 to 2020.

## Results

| Model | RMSE | R² |
|---|---|---|
| Random Forest | 37.03 | 0.925 |
| ANFIS (MATLAB) | — | — |

## Dataset

- **Source:** `data/raw/city_day.csv` — 29,531 daily records across multiple Indian cities
- **Features:** PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene, AQI
- **Processed:** `data/processed/cleaned_aqi.csv` — interpolated, scaled, with lag and moving-average features

> **Note:** Large CSV files are excluded from git tracking. Download the dataset from [Kaggle — Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) and place it in `data/raw/`.

## Project Structure

```
AQI-Prediction/
├── notebooks/
│   └── AQI_India.ipynb        # EDA, preprocessing, and modeling walkthrough
├── src/
│   ├── preprocessing.py       # Data loading, feature extraction, scaling
│   └── train.py               # Random Forest training and evaluation
├── matlab/
│   └── AQI_ANFIS.m            # ANFIS fuzzy inference model
├── data/
│   ├── raw/                   # Original dataset (not tracked in git)
│   └── processed/             # Cleaned dataset (not tracked in git)
├── docs/
│   └── Research_Paper.pdf     # Reference paper
├── requirements.txt
└── .gitignore
```

## Setup

```bash
git clone https://github.com/jchowdhery3/AQI-Prediction.git
cd AQI-Prediction
pip install -r requirements.txt
```

Then open the notebook:

```bash
jupyter notebook notebooks/AQI_India.ipynb
```

Or run training directly:

```bash
python src/train.py
```

## Features Engineered

- **Lag features:** `AQI_lag1`, `PM25_lag1`, `PM10_lag1`
- **Rolling averages:** `PM25_MA3` (3-day), `AQI_MA5` (5-day)
- **Temporal:** Year, Month, Day, DayOfWeek, Quarter
- **Encoding:** City label encoding, AQI bucket encoding (Good=0 … Severe=5)

## Models

### Random Forest (Python / scikit-learn)
- 200 trees, max depth 15
- 80/20 train-test split
- Features: PM2.5, PM10, NO2, SO2, CO, O3, AQI_lag1, PM25_MA3, AQI_MA5

### ANFIS (MATLAB)
- Sugeno fuzzy inference system
- 2 Gaussian membership functions per input
- 50 training epochs
- Inputs: PM2.5, PM10, NO2, CO, O3, AQI_lag1
