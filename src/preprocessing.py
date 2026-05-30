import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Quarter"] = df["Date"].dt.quarter
    return df


def encode_cities(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    le = LabelEncoder()
    df["City_encoded"] = le.fit_transform(df["City"])
    return df


def map_aqi_bucket(df: pd.DataFrame) -> pd.DataFrame:
    bucket_map = {"Good": 0, "Satisfactory": 1, "Moderate": 2,
                  "Poor": 3, "Very Poor": 4, "Severe": 5}
    df = df.copy()
    df["AQI_Bucket_encoded"] = df["AQI_Bucket"].map(bucket_map)
    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["AQI_lag1"] = df["AQI"].shift(1)
    df["PM25_lag1"] = df["PM2.5"].shift(1)
    df["PM10_lag1"] = df["PM10"].shift(1)
    df["PM25_MA3"] = df["PM2.5"].rolling(window=3).mean()
    df["AQI_MA5"] = df["AQI"].rolling(window=5).mean()
    return df


def interpolate_and_scale(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    df = df.copy()
    df[feature_cols] = df[feature_cols].interpolate(method="linear")
    df.dropna(inplace=True)
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df
