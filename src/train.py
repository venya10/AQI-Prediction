import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


FEATURE_COLS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3",
                "AQI_lag1", "PM25_MA3", "AQI_MA5"]
TARGET_COL = "AQI"


def train_random_forest(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Random Forest  RMSE: {rmse:.2f}  R²: {r2:.3f}")
    return model, X_test, y_test, y_pred


if __name__ == "__main__":
    df = pd.read_csv("data/processed/cleaned_aqi.csv")
    train_random_forest(df)
