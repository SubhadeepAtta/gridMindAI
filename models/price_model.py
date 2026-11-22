"""
GridMind AI — IEX Price Forecasting Module
==========================================
Forecasts day-ahead IEX electricity prices for procurement planning.

Note on price forecasting: Real IEX DAM price forecasting requires access to
actual merit-order data, plant availability, and system-level demand forecasts
which are not publicly available in granular form. This module trains on
synthetic price data following realistic seasonal and diurnal patterns. The
forecast is used for DAM scheduling decisions and its uncertainty is reflected
in the real-time balancing stage.

The model is intentionally lightweight — a gradient boosting regressor with
minimal feature engineering — to stay within the computational budget.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.assumptions import TRAIN_YEARS, HOURS_PER_YEAR, RANDOM_SEED, IEX_RTM_PREMIUM


PRICE_FEATURES = [
    "hour", "day_of_week", "month", "is_weekend",
    "temperature_c", "cloud_cover",
    "price_lag1", "price_lag24", "price_roll24",
]


def _build_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build lag and rolling features for IEX DAM price.
    Use only the Delhi NCR cluster as the reference price series
    (prices are shared across the national IEX merit order).
    """
    ref = df[df["cluster"] == "Delhi NCR"].copy().sort_values("timestamp")
    ref["price_lag1"] = ref["iex_dam_price_inr_kwh"].shift(1)
    ref["price_lag24"] = ref["iex_dam_price_inr_kwh"].shift(24)
    ref["price_roll24"] = ref["iex_dam_price_inr_kwh"].shift(1).rolling(24).mean()
    return ref.dropna(subset=PRICE_FEATURES)


def train_price_model(df: pd.DataFrame, save_dir: str = None):
    """
    Train IEX DAM price forecasting model.
    Returns fitted model and evaluation metrics.
    """
    print("[price_model] Training IEX DAM price forecasting model...")
    df_feat = _build_price_features(df)
    cutoff = df_feat["timestamp"].min() + pd.Timedelta(hours=TRAIN_YEARS * HOURS_PER_YEAR)
    train = df_feat[df_feat["timestamp"] < cutoff]
    test = df_feat[df_feat["timestamp"] >= cutoff]

    X_train = train[PRICE_FEATURES].values
    y_train = train["iex_dam_price_inr_kwh"].values
    X_test = test[PRICE_FEATURES].values
    y_test = test["iex_dam_price_inr_kwh"].values

    model = HistGradientBoostingRegressor(
        max_iter=150,
        max_leaf_nodes=25,
        learning_rate=0.1,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    y_pred = np.clip(model.predict(X_test), 1.0, 20.0)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}")

    results = {
        "MAE": mae, "RMSE": rmse, "R2": r2,
        "y_test": y_test[:500], "y_pred": y_pred[:500],
        "test_timestamps": test["timestamp"].values[:500],
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "price_model.pkl"), "wb") as f:
            pickle.dump(model, f)

    return model, results


def forecast_dam_prices(model, weather_forecast: pd.DataFrame) -> np.ndarray:
    """
    Generate 24-hour ahead DAM price forecast.
    Returns array of shape (24,) with forecast prices (INR/kWh).
    Note: RTM prices are assumed as DAM + IEX_RTM_PREMIUM, not forecast separately.
    """
    if model is None:
        # Fallback: use diurnal baseline
        hours = np.arange(24)
        return 4.5 + 1.5 * np.sin(2 * np.pi * (hours - 7) / 24)
    # Use available features; lag values approximated from mean
    if PRICE_FEATURES[0] in weather_forecast.columns:
        X = weather_forecast[PRICE_FEATURES].ffill().values[:24]
        return np.clip(model.predict(X), 1.0, 20.0)[:24]
    return np.full(24, 5.2)
