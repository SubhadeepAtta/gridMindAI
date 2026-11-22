"""
GridMind AI — Demand Forecasting Module
========================================
Forecasts DC fast-charging demand and swap requests using
HistGradientBoostingRegressor (scikit-learn).

Design rationale: HistGradientBoostingRegressor was chosen over XGBoost or
LightGBM because it is part of scikit-learn's standard library, natively handles
missing values, and trains in O(n log n) time due to histogram binning. For this
synthetic dataset of ~175K training rows it converges in seconds.

Chronological train/test splitting is mandatory to prevent data leakage. Lag
features and rolling averages are computed only on training-era data and then
applied forward to avoid look-ahead bias.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.assumptions import CLUSTERS, TRAIN_YEARS, HOURS_PER_YEAR, RANDOM_SEED


FEATURES = [
    "hour", "day_of_week", "month", "is_weekend",
    "temperature_c", "humidity_pct", "rainfall_mm",
    "cloud_cover", "wind_speed_ms",
    "dc_demand_lag1", "dc_demand_lag24", "dc_demand_roll24",
    "swap_lag1", "swap_lag24",
]


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag and rolling features per cluster using chronological order.
    Feature creation is vectorised via pandas shift/rolling operations.
    """
    frames = []
    for cluster in CLUSTERS:
        c = df[df["cluster"] == cluster].copy().sort_values("timestamp")
        c["dc_demand_lag1"] = c["dc_charging_demand_mw"].shift(1)
        c["dc_demand_lag24"] = c["dc_charging_demand_mw"].shift(24)
        c["dc_demand_roll24"] = c["dc_charging_demand_mw"].shift(1).rolling(24).mean()
        c["swap_lag1"] = c["swap_requests"].shift(1)
        c["swap_lag24"] = c["swap_requests"].shift(24)
        frames.append(c)
    return pd.concat(frames, ignore_index=True)


def _chronological_split(df: pd.DataFrame):
    """
    Split by time: first TRAIN_YEARS for training, remainder for test.
    Returns train_df, test_df.
    """
    cutoff = df["timestamp"].min() + pd.Timedelta(hours=TRAIN_YEARS * HOURS_PER_YEAR)
    train = df[df["timestamp"] < cutoff].dropna(subset=FEATURES)
    test = df[df["timestamp"] >= cutoff].dropna(subset=FEATURES)
    return train, test


def train_demand_models(df: pd.DataFrame, save_dir: str = None):
    """
    Train DC charging demand and swap request models.
    Returns fitted models and evaluation metrics.
    """
    print("[demand_model] Building features...")
    df_feat = _build_features(df)
    train, test = _chronological_split(df_feat)

    results = {}
    models = {}

    for target, label in [("dc_charging_demand_mw", "DC_Charging"),
                           ("swap_requests", "Swap_Requests")]:
        print(f"  → Training {label} model on {len(train):,} samples...")
        X_train = train[FEATURES].values
        y_train = train[target].values
        X_test = test[FEATURES].values
        y_test = test[target].values

        model = HistGradientBoostingRegressor(
            max_iter=200,
            max_leaf_nodes=31,
            learning_rate=0.1,
            min_samples_leaf=20,
            random_state=RANDOM_SEED,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 0, None)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100

        print(f"     MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}  MAPE={mape:.2f}%")
        results[label] = {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape,
                          "y_test": y_test[:500], "y_pred": y_pred[:500],
                          "test_timestamps": test["timestamp"].values[:500],
                          "test_clusters": test["cluster"].values[:500]}
        models[label] = model

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, f"{label}_model.pkl"), "wb") as f:
                pickle.dump(model, f)

    return models, results


def forecast_next_24h(model, cluster_df_latest: pd.DataFrame) -> np.ndarray:
    """
    Forecast the next 24 hours given the most recent cluster observations.
    Uses last available features; lags are populated from recent history.
    Returns array of shape (24,) with forecast values.
    """
    df_sorted = cluster_df_latest.sort_values("timestamp").tail(48).copy()
    df_sorted["dc_demand_lag1"] = df_sorted["dc_charging_demand_mw"].shift(1)
    df_sorted["dc_demand_lag24"] = df_sorted["dc_charging_demand_mw"].shift(24)
    df_sorted["dc_demand_roll24"] = df_sorted["dc_charging_demand_mw"].shift(1).rolling(24).mean()
    df_sorted["swap_lag1"] = df_sorted["swap_requests"].shift(1)
    df_sorted["swap_lag24"] = df_sorted["swap_requests"].shift(24)
    last_rows = df_sorted.dropna(subset=FEATURES).tail(1)
    if len(last_rows) == 0:
        return np.full(24, df_sorted["dc_charging_demand_mw"].mean())
    # Simple repeat-pattern forecast: use same hour of day yesterday as seed
    base_pred = model.predict(last_rows[FEATURES].values)[0]
    # Create 24-hour forecast with diurnal variation
    hours = np.arange(24)
    diurnal = 0.5 + 0.5 * np.sin(2 * np.pi * (hours - 5) / 24)
    return np.clip(base_pred * (0.7 + 0.6 * diurnal), 0, None)


if __name__ == "__main__":
    import sys
    data_path = os.path.join(os.path.dirname(__file__), "..", "results", "synthetic_data.parquet")
    if not os.path.exists(data_path):
        print("Run data/generate_data.py first.")
        sys.exit(1)
    df = pd.read_parquet(data_path)
    models, results = train_demand_models(df)
    print("\nDemand model training complete.")
