"""
GridMind AI — Renewable Generation Forecasting Module
======================================================
Forecasts solar and wind generation for each metropolitan cluster.

Solar generation is highly predictable from weather features (cloud cover,
solar irradiation, temperature). Wind is less predictable but still benefits
from weather inputs. We use a single HistGradientBoostingRegressor per
energy type rather than per-cluster models to reduce training time, with
cluster identity included as a categorical feature.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.assumptions import CLUSTERS, TRAIN_YEARS, HOURS_PER_YEAR, RANDOM_SEED


RENEWABLE_FEATURES = [
    "cluster_enc", "hour", "month", "day_of_year",
    "temperature_c", "cloud_cover", "wind_speed_ms",
    "solar_irradiation_kwh_m2", "humidity_pct", "rainfall_mm",
]


def _prepare_renewable_features(df: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder()
    df = df.copy()
    df["cluster_enc"] = le.fit_transform(df["cluster"])
    return df, le


def _chronological_split(df: pd.DataFrame):
    cutoff = df["timestamp"].min() + pd.Timedelta(hours=TRAIN_YEARS * HOURS_PER_YEAR)
    return df[df["timestamp"] < cutoff], df[df["timestamp"] >= cutoff]


def train_renewable_models(df: pd.DataFrame, save_dir: str = None):
    """
    Train solar and wind generation forecasting models.
    Returns models dict and evaluation results.
    """
    print("[renewable_model] Training renewable forecasting models...")
    df_feat, le = _prepare_renewable_features(df)
    train, test = _chronological_split(df_feat)

    results = {}
    models = {"label_encoder": le}

    for target, label in [("solar_gen_mw", "Solar"), ("wind_gen_mw", "Wind")]:
        print(f"  → Training {label} model...")
        X_train = train[RENEWABLE_FEATURES].values
        y_train = train[target].values
        X_test = test[RENEWABLE_FEATURES].values
        y_test = test[target].values

        model = HistGradientBoostingRegressor(
            max_iter=150,
            max_leaf_nodes=31,
            learning_rate=0.1,
            min_samples_leaf=25,
            random_state=RANDOM_SEED,
        )
        model.fit(X_train, y_train)
        y_pred = np.clip(model.predict(X_test), 0, None)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"     MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}")

        results[label] = {
            "MAE": mae, "RMSE": rmse, "R2": r2,
            "y_test": y_test[:500], "y_pred": y_pred[:500],
            "test_timestamps": test["timestamp"].values[:500],
            "test_clusters": test["cluster"].values[:500],
        }
        models[label] = model

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, f"{label}_model.pkl"), "wb") as f:
                pickle.dump(model, f)

    return models, results


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "results", "synthetic_data.parquet")
    df = pd.read_parquet(data_path)
    models, results = train_renewable_models(df)
    print("\nRenewable model training complete.")
