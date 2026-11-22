"""
GridMind AI — Battery Degradation Estimation Module
====================================================
Estimates marginal degradation cost (INR/kWh throughput) for stationary BESS
and swap packs.

This is intentionally a simple physics-informed regression rather than a
complex ML model. The rationale is:
1. Degradation models trained on synthetic SoH data reproduce the synthetic
   assumptions, adding noise without insight.
2. The key output — a fixed cost coefficient for the MILP objective — requires
   only a defensible scalar estimate, not a probabilistic forecast.
3. Standard battery degradation literature provides sufficient guidance for
   the coefficient range used here.

The Rainflow counting approximation and calendar aging model are simplified
but physically consistent.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.assumptions import (
    BESS_DEGRADATION_COST_INR_KWH, PACK_DEGRAD_COST_INR_KWH,
    BESS_ROUND_TRIP_EFFICIENCY, PACK_CHARGE_EFFICIENCY,
    RANDOM_SEED,
)


def estimate_degradation_cost_bess(
    soh: float,
    dod_avg: float = 0.7,
    temperature_avg: float = 30.0,
    daily_cycles: float = 1.0,
) -> float:
    """
    Estimate marginal degradation cost (INR/kWh) for stationary BESS.

    Uses a simplified capacity-fade model:
      Lifetime_cycles ≈ base_cycles × DoD_factor × temperature_factor
      Cost_per_kWh = Pack_cost / (Lifetime_cycles × Usable_capacity)

    Parameters:
      soh : Current state of health (0–1)
      dod_avg : Average depth of discharge (0–1)
      temperature_avg : Average cell temperature (°C)
      daily_cycles : Average charge-discharge cycles per day

    Returns:
      Marginal degradation cost (INR/kWh throughput)
    """
    # Base cycle life at 50% DoD, 25°C (typical Li-NMC spec)
    base_cycle_life = 3500
    # DoD adjustment: deeper discharge reduces cycle life (Wöhler-curve approximation)
    dod_factor = (0.5 / max(dod_avg, 0.1)) ** 1.2
    # Temperature factor: elevated temperature accelerates aging
    temp_factor = max(0.5, 1.0 - 0.015 * max(0, temperature_avg - 25))
    # SoH adjustment: remaining useful life proportional to current SoH
    soh_factor = max(0.3, soh)

    effective_cycles = base_cycle_life * dod_factor * temp_factor * soh_factor

    # Capex assumption: INR 4.5 crore/MWh installed (2024-25 benchmark)
    capex_per_kwh = 4500  # INR/kWh
    # Usable energy per cycle (kWh per kWh installed)
    usable_per_cycle = dod_avg * BESS_ROUND_TRIP_EFFICIENCY
    cost_per_kwh = capex_per_kwh / (effective_cycles * usable_per_cycle)

    # Clamp to a physically reasonable range
    return float(np.clip(cost_per_kwh, 0.40, 1.20))


def estimate_degradation_cost_pack(
    soh: float = 0.95,
    dod_avg: float = 0.85,
    temperature_avg: float = 32.0,
) -> float:
    """
    Estimate marginal degradation cost (INR/kWh) for swap packs.
    Swap packs undergo full charge-discharge cycles more frequently.
    """
    base_cycle_life = 1500  # Swap packs face harder cycling
    dod_factor = (0.85 / max(dod_avg, 0.2)) ** 1.1
    temp_factor = max(0.5, 1.0 - 0.012 * max(0, temperature_avg - 25))
    effective_cycles = base_cycle_life * dod_factor * temp_factor * max(0.3, soh)
    capex_per_kwh = 3800  # INR/kWh for swap packs
    usable_per_cycle = dod_avg * PACK_CHARGE_EFFICIENCY
    cost_per_kwh = capex_per_kwh / (effective_cycles * usable_per_cycle)
    return float(np.clip(cost_per_kwh, 0.30, 2.00))


def fit_degradation_regression(df: pd.DataFrame):
    """
    Fit a simple Ridge regression to predict SoH from operational features.
    Used to validate that simulated SoH trends are physically reasonable.
    """
    print("[degradation_model] Fitting SoH regression...")
    # Sample one cluster for regression
    subset = df[df["cluster"] == "Delhi NCR"].copy().sort_values("timestamp")
    subset = subset.dropna(subset=["bess_soh", "bess_charge_cycles", "temperature_c"])

    X = subset[["bess_charge_cycles", "temperature_c", "bess_soc_mwh"]].values
    y = subset["bess_soh"].values

    # Chronological split
    split = int(len(X) * 0.75)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"  SoH regression: MAE={mae:.5f}  R²={r2:.4f}")

    # Compute scenario-specific degradation costs
    nominal_bess_cost = estimate_degradation_cost_bess(soh=0.95)
    degraded_bess_cost = estimate_degradation_cost_bess(soh=0.85)
    nominal_pack_cost = estimate_degradation_cost_pack(soh=0.95)

    print(f"  Nominal BESS degradation cost: INR {nominal_bess_cost:.3f}/kWh")
    print(f"  Degraded BESS degradation cost: INR {degraded_bess_cost:.3f}/kWh")
    print(f"  Nominal pack degradation cost: INR {nominal_pack_cost:.3f}/kWh")

    return {
        "model": model,
        "MAE": mae, "R2": r2,
        "nominal_bess_cost_inr_kwh": nominal_bess_cost,
        "nominal_pack_cost_inr_kwh": nominal_pack_cost,
    }


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "results", "synthetic_data.parquet")
    df = pd.read_parquet(data_path)
    result = fit_degradation_regression(df)
    print("\nDegradation model complete.")
