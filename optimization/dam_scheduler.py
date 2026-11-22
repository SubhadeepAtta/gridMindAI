"""
GridMind AI — Day-Ahead Market Procurement Scheduler
=====================================================
Stage A of the two-stage market architecture.

At the end of day t-1, the operator must submit bids to:
  1. IEX Day-Ahead Market (DAM) — price-based auction clearing
  2. DISCOM bilateral contract — quantity nomination
  3. Renewable contract delivery nomination

Once cleared, DAM quantities become a locked schedule that cannot be changed
in real-time operations. This module generates the DAM commitment vector
based on demand and renewable forecasts.

Constraints:
  - Total supply ≥ (1 + reserve_fraction) × forecast_demand
  - Renewable contract must be drawn first (cheapest)
  - DAM quantity bounded by assumed maximum bid volume
  - DISCOM provides baseline floor supply
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.assumptions import (
    CLUSTERS, N_CLUSTERS, CLUSTER_CAPACITY_WEIGHTS,
    DC_FAST_CHARGING_CAPACITY_MW, SYSTEM_RESERVE_FRACTION,
    RENEWABLE_CONTRACT_TARIFF, DISCOM_TARIFF, IEX_DAM_BASE_PRICE,
    RENEWABLE_SHARE, GRID_SHARE,
)


def schedule_dam(
    demand_forecast_mw: np.ndarray,   # shape (N_CLUSTERS, 24)
    renewable_forecast_mw: np.ndarray, # shape (N_CLUSTERS, 24)
    iex_price_forecast: np.ndarray,    # shape (24,)
    bess_soc_init: np.ndarray,         # shape (N_CLUSTERS,)
) -> dict:
    """
    Generate day-ahead procurement schedule.

    Strategy:
    1. Determine net demand after renewable dispatch.
    2. Allocate DISCOM floor (minimum take-or-pay).
    3. Schedule IEX DAM for the remainder, weighted by price.
    4. Ensure total scheduled supply covers demand + reserve.

    Returns locked DAM commitment matrix (N_CLUSTERS × 24).
    """
    reserve_factor = 1.0 + SYSTEM_RESERVE_FRACTION
    target_supply = demand_forecast_mw * reserve_factor

    # Step 1: Renewable first dispatch
    ren_scheduled = np.minimum(renewable_forecast_mw, target_supply * RENEWABLE_SHARE * 1.2)

    # Step 2: Net demand after renewable
    net_demand = np.maximum(0, target_supply - ren_scheduled)

    # Step 3: DISCOM floor — fixed baseline (30% of cluster capacity)
    discom_floor = np.array([
        DC_FAST_CHARGING_CAPACITY_MW * CLUSTER_CAPACITY_WEIGHTS[c] * 0.30
        for c in CLUSTERS
    ]).reshape(-1, 1) * np.ones((1, 24))

    # Step 4: DAM schedule for remainder
    # Prefer lower-price hours for DAM procurement (price-weighted scheduling)
    price_weight = 1.0 / (iex_price_forecast + 0.5)  # higher weight = cheaper
    price_weight /= price_weight.sum()
    dam_total_per_cluster = np.maximum(0, net_demand - discom_floor)
    # Distribute hourly proportional to inverse price (buy more when cheap)
    dam_schedule = dam_total_per_cluster * price_weight.reshape(1, -1) * 24
    dam_schedule = np.minimum(dam_schedule,
                              DC_FAST_CHARGING_CAPACITY_MW *
                              np.array([CLUSTER_CAPACITY_WEIGHTS[c] for c in CLUSTERS]).reshape(-1, 1) * 0.5)

    # Total scheduled supply
    total_scheduled = ren_scheduled + discom_floor + dam_schedule

    # Procurement cost estimate (INR)
    ren_cost = (ren_scheduled * RENEWABLE_CONTRACT_TARIFF * 1000).sum()
    discom_cost = (discom_floor * DISCOM_TARIFF * 1000).sum()
    dam_cost = (dam_schedule * iex_price_forecast.reshape(1, -1) * 1000).sum()

    return {
        "dam_schedule_mw": dam_schedule,
        "discom_floor_mw": discom_floor,
        "renewable_scheduled_mw": ren_scheduled,
        "total_scheduled_mw": total_scheduled,
        "dam_cost_inr": dam_cost,
        "discom_cost_inr": discom_cost,
        "renewable_cost_inr": ren_cost,
        "total_cost_inr": ren_cost + discom_cost + dam_cost,
    }
