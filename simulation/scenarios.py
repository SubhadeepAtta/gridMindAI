"""
GridMind AI \u2014 Scenario Definitions and Simulation Runner
==========================================================
Implements six representative 24-hour operating scenarios as described in
the architecture specification. Each scenario is run under both the baseline
(conventional heuristic) strategy and the GridMind AI optimized strategy.

Scenario selection rationale:
  Normal day captures median operating conditions.
  Heatwave tests demand surge and elevated procurement costs.
  Low-renewable tests BESS reliance and procurement flexibility.
  Price spike tests arbitrage value and procurement strategy.
  Demand imbalance tests inter-cluster coordination mechanisms.
  Combined extreme tests system robustness under simultaneous stressors.

For each scenario, we construct representative demand and renewable profiles
by sampling from (or modifying) the synthetic dataset. This avoids the need
to run 365 separate MILPs while still producing meaningful comparisons.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.assumptions import (
    CLUSTERS, N_CLUSTERS, CLUSTER_CAPACITY_WEIGHTS,
    DC_FAST_CHARGING_CAPACITY_MW, BATTERY_SWAP_CAPACITY_MW,
    CLUSTER_BESS_MWH, CLUSTER_SWAP_PACKS, RANDOM_SEED,
    IEX_DAM_BASE_PRICE, DISCOM_TARIFF, PACK_CAPACITY_KWH,
)


SCENARIOS = {
    1: "Normal Operating Day",
    2: "Heatwave and Demand Surge",
    3: "Low Renewable (Cloudy/Low Wind)",
    4: "IEX Price Spike",
    5: "Cluster Demand Imbalance",
    6: "Combined Extreme",
}

T = 24  # 24-hour horizon


def _base_demand(cluster_idx: int) -> np.ndarray:
    """Typical diurnal demand profile (MW) for a cluster."""
    cap = DC_FAST_CHARGING_CAPACITY_MW * CLUSTER_CAPACITY_WEIGHTS[CLUSTERS[cluster_idx]]
    hours = np.arange(T)
    weekday_demand = cap * (
        0.35 + 0.45 * np.exp(-0.5 * ((hours - 9) / 1.5) ** 2) +
        0.60 * np.exp(-0.5 * ((hours - 19) / 2.0) ** 2)
    )
    return np.clip(weekday_demand, 0, cap)


def _base_renewable(cluster_idx: int) -> np.ndarray:
    """Typical diurnal solar+wind profile (MW)."""
    from config.assumptions import CLUSTER_SOLAR_CAPACITY_MW, CLUSTER_WIND_CAPACITY_MW
    c = CLUSTERS[cluster_idx]
    solar_cap = CLUSTER_SOLAR_CAPACITY_MW[c]
    wind_cap = CLUSTER_WIND_CAPACITY_MW[c]
    hours = np.arange(T)
    solar = solar_cap * 0.75 * np.maximum(0, np.sin(np.pi * (hours - 6) / 13))
    solar[(hours < 6) | (hours > 19)] = 0
    wind = wind_cap * 0.45 * np.ones(T)  # Moderate wind throughout day
    return solar + wind


def _base_iex_price() -> np.ndarray:
    """Typical diurnal IEX DAM price (INR/kWh)."""
    hours = np.arange(T)
    return np.clip(
        IEX_DAM_BASE_PRICE
        + 1.5 * np.exp(-0.5 * ((hours - 9) / 2) ** 2)
        + 2.0 * np.exp(-0.5 * ((hours - 20) / 2) ** 2),
        2.0, 15.0
    )


def build_scenario(scenario_id: int) -> dict:
    """
    Construct demand, renewable, and price arrays for a given scenario.
    Returns dict with numpy arrays ready for the dispatch optimizer.
    """
    rng = np.random.default_rng(RANDOM_SEED + scenario_id * 100)

    demand_mw = np.array([_base_demand(c) for c in range(N_CLUSTERS)])
    renewable_mw = np.array([_base_renewable(c) for c in range(N_CLUSTERS)])
    iex_price = _base_iex_price()

    if scenario_id == 1:
        # Normal day: add small noise
        demand_mw *= rng.uniform(0.95, 1.05, demand_mw.shape)
        renewable_mw *= rng.uniform(0.90, 1.10, renewable_mw.shape)
        description = "Typical weekday with moderate demand and normal renewable output."

    elif scenario_id == 2:
        # Heatwave: 15-25% demand surge, 10% solar reduction (high temperature reduces panel efficiency)
        demand_mw *= rng.uniform(1.15, 1.25, demand_mw.shape)
        renewable_mw[:, 8:19] *= 0.90  # Slight solar reduction at peak heat
        iex_price *= 1.20  # Higher market prices during peak demand
        description = "Heatwave conditions: 20% demand surge and elevated IEX prices."

    elif scenario_id == 3:
        # Low renewable: heavy cloud cover, low wind
        renewable_mw *= rng.uniform(0.20, 0.35, renewable_mw.shape)
        demand_mw *= rng.uniform(0.95, 1.05, demand_mw.shape)
        description = "Cloudy and low-wind day: renewable generation at 25-30% of normal."

    elif scenario_id == 4:
        # IEX price spike: factor of 2.5-3x on prices in evening
        price_spike = np.ones(T)
        price_spike[17:22] = rng.uniform(2.5, 3.0)  # Evening spike
        iex_price *= price_spike
        iex_price = np.clip(iex_price, 2, 20)
        description = "IEX price spike in evening hours (2.5\u20133.0x base price)."

    elif scenario_id == 5:
        # Cluster demand imbalance: Delhi NCR and Mumbai 30% above normal,
        # Bengaluru/Hyderabad/Chennai 20% below
        for c in range(N_CLUSTERS):
            if CLUSTERS[c] in ["Delhi NCR", "Mumbai MMR"]:
                demand_mw[c] *= 1.30
            else:
                demand_mw[c] *= 0.80
        description = "Demand imbalance: North/West clusters surge, South clusters below normal."

    elif scenario_id == 6:
        # Combined extreme: heatwave + low renewable + price spike
        demand_mw *= rng.uniform(1.20, 1.30, demand_mw.shape)
        renewable_mw *= rng.uniform(0.25, 0.35, renewable_mw.shape)
        price_spike = np.ones(T)
        price_spike[17:22] = 2.8
        iex_price *= price_spike * 1.20
        iex_price = np.clip(iex_price, 2, 22)
        description = "Combined extreme: heatwave demand surge + low renewable + price spike."

    # Initial conditions
    bess_soc_init = np.array([CLUSTER_BESS_MWH[c] * 0.55 for c in CLUSTERS])
    charged_packs_init = np.array([int(CLUSTER_SWAP_PACKS[c] * 0.65) for c in CLUSTERS])

    return {
        "scenario_id": scenario_id,
        "scenario_name": SCENARIOS[scenario_id],
        "description": description,
        "demand_mw": np.clip(demand_mw, 0, None),
        "renewable_mw": np.clip(renewable_mw, 0, None),
        "iex_price": iex_price,
        "bess_soc_init": bess_soc_init,
        "charged_packs_init": charged_packs_init,
    }
