"""
GridMind AI — Synthetic Dataset Generator
==========================================
Generates 5 years × 8,760 hours × 5 clusters = 219,000 cluster-hour records.
All operations are vectorised using NumPy/Pandas. No Python loops over timestamps.

Each generated feature is documented with its generation logic so that the
synthetic nature of the data is transparent throughout the analysis.
"""

import numpy as np
import pandas as pd
import os
import sys

# Allow running standalone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.assumptions import (
    CLUSTERS, SIMULATION_YEARS, HOURS_PER_YEAR, TOTAL_HOURS, RANDOM_SEED,
    CLUSTER_DEMAND_SCALE, CLUSTER_CAPACITY_WEIGHTS, CLUSTER_BASE_DEMAND_MW,
    CLUSTER_CLIMATE, CLUSTER_SOLAR_CAPACITY_MW, CLUSTER_WIND_CAPACITY_MW,
    CLUSTER_BESS_MWH, CLUSTER_SWAP_PACKS, PACK_CAPACITY_KWH,
    IEX_DAM_BASE_PRICE, DISCOM_TARIFF, RENEWABLE_CONTRACT_TARIFF,
    IEX_RTM_PREMIUM, BESS_MIN_SOC_FRACTION, BESS_MAX_SOC_FRACTION,
    DC_FAST_CHARGING_CAPACITY_MW, BATTERY_SWAP_CAPACITY_MW,
)


def generate_time_index(years: int = SIMULATION_YEARS) -> pd.DatetimeIndex:
    """Create an hourly DatetimeIndex covering the simulation period."""
    return pd.date_range("2020-01-01", periods=years * HOURS_PER_YEAR, freq="h")


def _temperature_profile(dt_index: pd.DatetimeIndex, climate: dict) -> np.ndarray:
    """
    Synthetic temperature (°C) using seasonal sinusoid + diurnal cycle + noise.
    Annual cycle: peaks in May–June, troughs in December–January.
    Diurnal cycle: peaks ~15:00, troughs ~05:00.
    """
    rng = np.random.default_rng(RANDOM_SEED + 1)
    day_of_year = dt_index.day_of_year.values
    hour = dt_index.hour.values
    t_summer = climate["mean_temp_summer"]
    t_winter = climate["mean_temp_winter"]
    t_mean = (t_summer + t_winter) / 2.0
    t_amp = (t_summer - t_winter) / 2.0
    # Peak around day 165 (mid-June)
    seasonal = t_mean + t_amp * np.cos(2 * np.pi * (day_of_year - 165) / 365)
    diurnal = 4.0 * np.sin(2 * np.pi * (hour - 5) / 24)
    noise = rng.normal(0, 1.5, size=len(dt_index))
    return np.clip(seasonal + diurnal + noise, 5, 50)


def _solar_generation(dt_index: pd.DatetimeIndex, capacity_mw: float, climate: dict,
                       seed_offset: int = 0) -> np.ndarray:
    """
    Synthetic solar generation (MWh/h = MW for Δt=1h).
    Non-zero only during daylight hours (6:00–19:00).
    Shape follows a half-sine bell curve centred at solar noon.
    Cloud cover attenuates output stochastically.
    """
    rng = np.random.default_rng(RANDOM_SEED + seed_offset + 10)
    hour = dt_index.hour.values
    day_of_year = dt_index.day_of_year.values
    # Seasonal clearness: peaks in winter/early summer
    clearness = 0.70 + 0.15 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
    # Stochastic cloud cover (0 = fully cloudy, 1 = clear)
    cloud_factor = np.clip(rng.beta(3, 1.5, size=len(dt_index)), 0.05, 1.0)
    # Daylight bell shape (solar noon at 13:00)
    solar_bell = np.maximum(0, np.sin(np.pi * (hour - 6) / 13))
    gen = capacity_mw * clearness * cloud_factor * solar_bell * climate["solar_peak_ghi"] / 7.0
    gen[hour < 6] = 0.0
    gen[hour > 19] = 0.0
    return np.clip(gen, 0, capacity_mw)


def _wind_generation(dt_index: pd.DatetimeIndex, capacity_mw: float, climate: dict,
                     seed_offset: int = 0) -> np.ndarray:
    """
    Synthetic wind generation (MW).
    Uses a Weibull-distributed wind speed (shape=2 = Rayleigh) with seasonal modulation.
    Power curve: cubic below rated speed, clipped at capacity.
    """
    rng = np.random.default_rng(RANDOM_SEED + seed_offset + 20)
    day_of_year = dt_index.day_of_year.values
    # Wind is typically higher in monsoon season (June–September)
    seasonal_wind = climate["avg_wind_ms"] * (1.0 + 0.3 * np.sin(2 * np.pi * (day_of_year - 180) / 365))
    # Weibull samples
    wind_speed = rng.weibull(2.0, size=len(dt_index)) * seasonal_wind
    # Simple power curve: cut-in 3 m/s, rated 12 m/s, cut-out 25 m/s
    rated_speed = 12.0
    cut_in = 3.0
    power_fraction = np.where(
        wind_speed < cut_in, 0.0,
        np.where(wind_speed > 25.0, 0.0,
                 np.where(wind_speed > rated_speed, 1.0,
                          ((wind_speed - cut_in) / (rated_speed - cut_in)) ** 3))
    )
    return np.clip(capacity_mw * power_fraction, 0, capacity_mw)


def _iex_price_profile(dt_index: pd.DatetimeIndex, seed_offset: int = 0) -> np.ndarray:
    """
    Synthetic IEX DAM price (INR/kWh).
    Follows a double-peak diurnal pattern (morning and evening peaks) with
    weekly and seasonal modulation plus stochastic volatility.
    """
    rng = np.random.default_rng(RANDOM_SEED + seed_offset + 30)
    hour = dt_index.hour.values
    day_of_year = dt_index.day_of_year.values
    # Diurnal: morning peak ~8–10, evening peak ~19–21
    diurnal = (1.5 * np.exp(-0.5 * ((hour - 9) / 2) ** 2) +
               2.0 * np.exp(-0.5 * ((hour - 20) / 2) ** 2))
    # Seasonal: higher in summer (cooling demand)
    seasonal = 1.0 + 0.3 * np.cos(2 * np.pi * (day_of_year - 190) / 365) * -1
    # Weekly: slightly lower on weekends
    weekday = dt_index.dayofweek.values
    weekend_discount = np.where(weekday >= 5, -0.5, 0.0)
    price = IEX_DAM_BASE_PRICE + diurnal * seasonal + weekend_discount
    # Add stochastic volatility with occasional spikes
    noise = rng.normal(0, 0.4, size=len(dt_index))
    spikes = rng.exponential(2.0, size=len(dt_index)) * (rng.random(len(dt_index)) < 0.03)
    return np.clip(price + noise + spikes, 2.0, 18.0)


def _charging_demand(dt_index: pd.DatetimeIndex, base_mw: float, cluster: str,
                     seed_offset: int = 0) -> np.ndarray:
    """
    Synthetic DC fast-charging demand (MW).
    Peaks during commuting hours (8–10, 17–21) and near malls/offices.
    Weekend pattern differs from weekday.
    """
    rng = np.random.default_rng(RANDOM_SEED + seed_offset + 40)
    hour = dt_index.hour.values
    weekday = dt_index.dayofweek.values
    day_of_year = dt_index.day_of_year.values

    # Weekday diurnal: morning rush + evening peak
    weekday_diurnal = (
        0.3 + 0.5 * np.exp(-0.5 * ((hour - 9) / 1.5) ** 2) +
        0.7 * np.exp(-0.5 * ((hour - 19) / 2.0) ** 2)
    )
    # Weekend: flatter, shifted slightly later
    weekend_diurnal = (
        0.2 + 0.6 * np.exp(-0.5 * ((hour - 12) / 3.0) ** 2)
    )
    is_weekend = (weekday >= 5).astype(float)
    diurnal = weekday_diurnal * (1 - is_weekend) + weekend_diurnal * is_weekend

    # Seasonal: higher in summer (AC charging demand increases)
    seasonal = 1.0 + 0.1 * np.cos(2 * np.pi * (day_of_year - 200) / 365) * -1

    scale = CLUSTER_DEMAND_SCALE[cluster]
    cap_mw = DC_FAST_CHARGING_CAPACITY_MW * CLUSTER_CAPACITY_WEIGHTS[cluster]
    noise = rng.normal(1.0, 0.08, size=len(dt_index))
    demand = base_mw * diurnal * seasonal * scale * noise
    return np.clip(demand, 0, cap_mw)


def _swap_requests(dt_index: pd.DatetimeIndex, n_packs: int, cluster: str,
                   seed_offset: int = 0) -> np.ndarray:
    """
    Synthetic swap request count (discrete).
    Peaks align with morning fleet dispatch and evening returns.
    """
    rng = np.random.default_rng(RANDOM_SEED + seed_offset + 50)
    hour = dt_index.hour.values
    weekday = dt_index.dayofweek.values

    # Fleet dispatch patterns: morning peak at 7, evening at 18
    demand_shape = (
        0.15 + 0.6 * np.exp(-0.5 * ((hour - 7) / 1.5) ** 2) +
        0.5 * np.exp(-0.5 * ((hour - 18) / 2.0) ** 2)
    )
    # Weekdays have higher fleet demand
    day_factor = np.where(weekday < 5, 1.0, 0.75)
    scale = CLUSTER_DEMAND_SCALE[cluster]
    max_hourly = n_packs * 0.08  # max 8% of inventory swapped per hour
    base_rate = n_packs * 0.03 * scale
    requests = base_rate * demand_shape * day_factor * rng.lognormal(0, 0.15, size=len(dt_index))
    return np.clip(requests, 0, max_hourly).astype(int)


def generate_cluster_dataset(cluster: str, dt_index: pd.DatetimeIndex,
                              cluster_idx: int) -> pd.DataFrame:
    """Generate all features for a single metropolitan cluster."""
    n = len(dt_index)
    climate = CLUSTER_CLIMATE[cluster]
    seed = cluster_idx * 1000

    # Time features
    hour = dt_index.hour.values
    dow = dt_index.dayofweek.values
    month = dt_index.month.values
    doy = dt_index.day_of_year.values
    is_weekend = (dow >= 5).astype(int)

    # Weather
    temp = _temperature_profile(dt_index, climate)
    rng = np.random.default_rng(RANDOM_SEED + seed + 5)
    humidity = np.clip(60 + 20 * np.sin(2 * np.pi * (doy - 150) / 365) + rng.normal(0, 8, n), 20, 100)
    rainfall = np.clip(rng.exponential(2.0, n) * (month_in_monsoon(month)), 0, 150)
    cloud_cover = np.clip(0.4 + 0.3 * (rainfall > 0).astype(float) + rng.uniform(-0.2, 0.2, n), 0, 1)
    wind_speed = np.clip(climate["avg_wind_ms"] + rng.normal(0, 1.5, n), 0, 20)
    solar_irr = np.clip(climate["solar_peak_ghi"] * (1 - 0.7 * cloud_cover) *
                        np.maximum(0, np.sin(np.pi * (hour - 6) / 13)) * (hour > 6) * (hour < 20),
                        0, climate["solar_peak_ghi"])

    # Generation
    solar_gen = _solar_generation(dt_index, CLUSTER_SOLAR_CAPACITY_MW[cluster], climate, seed)
    wind_gen = _wind_generation(dt_index, CLUSTER_WIND_CAPACITY_MW[cluster], climate, seed)
    renewable_gen = solar_gen + wind_gen
    renewable_allocated = renewable_gen * 0.95  # 5% curtailed for reserve

    # Demand
    base_mw = CLUSTER_BASE_DEMAND_MW[cluster]
    dc_demand = _charging_demand(dt_index, base_mw, cluster, seed)
    n_packs = CLUSTER_SWAP_PACKS[cluster]
    swap_req = _swap_requests(dt_index, n_packs, cluster, seed)

    # Fleet and retail split
    fleet_demand_mw = dc_demand * 0.35
    retail_demand_mw = dc_demand * 0.65
    queue_len = np.clip((dc_demand / (base_mw * 1.1) * 15).astype(int), 0, 50)
    abandoned = (queue_len > 10).astype(int) * np.clip(queue_len - 10, 0, 20)

    # Market prices
    iex_dam_price = _iex_price_profile(dt_index, seed)
    rtm_price = iex_dam_price + IEX_RTM_PREMIUM + rng.normal(0, 0.5, n)
    discom_tariff_arr = np.full(n, DISCOM_TARIFF)
    renewable_tariff_arr = np.full(n, RENEWABLE_CONTRACT_TARIFF)

    # BESS state (simulated heuristic for dataset — optimized separately)
    bess_capacity = CLUSTER_BESS_MWH[cluster]
    bess_soc = np.full(n, bess_capacity * 0.5)
    soh = np.clip(1.0 - (np.arange(n) / (n * 2.5)) * 0.1, 0.80, 1.0)  # Gradual degradation
    charge_cycles = np.cumsum(np.abs(np.diff(np.concatenate([[0], bess_soc]))) / bess_capacity * 0.5)

    # Swap inventory state
    charged_packs = np.full(n, int(n_packs * 0.6))
    empty_packs = np.full(n, int(n_packs * 0.3))
    partial_packs = np.full(n, int(n_packs * 0.1))
    pack_returns = np.clip((swap_req * 0.9).astype(int), 0, n_packs)

    df = pd.DataFrame({
        # Identity
        "cluster": cluster,
        "timestamp": dt_index,
        # Time features
        "hour": hour,
        "day_of_week": dow,
        "month": month,
        "day_of_year": doy,
        "is_weekend": is_weekend,
        # Weather
        "temperature_c": temp,
        "humidity_pct": humidity,
        "rainfall_mm": rainfall,
        "cloud_cover": cloud_cover,
        "wind_speed_ms": wind_speed,
        "solar_irradiation_kwh_m2": solar_irr,
        # Demand
        "dc_charging_demand_mw": dc_demand,
        "swap_requests": swap_req,
        "fleet_demand_mw": fleet_demand_mw,
        "retail_demand_mw": retail_demand_mw,
        "queue_length": queue_len,
        "abandoned_sessions": abandoned,
        # Renewable
        "solar_gen_mw": solar_gen,
        "wind_gen_mw": wind_gen,
        "renewable_gen_mw": renewable_gen,
        "renewable_allocated_mw": renewable_allocated,
        # Market
        "iex_dam_price_inr_kwh": iex_dam_price,
        "rtm_price_inr_kwh": np.clip(rtm_price, 0, 25),
        "discom_tariff_inr_kwh": discom_tariff_arr,
        "renewable_tariff_inr_kwh": renewable_tariff_arr,
        # BESS
        "bess_soc_mwh": bess_soc,
        "bess_soh": soh,
        "bess_charge_cycles": charge_cycles,
        "bess_capacity_mwh": bess_capacity,
        # Swap inventory
        "charged_packs": charged_packs,
        "empty_packs": empty_packs,
        "partial_packs": partial_packs,
        "swap_pack_returns": pack_returns,
    })
    return df


def month_in_monsoon(month: np.ndarray) -> np.ndarray:
    """Return 1 for monsoon months (June–September), 0 otherwise."""
    return np.isin(month, [6, 7, 8, 9]).astype(float)


def generate_full_dataset(save_path: str = None) -> pd.DataFrame:
    """
    Generate the complete 219,000-row synthetic dataset.
    Returns a DataFrame indexed by (cluster, timestamp).
    """
    print("[data] Generating synthetic dataset...")
    dt_index = generate_time_index(SIMULATION_YEARS)
    frames = []
    for idx, cluster in enumerate(CLUSTERS):
        print(f"  → {cluster} ({idx+1}/{len(CLUSTERS)})")
        df_c = generate_cluster_dataset(cluster, dt_index, idx)
        frames.append(df_c)

    df_full = pd.concat(frames, ignore_index=True)
    df_full = df_full.sort_values(["cluster", "timestamp"]).reset_index(drop=True)

    print(f"[data] Dataset shape: {df_full.shape}")
    print(f"[data] Date range: {df_full['timestamp'].min()} → {df_full['timestamp'].max()}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            df_full.to_parquet(save_path, index=False)
            print(f"[data] Saved to {save_path}")
        except Exception:
            csv_path = save_path.replace(".parquet", ".csv")
            df_full.to_csv(csv_path, index=False)
            print(f"[data] Parquet unavailable; saved CSV to {csv_path}")

    return df_full


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..", "results")
    df = generate_full_dataset(save_path=os.path.join(base, "synthetic_data.parquet"))
    print(df.describe())
