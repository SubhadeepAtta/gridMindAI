"""
GridMind AI - Central Assumptions and Configuration
====================================================
All parameters are either:
  [CASE-STUDY]  : Directly given in ERoH Round 2 problem statement
  [ASSUMPTION]  : Engineering assumption required due to unavailable company data
  [DERIVED]     : Calculated from case-study or assumption values

This file is the single source of truth for all numerical inputs.
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. SYSTEM IDENTITY
# ---------------------------------------------------------------------------
PROJECT_NAME = "GridMind AI"
TAGLINE = "Predict. Allocate. Optimize. Profit."
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# 2. METROPOLITAN CLUSTERS  [CASE-STUDY]
# ---------------------------------------------------------------------------
CLUSTERS = ["Delhi NCR", "Mumbai MMR", "Bengaluru", "Hyderabad", "Chennai"]
N_CLUSTERS = 5

# ---------------------------------------------------------------------------
# 3. INSTALLED CAPACITY  [CASE-STUDY]
# ---------------------------------------------------------------------------
TOTAL_CHARGING_CAPACITY_MW = 1200.0        # Installed charging/swap capacity
DC_FAST_CHARGING_CAPACITY_MW = 720.0       # DC fast-charging capacity
BATTERY_SWAP_CAPACITY_MW = 480.0           # Battery-swapping throughput capacity
TOTAL_STORAGE_ENERGY_MWH = 600.0           # Total electrical energy storage

# ---------------------------------------------------------------------------
# 4. STORAGE SPLIT  [ASSUMPTION - illustrative allocation]
# ---------------------------------------------------------------------------
# Case-study gives 600 MWh total storage but does not split stationary vs swap.
# We allocate 60/40 as a defensible engineering assumption.
STATIONARY_BESS_ENERGY_MWH = 360.0        # Bidirectional grid-tied BESS
SWAP_INVENTORY_ENERGY_MWH = 240.0         # Battery-swap pack inventory

PACK_CAPACITY_KWH = 30.0                   # [ASSUMPTION] kWh per swap pack
TOTAL_SWAP_PACKS = int(SWAP_INVENTORY_ENERGY_MWH * 1000 / PACK_CAPACITY_KWH)  # 8000

# Per-cluster allocation weights  [ASSUMPTION]
# Weights reflect approximate EV penetration and infrastructure size
CLUSTER_CAPACITY_WEIGHTS = {
    "Delhi NCR":   0.25,
    "Mumbai MMR":  0.22,
    "Bengaluru":   0.20,
    "Hyderabad":   0.17,
    "Chennai":     0.16,
}

# Per-cluster BESS energy (MWh)  [DERIVED]
CLUSTER_BESS_MWH = {c: STATIONARY_BESS_ENERGY_MWH * w for c, w in CLUSTER_CAPACITY_WEIGHTS.items()}

# Per-cluster swap packs  [DERIVED]
CLUSTER_SWAP_PACKS = {c: int(TOTAL_SWAP_PACKS * w) for c, w in CLUSTER_CAPACITY_WEIGHTS.items()}

# Per-cluster DC charging capacity (MW)  [DERIVED]
CLUSTER_DC_CAPACITY_MW = {c: DC_FAST_CHARGING_CAPACITY_MW * w for c, w in CLUSTER_CAPACITY_WEIGHTS.items()}

# Per-cluster swap throughput capacity (MW)  [DERIVED]
CLUSTER_SWAP_CAPACITY_MW = {c: BATTERY_SWAP_CAPACITY_MW * w for c, w in CLUSTER_CAPACITY_WEIGHTS.items()}

# ---------------------------------------------------------------------------
# 5. PROCUREMENT MIX  [CASE-STUDY]
# ---------------------------------------------------------------------------
RENEWABLE_SHARE = 0.35       # 35% from renewable contracts
GRID_SHARE = 0.65            # 65% from DISCOM + IEX

# ---------------------------------------------------------------------------
# 6. PERFORMANCE TARGETS  [CASE-STUDY]
# ---------------------------------------------------------------------------
EXISTING_RELIABILITY_PCT = 78.0
TARGET_RELIABILITY_PCT = 89.7
EXISTING_LOSS_PCT = 10.0
TARGET_LOSS_PCT = 8.0
TARGET_EBITDA_MARGIN_PCT = 12.0

# ---------------------------------------------------------------------------
# 7. PRICING  [ASSUMPTION + CASE-STUDY constraints]
# ---------------------------------------------------------------------------
# Case-study allows ±20% from base tariff
PRICING_BAND_MAX = 0.20

# Discrete pricing tiers (within ±20% band)
PRICE_TIERS = {
    "discount":  -0.15,   # -15%
    "standard":   0.00,   #   0%
    "premium":   +0.15,   # +15%
}

# Base retail tariffs (INR/kWh)  [ASSUMPTION - representative 2024-25 values]
BASE_CHARGING_TARIFF_INR_KWH = 12.0       # DC fast charge retail price
BASE_SWAP_FEE_INR_PACK = 350.0            # Fee per swap (30 kWh equivalent)

# Price-elastic demand response coefficients  [ASSUMPTION - synthetic]
# demand_multiplier = 1 - elasticity × (price_change / base_price)
PRICE_ELASTICITY = 0.4   # Assumed own-price elasticity magnitude
TIER_DEMAND_RESPONSE = {
    "discount":  1.0 + PRICE_ELASTICITY * 0.15,   # ~1.06x base demand
    "standard":  1.00,
    "premium":   1.0 - PRICE_ELASTICITY * 0.15,   # ~0.94x base demand
}

# ---------------------------------------------------------------------------
# 8. ENERGY COSTS (INR/kWh)  [ASSUMPTION - representative 2024-25]
# ---------------------------------------------------------------------------
RENEWABLE_CONTRACT_TARIFF = 3.50   # Signed PPA rate (solar/wind)
DISCOM_TARIFF = 7.80               # State DISCOM industrial tariff
IEX_DAM_BASE_PRICE = 5.20          # IEX DAM average price
IEX_RTM_PREMIUM = 1.80            # Additional cost for real-time procurement
EMERGENCY_PROCUREMENT_TARIFF = 12.50  # Fallback emergency supply

# ---------------------------------------------------------------------------
# 9. BESS PARAMETERS  [ASSUMPTION - representative Li-ion specifications]
# ---------------------------------------------------------------------------
BESS_CHARGE_EFFICIENCY = 0.95       # Round-trip charge efficiency
BESS_DISCHARGE_EFFICIENCY = 0.95    # Round-trip discharge efficiency
BESS_ROUND_TRIP_EFFICIENCY = BESS_CHARGE_EFFICIENCY * BESS_DISCHARGE_EFFICIENCY  # ~0.9025
BESS_MIN_SOC_FRACTION = 0.15        # Minimum SoC (DoD limit = 85%)
BESS_MAX_SOC_FRACTION = 0.95        # Maximum SoC
BESS_C_RATE_MAX = 0.5               # Maximum charge/discharge C-rate (0.5C)
BESS_SELF_DISCHARGE_HOURLY = 0.0002 # Hourly self-discharge fraction
BESS_DEGRADATION_COST_INR_KWH = 0.80  # Levelized degradation cost per kWh throughput

# ---------------------------------------------------------------------------
# 10. SWAP-PACK PARAMETERS  [ASSUMPTION]
# ---------------------------------------------------------------------------
PACK_CHARGE_EFFICIENCY = 0.93       # Pack charging efficiency
PACK_SELF_DISCHARGE_DAILY = 0.005   # Daily self-discharge of idle packs
PACK_RESERVE_FRACTION = 0.05        # 5% reserve inventory
PACK_DEGRAD_COST_INR_KWH = 0.60     # Pack degradation cost per kWh charged
PACK_TRANSPORT_COST_INR = 50.0      # Transport cost per pack per intra-metro move

# ---------------------------------------------------------------------------
# 11. LOSS PARAMETERS  [ASSUMPTION - hardware specs]
# ---------------------------------------------------------------------------
CHARGER_CONVERSION_LOSS = 0.04      # 4% DC/DC conversion loss
DISTRIBUTION_LOSS_FRACTION = 0.025  # 2.5% distribution cable losses
INVERTER_LOSS_FRACTION = 0.025      # 2.5% inverter loss (BESS interface)
IDLE_BESS_STANDBY_FRACTION = 0.001  # 0.1% per hour standby consumption

# ---------------------------------------------------------------------------
# 12. RESERVE REQUIREMENTS  [CASE-STUDY]
# ---------------------------------------------------------------------------
SYSTEM_RESERVE_FRACTION = 0.05      # 5% capacity reserve
SWAP_RESERVE_FRACTION = 0.05        # 5% pack inventory reserve

# ---------------------------------------------------------------------------
# 13. QUEUE MODEL PARAMETERS  [ASSUMPTION]
# ---------------------------------------------------------------------------
AVG_KWH_PER_SESSION = 25.0          # Average energy delivered per charging session
MAX_QUEUE_WAIT_MIN = 20             # Maximum acceptable queue wait (minutes)
ABANDONMENT_RATE_PER_MIN = 0.03     # Session abandonment rate per minute wait

# ---------------------------------------------------------------------------
# 14. OPERATIONAL COSTS  [ASSUMPTION - illustrative]
# ---------------------------------------------------------------------------
ANNUAL_OPEX_INR_CR = 850.0          # Annual operating expenses (crore INR)
HOURLY_OPEX_INR = ANNUAL_OPEX_INR_CR * 1e7 / 8760  # Per-hour equivalent

# ---------------------------------------------------------------------------
# 15. SIMULATION PARAMETERS
# ---------------------------------------------------------------------------
SIMULATION_YEARS = 5
HOURS_PER_YEAR = 8760
TOTAL_HOURS = SIMULATION_YEARS * HOURS_PER_YEAR
DELTA_T = 1.0                       # Time step = 1 hour

# Training/validation split (use last 2 years for primary work)
TRAIN_YEARS = 3
TEST_YEARS = 2

# ---------------------------------------------------------------------------
# 16. CLUSTER-SPECIFIC DEMAND SCALERS  [ASSUMPTION - relative EV demand]
# ---------------------------------------------------------------------------
CLUSTER_DEMAND_SCALE = {
    "Delhi NCR":   1.15,  # Highest demand (large metro, strong EV adoption)
    "Mumbai MMR":  1.10,
    "Bengaluru":   1.05,
    "Hyderabad":   0.90,
    "Chennai":     0.85,
}

# Base hourly demand (MW) per cluster for DC charging  [ASSUMPTION]
CLUSTER_BASE_DEMAND_MW = {
    c: DC_FAST_CHARGING_CAPACITY_MW * CLUSTER_CAPACITY_WEIGHTS[c] * 0.65  # 65% avg utilization
    for c in CLUSTERS
}

# ---------------------------------------------------------------------------
# 17. WEATHER PARAMETERS (city-specific)  [ASSUMPTION - climatological averages]
# ---------------------------------------------------------------------------
CLUSTER_CLIMATE = {
    "Delhi NCR":   {"mean_temp_summer": 38, "mean_temp_winter": 15, "solar_peak_ghi": 6.5, "avg_wind_ms": 3.5},
    "Mumbai MMR":  {"mean_temp_summer": 32, "mean_temp_winter": 24, "solar_peak_ghi": 5.5, "avg_wind_ms": 4.2},
    "Bengaluru":   {"mean_temp_summer": 28, "mean_temp_winter": 18, "solar_peak_ghi": 5.8, "avg_wind_ms": 3.0},
    "Hyderabad":   {"mean_temp_summer": 36, "mean_temp_winter": 20, "solar_peak_ghi": 6.0, "avg_wind_ms": 3.8},
    "Chennai":     {"mean_temp_summer": 35, "mean_temp_winter": 25, "solar_peak_ghi": 5.2, "avg_wind_ms": 5.5},
}

# ---------------------------------------------------------------------------
# 18. RENEWABLE GENERATION CAPACITY  [ASSUMPTION]
# ---------------------------------------------------------------------------
# Renewable contracted capacity per cluster (MW)
CLUSTER_SOLAR_CAPACITY_MW = {
    "Delhi NCR":   80,
    "Mumbai MMR":  60,
    "Bengaluru":   75,
    "Hyderabad":   65,
    "Chennai":     55,
}
CLUSTER_WIND_CAPACITY_MW = {
    "Delhi NCR":   40,
    "Mumbai MMR":  35,
    "Bengaluru":   30,
    "Hyderabad":   35,
    "Chennai":     55,  # Higher wind resource on coast
}

# ---------------------------------------------------------------------------
# 19. FINANCIAL PARAMETERS  [ASSUMPTION]
# ---------------------------------------------------------------------------
IMPLEMENTATION_COST_INR_CR = 150.0  # Capex for GridMind AI system (crore)
DISCOUNT_RATE = 0.12                 # Hurdle rate for NPV

print("[config] assumptions.py loaded successfully.")
