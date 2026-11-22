"""
GridMind AI — MILP Energy Dispatch Optimizer (Revised)
=======================================================
Critical A fix: Per-tier served-demand variables replace the bilinear
demand_served × tier_binary product in the original formulation.

Correct linear formulation:
  D[c,t] = sum_k  D_tier[c,t,k]          (total served demand)
  D_tier[c,t,k] <= BaseDemand[c,t] * DR[k] * x[c,t,k]   (linear: const × binary)
  Revenue[c,t]  =  sum_k Tariff_k * D_tier[c,t,k]        (linear: const × continuous)

Critical B fix: bess_dis is defined as the AC-side output power delivered to the
local bus. Conventions used consistently throughout:
  - bess_ch  = AC power drawn from the bus for charging [MW]
  - bess_dis = AC power delivered from the BESS to the bus [MW]
  SoC update:
    SoC[t+1] = SoC[t] + eta_c * bess_ch * dt - bess_dis/eta_d * dt - self_dis
  Energy balance supply side: ... + bess_dis[c,t] = ...
  (bess_dis is already net AC output; eta_d is accounted for in SoC equation)
"""

import numpy as np
import pulp
import os
import sys
from typing import Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.assumptions import (
    CLUSTERS, N_CLUSTERS, DELTA_T,
    CLUSTER_BESS_MWH, CLUSTER_SWAP_PACKS, CLUSTER_CAPACITY_WEIGHTS,
    BESS_CHARGE_EFFICIENCY, BESS_DISCHARGE_EFFICIENCY,
    BESS_MIN_SOC_FRACTION, BESS_MAX_SOC_FRACTION,
    BESS_SELF_DISCHARGE_HOURLY, BESS_DEGRADATION_COST_INR_KWH,
    PACK_CAPACITY_KWH, PACK_CHARGE_EFFICIENCY, PACK_DEGRAD_COST_INR_KWH,
    PACK_RESERVE_FRACTION, SYSTEM_RESERVE_FRACTION,
    RENEWABLE_CONTRACT_TARIFF, DISCOM_TARIFF, IEX_DAM_BASE_PRICE,
    IEX_RTM_PREMIUM, DISTRIBUTION_LOSS_FRACTION, CHARGER_CONVERSION_LOSS,
    PRICE_TIERS, TIER_DEMAND_RESPONSE, BASE_CHARGING_TARIFF_INR_KWH,
    DC_FAST_CHARGING_CAPACITY_MW, BATTERY_SWAP_CAPACITY_MW,
)

UNSERVED_PENALTY   = 25.0   # INR/kWh
CURTAILMENT_PENALTY = 0.5   # INR/kWh
TIER_NAMES = list(PRICE_TIERS.keys())   # ['discount', 'standard', 'premium']
N_TIERS = len(TIER_NAMES)


def _v(prob, name, lb=0.0, ub=None, cat="Continuous"):
    """Create a PuLP 4.x variable via prob.add_variable()."""
    return prob.add_variable(name, lowBound=lb, upBound=ub, cat=cat)


def solve_dispatch(
    horizon: int = 24,
    demand_mw: np.ndarray = None,
    renewable_mw: np.ndarray = None,
    dam_committed_mwh: np.ndarray = None,
    iex_price: np.ndarray = None,
    bess_soc_init: np.ndarray = None,
    charged_packs_init: np.ndarray = None,
    bess_degrad_cost: float = BESS_DEGRADATION_COST_INR_KWH,
    discom_tariff: float = DISCOM_TARIFF,
    verbose: bool = False,
) -> Dict:
    """
    Solve the 24-hour MILP dispatch for all 5 clusters.

    Variable naming convention (Critical B):
      bess_ch[c,t]  : AC power drawn from local bus for BESS charging [MW]
      bess_dis[c,t] : AC power delivered by BESS to local bus [MW]

    SoC transition:
      SoC[t+1] = SoC[t]  +  eta_c * bess_ch * dt
                           -  (bess_dis / eta_d) * dt
                           -  self_discharge

    Energy balance (supply = demand):
      ren_used + dam_delivered + discom_swing + rtm_buy + bess_dis
        = demand_served + bess_ch + pack_charge_energy + hw_losses + unserved

    Revenue linearisation (Critical A):
      Introduce D_tier[c,t,k] = energy served to customers under tier k.
      D_tier[c,t,k] <= BaseDemand[c,t] * DR[k] * x[c,t,k]   <- constant × binary = LINEAR
      demand_served[c,t] = sum_k D_tier[c,t,k]
      Revenue[c,t] = sum_k Tariff_k * D_tier[c,t,k]          <- constant × continuous = LINEAR
    """
    # --- Defaults ---
    if demand_mw is None:
        demand_mw = np.array([
            [DC_FAST_CHARGING_CAPACITY_MW * CLUSTER_CAPACITY_WEIGHTS[c] * 0.65] * horizon
            for c in CLUSTERS
        ])
    if renewable_mw is None:
        renewable_mw = np.zeros((N_CLUSTERS, horizon))
    if dam_committed_mwh is None:
        dam_committed_mwh = demand_mw * 0.40
    if iex_price is None:
        iex_price = np.full(horizon, IEX_DAM_BASE_PRICE)
    if bess_soc_init is None:
        bess_soc_init = np.array([CLUSTER_BESS_MWH[c] * 0.5 for c in CLUSTERS])
    if charged_packs_init is None:
        charged_packs_init = np.array([int(CLUSTER_SWAP_PACKS[c] * 0.6) for c in CLUSTERS])

    iex_price_mat = np.tile(iex_price, (N_CLUSTERS, 1)) if iex_price.ndim == 1 else iex_price
    C, T = N_CLUSTERS, horizon

    prob = pulp.LpProblem("GridMind_Dispatch", pulp.LpMinimize)

    # ---- BESS variables ----
    bess_ch   = [[_v(prob, f"bch_{c}_{t}")            for t in range(T)] for c in range(C)]
    bess_dis  = [[_v(prob, f"bdis_{c}_{t}")           for t in range(T)] for c in range(C)]
    bess_soc  = [[_v(prob, f"soc_{c}_{t}")            for t in range(T + 1)] for c in range(C)]
    bess_mode = [[_v(prob, f"bmode_{c}_{t}", cat="Binary") for t in range(T)] for c in range(C)]

    # ---- Procurement variables ----
    ren_used    = [[_v(prob, f"ren_{c}_{t}")  for t in range(T)] for c in range(C)]
    discom_sw   = [[_v(prob, f"dis_{c}_{t}")  for t in range(T)] for c in range(C)]
    rtm_buy     = [[_v(prob, f"rtm_{c}_{t}")  for t in range(T)] for c in range(C)]
    unserved    = [[_v(prob, f"uns_{c}_{t}")  for t in range(T)] for c in range(C)]
    ren_curtail = [[_v(prob, f"curt_{c}_{t}") for t in range(T)] for c in range(C)]

    # ---- Pricing tier variables (Critical A: per-tier demand) ----
    tier_sel  = [[[_v(prob, f"tier_{c}_{t}_{k}", cat="Binary")
                   for k in range(N_TIERS)] for t in range(T)] for c in range(C)]
    # D_tier[c][t][k] = energy served under tier k (MW)
    # This replaces the bilinear demand_served * x product.
    d_tier = [[[_v(prob, f"dt_{c}_{t}_{k}") for k in range(N_TIERS)]
               for t in range(T)] for c in range(C)]

    # ---- Swap pack variables (continuous relaxation) ----
    cpacks   = [[_v(prob, f"cp_{c}_{t}") for t in range(T + 1)] for c in range(C)]
    pissued  = [[_v(prob, f"pi_{c}_{t}") for t in range(T)] for c in range(C)]
    pcharged = [[_v(prob, f"pc_{c}_{t}") for t in range(T)] for c in range(C)]

    # ---- OBJECTIVE (fully linear) ----
    cost_terms = []
    rev_terms  = []
    for c in range(C):
        for t in range(T):
            cost_terms += [
                ren_used[c][t]    * RENEWABLE_CONTRACT_TARIFF * 1000,
                discom_sw[c][t]   * discom_tariff * 1000,
                rtm_buy[c][t]     * (iex_price_mat[c, t] + IEX_RTM_PREMIUM) * 1000,
                (bess_ch[c][t] + bess_dis[c][t]) * bess_degrad_cost * 1000,
                pcharged[c][t]    * PACK_CAPACITY_KWH * PACK_DEGRAD_COST_INR_KWH,
                unserved[c][t]    * UNSERVED_PENALTY * 1000,
                ren_curtail[c][t] * CURTAILMENT_PENALTY * 1000,
            ]
            # Revenue: Tariff_k (constant) × D_tier[c,t,k] (continuous) → LINEAR
            for k, tier_name in enumerate(TIER_NAMES):
                tariff_k = BASE_CHARGING_TARIFF_INR_KWH * (1.0 + PRICE_TIERS[tier_name])
                rev_terms.append(d_tier[c][t][k] * tariff_k * 1000)

    prob += pulp.lpSum(cost_terms) - pulp.lpSum(rev_terms)

    # ---- CONSTRAINTS ----
    for c in range(C):
        cap_mwh    = CLUSTER_BESS_MWH[CLUSTERS[c]]
        n_packs    = CLUSTER_SWAP_PACKS[CLUSTERS[c]]
        dc_cap_mw  = DC_FAST_CHARGING_CAPACITY_MW * CLUSTER_CAPACITY_WEIGHTS[CLUSTERS[c]]
        sw_cap_mw  = BATTERY_SWAP_CAPACITY_MW * CLUSTER_CAPACITY_WEIGHTS[CLUSTERS[c]]
        p_max      = cap_mwh * 0.5          # 0.5C rate limit
        pack_pwr   = sw_cap_mw * 0.3        # max pack charging power

        prob += bess_soc[c][0] == float(bess_soc_init[c])
        prob += cpacks[c][0]   == float(charged_packs_init[c])

        for t in range(T):
            # --- BESS state transition (Critical B: bess_dis = AC output) ---
            # SoC increases by eta_c * bess_ch (AC→battery conversion loss inside battery)
            # SoC decreases by bess_dis/eta_d (battery must supply more than what exits as AC)
            self_dis = bess_soc[c][t] * BESS_SELF_DISCHARGE_HOURLY
            prob += (bess_soc[c][t + 1] ==
                     bess_soc[c][t]
                     + BESS_CHARGE_EFFICIENCY * bess_ch[c][t] * DELTA_T
                     - bess_dis[c][t] / BESS_DISCHARGE_EFFICIENCY * DELTA_T
                     - self_dis)
            prob += bess_soc[c][t + 1] >= cap_mwh * BESS_MIN_SOC_FRACTION
            prob += bess_soc[c][t + 1] <= cap_mwh * BESS_MAX_SOC_FRACTION
            prob += bess_soc[c][t + 1] >= cap_mwh * SYSTEM_RESERVE_FRACTION  # reserve

            # Binary BESS mode: 1 = charging, 0 = discharging
            prob += bess_ch[c][t]  <= p_max * bess_mode[c][t]
            prob += bess_dis[c][t] <= p_max * (1 - bess_mode[c][t])

            # Renewable balance
            prob += ren_used[c][t] + ren_curtail[c][t] == float(renewable_mw[c, t])

            # Procurement limits
            prob += discom_sw[c][t] <= dc_cap_mw * 0.30
            prob += rtm_buy[c][t]   <= dc_cap_mw * 0.10

            # --- Pricing tier constraints (Critical A) ---
            prob += pulp.lpSum(tier_sel[c][t]) == 1   # exactly one tier active

            for k, tier_name in enumerate(TIER_NAMES):
                dr = TIER_DEMAND_RESPONSE[tier_name]
                cap_k = float(demand_mw[c, t]) * dr   # max demand under this tier (constant)
                # d_tier[k] > 0 only if tier k is selected; enforced by upper bound:
                # d_tier[k] <= cap_k * x[k]  (constant × binary → LINEAR)
                prob += d_tier[c][t][k] <= cap_k * tier_sel[c][t][k]
                prob += d_tier[c][t][k] >= 0

            # Total served demand = sum of tier demands
            demand_served_expr = pulp.lpSum(d_tier[c][t])
            prob += demand_served_expr <= dc_cap_mw

            # --- Energy balance (Critical B: bess_dis is AC output directly) ---
            dam_del   = float(dam_committed_mwh[c, t])
            hw_loss   = demand_served_expr * (CHARGER_CONVERSION_LOSS + DISTRIBUTION_LOSS_FRACTION)
            pack_e    = pcharged[c][t] * PACK_CAPACITY_KWH / 1000.0 / PACK_CHARGE_EFFICIENCY

            prob += (ren_used[c][t] + dam_del + discom_sw[c][t] + rtm_buy[c][t] + bess_dis[c][t]
                     == demand_served_expr + bess_ch[c][t] + pack_e + hw_loss + unserved[c][t])

            # --- Swap inventory ---
            prob += cpacks[c][t + 1] == cpacks[c][t] + pcharged[c][t] - pissued[c][t]
            prob += pissued[c][t]  <= cpacks[c][t]
            swap_dem = float(demand_mw[c, t]) / (PACK_CAPACITY_KWH / 1000.0) * 0.4
            prob += pissued[c][t]  <= swap_dem
            prob += cpacks[c][t + 1] >= float(n_packs) * PACK_RESERVE_FRACTION
            prob += pcharged[c][t] * PACK_CAPACITY_KWH / 1000.0 <= pack_pwr
            prob += cpacks[c][t]   <= float(n_packs)

    # ---- SOLVE ----
    solver = pulp.HiGHS(msg=1 if verbose else 0, timeLimit=120)
    stats  = prob.solve(solver)

    solve_status = str(stats).split("\n")[1] if stats else "Unknown"

    def val(v):
        try:
            return pulp.value(v) or 0.0
        except Exception:
            return 0.0

    # Reconstruct demand_served from per-tier components
    demand_served_arr = np.array([
        [sum(val(d_tier[c][t][k]) for k in range(N_TIERS)) for t in range(T)]
        for c in range(C)
    ])
    tier_selected_arr = np.array([
        [np.argmax([val(tier_sel[c][t][k]) for k in range(N_TIERS)]) for t in range(T)]
        for c in range(C)
    ])
    # Revenue from linearised per-tier formulation
    revenue_arr = np.array([
        [sum(val(d_tier[c][t][k]) * BASE_CHARGING_TARIFF_INR_KWH *
             (1.0 + PRICE_TIERS[TIER_NAMES[k]]) * 1000
             for k in range(N_TIERS))
         for t in range(T)]
        for c in range(C)
    ])

    return {
        "status":              solve_status,
        "objective":           val(prob.objective),
        "demand_served_mw":    demand_served_arr,
        "bess_soc_mwh":        np.array([[val(bess_soc[c][t])  for t in range(T + 1)] for c in range(C)]),
        "bess_charge_mw":      np.array([[val(bess_ch[c][t])   for t in range(T)] for c in range(C)]),
        "bess_discharge_mw":   np.array([[val(bess_dis[c][t])  for t in range(T)] for c in range(C)]),
        "renewable_used_mw":   np.array([[val(ren_used[c][t])  for t in range(T)] for c in range(C)]),
        "discom_swing_mw":     np.array([[val(discom_sw[c][t]) for t in range(T)] for c in range(C)]),
        "rtm_buy_mw":          np.array([[val(rtm_buy[c][t])   for t in range(T)] for c in range(C)]),
        "unserved_mw":         np.array([[val(unserved[c][t])  for t in range(T)] for c in range(C)]),
        "renewable_curtail_mw":np.array([[val(ren_curtail[c][t]) for t in range(T)] for c in range(C)]),
        "packs_issued":        np.array([[int(val(pissued[c][t]))  for t in range(T)] for c in range(C)]),
        "packs_charged":       np.array([[int(val(pcharged[c][t])) for t in range(T)] for c in range(C)]),
        "tier_selected":       tier_selected_arr,
        "revenue_arr":         revenue_arr,
        "demand_mw_input":     demand_mw,
        "renewable_mw_input":  renewable_mw,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    demand = rng.uniform(50, 150, (N_CLUSTERS, 24))
    renewable = rng.uniform(10, 80, (N_CLUSTERS, 24))
    renewable[:, :6] = 0; renewable[:, 20:] = 0
    r = solve_dispatch(demand_mw=demand, renewable_mw=renewable)
    print(f"Status : {r['status']}")
    print(f"Revenue: INR {r['revenue_arr'].sum():,.0f}")
    print(f"Unserved: {r['unserved_mw'].sum():.4f} MWh")
