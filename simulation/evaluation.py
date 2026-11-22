"""
GridMind AI — Performance Evaluation Engine (Revised)
======================================================
Critical C fix: loss categories are now strictly separated.

PHYSICAL LOSSES (included in loss_pct):
  1. Charger conversion loss  : fraction of energy delivered to EV
  2. Distribution loss        : LV cable and transformer losses
  3. Inverter / power electronics loss : AC↔DC conversion for BESS
  4. BESS round-trip loss     : energy left inside battery during charge/discharge
  5. Self-discharge           : standing BESS energy loss per hour

NOT INCLUDED in physical loss_pct (reported separately):
  6. Renewable curtailment    : available generation not consumed; not a physical
                                 conversion loss — it represents a planning gap.
  7. Unserved demand          : demand not met; also reported separately.

Loss denominator:
  Total electrical input = renewable_used + dam_delivered + discom + rtm + bess_dis
  (all AC-side supply into the local bus)

This denominator is consistent with the energy balance equation in the MILP.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.assumptions import (
    CLUSTERS, N_CLUSTERS, CLUSTER_CAPACITY_WEIGHTS,
    DC_FAST_CHARGING_CAPACITY_MW, PACK_CAPACITY_KWH,
    DISTRIBUTION_LOSS_FRACTION, CHARGER_CONVERSION_LOSS,
    INVERTER_LOSS_FRACTION, BESS_ROUND_TRIP_EFFICIENCY,
    BESS_CHARGE_EFFICIENCY, BESS_DISCHARGE_EFFICIENCY,
    BESS_SELF_DISCHARGE_HOURLY, HOURLY_OPEX_INR,
    BASE_CHARGING_TARIFF_INR_KWH, BASE_SWAP_FEE_INR_PACK,
    TARGET_RELIABILITY_PCT, TARGET_LOSS_PCT, TARGET_EBITDA_MARGIN_PCT,
    EXISTING_RELIABILITY_PCT, EXISTING_LOSS_PCT,
)

# Physical loss floor: minimum unavoidable losses given hardware specs
# Charger conversion: 4% of energy delivered to EV
# Distribution:       2.5% of energy delivered
# Inverter (BESS):    2.5% of BESS throughput
# Total floor estimate (approximation at 50% BESS utilisation):
_HARDWARE_FLOOR_PCT = (CHARGER_CONVERSION_LOSS + DISTRIBUTION_LOSS_FRACTION) * 100  # ~6.5%


def compute_reliability(demand_served_mw: np.ndarray,
                        unserved_mw: np.ndarray,
                        packs_issued: np.ndarray,
                        total_demand_mw: np.ndarray) -> dict:
    """
    Reliability = (Energy served by DC chargers + Energy-equivalent of swaps)
                  / Total forecast customer demand × 100
    Shape of all arrays: (N_CLUSTERS, T)
    """
    pack_energy_mwh = packs_issued * PACK_CAPACITY_KWH / 1000.0
    total_delivered = demand_served_mw.sum() + pack_energy_mwh.sum()
    total_demand    = total_demand_mw.sum()
    aggregate_pct   = total_delivered / max(total_demand, 1.0) * 100

    realized_demand = demand_served_mw + unserved_mw
    supplementary_pct = demand_served_mw.sum() / max(realized_demand.sum(), 1.0) * 100

    per_cluster = {}
    for c, cluster in enumerate(CLUSTERS):
        c_del = demand_served_mw[c].sum() + pack_energy_mwh[c].sum()
        per_cluster[cluster] = c_del / max(total_demand_mw[c].sum(), 1.0) * 100

    return {
        "aggregate_pct":        aggregate_pct,
        "supplementary_pct":    supplementary_pct,
        "per_cluster_pct":      per_cluster,
        "total_delivered_mwh":  total_delivered,
        "total_demand_mwh":     total_demand,
        "total_unserved_mwh":   unserved_mw.sum(),
        "swap_energy_mwh":      pack_energy_mwh.sum(),
    }


def compute_losses(demand_served_mw: np.ndarray,
                   bess_charge_mw: np.ndarray,
                   bess_discharge_mw: np.ndarray,
                   bess_soc_mwh: np.ndarray,
                   renewable_curtail_mw: np.ndarray,
                   total_supply_mwh: float) -> dict:
    """
    Physical energy loss breakdown.

    Denominator: total_supply_mwh = all AC-side electrical input to the local bus
    (renewable used + DAM delivered + DISCOM + RTM + BESS discharge).
    This is consistent with the MILP energy balance.

    Curtailment is EXCLUDED from physical loss_pct and reported separately.
    It represents unused available generation, not a conversion loss.
    """
    T = demand_served_mw.shape[1]

    # 1. Charger conversion + distribution: fraction of energy reaching customers
    hw_loss_mwh = demand_served_mw.sum() * (CHARGER_CONVERSION_LOSS + DISTRIBUTION_LOSS_FRACTION)

    # 2. Inverter losses on BESS charge and discharge cycles
    inverter_loss_mwh = (bess_charge_mw.sum() + bess_discharge_mw.sum()) * INVERTER_LOSS_FRACTION

    # 3. BESS round-trip losses
    # Charge losses: energy entering battery = eta_c * bess_ch; loss = (1-eta_c) * bess_ch
    # Discharge losses: AC output = eta_d * bess_dis_battery; we track AC output bess_dis,
    #   so battery drew bess_dis/eta_d. Loss = bess_dis*(1/eta_d - 1)
    charge_loss_mwh    = bess_charge_mw.sum() * (1 - BESS_CHARGE_EFFICIENCY)
    discharge_loss_mwh = bess_discharge_mw.sum() * (1 / BESS_DISCHARGE_EFFICIENCY - 1)

    # 4. Self-discharge: estimated from mean SoC × hourly rate × hours
    self_dis_mwh = bess_soc_mwh[:, :-1].mean(axis=1).sum() * BESS_SELF_DISCHARGE_HOURLY * T

    total_physical_loss_mwh = (hw_loss_mwh + inverter_loss_mwh +
                                charge_loss_mwh + discharge_loss_mwh + self_dis_mwh)

    physical_loss_pct = (total_physical_loss_mwh / max(total_supply_mwh, 1.0)) * 100

    # Curtailment reported separately — not physical loss
    curtailment_mwh = renewable_curtail_mw.sum()

    return {
        # Physical losses (used for target comparison against 8%)
        "physical_loss_mwh":        total_physical_loss_mwh,
        "loss_pct":                  physical_loss_pct,
        "hw_loss_mwh":               hw_loss_mwh,
        "inverter_loss_mwh":         inverter_loss_mwh,
        "bess_charge_loss_mwh":      charge_loss_mwh,
        "bess_discharge_loss_mwh":   discharge_loss_mwh,
        "self_discharge_loss_mwh":   self_dis_mwh,
        # Curtailment (separate category)
        "curtailment_mwh":           curtailment_mwh,
        "curtailment_pct_of_avail":  curtailment_mwh / max(curtailment_mwh + renewable_curtail_mw.shape[0] * 0.001, 1.0) * 100,
        # Reference
        "total_supply_mwh":          total_supply_mwh,
        "hardware_floor_pct":        _HARDWARE_FLOOR_PCT,
    }


def compute_ebitda(revenue_inr: float,
                   procurement_cost_inr: float,
                   degrad_cost_inr: float,
                   opex_inr: float) -> dict:
    """
    EBITDA = Revenue − Procurement − Degradation − Operating Expenses

    All inputs are for the SAME time window (one 24-hour scenario slice).
    The 24-hour OpEx is a pro-rata share of annual OpEx:
      daily_opex = annual_opex / 365

    EBITDA margin = EBITDA / Revenue × 100

    Note: Interest, Tax, Depreciation/Amortisation are excluded by definition.
    """
    ebitda = revenue_inr - procurement_cost_inr - degrad_cost_inr - opex_inr
    margin = ebitda / max(revenue_inr, 1.0) * 100
    return {
        "revenue_inr":          revenue_inr,
        "procurement_cost_inr": procurement_cost_inr,
        "degrad_cost_inr":      degrad_cost_inr,
        "opex_inr":             opex_inr,
        "ebitda_inr":           ebitda,
        "ebitda_margin_pct":    margin,
        "target_met":           margin >= TARGET_EBITDA_MARGIN_PCT,
    }


def evaluate_scenario(baseline_result: dict, ai_result: dict, scenario: dict) -> dict:
    """
    Full KPI evaluation for one 24-hour scenario.
    Returns nested dict with 'baseline', 'ai', and 'improvement' keys.
    """
    T = 24
    demand_mw = scenario["demand_mw"]

    # Daily OpEx = annual / 365 (applied equally to baseline and AI for fair comparison)
    daily_opex = HOURLY_OPEX_INR * 24

    metrics = {}
    for label, result in [("baseline", baseline_result), ("ai", ai_result)]:
        # Reliability
        rel = compute_reliability(
            demand_served_mw=result["demand_served_mw"],
            unserved_mw=result["unserved_mw"],
            packs_issued=result["packs_issued"],
            total_demand_mw=demand_mw,
        )

        # Total AC-side supply to local bus
        total_supply = (result["renewable_used_mw"].sum()
                        + result.get("dam_used_mw", np.zeros((N_CLUSTERS, T))).sum()
                        + result["discom_swing_mw"].sum()
                        + result["rtm_buy_mw"].sum()
                        + result["bess_discharge_mw"].sum())

        loss = compute_losses(
            demand_served_mw=result["demand_served_mw"],
            bess_charge_mw=result["bess_charge_mw"],
            bess_discharge_mw=result["bess_discharge_mw"],
            bess_soc_mwh=result["bess_soc_mwh"],
            renewable_curtail_mw=result.get("renewable_curtail_mw", np.zeros((N_CLUSTERS, T))),
            total_supply_mwh=max(total_supply, 1.0),
        )

        ebitda = compute_ebitda(
            revenue_inr=result["revenue_inr"],
            procurement_cost_inr=result["procurement_cost_inr"],
            degrad_cost_inr=result["degrad_cost_inr"],
            opex_inr=daily_opex,
        )

        # Renewable utilisation
        total_avail = scenario["renewable_mw"].sum()
        ren_used    = result["renewable_used_mw"].sum()
        ren_util    = ren_used / max(total_avail, 1.0) * 100

        metrics[label] = {
            "reliability":              rel,
            "losses":                   loss,
            "ebitda":                   ebitda,
            "renewable_utilization_pct":ren_util,
            "total_unserved_mwh":       result["unserved_mw"].sum(),
            "bess_throughput_mwh":      (result["bess_charge_mw"] + result["bess_discharge_mw"]).sum(),
            "packs_issued_total":       result["packs_issued"].sum(),
            "procurement_cost_inr":     result["procurement_cost_inr"],
            "revenue_inr":              result["revenue_inr"],
            "curtailment_mwh":          result.get("renewable_curtail_mw", np.zeros((N_CLUSTERS, T))).sum(),
        }

    b, a = metrics["baseline"], metrics["ai"]
    metrics["improvement"] = {
        "reliability_delta_pp":    a["reliability"]["aggregate_pct"] - b["reliability"]["aggregate_pct"],
        "loss_delta_pp":           b["losses"]["loss_pct"] - a["losses"]["loss_pct"],
        "ebitda_delta_pp":         a["ebitda"]["ebitda_margin_pct"] - b["ebitda"]["ebitda_margin_pct"],
        "ren_util_delta_pp":       a["renewable_utilization_pct"] - b["renewable_utilization_pct"],
        "unserved_reduction_mwh":  b["total_unserved_mwh"] - a["total_unserved_mwh"],
        "cost_saving_inr":         b["procurement_cost_inr"] - a["procurement_cost_inr"],
    }
    return metrics


def build_kpi_table(all_scenario_metrics: list) -> pd.DataFrame:
    """Single authoritative KPI table — all values from evaluate_scenario."""
    rows = []
    for entry in all_scenario_metrics:
        m  = entry["metrics"]
        b  = m["baseline"]
        a  = m["ai"]
        imp = m["improvement"]
        rows.append({
            "Scenario":               entry["scenario_name"],
            "B.Reliability(%)":       round(b["reliability"]["aggregate_pct"], 1),
            "AI.Reliability(%)":      round(a["reliability"]["aggregate_pct"], 1),
            "Rel.Delta(pp)":          round(imp["reliability_delta_pp"], 1),
            "B.PhysLoss(%)":          round(b["losses"]["loss_pct"], 1),
            "AI.PhysLoss(%)":         round(a["losses"]["loss_pct"], 1),
            "Loss.Delta(pp)":         round(imp["loss_delta_pp"], 1),
            "B.EBITDA(%)":            round(b["ebitda"]["ebitda_margin_pct"], 1),
            "AI.EBITDA(%)":           round(a["ebitda"]["ebitda_margin_pct"], 1),
            "EBITDA.Delta(pp)":       round(imp["ebitda_delta_pp"], 1),
            "AI.RenUtil(%)":          round(a["renewable_utilization_pct"], 1),
            "AI.Curtail(MWh)":        round(a["curtailment_mwh"], 1),
            "CostSaving(INR Cr)":     round(imp["cost_saving_inr"] / 1e7, 2),
        })
    return pd.DataFrame(rows)
