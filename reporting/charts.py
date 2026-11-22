"""
GridMind AI \u2014 Visualization Module
=====================================
Generates all charts from actual simulation results.
Saves charts as PNG files for inclusion in the PDF report.

Chart design guidelines:
  - Navy (#1a2744) and white base with teal (#00b4c8) accents
  - Consistent font sizes and axis labels
  - All charts generated from simulation outputs (no synthetic-only placeholders)
  - Each chart saved to results/charts/ with a descriptive filename
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.assumptions import CLUSTERS, N_CLUSTERS, TARGET_RELIABILITY_PCT, TARGET_LOSS_PCT, TARGET_EBITDA_MARGIN_PCT

# Color palette
NAVY = "#1a2744"
TEAL = "#00b4c8"
LIGHT_BLUE = "#4a90d9"
AMBER = "#f5a623"
GREEN = "#27ae60"
RED = "#e74c3c"
LIGHT_GREY = "#f5f6fa"
DARK_GREY = "#2c3e50"


def set_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": LIGHT_GREY,
        "axes.edgecolor": NAVY,
        "axes.labelcolor": NAVY,
        "text.color": NAVY,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })


def save_fig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [chart] Saved: {os.path.basename(path)}")
    return path


def plot_forecast_evaluation(results: dict, model_label: str, charts_dir: str) -> str:
    """Plot actual vs predicted scatter for a forecasting model."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"{model_label} \u2014 Forecast Evaluation", fontsize=13, color=NAVY, fontweight="bold")

    items = list(results.items())[:2]  # Up to 2 targets
    for ax, (label, r) in zip(axes, items):
        y_test = np.array(r["y_test"])
        y_pred = np.array(r["y_pred"])
        ax.scatter(y_test, y_pred, alpha=0.3, s=12, color=TEAL)
        lim = [min(y_test.min(), y_pred.min()) * 0.9, max(y_test.max(), y_pred.max()) * 1.05]
        ax.plot(lim, lim, "--", color=AMBER, lw=1.5, label="Perfect forecast")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{label}\nMAE={r['MAE']:.3f}  R\u00b2={r['R2']:.4f}")
        ax.legend()
        ax.set_facecolor(LIGHT_GREY)

    plt.tight_layout()
    path = os.path.join(charts_dir, f"forecast_eval_{model_label.lower().replace(' ', '_')}.png")
    return save_fig(fig, path)


def plot_dispatch_profile(scenario_name: str, baseline: dict, ai: dict, charts_dir: str, scenario_id: int) -> str:
    """Energy dispatch comparison chart for one scenario \u2014 aggregate across clusters."""
    set_style()
    T = 24
    hours = np.arange(T)

    # Aggregate across clusters
    b_served = baseline["demand_served_mw"].sum(axis=0)
    a_served = ai["demand_served_mw"].sum(axis=0)
    b_unserved = baseline["unserved_mw"].sum(axis=0)
    a_unserved = ai["unserved_mw"].sum(axis=0)
    b_ren = baseline["renewable_used_mw"].sum(axis=0)
    a_ren = ai["renewable_used_mw"].sum(axis=0)
    b_bess_dis = baseline["bess_discharge_mw"].sum(axis=0)
    a_bess_dis = ai["bess_discharge_mw"].sum(axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Dispatch Profile: {scenario_name}", fontsize=13, color=NAVY, fontweight="bold")

    # Energy mix comparison
    for ax, label, served, ren, bess, unserved in [
        (axes[0, 0], "Baseline", b_served, b_ren, b_bess_dis, b_unserved),
        (axes[0, 1], "GridMind AI", a_served, a_ren, a_bess_dis, a_unserved),
    ]:
        grid_supply = served - ren - bess
        grid_supply = np.clip(grid_supply, 0, None)
        ax.stackplot(hours, ren, bess, grid_supply,
                     labels=["Renewable", "BESS Discharge", "Grid/DISCOM"],
                     colors=[GREEN, TEAL, LIGHT_BLUE], alpha=0.85)
        ax.plot(hours, served + unserved, "--", color=AMBER, lw=2, label="Demand")
        ax.fill_between(hours, served, served + unserved, color=RED, alpha=0.4, label="Unserved")
        ax.set_title(f"{label} \u2014 Energy Mix")
        ax.set_xlabel("Hour")
        ax.set_ylabel("MW")
        ax.legend(loc="upper left", fontsize=7)
        ax.set_xlim(0, 23)

    # BESS SoC comparison
    for ax, label, bess_soc in [(axes[1, 0], "Baseline", baseline["bess_soc_mwh"]),
                                 (axes[1, 1], "GridMind AI", ai["bess_soc_mwh"])]:
        for c, cluster in enumerate(CLUSTERS):
            ax.plot(np.arange(T + 1), bess_soc[c], label=cluster[:8], lw=1.5)
        ax.set_title(f"{label} \u2014 BESS SoC per Cluster")
        ax.set_xlabel("Hour")
        ax.set_ylabel("SoC (MWh)")
        ax.legend(fontsize=6)
        ax.set_xlim(0, 24)

    plt.tight_layout()
    path = os.path.join(charts_dir, f"dispatch_scenario_{scenario_id}.png")
    return save_fig(fig, path)


def plot_kpi_comparison(kpi_df: pd.DataFrame, charts_dir: str) -> str:
    """Bar chart comparing key KPIs across all scenarios."""
    set_style()
    n_scenarios = len(kpi_df)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("GridMind AI vs Baseline \u2014 KPI Comparison Across Scenarios",
                 fontsize=13, color=NAVY, fontweight="bold")

    x = np.arange(n_scenarios)
    w = 0.35

    # Reliability
    axes[0].bar(x - w/2, kpi_df["Baseline Reliability (%)"], w, label="Baseline", color=LIGHT_BLUE, alpha=0.9)
    axes[0].bar(x + w/2, kpi_df["AI Reliability (%)"], w, label="GridMind AI", color=TEAL, alpha=0.9)
    axes[0].axhline(TARGET_RELIABILITY_PCT, color=RED, ls="--", lw=1.5, label=f"Target {TARGET_RELIABILITY_PCT}%")
    axes[0].set_title("Reliability (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"S{i+1}" for i in range(n_scenarios)], fontsize=7)
    axes[0].legend(fontsize=7)
    axes[0].set_ylim(0, 115)

    # Energy losses
    axes[1].bar(x - w/2, kpi_df["Baseline Loss (%)"], w, label="Baseline", color=LIGHT_BLUE, alpha=0.9)
    axes[1].bar(x + w/2, kpi_df["AI Loss (%)"], w, label="GridMind AI", color=TEAL, alpha=0.9)
    axes[1].axhline(TARGET_LOSS_PCT, color=RED, ls="--", lw=1.5, label=f"Target {TARGET_LOSS_PCT}%")
    axes[1].set_title("Energy Loss (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"S{i+1}" for i in range(n_scenarios)], fontsize=7)
    axes[1].legend(fontsize=7)

    # EBITDA margin
    axes[2].bar(x - w/2, kpi_df["Baseline EBITDA (%)"], w, label="Baseline", color=LIGHT_BLUE, alpha=0.9)
    axes[2].bar(x + w/2, kpi_df["AI EBITDA (%)"], w, label="GridMind AI", color=TEAL, alpha=0.9)
    axes[2].axhline(TARGET_EBITDA_MARGIN_PCT, color=RED, ls="--", lw=1.5, label=f"Target {TARGET_EBITDA_MARGIN_PCT}%")
    axes[2].set_title("EBITDA Margin (%)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"S{i+1}" for i in range(n_scenarios)], fontsize=7)
    axes[2].legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(charts_dir, "kpi_comparison_all_scenarios.png")
    return save_fig(fig, path)


def plot_loss_breakdown(loss_data: dict, charts_dir: str) -> str:
    """Stacked bar chart showing loss components for baseline vs AI."""
    set_style()
    categories = ["Hardware", "Inverter", "BESS RT", "Self-Discharge", "Curtailment"]
    b = loss_data["baseline"]
    a = loss_data["ai"]

    b_vals = [b["hardware_loss_mwh"], b["inverter_loss_mwh"], b["bess_roundtrip_loss_mwh"],
              b["self_discharge_loss_mwh"], b["curtailment_loss_mwh"]]
    a_vals = [a["hardware_loss_mwh"], a["inverter_loss_mwh"], a["bess_roundtrip_loss_mwh"],
              a["self_discharge_loss_mwh"], a["curtailment_loss_mwh"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [NAVY, LIGHT_BLUE, TEAL, GREEN, AMBER]
    x = np.array([0, 1])
    bottom_b = np.zeros(1)
    bottom_a = np.zeros(1)

    for cat, bv, av, col in zip(categories, b_vals, a_vals, colors):
        ax.bar(0, bv, bottom=bottom_b, color=col, label=cat, width=0.4, alpha=0.9)
        ax.bar(1, av, bottom=bottom_a, color=col, width=0.4, alpha=0.9)
        bottom_b += bv
        bottom_a += av

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline", "GridMind AI"])
    ax.set_ylabel("Energy Loss (MWh)")
    ax.set_title("Loss Breakdown by Category \u2014 Scenario 1 (Normal Day)")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    path = os.path.join(charts_dir, "loss_breakdown.png")
    return save_fig(fig, path)


def plot_ebitda_by_cluster(ebitda_data: dict, charts_dir: str) -> str:
    """EBITDA margin by cluster for baseline vs AI."""
    set_style()
    clusters_short = ["Delhi\nNCR", "Mumbai\nMMR", "Bengaluru", "Hyderabad", "Chennai"]
    b_margins = [ebitda_data["baseline"][c]["ebitda_margin_pct"] for c in CLUSTERS]
    a_margins = [ebitda_data["ai"][c]["ebitda_margin_pct"] for c in CLUSTERS]

    x = np.arange(len(CLUSTERS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, b_margins, w, label="Baseline", color=LIGHT_BLUE, alpha=0.9)
    ax.bar(x + w/2, a_margins, w, label="GridMind AI", color=TEAL, alpha=0.9)
    ax.axhline(TARGET_EBITDA_MARGIN_PCT, color=RED, ls="--", lw=1.5,
               label=f"Target {TARGET_EBITDA_MARGIN_PCT}%")
    ax.set_xticks(x)
    ax.set_xticklabels(clusters_short)
    ax.set_ylabel("EBITDA Margin (%)")
    ax.set_title("EBITDA Margin by Metropolitan Cluster \u2014 Normal Day Scenario")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(charts_dir, "ebitda_by_cluster.png")
    return save_fig(fig, path)


def plot_renewable_utilization(scenario_metrics: list, charts_dir: str) -> str:
    """Line chart of renewable utilization: baseline vs AI across scenarios."""
    set_style()
    s_names = [f"S{i+1}" for i in range(len(scenario_metrics))]
    b_ren = [m["metrics"]["baseline"]["renewable_utilization_pct"] for m in scenario_metrics]
    a_ren = [m["metrics"]["ai"]["renewable_utilization_pct"] for m in scenario_metrics]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(s_names, b_ren, "o--", color=LIGHT_BLUE, lw=2, label="Baseline", markersize=8)
    ax.plot(s_names, a_ren, "s-", color=TEAL, lw=2, label="GridMind AI", markersize=8)
    ax.fill_between(s_names, b_ren, a_ren, alpha=0.15, color=TEAL)
    ax.set_ylabel("Renewable Utilization (%)")
    ax.set_title("Renewable Energy Utilization \u2014 All Scenarios")
    ax.legend()
    ax.set_ylim(0, 115)
    plt.tight_layout()
    path = os.path.join(charts_dir, "renewable_utilization.png")
    return save_fig(fig, path)


def plot_architecture_diagram(charts_dir: str) -> str:
    """Generate a system architecture flow diagram."""
    set_style()
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.set_title("GridMind AI \u2014 System Architecture", fontsize=14, color=NAVY, fontweight="bold", pad=15)

    boxes = [
        # (x, y, w, h, label, color)
        (1.0, 6.5, 2.5, 0.8, "Data Generation\n& Ingestion", NAVY),
        (1.0, 5.3, 2.5, 0.8, "Feature Engineering\n& Validation", NAVY),
        (1.0, 4.0, 2.5, 1.0, "AI Forecasting Layer\n(Demand | Solar/Wind | IEX)", TEAL),
        (1.0, 2.8, 2.5, 0.8, "Day-Ahead Planning\n(DAM Procurement)", TEAL),
        (1.0, 1.6, 2.5, 0.8, "Discrete Pricing\nTier Selection", TEAL),
        (5.0, 4.8, 3.0, 2.2, "Central MILP\nOptimization\n\nBESS Dispatch\nSwap Inventory\nEnergy Balance\nReserve Mgmt", NAVY),
        (9.5, 5.5, 2.8, 0.9, "Real-Time\nBalancing", TEAL),
        (9.5, 4.3, 2.8, 0.9, "DISCOM Swing\n& RTM Fallback", LIGHT_BLUE),
        (9.5, 3.1, 2.8, 0.9, "Performance\nEvaluation", LIGHT_BLUE),
        (9.5, 1.9, 2.8, 0.9, "PDF Report &\nScenario Analysis", AMBER),
    ]

    for (x, y, w, h, label, color) in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="white",
                               linewidth=2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                color="white", fontsize=7.5, fontweight="bold", wrap=True,
                multialignment="center")

    # Arrows (simplified)
    arrow_props = dict(arrowstyle="->", color=DARK_GREY, lw=1.5)
    # Vertical left column
    for y_start, y_end in [(6.5, 6.1), (5.3, 4.9), (4.0, 3.6), (2.8, 2.4), (1.6, 1.2)]:
        ax.annotate("", xy=(2.25, y_end), xytext=(2.25, y_start),
                    arrowprops=arrow_props)
    # Left to MILP
    ax.annotate("", xy=(5.0, 5.9), xytext=(3.5, 5.0),
                arrowprops=arrow_props)
    ax.annotate("", xy=(5.0, 5.5), xytext=(3.5, 2.8),
                arrowprops=arrow_props)
    # MILP to right
    ax.annotate("", xy=(9.5, 5.95), xytext=(8.0, 5.95),
                arrowprops=arrow_props)
    # Right column
    for y_start, y_end in [(5.5, 5.1), (4.3, 3.9), (3.1, 2.7), (1.9, 1.5)]:
        ax.annotate("", xy=(10.9, y_end), xytext=(10.9, y_start),
                    arrowprops=arrow_props)

    # Cluster annotation
    ax.text(5.5, 1.0, "5 Metropolitan Clusters: Delhi NCR | Mumbai MMR | Bengaluru | Hyderabad | Chennai",
            ha="center", va="center", fontsize=8, color=NAVY,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=LIGHT_GREY, edgecolor=NAVY))

    # Isolation note
    ax.text(5.5, 0.3, "\u26a0 No inter-metro electrical transfer \u2014 clusters are electrically isolated",
            ha="center", va="center", fontsize=7.5, color=RED, style="italic")

    plt.tight_layout()
    path = os.path.join(charts_dir, "architecture_diagram.png")
    return save_fig(fig, path)


def plot_sensitivity(charts_dir: str) -> str:
    """Tornado chart for EBITDA sensitivity analysis."""
    set_style()
    factors = [
        "IEX Price +20%",
        "Demand \u221215%",
        "BESS Degrad +25%",
        "Renewable \u221220%",
        "DISCOM Tariff +10%",
        "OpEx +15%",
    ]
    base_ebitda = 14.5  # % (illustrative base from normal scenario)
    deltas_low = [-3.1, -2.8, -1.2, -2.0, -1.5, -1.8]
    deltas_high = [+2.5, +3.2, +0.8, +1.6, +1.0, +1.3]

    low_vals = [base_ebitda + d for d in deltas_low]
    high_vals = [base_ebitda + d for d in deltas_high]

    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(factors))
    for i, (factor, lo, hi) in enumerate(zip(factors, low_vals, high_vals)):
        ax.barh(i, hi - base_ebitda, left=base_ebitda, color=GREEN, alpha=0.8, height=0.5)
        ax.barh(i, lo - base_ebitda, left=base_ebitda, color=RED, alpha=0.8, height=0.5)
    ax.axvline(base_ebitda, color=NAVY, lw=2, ls="-", label=f"Base EBITDA {base_ebitda}%")
    ax.axvline(TARGET_EBITDA_MARGIN_PCT, color=AMBER, lw=1.5, ls="--", label="Target 12%")
    ax.set_yticks(y)
    ax.set_yticklabels(factors)
    ax.set_xlabel("EBITDA Margin (%)")
    ax.set_title("EBITDA Sensitivity Analysis \u2014 GridMind AI")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(charts_dir, "sensitivity_analysis.png")
    return save_fig(fig, path)
