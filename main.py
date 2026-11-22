"""
GridMind AI \u2014 Main Execution Pipeline
========================================
Runs the complete GridMind AI prototype end-to-end:
  Phase 1: Data generation
  Phase 2: Forecasting model training
  Phase 3: Scenario simulation (baseline + AI strategy)
  Phase 4: Performance evaluation
  Phase 5: Chart generation
  Phase 6: PDF report generation

Run with: python main.py
Estimated runtime: 5\u201315 minutes depending on hardware.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Fix import paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHARTS_DIR = os.path.join(RESULTS_DIR, "charts")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


def phase1_data():
    print("\n" + "="*60)
    print("PHASE 1: DATA GENERATION")
    print("="*60)
    from data.generate_data import generate_full_dataset
    data_path_parquet = os.path.join(RESULTS_DIR, "synthetic_data.parquet")
    data_path_csv = os.path.join(RESULTS_DIR, "synthetic_data.csv")
    if os.path.exists(data_path_parquet):
        print(f"[main] Loading dataset from parquet...")
        df = pd.read_parquet(data_path_parquet)
    elif os.path.exists(data_path_csv):
        print(f"[main] Loading dataset from CSV (parquet not available)...")
        df = pd.read_csv(data_path_csv, parse_dates=["timestamp"])
    else:
        df = generate_full_dataset(save_path=data_path_parquet)
        # Fallback save as CSV if parquet engine missing
        if not os.path.exists(data_path_parquet):
            df.to_csv(data_path_csv, index=False)
            print(f"[main] Saved as CSV to {data_path_csv}")
    print(f"[main] Dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def phase2_forecasting(df):
    print("\n" + "="*60)
    print("PHASE 2: FORECASTING MODEL TRAINING")
    print("="*60)
    from models.demand_model import train_demand_models
    from models.renewable_model import train_renewable_models
    from models.price_model import train_price_model
    from models.degradation_model import fit_degradation_regression, estimate_degradation_cost_bess
    from config.assumptions import BESS_DEGRADATION_COST_INR_KWH

    models_dir = os.path.join(RESULTS_DIR, "models")

    demand_models, demand_results = train_demand_models(df, save_dir=models_dir)
    renewable_models, renewable_results = train_renewable_models(df, save_dir=models_dir)
    price_model, price_results = train_price_model(df, save_dir=models_dir)
    degrad_result = fit_degradation_regression(df)

    bess_degrad_cost = min(estimate_degradation_cost_bess(soh=0.92, dod_avg=0.70, temperature_avg=32),
                           BESS_DEGRADATION_COST_INR_KWH)
    print(f"[main] BESS degradation cost used in MILP: INR {bess_degrad_cost:.4f}/kWh")

    forecast_metrics = {
        "Demand": demand_results,
        "Renewable": renewable_results,
        "Price": {"IEX_DAM": price_results},
    }

    # Save forecast metrics
    rows = []
    for model_name, model_results in forecast_metrics.items():
        for target, r in model_results.items():
            if isinstance(r, dict) and "MAE" in r:
                rows.append({
                    "Model": model_name, "Target": target,
                    "MAE": round(r["MAE"], 4), "RMSE": round(r["RMSE"], 4),
                    "R2": round(r["R2"], 4)
                })
    if rows:
        metrics_df = pd.DataFrame(rows)
        metrics_df.to_csv(os.path.join(METRICS_DIR, "forecast_metrics.csv"), index=False)
        print(f"\n[main] Forecast metrics:\n{metrics_df.to_string(index=False)}")

    return demand_models, renewable_models, price_model, bess_degrad_cost, forecast_metrics


def phase3_simulation(demand_models, renewable_models, price_model, bess_degrad_cost):
    print("\n" + "="*60)
    print("PHASE 3: SCENARIO SIMULATION")
    print("="*60)
    from simulation.scenarios import build_scenario, SCENARIOS
    from simulation.baseline import run_baseline
    from simulation.ai_strategy import run_ai_strategy
    from simulation.evaluation import evaluate_scenario, build_kpi_table

    all_scenario_metrics = []
    sim_results = {}

    for scenario_id in range(1, 7):
        print(f"\n--- Scenario {scenario_id}: {SCENARIOS[scenario_id]} ---")
        scenario = build_scenario(scenario_id)

        print("  Running baseline...")
        t0 = time.time()
        baseline = run_baseline(scenario)
        print(f"  Baseline done in {time.time()-t0:.1f}s")

        print("  Running GridMind AI (MILP)...")
        t0 = time.time()
        ai = run_ai_strategy(
            scenario,
            demand_models=demand_models,
            renewable_models=renewable_models,
            price_model=price_model,
            bess_degrad_cost=bess_degrad_cost,
            verbose=False,
        )
        print(f"  AI strategy done in {time.time()-t0:.1f}s | Solver: {ai.get('status', 'N/A')}")

        metrics = evaluate_scenario(baseline, ai, scenario)
        sim_results[scenario_id] = {
            "scenario_id": scenario_id,
            "scenario_name": SCENARIOS[scenario_id],
            "metrics": metrics,
            "baseline": baseline,
            "ai": ai,
            "scenario": scenario,
        }
        all_scenario_metrics.append({
            "scenario_id": scenario_id,
            "scenario_name": SCENARIOS[scenario_id],
            "metrics": metrics,
        })

        # Print quick summary
        b_rel = metrics["baseline"]["reliability"]["aggregate_pct"]
        a_rel = metrics["ai"]["reliability"]["aggregate_pct"]
        b_ebitda = metrics["baseline"]["ebitda"]["ebitda_margin_pct"]
        a_ebitda = metrics["ai"]["ebitda"]["ebitda_margin_pct"]
        print(f"  Reliability: {b_rel:.1f}% (baseline) \u2192 {a_rel:.1f}% (AI)")
        print(f"  EBITDA:      {b_ebitda:.1f}% (baseline) \u2192 {a_ebitda:.1f}% (AI)")

    kpi_df = build_kpi_table(all_scenario_metrics)
    kpi_df.to_csv(os.path.join(METRICS_DIR, "kpi_comparison.csv"), index=False)
    print(f"\n[main] KPI table saved.")
    print(kpi_df.to_string(index=False))

    return sim_results, kpi_df, all_scenario_metrics


def phase4_charts(sim_results, kpi_df, all_scenario_metrics, forecast_metrics):
    print("\n" + "="*60)
    print("PHASE 4: CHART GENERATION")
    print("="*60)
    from reporting.charts import (
        plot_forecast_evaluation, plot_dispatch_profile, plot_kpi_comparison,
        plot_loss_breakdown, plot_ebitda_by_cluster, plot_renewable_utilization,
        plot_architecture_diagram, plot_sensitivity,
    )

    charts_paths = {}

    # Architecture diagram
    charts_paths["architecture"] = plot_architecture_diagram(CHARTS_DIR)

    # Forecast evaluation
    if "Demand" in forecast_metrics:
        charts_paths["forecast_demand"] = plot_forecast_evaluation(
            forecast_metrics["Demand"], "Demand", CHARTS_DIR)
    if "Renewable" in forecast_metrics:
        charts_paths["forecast_renewable"] = plot_forecast_evaluation(
            forecast_metrics["Renewable"], "Renewable", CHARTS_DIR)

    # KPI comparison
    charts_paths["kpi_comparison"] = plot_kpi_comparison(kpi_df, CHARTS_DIR)

    # Dispatch profile for scenario 1 (normal day)
    if 1 in sim_results:
        charts_paths["dispatch_1"] = plot_dispatch_profile(
            sim_results[1]["scenario_name"],
            sim_results[1]["baseline"],
            sim_results[1]["ai"],
            CHARTS_DIR, 1,
        )

    # Loss breakdown
    if 1 in sim_results:
        loss_data = {
            "baseline": sim_results[1]["metrics"]["baseline"]["losses"],
            "ai": sim_results[1]["metrics"]["ai"]["losses"],
        }
        charts_paths["loss_breakdown"] = plot_loss_breakdown(loss_data, CHARTS_DIR)

    # EBITDA by cluster
    from config.assumptions import CLUSTERS
    from simulation.evaluation import compute_ebitda
    from config.assumptions import HOURLY_OPEX_INR, N_CLUSTERS
    if 1 in sim_results:
        ebitda_data = {"baseline": {}, "ai": {}}
        b_res = sim_results[1]["baseline"]
        a_res = sim_results[1]["ai"]
        # Approximate per-cluster from shares
        from config.assumptions import CLUSTER_CAPACITY_WEIGHTS
        for c, cluster in enumerate(CLUSTERS):
            w = CLUSTER_CAPACITY_WEIGHTS[cluster]
            for label, res in [("baseline", b_res), ("ai", a_res)]:
                ebitda_data[label][cluster] = compute_ebitda(
                    revenue_inr=res["revenue_inr"] * w,
                    procurement_cost_inr=res["procurement_cost_inr"] * w,
                    degrad_cost_inr=res["degrad_cost_inr"] * w,
                    opex_inr=HOURLY_OPEX_INR * 24 * w,
                )
        charts_paths["ebitda_cluster"] = plot_ebitda_by_cluster(ebitda_data, CHARTS_DIR)

    # Renewable utilization
    charts_paths["renewable_util"] = plot_renewable_utilization(all_scenario_metrics, CHARTS_DIR)

    # Sensitivity
    charts_paths["sensitivity"] = plot_sensitivity(CHARTS_DIR)

    # Dispatch profiles for all scenarios
    for sid in range(2, 7):
        if sid in sim_results:
            key = f"dispatch_{sid}"
            charts_paths[key] = plot_dispatch_profile(
                sim_results[sid]["scenario_name"],
                sim_results[sid]["baseline"],
                sim_results[sid]["ai"],
                CHARTS_DIR, sid,
            )

    print(f"[main] Generated {len(charts_paths)} charts.")
    return charts_paths


def phase5_report(sim_results, kpi_df, forecast_metrics, charts_paths):
    print("\n" + "="*60)
    print("PHASE 5: PDF REPORT GENERATION")
    print("="*60)
    from reporting.generate_pdf import build_report
    pdf_path = os.path.join(BASE_DIR, "GridMind_AI_ERoH_Round2_Report.pdf")
    build_report(
        output_path=pdf_path,
        sim_results=sim_results,
        kpi_df=kpi_df,
        forecast_metrics=forecast_metrics,
        charts_paths=charts_paths,
    )
    return pdf_path


def phase6_validate(pdf_path, kpi_df):
    print("\n" + "="*60)
    print("PHASE 6: VALIDATION")
    print("="*60)
    checks = []

    # PDF exists
    pdf_exists = os.path.exists(pdf_path)
    checks.append(("PDF file exists", pdf_exists))
    if pdf_exists:
        size_kb = os.path.getsize(pdf_path) / 1024
        checks.append((f"PDF size ({size_kb:.0f} KB)", size_kb > 50))

    # KPI CSV
    kpi_path = os.path.join(METRICS_DIR, "kpi_comparison.csv")
    checks.append(("KPI CSV exists", os.path.exists(kpi_path)))

    # Charts
    chart_files = [f for f in os.listdir(CHARTS_DIR) if f.endswith(".png")] if os.path.exists(CHARTS_DIR) else []
    checks.append((f"Charts generated ({len(chart_files)})", len(chart_files) >= 5))

    # Data file
    data_path = os.path.join(RESULTS_DIR, "synthetic_data.parquet")
    checks.append(("Synthetic data file exists", os.path.exists(data_path)))

    # KPI rows
    checks.append((f"KPI table has {len(kpi_df)} scenarios", len(kpi_df) == 6))

    print("\n[Validation Results]")
    all_pass = True
    for desc, passed in checks:
        status = "\u2713 PASS" if passed else "\u2717 FAIL"
        print(f"  {status}  {desc}")
        if not passed:
            all_pass = False

    return all_pass


def main():
    t_start = time.time()
    print("\n" + "="*60)
    print("  GRIDMIND AI \u2014 EROH ROUND 2")
    print("  Predict. Allocate. Optimize. Profit.")
    print("="*60)

    df = phase1_data()
    demand_models, renewable_models, price_model, bess_degrad_cost, forecast_metrics = phase2_forecasting(df)
    sim_results, kpi_df, all_scenario_metrics = phase3_simulation(demand_models, renewable_models, price_model, bess_degrad_cost)
    charts_paths = phase4_charts(sim_results, kpi_df, all_scenario_metrics, forecast_metrics)
    pdf_path = phase5_report(sim_results, kpi_df, forecast_metrics, charts_paths)
    all_pass = phase6_validate(pdf_path, kpi_df)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  COMPLETE in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")
    print(f"\n  PDF Report : {pdf_path}")
    print(f"  Charts     : {CHARTS_DIR}")
    print(f"  KPI CSV    : {os.path.join(METRICS_DIR, 'kpi_comparison.csv')}")
    print(f"  Dataset    : {os.path.join(RESULTS_DIR, 'synthetic_data.parquet')}")
    if not all_pass:
        print("\n  [WARNING] Some validation checks failed. Review output above.")
    else:
        print("\n  [OK] All validation checks passed.")


if __name__ == "__main__":
    main()
