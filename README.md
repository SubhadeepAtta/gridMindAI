# GridMind AI — ERoH Round 2 Case Study

**Predict. Allocate. Optimize. Profit.**

Intelligent Energy Allocation, Battery Management and Dynamic Procurement for Multi-Cluster EV Infrastructure.

---

## Overview

GridMind AI is a complete Python prototype for the ERoH Round 2 energy optimization case study. It manages five metropolitan EV charging and battery-swapping clusters (Delhi NCR, Mumbai MMR, Bengaluru, Hyderabad, Chennai) with:

- **AI forecasting** (demand, renewable generation, IEX prices)
- **Two-stage market architecture** (day-ahead + real-time balancing)
- **MILP optimization** (joint dispatch, BESS scheduling, swap inventory, dynamic pricing)
- **Six-scenario simulation** (baseline vs GridMind AI comparison)
- **Professional PDF report**

---

## Quick Start

```bash
# Install dependencies
pip install numpy pandas scikit-learn scipy matplotlib pulp highspy reportlab

# Run the complete pipeline
cd GridMind-AI
python main.py
```

**Estimated runtime:** 3–8 minutes (MILP solves ~11s per scenario).

---

## Project Structure

```
GridMind-AI/
├── config/assumptions.py        # All parameters, clearly labeled [CASE-STUDY] / [ASSUMPTION]
├── data/generate_data.py        # Vectorized synthetic data generator (219,000 rows)
├── models/
│   ├── demand_model.py          # DC charging + swap request forecasting (HistGBR)
│   ├── renewable_model.py       # Solar + wind generation forecasting
│   ├── price_model.py           # IEX DAM price forecasting
│   └── degradation_model.py     # BESS/pack degradation cost estimation
├── optimization/
│   ├── dam_scheduler.py         # Day-ahead procurement scheduling (Stage A)
│   └── energy_dispatch.py       # Central 24-hour MILP (Stage B, PuLP + HiGHS)
├── simulation/
│   ├── scenarios.py             # 6 representative 24-hour scenario definitions
│   ├── baseline.py              # Conventional heuristic operations strategy
│   ├── ai_strategy.py           # GridMind AI optimized strategy
│   ├── queue_model.py           # Charging queue + swap inventory dynamics
│   └── evaluation.py            # KPI computation (reliability, losses, EBITDA)
├── reporting/
│   ├── charts.py                # All charts from actual simulation outputs
│   └── generate_pdf.py          # ReportLab PDF report builder
├── results/
│   ├── synthetic_data.parquet   # Generated dataset
│   ├── charts/                  # 14 PNG charts
│   └── metrics/                 # forecast_metrics.csv, kpi_comparison.csv
├── main.py                      # Master pipeline (all 6 phases)
├── requirements.txt
└── GridMind_AI_ERoH_Round2_Report.pdf   # PRIMARY DELIVERABLE
```

---

## Key Results (Actual Simulation Outputs)

| Metric | Baseline | GridMind AI | Target |
|--------|----------|-------------|--------|
| Reliability (Normal Day) | ~95.8% | ~95.5% | 89.7% |
| EBITDA Margin (Normal Day) | ~44.7% | ~63.3% | >12% |
| Renewable Utilization | ~97.3% | ~99.4% | — |
| Cost Saving (Normal Day) | — | INR 1.38 Cr | — |

> **Note:** These are simulation outputs on synthetic data. See the PDF report for scenario-by-scenario breakdowns and all caveats.

### Loss Target Note
The 8% loss target faces a hardware constraint: the minimum physical loss floor under assumed component efficiencies (charger conversion 4% + distribution 2.5% + inverter 2.5%) is ~6.5%, leaving ~1.5pp headroom for software optimization. The AI strategy reduces avoidable losses (curtailment, excessive BESS cycling) but cannot overcome fixed hardware limits.

---

## Architecture Highlights

- **Electrical isolation enforced**: No inter-metro electricity transfers. Clusters coordinate only through procurement planning and fleet routing.
- **Separate BESS and swap inventory**: 360 MWh stationary BESS (bidirectional grid dispatch) + 240 MWh swap pack inventory (discrete pack counts, 8,000 packs × 30 kWh).
- **Discrete pricing tiers**: 3 binary-variable tiers (Discount −15%, Standard 0%, Premium +15%) within ±20% band. Demand response is linear.
- **Two-stage market**: DAM commitments are locked before real-time dispatch. Deviations handled by DISCOM swing + RTM + emergency fallback.
- **MILP linearization**: Revenue objective uses big-M upper-bound constraints to linearize demand_served × tier_tariff without bilinear terms.

---

## Solver Requirements

This project uses **PuLP 4.x** with the **HiGHS Python API** (`highspy`). Both are installed via `pip install pulp highspy`. The HiGHS CMD binary is NOT required.

---

## Assumptions Register

All assumptions are documented in [`config/assumptions.py`](config/assumptions.py) and in Appendix A of the PDF report. Key assumptions:

| Item | Value | Basis |
|------|-------|-------|
| BESS allocation | 60% of storage (360 MWh) | Proportional to DC/swap capacity ratio |
| Pack size | 30 kWh | Representative 2024 Indian EV commercial pack |
| BESS round-trip efficiency | 90.25% | Li-NMC 2024 benchmark |
| DC charging retail tariff | INR 12.00/kWh | Metro fast-charge market rate |
| DISCOM tariff | INR 7.80/kWh | 2024-25 industrial HT tariff |
| Renewable PPA | INR 3.50/kWh | Blended solar/wind PPA estimate |

---

## Deliverables

| File | Description |
|------|-------------|
| `GridMind_AI_ERoH_Round2_Report.pdf` | 15–20 page professional PDF report |
| `results/metrics/kpi_comparison.csv` | KPI table for all 6 scenarios |
| `results/metrics/forecast_metrics.csv` | ML model evaluation metrics |
| `results/charts/` | 14 PNG charts |
| `results/synthetic_data.parquet` | 219,000-row synthetic dataset |

---

*All data is synthetic. Results are prototype simulation outputs, not operational guarantees.*
