# Drone-Based Blood Delivery Dispatch Optimization

A forecast-driven dispatch optimization pipeline for drone-based blood delivery in healthcare logistics, implementing a modular Modeling–Simulation–Optimization (MSO) framework.

**All data used in this project is fully synthetic. No real hospital data, patient data, or external APIs are used.**

---

## Overview

This project addresses the challenge of dispatching blood products from centralized blood banks to hospitals via drone, where demand is stochastic, time-sensitive, and heterogeneous across facilities. The pipeline integrates three stages:

1. **Forecasting (M):** Hierarchical Poisson factor model and SARIMA per-hospital demand forecasting over a 24-hour horizon.
2. **Optimization (O):** Multi-period linear program that jointly assigns bank-to-hospital delivery allocations across all 24 windows, with cumulative inventory constraints and priority-tiered penalties.
3. **Simulation (S):** SimPy discrete-event simulation comparing four dispatch policies (Random, Greedy, Static-LP, MPC-LP) across 50 independent replications using Common Random Numbers.

Evaluation uses paired t-tests with Cohen's d effect sizes, minimum detectable effects, and operating curves across demand intensities.

**Scenario:** 12 hospitals, 3 blood banks, 8 drones, Orlando FL-scale geographic network.

---

## Architecture

```
src/
├── data_gen/          # SyntheticDemandGenerator — Poisson demand with hospital-specific scaling
├── forecasting/       # HierarchicalFactorForecaster, SARIMAForecaster, grid search, metrics
├── optimization/      # LPDispatchSolver (24-hr joint LP), RollingHorizonLP (MPC)
├── simulation/
│   ├── dispatch/      # RandomDispatch, NearestFeasibleGreedy, LPOptimizedDispatch, MPCDispatch
│   └── ...            # SimPy entities, processes, metrics collector, runner
├── experiment/        # ExperimentRunner — four-condition × demand-stress sweep
├── evaluation/        # Paired t-tests, Cohen's d, forest plots, operating curves
└── utils/             # Config loader, Haversine distance, RNG seeding, logging
config/                # YAML configuration for all pipeline stages
notebooks/             # Four sequential Jupyter notebooks (M → O → S → Evaluation)
tests/                 # pytest suite (31 tests)
data/                  # synthetic/, processed/, results/
```

---

## Method Summary

### Forecasting
A rank-1 Poisson factor model decomposes demand as `λ[h,t] = σ_h · π_t`, estimated via alternating MLE. This is the correctly-specified estimator for the multiplicative-separable DGP and outperforms SARIMA in MAPE and bias by pooling all hospitals and hours. SARIMA is also available as an alternative model; results are blended (60/40) when both converge.

### Optimization
The LP jointly minimizes total flight time plus priority-weighted unmet-demand penalties subject to:
- Soft demand coverage constraints (with per-priority slack variables)
- Hard cumulative inventory constraints across all 24 windows
- Per-window fleet capacity

Unlike a sequential per-window LP (which is equivalent to greedy), the cumulative inventory constraint forces the solver to spread load proactively — preventing early depletion of nearby banks during peak hours.

### Simulation
Four dispatch policies under a pooled-fleet SimPy DES:
- **Random:** Uniform bank selection (lower bound)
- **Greedy:** Nearest-feasible bank by flight time
- **Static-LP:** Follows the 24-hour LP plan committed at t=0; falls back to greedy when budget is exhausted
- **MPC-LP:** Re-solves the rolling-horizon LP hourly with current inventory state and online demand scaling

### Evaluation
Pre-registered primary hypotheses tested with paired t-tests on three headline metrics (FR\_weighted, ERR\_peak, Expiration\_cost) at α=0.05. Reference pairs use Holm–Bonferroni correction.

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10 – 3.12 |
| pip | ≥ 23 |
| Git | any recent |
| conda *(optional)* | ≥ 23 |

---

## Environment Setup

### Option A: pip + virtual environment

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

```cmd
:: Windows (Command Prompt)
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

```powershell
# Windows (PowerShell)
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: conda

```bash
conda env create -f environment.yml
conda activate mso_blood
```

---

## Running the Pipeline

Notebooks must be executed in order — each writes outputs consumed by the next. Launch JupyterLab from the **project root**:

```bash
jupyter lab notebooks/
```

| # | Notebook | Description | Runtime |
|---|----------|-------------|---------|
| 1 | `notebook_01_forecasting.ipynb` | Synthetic demand generation; factor model and SARIMA fitting | 3–8 min |
| 2 | `notebook_02_optimization.ipynb` | LP solve for all demand models; saves `lp_assignment_*.json` | ~1 s |
| 3 | `notebook_03_simulation.ipynb` | Four-condition DES × 50 replications × 3 demand levels | 15–45 min |
| 4 | `notebook_04_evaluation.ipynb` | Paired t-tests, effect sizes, 8 publication-ready figures | ~1 min |

---

## Running Tests

```bash
# Linux / macOS
python3 -m pytest tests/ -v

# Windows
python -m pytest tests/ -v
```

All **31 tests** should pass.

---

## Generated Artifacts

| Path | Description |
|------|-------------|
| `data/synthetic/demand_train.csv` | Training demand counts (6 days × 12 hospitals × 24 hours) |
| `data/synthetic/demand_test.csv` | Test demand counts (1 day) |
| `data/synthetic/lambda_true.csv` | Ground-truth Poisson rates |
| `data/processed/lambda_factor.csv` | Factor model forecast (12 × 24) |
| `data/processed/lp_assignment_static.json` | Static LP allocation tensor |
| `data/results/sim_results_{cond}_d{scale}.csv` | Per-replication simulation metrics |
| `data/results/summary_stats.csv` | Mean, std, 95% CI per condition × metric |
| `data/results/pairwise_tests.csv` | t-test results, Cohen's d, MDE |
| `data/results/figures/` | 8 publication-ready figures |

---

## Dependencies

Core: `numpy`, `pandas`, `scipy`, `statsmodels`, `simpy`, `pulp` (CBC solver), `matplotlib`, `seaborn`, `pyyaml`

See `requirements.txt` for pinned versions.

---

## Data Statement

All data is entirely synthetic. No real hospital operational data, patient records, blood bank inventories, or geographic routing data are used. Demand parameters are drawn from published healthcare queueing literature for illustrative purposes only.

---

## License

MIT License. See [LICENSE](LICENSE).
