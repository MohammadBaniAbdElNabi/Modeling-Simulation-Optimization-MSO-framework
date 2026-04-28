# MSO Blood Delivery Pipeline

A Forecast-Driven Dispatch Optimization Pipeline for Drone-Based Blood Delivery in Healthcare Logistics.

**All data used in this project is fully synthetic. No real hospital data, patient data, or external APIs are used.**

## Project Overview

This project implements a three-stage Modeling-Simulation-Optimization (M-S-O) pipeline:

- **Stage M (Forecasting):** SARIMA time series forecasting of hourly blood request rates for 12 hospitals over a 24-hour window.
- **Stage O (Optimization):** Integer linear program (ILP) solved with PuLP/CBC that assigns blood bank-to-hospital deliveries per hourly window.
- **Stage S (Simulation):** SimPy discrete-event simulation comparing three dispatch policies (Random, Nearest-Feasible Greedy, LP-Optimized) over 20 replications each.
- **Evaluation:** Statistical comparison (paired t-tests, 95% CIs, Cohen's d) across five performance metrics with 10 visualizations.

**Scenario:** 12 hospitals, 3 blood banks, 8 drones, Orlando FL-scale geographic network.

---

## Repository Structure

```
mso_blood_delivery/
├── README.md
├── requirements.txt
├── environment.yml
├── config/
│   ├── network.yaml          # Hospital/bank coordinates, distance matrix
│   ├── simulation.yaml       # Drone, timing, experiment parameters
│   ├── sarima.yaml           # SARIMA candidate orders, data gen settings
│   ├── lp.yaml               # LP solver settings
│   └── evaluation.yaml       # Metrics, CI level, alpha, replication count
├── data/
│   ├── synthetic/            # demand_train.csv, demand_test.csv, lambda_true.csv
│   ├── processed/            # lambda_hat.csv, lp_assignment.json, forecast_metrics.csv
│   └── results/              # sim_results_*.csv, summary_stats.csv, pairwise_tests.csv
├── src/
│   ├── data_gen/             # SyntheticDemandGenerator
│   ├── forecasting/          # SARIMAForecaster, grid search, metrics
│   ├── simulation/           # SimPy DES entities, processes, dispatch policies
│   ├── optimization/         # LPDispatchSolver, inventory, fleet capacity
│   ├── evaluation/           # Statistics, visualizations, report builder
│   └── utils/                # Config loader, distance, RNG, logging
├── notebooks/
│   ├── notebook_01_forecasting.ipynb
│   ├── notebook_02_optimization.ipynb
│   ├── notebook_03_simulation.ipynb
│   └── notebook_04_evaluation.ipynb
├── tests/
│   ├── test_data_gen.py
│   ├── test_sarima.py
│   ├── test_lp.py
│   ├── test_simulation.py
│   └── test_metrics.py
└── docs/
    ├── spec/design_spec.docx
    ├── pipeline_diagram.png
    └── parameter_reference.md
```

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10 – 3.12 | [python.org](https://www.python.org/downloads/) |
| pip | ≥ 23 | Bundled with Python 3.10+ |
| Git | any recent | To clone the repository |
| conda *(optional)* | ≥ 23 | Only needed for the conda setup path |

> **Windows note:** During Python installation, check **"Add Python to PATH"** on the first installer screen. If you already installed Python without it, re-run the installer and choose **Modify → Add to PATH**.

---

## Environment Setup

Clone the repository first, then choose one of the two setup options below.

### Clone

**Linux / macOS**
```bash
git clone <repository-url> mso_blood_delivery
cd mso_blood_delivery
```

**Windows (Command Prompt or PowerShell)**
```cmd
git clone <repository-url> mso_blood_delivery
cd mso_blood_delivery
```

---

### Option A: pip + virtual environment (recommended)

#### Linux / macOS

```bash
# Create the virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Upgrade pip, then install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

To deactivate when finished:
```bash
deactivate
```

#### Windows — Command Prompt

```cmd
:: Create the virtual environment
python -m venv .venv

:: Activate it
.venv\Scripts\activate.bat

:: Upgrade pip, then install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

To deactivate when finished:
```cmd
deactivate
```

#### Windows — PowerShell

```powershell
# Create the virtual environment
python -m venv .venv

# Allow script execution for this session (first time only)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Activate it
.venv\Scripts\Activate.ps1

# Upgrade pip, then install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

To deactivate when finished:
```powershell
deactivate
```

> **PowerShell execution policy:** If you see `"cannot be loaded because running scripts is disabled"`, run the `Set-ExecutionPolicy` line above once. This only affects the current user and is safe for development.

---

### Option B: conda

Conda handles the environment the same way on Linux and Windows.

```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate mso_blood

# Verify
python --version          # should print Python 3.10.x
pip show pulp simpy       # confirm pip-installed packages
```

To deactivate when finished:
```bash
conda deactivate
```

> **Windows note:** Run these commands inside **Anaconda Prompt** or a terminal where `conda` is on the PATH. If using PowerShell, run `conda init powershell` once (then restart the terminal) before using `conda activate`.

---

## Verifying the Installation

After setup, confirm everything installed correctly by running the test suite:

**Linux / macOS**
```bash
python3 -m pytest tests/ -v
```

**Windows (Command Prompt or PowerShell)**
```cmd
python -m pytest tests/ -v
```

All **31 tests** should pass. Expected output:
```
======================== 31 passed in ~105s =============================
```

---

## Running the Pipeline

Notebooks **must** be run in order — each notebook reads outputs written by the previous one. Launch JupyterLab from the project root directory (not from inside `notebooks/`).

### Start JupyterLab

**Linux / macOS**
```bash
# Make sure your virtual environment is active, then:
cd mso_blood_delivery        # if not already there
jupyter lab notebooks/
```

**Windows (Command Prompt)**
```cmd
:: Make sure your virtual environment is active, then:
cd mso_blood_delivery
jupyter lab notebooks\
```

**Windows (PowerShell)**
```powershell
# Make sure your virtual environment is active, then:
cd mso_blood_delivery
jupyter lab notebooks\
```

JupyterLab will open in your default browser. Run each notebook top-to-bottom using **Kernel → Restart Kernel and Run All Cells**.

### Notebook Execution Order

| # | Notebook | What it does | Approx. runtime |
|---|----------|--------------|-----------------|
| 1 | `notebook_01_forecasting.ipynb` | Generates synthetic demand data, fits SARIMA models per hospital, saves `data/processed/lambda_hat.csv` and `data/processed/forecast_metrics.csv` | 3 – 8 min |
| 2 | `notebook_02_optimization.ipynb` | Reads `lambda_hat.csv`, solves the integer LP for all 24 hourly windows, saves `data/processed/lp_assignment.json` | ~1 sec |
| 3 | `notebook_03_simulation.ipynb` | Runs 60 SimPy replications (3 policies × 20 seeds), saves `data/results/sim_results_*.csv` | 5 – 15 min |
| 4 | `notebook_04_evaluation.ipynb` | Reads all results, runs paired t-tests, generates 10 figures under `data/results/figures/` | ~1 min |

**Estimated total runtime: 10 – 25 minutes on standard hardware.**

---

## Running Tests

### Linux / macOS

```bash
# Basic run
python3 -m pytest tests/ -v

# With coverage report
python3 -m pytest tests/ --cov=src --cov-report=term-missing

# Run a single test file
python3 -m pytest tests/test_simulation.py -v
```

### Windows (Command Prompt or PowerShell)

```cmd
:: Basic run
python -m pytest tests/ -v

:: With coverage report
python -m pytest tests/ --cov=src --cov-report=term-missing

:: Run a single test file
python -m pytest tests\test_simulation.py -v
```

---

## Expected Generated Artifacts

| File | Description |
|------|-------------|
| `data/synthetic/demand_train.csv` | Training demand counts (6 days × 12 hospitals × 24 hours) |
| `data/synthetic/demand_test.csv` | Test demand counts (1 day × 12 hospitals × 24 hours) |
| `data/synthetic/lambda_true.csv` | Ground truth arrival rates |
| `data/processed/lambda_hat.csv` | SARIMA forecasts (12 × 24) |
| `data/processed/lp_assignment.json` | LP assignment tensor (bank × hospital × window) |
| `data/processed/forecast_metrics.csv` | MAE, RMSE, MAPE, Bias per hospital |
| `data/results/sim_results_random.csv` | Simulation metrics, Random policy (20 rows) |
| `data/results/sim_results_greedy.csv` | Simulation metrics, Greedy policy (20 rows) |
| `data/results/sim_results_lp.csv` | Simulation metrics, LP policy (20 rows) |
| `data/results/summary_stats.csv` | Mean, std, CI per policy per metric |
| `data/results/pairwise_tests.csv` | t-test results and Cohen's d |
| `data/results/figures/fig1_*.png` … `fig10_*.png` | All 10 visualization figures |

---

## Troubleshooting

### `python` not found (Linux / macOS)
Use `python3` instead of `python`. On modern Linux/macOS distributions the unversioned `python` alias may not exist.

### `python` not found (Windows)
Reinstall Python from [python.org](https://www.python.org/downloads/) and check **"Add Python to PATH"** during installation, or add it manually:
- Search for **"Edit the system environment variables"** → **Environment Variables** → edit `Path` → add the folder containing `python.exe` (e.g. `C:\Users\<you>\AppData\Local\Programs\Python\Python310\`).

### `ModuleNotFoundError` for any package
Ensure your virtual environment is activated (you should see `(.venv)` or `(mso_blood)` at the start of the prompt) and re-run `pip install -r requirements.txt`.

### PuLP / CBC solver error on Windows
PuLP ships its own CBC binary for Windows. If you see a solver error, try:
```cmd
python -m pulp.tests.test_pulp
```
If that fails, install the Microsoft Visual C++ Redistributable (x64) from the Microsoft website, then reinstall PuLP:
```cmd
pip install --force-reinstall pulp
```

### `jupyter` not found after install
The virtual environment's `Scripts` (Windows) or `bin` (Linux) directory must be on your PATH. Reactivate the environment and try again. Alternatively, run:
```bash
python -m jupyter lab notebooks/
```

### Long SARIMA runtime
Notebook 01 runs an AIC grid search over SARIMA orders for each of the 12 hospitals. Runtime varies with CPU speed (typically 3–8 minutes). This is expected behaviour.

---

## Data Statement

**All data in this project is entirely synthetic.** No real hospital operational data, patient records, blood bank inventories, or geographic routing data are used. The synthetic demand patterns are parameterized using published healthcare queueing literature values for illustrative purposes only.
