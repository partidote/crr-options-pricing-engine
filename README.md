# American Option Pricing Engine & Greeks Valuation via Binomial Trees

[![CI](https://github.com/partidote/crr-options-pricing-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/partidote/crr-options-pricing-engine/actions/workflows/ci.yml)

A modular Python quantitative finance pipeline that ingests equity option chains, filters illiquid market noise, prices American and European contracts via a vectorized Cox-Ross-Rubinstein (CRR) binomial lattice, computes Greek risk sensitivities ($\Delta, \Gamma, \Theta, \mathcal{V}, \rho$), and benchmarks model-vs-market pricing error.

## Architecture
- `configs/config.yaml`: Centralized tickers, liquidity filter thresholds, and CRR lattice step budget ($N = 100\text{--}250$).
- `src/options_engine/`: Installable Python package:
  - `data/`: Live `yfinance` option chain & 13-week T-Bill (`^IRX`) ingestion (`ingest.py`) and liquidity/spread filters (`clean.py`).
  - `models/`: Analytical Black-Scholes-Merton benchmark + Brent IV solver (`black_scholes.py`) and 1D-vectorized CRR binomial lattice (`binomial.py`).
  - `risk/`: Batch chain valuation and desk-level RMSE/MAPE & net Greeks aggregation (`evaluate.py`).
- `pipelines/run_eod_valuation.py`: End-to-end CLI runner (`ingest -> clean -> price -> Parquet export`).
- `app/dashboard.py`: Interactive Streamlit app for single-contract pricing, convergence inspection, and live chain risk monitoring.
- `notebooks/01_crr_convergence_and_greeks.ipynb`: Executive walkthrough of BSM convergence, volatility smile, and Greeks profiles.
- `tests/`: Automated `pytest` suite (run via GitHub Actions CI) validating $< \$0.01$ BSM convergence, put-call parity, early-exercise dominance, and data filters.
- `docs/stakeholder_memo.md`: Desk requirements, modeling assumptions, and operational risk guardrails.

## Quickstart
```bash
conda create -n options-engine python=3.11 -y
conda activate options-engine
pip install -e ".[dev,app]"
pytest -v