# American Option Pricing Engine & Greeks Valuation via Binomial Trees

A modular Python quantitative finance pipeline that ingests equity option chains, filters illiquid market noise, prices American and European contracts via a vectorized Cox-Ross-Rubinstein (CRR) binomial lattice, computes Greek risk sensitivities ($\Delta, \Gamma, \Theta, \mathcal{V}, \rho$), and benchmarks model-vs-market pricing error.

## Architecture
- `configs/config.yaml`: Centralized tickers, liquidity filter thresholds, and CRR lattice parameters.
- `src/options_engine/`: Installable Python package (`data/`, `models/`, `risk/`).
- `pipelines/`: End-to-end CLI runner (`ingest -> clean -> price -> evaluate`).
- `tests/`: Automated `pytest` suite validating Black-Scholes convergence ($< \$0.01$), put-call parity, and data filters.
- `docs/stakeholder_memo.md`: Desk requirements, assumptions, and operational risk guardrails.

## Quickstart
```bash
conda create -n options-engine python=3.11 -y
conda activate options-engine
pip install -e ".[dev]"
pytest
