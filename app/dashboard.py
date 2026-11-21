"""Interactive Streamlit dashboard for CRR American Option Pricing & Desk Risk."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from options_engine.config import load_config
from options_engine.data.clean import clean_option_chain
from options_engine.data.ingest import fetch_option_chain, fetch_risk_free_rate
from options_engine.models.binomial import crr_price, crr_valuation
from options_engine.models.black_scholes import bsm_price
from options_engine.risk.evaluate import evaluate_chain, summarize_risk

st.set_page_config(page_title="CRR American Option Pricing Engine", layout="wide")
st.title("American Option Pricing Engine & Greeks Valuation")
st.caption("Vectorized Cox-Ross-Rubinstein (CRR) Binomial Lattice | Analytical BSM Benchmark")

tab1, tab2 = st.tabs(["Single-Contract Pricer", "Live Market Chain & Desk Risk"])

# ---------------------------------------------------------------------------
# TAB 1: Interactive Single-Contract Pricer
# ---------------------------------------------------------------------------
with tab1:
    col_in, col_out = st.columns([1, 2], gap="large")

    with col_in:
        st.subheader("Contract Parameters")
        option_type = st.radio("Option Type", ["put", "call"], horizontal=True)
        exercise = st.radio("Exercise Style", ["american", "european"], horizontal=True)
        S = st.number_input("Spot Price (S)", min_value=1.0, value=100.0, step=1.0)
        K = st.number_input("Strike Price (K)", min_value=1.0, value=100.0, step=1.0)
        T = st.slider("Time to Maturity (Years)", min_value=0.05, max_value=2.0, value=0.50, step=0.05)
        sigma = st.slider("Volatility (σ)", min_value=0.05, max_value=1.00, value=0.25, step=0.01)
        r = st.slider("Risk-Free Rate (r)", min_value=0.00, max_value=0.12, value=0.045, step=0.005)
        q = st.slider("Dividend Yield (q)", min_value=0.00, max_value=0.08, value=0.00, step=0.005)
        N = st.slider("CRR Tree Steps (N)", min_value=20, max_value=250, value=200, step=10)

    res = crr_valuation(S=S, K=K, T=T, r=r, sigma=sigma, N=N, option_type=option_type, exercise=exercise, q=q)
    bsm_ref = bsm_price(S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type, q=q)

    with col_out:
        st.subheader("Valuation & Benchmark")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CRR Price", f"${res.price:.4f}")
        m2.metric("European CRR", f"${res.european_price:.4f}")
        m3.metric("Early-Exercise Prem.", f"${res.early_exercise_premium:.4f}")
        m4.metric("Analytical BSM (Euro)", f"${bsm_ref:.4f}")

        st.subheader("Greek Risk Sensitivities")
        g1, g2, g3, g4, g5 = st.columns(5)
        g1.metric("Delta (Δ)", f"{res.delta:.4f}")
        g2.metric("Gamma (Γ)", f"{res.gamma:.4f}")
        g3.metric("Theta (Θ/day)", f"{res.theta:.4f}")
        g4.metric("Vega (ν/1%)", f"{res.vega:.4f}")
        g5.metric("Rho (ρ/1%)", f"{res.rho:.4f}")

        # Convergence plot
        step_grid = list(range(10, max(N, 50) + 1, 10))
        conv_prices = [
            crr_price(S, K, T, r, sigma, N=n, option_type=option_type, exercise="european", q=q)
            for n in step_grid
        ]
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(step_grid, conv_prices, marker="o", ms=3, label="CRR European")
        ax.axhline(bsm_ref, color="crimson", ls="--", label=f"BSM Exact (${bsm_ref:.4f})")
        ax.set_title("European Lattice Convergence to Analytical Black-Scholes")
        ax.set_xlabel("Steps (N)")
        ax.set_ylabel("Price ($)")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

# ---------------------------------------------------------------------------
# TAB 2: Live Market Chain & Desk Risk
# ---------------------------------------------------------------------------
with tab2:
    cfg = load_config()
    c1, c2, c3 = st.columns([1, 1, 1])
    ticker = c1.text_input("Ticker", value="SPY").upper().strip()
    steps_live = c2.slider("Batch CRR Steps", 50, 250, cfg["pricing"]["crr_steps"], step=25)
    run_live = c3.button("Fetch & Price Live Chain", type="primary")

    val_df = pd.DataFrame()
    if run_live:
        with st.spinner(f"Ingesting and pricing {ticker} option chain..."):
            r_live = fetch_risk_free_rate(
                cfg["market_data"]["risk_free_ticker"],
                cfg["market_data"]["default_risk_free_rate"],
            )
            raw_df = fetch_option_chain(ticker, max_expiries=2, risk_free_rate=r_live)
            clean_df = clean_option_chain(raw_df, cfg["filters"])
            val_df = evaluate_chain(clean_df, crr_steps=steps_live)
    else:
        proc_dir = Path("data/processed")
        pq_files = sorted(proc_dir.glob("valuations_*.parquet"))
        if pq_files:
            val_df = pd.read_parquet(pq_files[-1])
            st.info(f"Showing latest cached EOD run: `{pq_files[-1].name}`. Click **Fetch & Price Live Chain** to refresh.")

    if not val_df.empty:
        summary = summarize_risk(val_df)
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Contracts Priced", summary["contracts_priced"])
        k2.metric("RMSE ($)", f"${summary['rmse_usd']:.4f}")
        k3.metric("MAPE (%)", f"{summary['mape_pct']:.2f}%")
        k4.metric("Net Delta", f"{summary['net_delta']:.2f}")
        k5.metric("Net Daily Theta", f"{summary['net_theta_daily']:.2f}")

        expiries = sorted(val_df["expiration"].unique())
        sel_exp = st.selectbox("Select Expiration", expiries)
        sub = val_df[val_df["expiration"] == sel_exp].sort_values("strike")

        p1, p2 = st.columns(2)
        with p1:
            fig1, ax1 = plt.subplots(figsize=(6, 3.5))
            for opt, grp in sub.groupby("option_type"):
                ax1.plot(grp["moneyness"], grp["iv"], marker="o", ms=3, label=opt)
            ax1.set_title(f"Implied Volatility Smile ({sel_exp})")
            ax1.set_xlabel("Moneyness (K / S)")
            ax1.set_ylabel("Implied Volatility")
            ax1.legend()
            ax1.grid(alpha=0.3)
            st.pyplot(fig1)

        with p2:
            fig2, ax2 = plt.subplots(figsize=(6, 3.5))
            for opt, grp in sub.groupby("option_type"):
                ax2.plot(grp["moneyness"], grp["delta"], marker="o", ms=3, label=opt)
            ax2.set_title(f"CRR Delta Profile ({sel_exp})")
            ax2.set_xlabel("Moneyness (K / S)")
            ax2.set_ylabel("Delta")
            ax2.legend()
            ax2.grid(alpha=0.3)
            st.pyplot(fig2)

        st.dataframe(sub.reset_index(drop=True), use_container_width=True)
    else:
        st.warning("No cached valuations found yet. Click **Fetch & Price Live Chain** above to run.")