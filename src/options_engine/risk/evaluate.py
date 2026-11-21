"""Batch CRR chain valuation and model-vs-market risk evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from options_engine.models.binomial import crr_valuation
from options_engine.models.black_scholes import implied_volatility


def evaluate_chain(df: pd.DataFrame, crr_steps: int = 200) -> pd.DataFrame:
    """Price every contract in a cleaned chain and compute Greeks + pricing errors."""
    if df.empty:
        return df.copy()

    records: list[dict] = []
    for row in df.itertuples(index=False):
        S = float(row.spot)
        K = float(row.strike)
        T = float(row.T)
        r = float(row.risk_free_rate)
        opt_type = str(row.option_type)
        mid = float(row.mid_price)

        # Use market IV if valid, otherwise invert BSM from mid-price
        iv = float(getattr(row, "impliedVolatility", np.nan))
        if not (0.01 <= iv <= 3.0):
            iv = implied_volatility(mid, S, K, T, r, option_type=opt_type)

        if np.isnan(iv) or iv <= 0:
            continue

        res = crr_valuation(
            S=S, K=K, T=T, r=r, sigma=iv, N=crr_steps, option_type=opt_type, exercise="american"
        )

        err = res.price - mid
        records.append(
            {
                "ticker": row.ticker,
                "expiration": row.expiration,
                "option_type": opt_type,
                "strike": K,
                "spot": S,
                "T": T,
                "moneyness": float(row.moneyness),
                "iv": iv,
                "market_mid": mid,
                "crr_american": res.price,
                "crr_european": res.european_price,
                "early_exercise_premium": res.early_exercise_premium,
                "pricing_error": err,
                "abs_error": abs(err),
                "pct_error": abs(err) / mid if mid > 0 else np.nan,
                "delta": res.delta,
                "gamma": res.gamma,
                "theta": res.theta,
                "vega": res.vega,
                "rho": res.rho,
            }
        )

    return pd.DataFrame(records)


def summarize_risk(val_df: pd.DataFrame) -> dict[str, float]:
    """Compute desk-level pricing accuracy (RMSE/MAPE) and aggregate Greek exposures."""
    if val_df.empty:
        return {}

    rmse = float(np.sqrt(np.mean(val_df["pricing_error"] ** 2)))
    mae = float(val_df["abs_error"].mean())
    mape = float(val_df["pct_error"].mean() * 100.0)

    return {
        "contracts_priced": int(len(val_df)),
        "rmse_usd": round(rmse, 4),
        "mae_usd": round(mae, 4),
        "mape_pct": round(mape, 2),
        "avg_early_ex_premium": round(float(val_df["early_exercise_premium"].mean()), 4),
        "net_delta": round(float(val_df["delta"].sum()), 4),
        "net_gamma": round(float(val_df["gamma"].sum()), 4),
        "net_theta_daily": round(float(val_df["theta"].sum()), 4),
        "net_vega": round(float(val_df["vega"].sum()), 4),
    }