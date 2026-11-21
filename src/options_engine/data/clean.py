"""Option chain cleaning, liquidity filtering, and feature enrichment."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def clean_option_chain(df: pd.DataFrame, filter_cfg: dict[str, Any]) -> pd.DataFrame:
    """Filter illiquid/noisy quotes and compute mid-price, moneyness, and time to expiry."""
    if df.empty:
        return df.copy()

    out = df.copy()

    # Standardize numeric columns
    for col in ("strike", "bid", "ask", "volume", "openInterest", "impliedVolatility", "spot"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["volume"] = out["volume"].fillna(0)
    out["openInterest"] = out["openInterest"].fillna(0)

    # Drop invalid or inverted quotes
    out = out.dropna(subset=["strike", "bid", "ask", "spot", "expiration"])
    out = out[(out["bid"] >= filter_cfg.get("min_bid", 0.05)) & (out["ask"] >= out["bid"])]

    # Compute mid-price, relative bid-ask spread, and moneyness (K / S)
    out["mid_price"] = 0.5 * (out["bid"] + out["ask"])
    out["rel_spread"] = (out["ask"] - out["bid"]) / out["mid_price"].replace(0, np.nan)
    out["moneyness"] = out["strike"] / out["spot"]

    # Compute annualized time to maturity T (in years)
    as_of = pd.to_datetime(out["as_of_utc"], utc=True)
    exp_dt = pd.to_datetime(out["expiration"], utc=True) + pd.Timedelta(hours=20)  # ~4pm ET
    out["T"] = (exp_dt - as_of).dt.total_seconds() / (365.0 * 24.0 * 3600.0)

    # Apply liquidity and moneyness filters from config.yaml
    min_vol = filter_cfg.get("min_volume", 1)
    min_oi = filter_cfg.get("min_open_interest", 10)
    max_spread = filter_cfg.get("max_rel_spread", 0.20)
    lo_m, hi_m = filter_cfg.get("moneyness_bounds", [0.80, 1.20])

    mask = (
        (out["volume"] >= min_vol)
        & (out["openInterest"] >= min_oi)
        & (out["rel_spread"] <= max_spread)
        & (out["moneyness"].between(lo_m, hi_m))
        & (out["T"] > 1.0 / 365.0)
    )

    return out.loc[mask].reset_index(drop=True)