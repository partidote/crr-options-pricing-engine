"""Live equity option chain and risk-free rate ingestion via yfinance."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import yfinance as yf


def fetch_risk_free_rate(
    rate_ticker: str = "^IRX", default_rate: float = 0.045
) -> float:
    """Fetch annualized 13-week US Treasury Bill yield as a decimal."""
    try:
        hist = yf.Ticker(rate_ticker).history(period="5d")
        if not hist.empty and "Close" in hist.columns:
            latest = float(hist["Close"].dropna().iloc[-1])
            return latest / 100.0 if latest > 1.0 else latest
    except Exception:
        pass
    return default_rate


def fetch_option_chain(
    ticker: str,
    max_expiries: int = 4,
    risk_free_rate: float | None = None,
) -> pd.DataFrame:
    """Download calls and puts for the nearest `max_expiries` expiration dates."""
    tk = yf.Ticker(ticker)
    hist = tk.history(period="5d")
    if hist.empty:
        raise ValueError(f"Could not retrieve spot price for ticker '{ticker}'.")

    spot = float(hist["Close"].dropna().iloc[-1])
    r = risk_free_rate if risk_free_rate is not None else fetch_risk_free_rate()
    expiries = list(tk.options)[:max_expiries]

    if not expiries:
        raise ValueError(f"No option expirations available for ticker '{ticker}'.")

    now_utc = datetime.now(timezone.utc)
    frames: list[pd.DataFrame] = []

    for exp in expiries:
        chain = tk.option_chain(exp)
        for opt_type, df_side in [("call", chain.calls), ("put", chain.puts)]:
            if df_side.empty:
                continue
            part = df_side.copy()
            part["ticker"] = ticker.upper()
            part["option_type"] = opt_type
            part["expiration"] = exp
            part["spot"] = spot
            part["risk_free_rate"] = r
            part["as_of_utc"] = now_utc.isoformat()
            frames.append(part)

    if not frames:
        raise ValueError(f"Empty option chain returned for ticker '{ticker}'.")

    return pd.concat(frames, ignore_index=True)


def save_raw_chain(df: pd.DataFrame, ticker: str, raw_dir: Path) -> Path:
    """Persist raw option chain snapshot as a timestamped CSV."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = raw_dir / f"chain_{ticker.upper()}_{ts}.csv"
    df.to_csv(out_path, index=False)
    return out_path