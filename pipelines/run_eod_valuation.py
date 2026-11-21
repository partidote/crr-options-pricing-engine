"""End-to-end EOD valuation pipeline: ingest -> clean -> CRR price -> persist -> report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import pandas as pd

from options_engine.config import load_config, resolve_path
from options_engine.data.clean import clean_option_chain
from options_engine.data.ingest import fetch_option_chain, fetch_risk_free_rate, save_raw_chain
from options_engine.risk.evaluate import evaluate_chain, summarize_risk


def run_pipeline(tickers: list[str] | None = None) -> pd.DataFrame:
    cfg = load_config()
    mkt_cfg = cfg["market_data"]
    flt_cfg = cfg["filters"]
    prc_cfg = cfg["pricing"]

    target_tickers = tickers or mkt_cfg["tickers"]
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    proc_dir = resolve_path(cfg["paths"]["processed_dir"])

    r = fetch_risk_free_rate(
        rate_ticker=mkt_cfg["risk_free_ticker"],
        default_rate=mkt_cfg["default_risk_free_rate"],
    )
    print(f"[INFO] Risk-free rate ({mkt_cfg['risk_free_ticker']}): {r:.4%}")

    all_valuations: list[pd.DataFrame] = []

    for ticker in target_tickers:
        print(f"\n[INFO] Ingesting option chain for {ticker}...")
        raw_df = fetch_option_chain(
            ticker=ticker,
            max_expiries=mkt_cfg["max_expiries"],
            risk_free_rate=r,
        )
        raw_path = save_raw_chain(raw_df, ticker, raw_dir)
        print(f"[INFO] Saved raw snapshot ({len(raw_df)} rows) -> {raw_path.name}")

        clean_df = clean_option_chain(raw_df, flt_cfg)
        print(f"[INFO] Filtered to {len(clean_df)} liquid contracts.")

        val_df = evaluate_chain(clean_df, crr_steps=prc_cfg["crr_steps"])
        summary = summarize_risk(val_df)
        print(f"[SUMMARY] {ticker}: {summary}")
        all_valuations.append(val_df)

    combined = pd.concat(all_valuations, ignore_index=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = proc_dir / f"valuations_{ts}.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"\n[DONE] Saved {len(combined)} valued contracts -> {out_path}")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EOD CRR American Option Valuation Pipeline.")
    parser.add_argument("--tickers", nargs="+", help="Optional override list of tickers (e.g. SPY AAPL)")
    args = parser.parse_args()
    run_pipeline(tickers=args.tickers)