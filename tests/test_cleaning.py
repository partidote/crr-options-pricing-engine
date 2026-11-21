"""Unit tests for option chain liquidity and data quality filters."""

import pandas as pd
from options_engine.data.clean import clean_option_chain

FILTER_CFG = {
    "min_volume": 1,
    "min_open_interest": 10,
    "min_bid": 0.05,
    "max_rel_spread": 0.20,
    "moneyness_bounds": [0.80, 1.20],
}


def _sample_chain() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Valid liquid ATM call
            {
                "strike": 100.0, "bid": 4.90, "ask": 5.10, "volume": 50,
                "openInterest": 200, "impliedVolatility": 0.25, "spot": 100.0,
                "expiration": "2027-06-18", "as_of_utc": "2026-11-21T00:00:00+00:00",
                "option_type": "call",
            },
            # Zero volume -> should be dropped
            {
                "strike": 100.0, "bid": 4.90, "ask": 5.10, "volume": 0,
                "openInterest": 200, "impliedVolatility": 0.25, "spot": 100.0,
                "expiration": "2027-06-18", "as_of_utc": "2026-11-21T00:00:00+00:00",
                "option_type": "call",
            },
            # Inverted spread (bid > ask) -> should be dropped
            {
                "strike": 100.0, "bid": 5.20, "ask": 5.00, "volume": 50,
                "openInterest": 200, "impliedVolatility": 0.25, "spot": 100.0,
                "expiration": "2027-06-18", "as_of_utc": "2026-11-21T00:00:00+00:00",
                "option_type": "put",
            },
            # Wide spread (> 20%) -> should be dropped
            {
                "strike": 105.0, "bid": 1.00, "ask": 1.50, "volume": 20,
                "openInterest": 100, "impliedVolatility": 0.25, "spot": 100.0,
                "expiration": "2027-06-18", "as_of_utc": "2026-11-21T00:00:00+00:00",
                "option_type": "call",
            },
            # Deep OTM outside moneyness bounds -> should be dropped
            {
                "strike": 150.0, "bid": 0.50, "ask": 0.55, "volume": 20,
                "openInterest": 100, "impliedVolatility": 0.25, "spot": 100.0,
                "expiration": "2027-06-18", "as_of_utc": "2026-11-21T00:00:00+00:00",
                "option_type": "call",
            },
        ]
    )


def test_clean_option_chain_drops_noisy_quotes() -> None:
    cleaned = clean_option_chain(_sample_chain(), FILTER_CFG)
    assert len(cleaned) == 1
    assert cleaned.loc[0, "mid_price"] == 5.0
    assert cleaned.loc[0, "moneyness"] == 1.0
    assert cleaned.loc[0, "T"] > 0.5