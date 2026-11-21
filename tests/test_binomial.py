import numpy as np
import pytest

from options_engine.models.binomial import crr_price, crr_valuation
from options_engine.models.black_scholes import bsm_greeks, bsm_price, implied_volatility


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
def test_crr_european_converges_to_bsm(option_type: str, K: float) -> None:
    """CRR European price at N=200 must be within $0.01 of analytical BSM."""
    S, T, r, sigma = 100.0, 0.5, 0.05, 0.25
    bsm = bsm_price(S, K, T, r, sigma, option_type=option_type)
    crr = crr_price(S, K, T, r, sigma, N=200, option_type=option_type, exercise="european")
    assert abs(crr - bsm) < 0.01


def test_put_call_parity_european() -> None:
    """Verify C - P = S - K * exp(-rT) on European CRR prices."""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.045, 0.20
    call = crr_price(S, K, T, r, sigma, N=200, option_type="call", exercise="european")
    put = crr_price(S, K, T, r, sigma, N=200, option_type="put", exercise="european")
    lhs = call - put
    rhs = S - K * np.exp(-r * T)
    assert abs(lhs - rhs) < 1e-4


def test_american_put_has_positive_early_exercise_premium() -> None:
    """Deep ITM American put with positive interest rates must exceed European value."""
    res = crr_valuation(S=85.0, K=100.0, T=1.0, r=0.05, sigma=0.20, N=200, option_type="put")
    assert res.price > res.european_price
    assert res.early_exercise_premium > 0.05


def test_non_dividend_american_call_equals_european() -> None:
    """Without dividends (q=0), optimal early exercise of an American call never occurs."""
    res = crr_valuation(S=115.0, K=100.0, T=1.0, r=0.05, sigma=0.20, N=200, option_type="call", q=0.0)
    assert abs(res.early_exercise_premium) < 1e-10


def test_greeks_signs_and_bsm_alignment() -> None:
    """CRR European Greeks should closely track analytical BSM Greeks."""
    S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.25
    crr_call = crr_valuation(S, K, T, r, sigma, N=200, option_type="call", exercise="european")
    bsm_call = bsm_greeks(S, K, T, r, sigma, option_type="call")

    assert 0.0 < crr_call.delta < 1.0
    assert crr_call.gamma > 0.0
    assert crr_call.vega > 0.0
    assert abs(crr_call.delta - bsm_call.delta) < 0.01
    assert abs(crr_call.gamma - bsm_call.gamma) < 0.005


def test_implied_volatility_round_trip() -> None:
    """Inverting BSM price should recover the original volatility."""
    S, K, T, r, true_vol = 100.0, 105.0, 0.5, 0.04, 0.32
    mkt_price = bsm_price(S, K, T, r, true_vol, option_type="call")
    recovered = implied_volatility(mkt_price, S, K, T, r, option_type="call")
    assert abs(recovered - true_vol) < 1e-5