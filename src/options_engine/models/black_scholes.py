from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class BSMResult:
    price: float
    delta: float
    gamma: float
    theta: float  # per calendar day
    vega: float   # per 1% vol move
    rho: float    # per 1% rate move


def _validate_inputs(S: float, K: float, T: float, sigma: float, option_type: str) -> None:
    if S <= 0 or K <= 0:
        raise ValueError("Spot price S and strike K must be strictly positive.")
    if T < 0:
        raise ValueError("Time to maturity T cannot be negative.")
    if sigma < 0:
        raise ValueError("Volatility sigma cannot be negative.")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")


def bsm_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    q: float = 0.0,
) -> float:
    """Compute analytical European option price under Black-Scholes-Merton."""
    _validate_inputs(S, K, T, sigma, option_type)

    if T == 0 or sigma == 0:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return float(intrinsic)

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if option_type == "call":
        return float(S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1))


def bsm_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    q: float = 0.0,
) -> BSMResult:
    """Compute analytical BSM price and Greeks."""
    _validate_inputs(S, K, T, sigma, option_type)
    price = bsm_price(S, K, T, r, sigma, option_type, q)

    if T == 0 or sigma == 0:
        delta = (1.0 if S > K else 0.0) if option_type == "call" else (-1.0 if S < K else 0.0)
        return BSMResult(price=price, delta=delta, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    pdf_d1 = norm.pdf(d1)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    gamma = (disc_q * pdf_d1) / (S * sigma * sqrt_T)
    vega = (S * disc_q * pdf_d1 * sqrt_T) / 100.0

    if option_type == "call":
        delta = disc_q * norm.cdf(d1)
        theta_annual = (
            -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrt_T)
            - r * K * disc_r * norm.cdf(d2)
            + q * S * disc_q * norm.cdf(d1)
        )
        rho = (K * T * disc_r * norm.cdf(d2)) / 100.0
    else:
        delta = disc_q * (norm.cdf(d1) - 1.0)
        theta_annual = (
            -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrt_T)
            + r * K * disc_r * norm.cdf(-d2)
            - q * S * disc_q * norm.cdf(-d1)
        )
        rho = (-K * T * disc_r * norm.cdf(-d2)) / 100.0

    return BSMResult(
        price=float(price),
        delta=float(delta),
        gamma=float(gamma),
        theta=float(theta_annual / 365.0),
        vega=float(vega),
        rho=float(rho),
    )


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType = "call",
    q: float = 0.0,
    tol: float = 1e-6,
) -> float:
    """Invert BSM price for implied volatility using Brent's method."""
    if market_price <= 0 or T <= 0:
        return float("nan")

    def objective(vol: float) -> float:
        return bsm_price(S, K, T, r, vol, option_type, q) - market_price

    try:
        return float(brentq(objective, 1e-5, 5.0, xtol=tol, maxiter=200))
    except ValueError:
        return float("nan")