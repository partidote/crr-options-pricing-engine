from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np

OptionType = Literal["call", "put"]
ExerciseStyle = Literal["american", "european"]


@dataclass(frozen=True)
class CRRResult:
    price: float
    european_price: float
    early_exercise_premium: float
    delta: float
    gamma: float
    theta: float  # per calendar day
    vega: float   # per 1% vol move
    rho: float    # per 1% rate move
    steps: int


def _crr_lattice_core(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N: int,
    option_type: OptionType,
    exercise: ExerciseStyle,
    q: float = 0.0,
) -> tuple[float, float, float, float]:
    """Run backward induction on a CRR recombining lattice.

    Returns (V0, delta, gamma, theta_annual) extracted directly from tree levels 0, 1, 2.
    """
    if S <= 0 or K <= 0:
        raise ValueError("Spot S and strike K must be strictly positive.")
    if N < 2:
        raise ValueError("Step count N must be at least 2 to extract Greeks.")
    if T <= 0 or sigma <= 0:
        payoff = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return float(payoff), 0.0, 0.0, 0.0

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)

    if not (0.0 <= p <= 1.0):
        raise ValueError(
            f"Risk-neutral probability p={p:.4f} out of [0, 1]. Increase N or check inputs."
        )

    # Terminal asset prices at step N: S * u^j * d^(N-j) for j = 0..N
    j = np.arange(N + 1)
    ST = S * (u**j) * (d ** (N - j))

    if option_type == "call":
        V = np.maximum(ST - K, 0.0)
    else:
        V = np.maximum(K - ST, 0.0)

    # Store level-2 and level-1 values for Delta, Gamma, Theta
    V_step2 = None
    V_step1 = None

    for step in range(N - 1, -1, -1):
        V = disc * (p * V[1:] + (1.0 - p) * V[:-1])

        if exercise == "american":
            idx = np.arange(step + 1)
            S_step = S * (u**idx) * (d ** (step - idx))
            intrinsic = np.maximum(S_step - K, 0.0) if option_type == "call" else np.maximum(K - S_step, 0.0)
            V = np.maximum(V, intrinsic)

        if step == 2:
            V_step2 = V.copy()
        elif step == 1:
            V_step1 = V.copy()

    V0 = float(V[0])

    # Extract Delta, Gamma, Theta directly from the lattice
    S_u, S_d = S * u, S * d
    delta = (V_step1[1] - V_step1[0]) / (S_u - S_d)

    S_uu, S_ud, S_dd = S * u * u, S, S * d * d
    delta_up = (V_step2[2] - V_step2[1]) / (S_uu - S_ud)
    delta_dn = (V_step2[1] - V_step2[0]) / (S_ud - S_dd)
    gamma = (delta_up - delta_dn) / (0.5 * (S_uu - S_dd))

    theta_annual = (V_step2[1] - V0) / (2.0 * dt)

    return V0, float(delta), float(gamma), float(theta_annual)


def crr_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N: int = 200,
    option_type: OptionType = "call",
    exercise: ExerciseStyle = "american",
    q: float = 0.0,
) -> float:
    """Compute option fair value using the CRR binomial lattice."""
    price, _, _, _ = _crr_lattice_core(S, K, T, r, sigma, N, option_type, exercise, q)
    return price


def crr_valuation(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N: int = 200,
    option_type: OptionType = "call",
    exercise: ExerciseStyle = "american",
    q: float = 0.0,
    vol_bump: float = 0.01,
    rate_bump: float = 0.0001,
) -> CRRResult:
    """Compute CRR option price, early-exercise premium, and full Greeks."""
    price, delta, gamma, theta_annual = _crr_lattice_core(
        S, K, T, r, sigma, N, option_type, exercise, q
    )

    if exercise == "american":
        euro_price, _, _, _ = _crr_lattice_core(S, K, T, r, sigma, N, option_type, "european", q)
    else:
        euro_price = price

    # Vega (per 1% vol move) via central finite difference
    p_vol_up = crr_price(S, K, T, r, sigma + vol_bump, N, option_type, exercise, q)
    p_vol_dn = crr_price(S, K, T, r, max(sigma - vol_bump, 1e-4), N, option_type, exercise, q)
    vega = (p_vol_up - p_vol_dn) / (2.0 * vol_bump * 100.0)

    # Rho (per 1% rate move) via central finite difference
    p_r_up = crr_price(S, K, T, r + rate_bump, sigma, N, option_type, exercise, q)
    p_r_dn = crr_price(S, K, T, r - rate_bump, sigma, N, option_type, exercise, q)
    rho = (p_r_up - p_r_dn) / (2.0 * rate_bump * 100.0)

    return CRRResult(
        price=price,
        european_price=euro_price,
        early_exercise_premium=max(price - euro_price, 0.0),
        delta=delta,
        gamma=gamma,
        theta=theta_annual / 365.0,
        vega=float(vega),
        rho=float(rho),
        steps=N,
    )