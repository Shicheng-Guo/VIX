"""Tradable volatility strategies.

Each strategy consumes the signal DataFrame (see :mod:`svix.signals`) and emits
a **target-weight DataFrame** whose columns are instruments in the tradable
universe. Weights are *target exposures decided at the close of day t*; the
backtester lags them one day before applying returns, so there is no
look-ahead. Anything not allocated sits in cash (0% return).

Tradable universe
-----------------
    SVXY  short-vol  (harvests contango / VRP; the carry engine)
    VXX   long-vol   (tail capture; bleeds in contango)
    SPX   equity     (benchmark / overlay host)
    CASH  risk-free-ish 0%

Strategies
----------
    roll_yield_carry   short vol while curve is in steep contango + uptrend
    vrp_carry          short vol scaled by the variance risk premium
    mean_reversion     short vol when VIX is stretched high (reverts down)
    long_vol_tactical  long vol only while curve is in backwardation
    regime_switch      flagship: allocate by CALM/NORMAL/STRESS/CRISIS regime
    tail_hedged_equity SPX with a permanent small long-vol hedge sleeve
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

UNIVERSE = ["SVXY", "VXX", "SPX", "CASH"]


@dataclass
class Strategy:
    key: str
    name: str
    description: str
    func: Callable[[pd.DataFrame], pd.DataFrame]

    def weights(self, sig: pd.DataFrame) -> pd.DataFrame:
        w = self.func(sig)
        # ensure all universe columns exist, fill cash with the remainder
        for col in UNIVERSE:
            if col not in w.columns:
                w[col] = 0.0
        w = w[UNIVERSE].fillna(0.0)
        invested = w.drop(columns=["CASH"]).clip(lower=-1, upper=1).sum(axis=1)
        w["CASH"] = (1.0 - invested).clip(lower=0.0)
        return w


def _frame(index) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=index, columns=["SVXY", "VXX", "SPX"])


# --------------------------------------------------------------------------- #
# Individual strategies
# --------------------------------------------------------------------------- #
def _roll_yield_carry(sig: pd.DataFrame, enter=0.95, exit_=0.99) -> pd.DataFrame:
    """Long SVXY while the curve is in steep contango and SPX trends up.

    Hysteresis: enter when VIX/VIX3M < `enter`, stay until it crosses `exit_`,
    forcing flat the moment the curve flattens/inverts (the dangerous regime).
    """
    w = _frame(sig.index)
    ratio = sig["ts_ratio"]
    trend = sig["spx_above_200"].fillna(False)
    state = np.zeros(len(sig))
    on = False
    rv, tv = ratio.to_numpy(), trend.to_numpy()
    for i in range(len(sig)):
        if np.isnan(rv[i]):
            on = False
        elif not on and rv[i] < enter and tv[i]:
            on = True
        elif on and (rv[i] > exit_ or not tv[i]):
            on = False
        state[i] = 1.0 if on else 0.0
    w["SVXY"] = state
    return w


def _vrp_carry(sig: pd.DataFrame, cap=1.0, scale=10.0) -> pd.DataFrame:
    """Short vol sized by the variance risk premium (VIX - realized vol).

    Bigger premium -> bigger short-vol position; flat when the premium is
    negative (realized > implied = no edge / vol underpriced).
    """
    w = _frame(sig.index)
    vrp = sig["vrp"]
    contango = sig["ts_ratio"] < 1.0
    pos = (vrp / scale).clip(lower=0, upper=cap)
    pos = pos.where(contango, 0.0)  # only when carry is on your side
    w["SVXY"] = pos.fillna(0.0)
    return w


def _mean_reversion(sig: pd.DataFrame, z_hi=1.5, z_lo=-1.0) -> pd.DataFrame:
    """VIX is mean-reverting: fade spikes (short vol) and respect crushes.

    When the 1y z-score of VIX is high AND the curve has not inverted, short
    vol expecting reversion down. When VIX is unusually low, step aside (carry
    is thin and a spike is overdue).
    """
    w = _frame(sig.index)
    z = sig["vix_z"]
    not_crisis = sig["ts_ratio"] < 1.02
    short_vol = ((z > z_hi) & not_crisis).astype(float)
    w["SVXY"] = short_vol
    return w


def _long_vol_tactical(sig: pd.DataFrame) -> pd.DataFrame:
    """Hold VXX only while the curve is in backwardation (stress is paying)."""
    w = _frame(sig.index)
    backwardation = (sig["ts_ratio"] >= 1.0).astype(float)
    w["VXX"] = backwardation
    return w


def _regime_switch(sig: pd.DataFrame) -> pd.DataFrame:
    """Flagship: position by regime, the synthesis of every signal.

        CALM    -> 100% short vol  (harvest steep contango carry)
        NORMAL  ->  50% short vol  (carry on, half size)
        STRESS  -> flat / cash     (capital preservation)
        CRISIS  ->  30% long vol   (small tactical tail capture)
    """
    w = _frame(sig.index)
    reg = sig["regime"]
    w.loc[reg == "CALM", "SVXY"] = 1.0
    w.loc[reg == "NORMAL", "SVXY"] = 0.5
    w.loc[reg == "STRESS", "SVXY"] = 0.0
    w.loc[reg == "CRISIS", "VXX"] = 0.30
    return w


def _tail_hedged_equity(sig: pd.DataFrame, hedge=0.05) -> pd.DataFrame:
    """Buy-and-hold SPX with a constant small long-vol hedge sleeve.

    Demonstrates the classic trade-off: the VXX sleeve bleeds in calm markets
    but cushions crashes. Compare against plain SPX in the dashboard.
    """
    w = _frame(sig.index)
    w["SPX"] = 1.0 - hedge
    w["VXX"] = hedge
    return w


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
REGISTRY: dict[str, Strategy] = {
    "regime_switch": Strategy(
        "regime_switch", "Regime Switch (flagship)",
        "Allocate short/long vol by CALM/NORMAL/STRESS/CRISIS regime.",
        _regime_switch,
    ),
    "roll_yield_carry": Strategy(
        "roll_yield_carry", "Roll-Yield Carry",
        "Short vol while the curve is in steep contango and SPX trends up.",
        _roll_yield_carry,
    ),
    "vrp_carry": Strategy(
        "vrp_carry", "VRP Carry",
        "Short vol sized by the variance risk premium (VIX - realized vol).",
        _vrp_carry,
    ),
    "mean_reversion": Strategy(
        "mean_reversion", "VIX Mean Reversion",
        "Fade VIX spikes (short vol) when the 1y z-score is stretched high.",
        _mean_reversion,
    ),
    "long_vol_tactical": Strategy(
        "long_vol_tactical", "Tactical Long Vol",
        "Hold VXX only while the term structure is in backwardation.",
        _long_vol_tactical,
    ),
    "tail_hedged_equity": Strategy(
        "tail_hedged_equity", "Tail-Hedged Equity",
        "Buy-and-hold SPX with a permanent 5% long-vol hedge sleeve.",
        _tail_hedged_equity,
    ),
}


def get(key: str) -> Strategy:
    return REGISTRY[key]


def all_strategies() -> list[Strategy]:
    return list(REGISTRY.values())
