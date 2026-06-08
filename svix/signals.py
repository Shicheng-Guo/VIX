"""Signal engine for the VIX complex.

These are the quantitative inputs every volatility strategy keys off of. The
central insight of vol trading: VIX futures spend ~80% of the time in *contango*
(front < back), so a long-vol holder bleeds roll yield while a short-vol holder
harvests it — until the term structure inverts into *backwardation* during a
stress event and the short-vol position takes a violent loss.

Signals produced (all returned as a single aligned DataFrame):

    ts_slope       VIX3M - VIX            (>0 contango, <0 backwardation)
    ts_ratio       VIX / VIX3M            (<1 contango, >1 backwardation)
    near_slope     VIX - VIX9D            (front-end roll)
    contango       boolean: ts_ratio < 1
    roll_yield      annualized carry implied by the 1M/3M slope
    vix_z          rolling z-score of VIX level (mean reversion)
    vix_pctile     rolling percentile rank of VIX
    rv_20          20d realized vol of SPX (annualized, %)
    vrp            variance risk premium = VIX - realized vol
    vvix_z         z-score of vol-of-vol
    regime         categorical: CALM / NORMAL / STRESS / CRISIS
    spx_above_200  SPX > its 200d MA (risk-on trend filter)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import MarketData

TRADING_DAYS = 252


def realized_vol(spx: pd.Series, window: int = 20) -> pd.Series:
    """Close-to-close annualized realized volatility in vol points (e.g. 18.5).

    Computed on the gap-free series so a single missing print (a calendar
    mismatch introduced by the outer join) doesn't blank out 20 windows.
    """
    clean = spx.dropna()
    logret = np.log(clean / clean.shift(1))
    rv = logret.rolling(window).std() * np.sqrt(TRADING_DAYS) * 100
    return rv.reindex(spx.index).ffill()


def rolling_z(x: pd.Series, window: int = 252) -> pd.Series:
    mu = x.rolling(window).mean()
    sd = x.rolling(window).std()
    return (x - mu) / sd


def rolling_pctile(x: pd.Series, window: int = 252) -> pd.Series:
    return x.rolling(window).apply(
        lambda w: (w[-1] >= w).mean() * 100, raw=True
    )


def classify_regime(vix: pd.Series, ts_ratio: pd.Series) -> pd.Series:
    """Map (level, term-structure) into a four-state regime.

    CRISIS  : backwardation AND high VIX            (long vol wins)
    STRESS  : elevated VIX or flat/inverted curve   (de-risk)
    NORMAL  : ordinary contango
    CALM    : low VIX, steep contango               (short vol harvests carry)
    """
    regime = pd.Series("NORMAL", index=vix.index, dtype=object)
    calm = (vix < 15) & (ts_ratio < 0.90)
    stress = (vix >= 20) | (ts_ratio >= 0.97)
    crisis = (ts_ratio >= 1.0) & (vix >= 25)
    regime[calm] = "CALM"
    regime[stress] = "STRESS"
    regime[crisis] = "CRISIS"
    return regime


def compute(md: MarketData, z_window: int = 252) -> pd.DataFrame:
    c = md.close
    out = pd.DataFrame(index=c.index)

    vix = c["VIX"]
    vix3m = c.get("VIX3M")
    vix9d = c.get("VIX9D")
    vix6m = c.get("VIX6M")
    vvix = c.get("VVIX")
    spx = c["SPX"]

    out["VIX"] = vix
    out["VIX3M"] = vix3m
    out["VIX9D"] = vix9d
    out["VIX6M"] = vix6m

    # ----- term structure -----
    if vix3m is not None:
        out["ts_slope"] = vix3m - vix
        out["ts_ratio"] = vix / vix3m
        out["contango"] = out["ts_ratio"] < 1.0
        # annualized roll yield implied by the 1M->3M segment (~42 calendar days)
        # carry a short-vol holder earns per year if the curve stays put.
        out["roll_yield"] = (vix3m / vix - 1.0) * (TRADING_DAYS / 42)
    if vix9d is not None:
        out["near_slope"] = vix - vix9d

    # ----- level / mean reversion -----
    out["vix_z"] = rolling_z(vix, z_window)
    out["vix_pctile"] = rolling_pctile(vix, z_window)

    # ----- variance risk premium -----
    rv = realized_vol(spx, 20)
    out["rv_20"] = rv
    out["vrp"] = vix - rv  # positive = implied richer than realized = sell vol edge

    # ----- vol of vol -----
    if vvix is not None:
        out["vvix"] = vvix
        out["vvix_z"] = rolling_z(vvix, z_window)

    # ----- trend filter -----
    sma200 = spx.rolling(200).mean()
    out["spx"] = spx
    out["spx_sma200"] = sma200
    out["spx_above_200"] = spx > sma200

    # ----- regime -----
    if "ts_ratio" in out:
        out["regime"] = classify_regime(vix, out["ts_ratio"])
    else:
        out["regime"] = np.where(vix < 15, "CALM", np.where(vix > 25, "CRISIS", "NORMAL"))

    return out


def latest_snapshot(sig: pd.DataFrame) -> dict:
    """Human-readable summary of the most recent observation — the 'what now?'."""
    row = sig.dropna(subset=["VIX"]).iloc[-1]
    snap = {
        "date": str(sig.index[-1].date()),
        "VIX": round(float(row["VIX"]), 2),
        "regime": str(row.get("regime", "NA")),
    }
    if "ts_ratio" in sig:
        r = float(row["ts_ratio"])
        snap["ts_ratio"] = round(r, 3)
        snap["structure"] = "CONTANGO" if r < 1 else "BACKWARDATION"
        snap["roll_yield_ann_%"] = round(float(row["roll_yield"]) * 100, 1)
    if "vrp" in sig and pd.notna(row.get("vrp")):
        snap["VRP"] = round(float(row["vrp"]), 2)
    if pd.notna(row.get("vix_pctile")):
        snap["vix_1y_pctile"] = round(float(row["vix_pctile"]), 0)
    snap["spx_above_200dma"] = bool(row.get("spx_above_200", False))
    return snap


if __name__ == "__main__":
    from .data import fetch

    md = fetch()
    sig = compute(md)
    import json

    print(json.dumps(latest_snapshot(sig), indent=2))
    print(sig.tail()[["VIX", "ts_ratio", "roll_yield", "vrp", "regime"]])
