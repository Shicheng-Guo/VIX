"""Vectorized daily backtester.

Takes a strategy's target weights, lags them one trading day (decide at close
t, hold over t+1) to avoid look-ahead, applies per-instrument daily returns,
and subtracts transaction costs proportional to turnover.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics
from .data import MarketData
from .strategies import Strategy, UNIVERSE


@dataclass
class BacktestResult:
    key: str
    name: str
    returns: pd.Series          # daily net strategy returns
    equity: pd.Series           # growth of $1
    weights: pd.DataFrame       # realized (lagged) weights
    drawdown: pd.Series
    stats: dict

    def gross_exposure(self) -> pd.Series:
        return self.weights.drop(columns=["CASH"]).abs().sum(axis=1)


def asset_returns(md: MarketData) -> pd.DataFrame:
    """Daily simple returns for every tradable instrument, incl. SPX and CASH."""
    cols = {}
    for c in ["SVXY", "VXX", "SPX"]:
        if c in md.close.columns:
            cols[c] = md.close[c].pct_change()
    df = pd.DataFrame(cols)
    df["CASH"] = 0.0
    return df


def run(
    strategy: Strategy,
    sig: pd.DataFrame,
    md: MarketData,
    cost_bps: float = 5.0,
    start: str | None = None,
) -> BacktestResult:
    """Backtest one strategy. `cost_bps` is one-way cost per unit turnover."""
    rets = asset_returns(md)
    w = strategy.weights(sig)

    # align to the period where the traded instruments actually have data
    traded = [c for c in UNIVERSE if c != "CASH" and c in rets.columns]
    valid = rets[traded].dropna(how="all").index
    if start:
        valid = valid[valid >= pd.Timestamp(start)]
    w = w.reindex(valid).fillna(0.0)
    rets = rets.reindex(valid).fillna(0.0)

    # hold yesterday's target over today's return
    held = w.shift(1).fillna(0.0)
    gross = (held[UNIVERSE] * rets[UNIVERSE]).sum(axis=1)

    # transaction cost on traded notional
    turn = held.diff().abs().sum(axis=1).fillna(0.0)
    cost = turn * (cost_bps / 1e4)
    net = gross - cost

    equity = metrics.equity_curve(net)
    dd = metrics.drawdown_series(net)
    invested = held.drop(columns=["CASH"]).abs().sum(axis=1)
    stats = metrics.summary(net, weights=invested)
    stats["Trades/yr"] = round(float((turn > 1e-6).mean() * 252), 0)

    return BacktestResult(
        key=strategy.key, name=strategy.name, returns=net,
        equity=equity, weights=held, drawdown=dd, stats=stats,
    )


def benchmark(md: MarketData, sig: pd.DataFrame, col: str = "SPX",
              start: str | None = None) -> BacktestResult:
    """Buy-and-hold benchmark (default SPX) on the same calendar."""
    rets = asset_returns(md)
    idx = rets[col].dropna().index
    if start:
        idx = idx[idx >= pd.Timestamp(start)]
    net = rets[col].reindex(idx).fillna(0.0)
    w = pd.DataFrame(0.0, index=idx, columns=UNIVERSE)
    w[col] = 1.0
    equity = metrics.equity_curve(net)
    return BacktestResult(
        key=f"bench_{col.lower()}", name=f"Buy & Hold {col}",
        returns=net, equity=equity, weights=w,
        drawdown=metrics.drawdown_series(net), stats=metrics.summary(net),
    )


def run_all(md: MarketData, sig: pd.DataFrame, cost_bps: float = 5.0,
            start: str | None = None) -> dict[str, BacktestResult]:
    from .strategies import all_strategies

    results: dict[str, BacktestResult] = {}
    for strat in all_strategies():
        try:
            results[strat.key] = run(strat, sig, md, cost_bps=cost_bps, start=start)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] strategy {strat.key} failed: {exc}")
    results["bench_spx"] = benchmark(md, sig, "SPX", start=start)
    return results


if __name__ == "__main__":
    from .data import fetch
    from .signals import compute

    md = fetch()
    sig = compute(md)
    res = run_all(md, sig, start="2012-01-01")
    print(f"{'strategy':<26}{'CAGR%':>8}{'Vol%':>8}{'Sharpe':>8}{'MaxDD%':>9}{'Calmar':>8}")
    for r in res.values():
        s = r.stats
        print(f"{r.name:<26}{s['CAGR_%']:>8}{s['Vol_%']:>8}{s['Sharpe']:>8}{s['MaxDD_%']:>9}{s['Calmar']:>8}")
