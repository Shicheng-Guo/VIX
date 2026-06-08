"""Performance & risk statistics for a daily return / equity series."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def equity_curve(returns: pd.Series, start: float = 1.0) -> pd.Series:
    return start * (1 + returns.fillna(0)).cumprod()


def cagr(returns: pd.Series) -> float:
    eq = equity_curve(returns)
    if len(eq) < 2:
        return 0.0
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    if years <= 0 or eq.iloc[-1] <= 0:
        return 0.0
    return eq.iloc[-1] ** (1 / years) - 1


def ann_vol(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    ex = returns - rf / TRADING_DAYS
    sd = ex.std()
    return (ex.mean() / sd) * np.sqrt(TRADING_DAYS) if sd > 0 else 0.0


def sortino(returns: pd.Series, rf: float = 0.0) -> float:
    ex = returns - rf / TRADING_DAYS
    downside = ex[ex < 0].std()
    return (ex.mean() / downside) * np.sqrt(TRADING_DAYS) if downside and downside > 0 else 0.0


def max_drawdown(returns: pd.Series) -> float:
    eq = equity_curve(returns)
    return (eq / eq.cummax() - 1).min()


def drawdown_series(returns: pd.Series) -> pd.Series:
    eq = equity_curve(returns)
    return eq / eq.cummax() - 1


def calmar(returns: pd.Series) -> float:
    mdd = abs(max_drawdown(returns))
    return cagr(returns) / mdd if mdd > 0 else 0.0


def win_rate(returns: pd.Series) -> float:
    r = returns.dropna()
    r = r[r != 0]
    return (r > 0).mean() if len(r) else 0.0


def turnover(weights: pd.Series) -> float:
    """Average daily absolute change in position (one-way), annualized."""
    return weights.diff().abs().mean() * TRADING_DAYS


def summary(returns: pd.Series, weights: pd.Series | None = None, rf: float = 0.0) -> dict:
    returns = returns.dropna()
    s = {
        "CAGR_%": round(cagr(returns) * 100, 2),
        "Vol_%": round(ann_vol(returns) * 100, 2),
        "Sharpe": round(sharpe(returns, rf), 2),
        "Sortino": round(sortino(returns, rf), 2),
        "MaxDD_%": round(max_drawdown(returns) * 100, 2),
        "Calmar": round(calmar(returns), 2),
        "WinRate_%": round(win_rate(returns) * 100, 1),
        "BestDay_%": round(returns.max() * 100, 2),
        "WorstDay_%": round(returns.min() * 100, 2),
    }
    if weights is not None:
        s["Turnover_x"] = round(turnover(weights.reindex(returns.index).fillna(0)), 1)
        s["TimeInMkt_%"] = round((weights.reindex(returns.index).fillna(0) != 0).mean() * 100, 1)
    return s
