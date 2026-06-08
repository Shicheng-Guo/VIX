"""Data layer for the VIX complex.

Fetches and caches the instruments that matter for volatility trading:

Indices (CBOE, via Yahoo):
    ^VIX     30-day implied vol of SPX            (1990-)
    ^VIX9D   9-day  implied vol  (very short end) (2011-)
    ^VIX3M   93-day implied vol  (formerly ^VXV)  (2007-)
    ^VIX6M   6-month implied vol                  (2008-)
    ^VVIX    vol-of-vol (implied vol of VIX)      (2007-)
    ^GSPC    S&P 500 spot                         (1950-)

Tradable vol ETPs (what a strategy actually buys/sells):
    VXX      iPath long  front-month VIX futures  (2018 reissue)
    VIXY     ProShares long VIX short-term futs   (2011-)
    UVXY     ProShares 1.5x long VIX futures      (2011-)
    SVXY     ProShares -0.5x short VIX futures    (2011-)

Everything is cached to ./data as parquet so the tool runs offline after the
first fetch. If the network is unavailable AND there is no cache, a realistic
synthetic dataset is generated so the dashboard always renders.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

INDICES = {
    "VIX": "^VIX",
    "VIX9D": "^VIX9D",
    "VIX3M": "^VIX3M",
    "VIX6M": "^VIX6M",
    "VVIX": "^VVIX",
    "SPX": "^GSPC",
}

ETFS = {
    "VXX": "VXX",
    "VIXY": "VIXY",
    "UVXY": "UVXY",
    "SVXY": "SVXY",
}


@dataclass
class MarketData:
    """Container for the full VIX complex, all aligned on a business-day index."""

    close: pd.DataFrame              # close price / level of every series
    etf_returns: pd.DataFrame        # daily simple returns of the tradable ETPs
    source: str = "unknown"          # "live", "cache", or "synthetic"
    meta: dict = field(default_factory=dict)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.close.index

    def column(self, name: str) -> pd.Series:
        return self.close[name].dropna()


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
def _download(tickers: dict[str, str], period: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        list(tickers.values()),
        period=period,
        progress=False,
        auto_adjust=True,
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("empty download")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    # rename yahoo tickers -> friendly names
    inv = {v: k for k, v in tickers.items()}
    close = close.rename(columns=inv)
    return close


def fetch(period: str = "max", use_cache: bool = True, refresh: bool = False,
          offline: bool = False) -> MarketData:
    """Load the VIX complex, preferring live data, then cache, then synthetic.

    offline=True never touches the network: it serves the cache if present,
    otherwise synthetic data.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    close_path = os.path.join(CACHE_DIR, "close.csv")

    # 1) try live (skipped entirely when offline)
    if not offline and (refresh or not (use_cache and os.path.exists(close_path))):
        try:
            idx = _download(INDICES, period)
            etf = _download(ETFS, period)
            close = idx.join(etf, how="outer").sort_index()
            close = close.loc[close.index >= "1990-01-01"]
            close.to_csv(close_path)
            return _assemble(close, "live")
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"live fetch failed ({exc}); falling back")

    # 2) cache
    if use_cache and os.path.exists(close_path):
        close = pd.read_csv(close_path, index_col=0, parse_dates=True)
        return _assemble(close, "cache")

    # 3) synthetic
    return _assemble(_synthetic(), "synthetic")


def _assemble(close: pd.DataFrame, source: str) -> MarketData:
    close = close.sort_index()
    close.index = pd.to_datetime(close.index)
    # ETF daily simple returns
    etf_cols = [c for c in ETFS if c in close.columns]
    etf_returns = close[etf_cols].pct_change() if etf_cols else pd.DataFrame(index=close.index)
    meta = {
        "start": str(close.index.min().date()),
        "end": str(close.index.max().date()),
        "n_obs": len(close),
        "columns": list(close.columns),
    }
    return MarketData(close=close, etf_returns=etf_returns, source=source, meta=meta)


# --------------------------------------------------------------------------- #
# Synthetic fallback — a stochastic but realistic VIX complex
# --------------------------------------------------------------------------- #
def _synthetic(n_years: int = 18, seed: int = 7) -> pd.DataFrame:
    """Generate a plausible VIX complex via a mean-reverting (CIR-like) process.

    Produces a term structure that is usually in contango but spikes into
    backwardation during engineered 'crisis' windows, plus an SPX series that
    is negatively correlated with vol shocks. Good enough to exercise every
    signal and strategy when no network/cache is available.
    """
    rng = np.random.default_rng(seed)
    n = n_years * 252
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)

    # CIR-style mean reverting VIX
    kappa, theta, xi = 5.0, 17.0, 2.4
    dt = 1 / 252
    vix = np.empty(n)
    vix[0] = theta
    # inject a few crises (spikes)
    crisis = np.zeros(n)
    for c in rng.choice(np.arange(252, n - 60), size=n_years // 3 + 1, replace=False):
        length = rng.integers(15, 60)
        crisis[c : c + length] += rng.uniform(15, 45) * np.exp(
            -np.arange(length) / rng.uniform(8, 20)
        )
    for t in range(1, n):
        dv = kappa * (theta - vix[t - 1]) * dt + xi * np.sqrt(max(vix[t - 1], 1e-3) * dt) * rng.standard_normal()
        vix[t] = max(vix[t - 1] + dv, 8.0)
    vix = vix + crisis
    vix = np.clip(vix, 8.5, 90)

    # term structure: longer tenors are smoother & mean-revert toward theta.
    def tenor(vols, w):
        s = pd.Series(vols).ewm(span=w).mean().to_numpy()
        # in crises the front end inverts above the back end
        return s

    vix9d = np.clip(vix * 1.0 + (vix - theta) * 0.25 + rng.normal(0, 0.4, n), 8, 110)
    vix3m = np.clip(tenor(vix, 12) * 0.4 + theta * 0.6 + 1.5, 9, 70)
    vix6m = np.clip(tenor(vix, 22) * 0.3 + theta * 0.7 + 2.5, 10, 60)
    vvix = np.clip(80 + (vix - theta) * 2.2 + rng.normal(0, 4, n), 60, 200)

    # SPX: drift up, big down moves when vix spikes
    dvix = np.diff(vix, prepend=vix[0])
    spx_ret = 0.0003 - 0.0016 * (dvix / theta) + rng.normal(0, 0.008, n)
    spx = 1500 * np.exp(np.cumsum(spx_ret))

    # ETPs derived from front-end roll. Daily roll cost in contango.
    # Scale the roll to a realistic *daily* drag (~ -0.2%/day in calm contango)
    # and cap to keep the geometric compounding sane.
    roll = (vix - vix3m) / vix3m / 21.0  # per-day roll, negative in contango
    vxx_ret = np.clip(roll * 0.6 + 0.9 * (dvix / vix), -0.35, 0.5) + rng.normal(0, 0.004, n)
    df = pd.DataFrame(index=idx)
    df["VIX"], df["VIX9D"], df["VIX3M"], df["VIX6M"], df["VVIX"], df["SPX"] = (
        vix, vix9d, vix3m, vix6m, vvix, spx,
    )
    df["VXX"] = 100 * np.exp(np.cumsum(vxx_ret))
    df["VIXY"] = 100 * np.exp(np.cumsum(vxx_ret * 0.98))
    df["UVXY"] = 100 * np.exp(np.cumsum(vxx_ret * 1.5))
    df["SVXY"] = 100 * np.exp(np.cumsum(-vxx_ret * 0.5))
    return df


if __name__ == "__main__":
    md = fetch()
    print(f"source={md.source}  {md.meta['start']} -> {md.meta['end']}  ({md.meta['n_obs']} obs)")
    print(md.close.tail())
