# sVIX — VIX Volatility Strategy & Visualization Toolkit

A comprehensive, end-to-end research tool for **volatility (VIX) investing**. It
pulls the full VIX complex, turns it into the signals that drive vol trading,
backtests six strategies on real ETF returns, and renders a single interactive
HTML dashboard that tells you *where the volatility market is now and what the
model would do about it*.

> ⚠️ **Research & education only — not investment advice.** Short-volatility
> strategies carry severe tail risk (e.g. SVXY lost ~90% in a day on
> 5 Feb 2018). Backtests include those events and are net of trading costs, but
> past performance does not predict future results.

---

## What it does

```
data ─▶ signals ─▶ strategies ─▶ backtest ─▶ dashboard.html
```

### 1. Data (`svix/data.py`)
Fetches and caches (CSV, so it runs offline after first use) the instruments
that matter for vol trading:

| Group | Series |
|-------|--------|
| Term structure | `^VIX9D` `^VIX` `^VIX3M` `^VIX6M` |
| Vol-of-vol | `^VVIX` |
| Equity | `^GSPC` (S&P 500) |
| Tradable ETPs | `SVXY` (short vol) · `VXX`/`VIXY` (long vol) · `UVXY` (1.5× long) |

If the network and cache are both unavailable, a realistic **synthetic** VIX
complex (mean-reverting CIR process with engineered crises) is generated so the
dashboard always renders.

### 2. Signals (`svix/signals.py`)
The quantitative inputs every vol strategy keys off of:

- **Term-structure slope & ratio** (`VIX/VIX3M`) — contango vs backwardation
- **Roll yield** — the annualized carry a short-vol holder earns
- **Variance risk premium (VRP)** — implied (VIX) minus 20-day realized vol
- **Mean-reversion z-score & percentile** of the VIX level
- **Vol-of-vol** z-score (VVIX)
- **SPX 200-day trend filter**
- **Regime classifier** → `CALM / NORMAL / STRESS / CRISIS`

### 3. Strategies (`svix/strategies.py`)
Each emits target weights over `{SVXY, VXX, SPX, CASH}` with **no look-ahead**
(weights are lagged one day before returns apply):

| Strategy | Idea |
|----------|------|
| **Regime Switch** (flagship) | Allocate short/long vol by the four-state regime |
| **Roll-Yield Carry** | Short vol while the curve is in steep contango + uptrend |
| **VRP Carry** | Short vol sized by the variance risk premium |
| **VIX Mean Reversion** | Fade stretched-high VIX (z-score) |
| **Tactical Long Vol** | Hold VXX only during backwardation |
| **Tail-Hedged Equity** | SPX + permanent 5% long-vol hedge sleeve |

### 4. Backtest & metrics (`svix/backtest.py`, `svix/metrics.py`)
Vectorized daily backtester with transaction costs. Reports CAGR, annualized
vol, Sharpe, Sortino, max drawdown, Calmar, win rate, time-in-market and
turnover, all benchmarked against buy-and-hold SPX.

### 5. Dashboard (`svix/dashboard.py`)
One self-contained `output/dashboard.html` (Plotly via CDN): KPI header, the
live signal call, term-structure curve, regime-shaded VIX history, the
contango/backwardation ratio, the VRP, strategy equity curves, drawdowns, the
flagship's live allocation, and a full metrics table.

---

## Quick start

```bash
pip install -r requirements.txt

python run.py                      # live data → backtest → output/dashboard.html
python run.py --signal             # just print the current signal snapshot
python run.py --start 2010-01-01   # custom backtest window
python run.py --flagship vrp_carry # pick the headline strategy
python run.py --offline            # use cache / synthetic only
python run.py --refresh            # force a fresh download
python run.py --open               # open the dashboard in a browser
```

Then open `output/dashboard.html` in any browser.

## Project layout

```
VIX/
├── run.py               # CLI entrypoint
├── requirements.txt
├── svix/
│   ├── data.py          # fetch + cache the VIX complex (offline fallback)
│   ├── signals.py       # term structure, VRP, z-scores, regimes
│   ├── strategies.py    # 6 tradable strategies
│   ├── backtest.py      # vectorized daily backtester
│   ├── metrics.py       # performance & risk stats
│   └── dashboard.py     # interactive Plotly HTML report
├── data/                # cached CSV (auto-created)
└── output/              # dashboard.html (auto-created)
```

Each module is runnable standalone for debugging, e.g. `python -m svix.signals`.
