"""sVIX — a comprehensive VIX / volatility investment strategy & visualization toolkit.

Modules
-------
data        : fetch + cache the VIX complex (term structure, VVIX, SPX, vol ETFs)
signals     : term-structure slope, contango/backwardation, VRP, z-scores, regimes
strategies  : tradable strategies built on those signals
backtest     : vectorized daily backtester with transaction costs
metrics     : performance & risk statistics
dashboard   : self-contained interactive Plotly HTML report
"""

__version__ = "1.0.0"

# Submodules are imported lazily by callers (they import each other), so we
# deliberately do NOT eagerly import them here to avoid circular-import races.
