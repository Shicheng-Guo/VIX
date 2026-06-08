#!/usr/bin/env python3
"""sVIX — VIX volatility strategy & visualization toolkit (CLI entrypoint).

Examples
--------
    python run.py                       # fetch live data, backtest all, build dashboard
    python run.py --start 2012-01-01    # restrict backtest window
    python run.py --offline             # use cache / synthetic data only
    python run.py --refresh             # force fresh download
    python run.py --signal              # just print the current signal snapshot
    python run.py --flagship vrp_carry  # choose which strategy drives the headline call
"""
from __future__ import annotations

import argparse
import json
import webbrowser

from svix import backtest, dashboard
from svix.data import fetch
from svix.signals import compute, latest_snapshot


def main() -> None:
    p = argparse.ArgumentParser(description="sVIX volatility strategy toolkit")
    p.add_argument("--start", default="2012-01-01",
                   help="backtest start date (YYYY-MM-DD); default 2012-01-01")
    p.add_argument("--cost-bps", type=float, default=5.0,
                   help="one-way transaction cost in bps per unit turnover")
    p.add_argument("--flagship", default="regime_switch",
                   help="strategy key used for the headline signal/allocation")
    p.add_argument("--offline", action="store_true", help="use cache/synthetic only")
    p.add_argument("--refresh", action="store_true", help="force a fresh download")
    p.add_argument("--signal", action="store_true",
                   help="print the current signal snapshot and exit (no dashboard)")
    p.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    args = p.parse_args()

    print("• Loading VIX complex…")
    md = fetch(period="max", use_cache=True, refresh=args.refresh, offline=args.offline)
    print(f"  source={md.source}  {md.meta['start']} → {md.meta['end']}  ({md.meta['n_obs']} obs)")

    print("• Computing signals…")
    sig = compute(md)
    snap = latest_snapshot(sig)
    print("  current snapshot:")
    print("    " + json.dumps(snap, indent=2).replace("\n", "\n    "))

    if args.signal:
        return

    print(f"• Backtesting strategies from {args.start} (cost {args.cost_bps}bps)…")
    results = backtest.run_all(md, sig, cost_bps=args.cost_bps, start=args.start)
    print(f"  {'strategy':<26}{'CAGR%':>8}{'Sharpe':>8}{'MaxDD%':>9}{'Calmar':>8}")
    for r in results.values():
        s = r.stats
        print(f"  {r.name:<26}{s['CAGR_%']:>8}{s['Sharpe']:>8}{s['MaxDD_%']:>9}{s['Calmar']:>8}")

    print("• Building dashboard…")
    out = dashboard.build(results, sig, md, flagship_key=args.flagship, cost_bps=args.cost_bps)
    print(f"✓ Dashboard written to {out}")
    if args.open:
        webbrowser.open(f"file://{out}")


if __name__ == "__main__":
    main()
