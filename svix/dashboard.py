"""Self-contained interactive Plotly HTML dashboard.

Renders a single ``dashboard.html`` with:
  * a KPI header (current regime, VIX, term-structure state, the live signal call)
  * the live VIX term-structure curve
  * VIX history with regime shading
  * the term-structure ratio (contango vs backwardation zones)
  * the variance risk premium
  * strategy equity curves vs SPX (log scale)
  * the drawdown comparison
  * a sortable performance-metrics table
  * the flagship strategy's current target allocation

Everything is embedded in one file (Plotly via CDN) so it opens in any browser.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .backtest import BacktestResult
from .signals import latest_snapshot

# colour system ------------------------------------------------------------- #
BG = "#0e1117"
PANEL = "#161b26"
GRID = "#243044"
TEXT = "#e6edf3"
MUTED = "#8b98a9"
ACCENT = "#4dabf7"
GREEN = "#51cf66"
RED = "#ff6b6b"
AMBER = "#ffd43b"
ORANGE = "#ff922b"

REGIME_COLORS = {
    "CALM": "rgba(81,207,102,0.10)",
    "NORMAL": "rgba(77,171,247,0.06)",
    "STRESS": "rgba(255,212,59,0.12)",
    "CRISIS": "rgba(255,107,107,0.16)",
}
STRAT_COLORS = [ACCENT, GREEN, ORANGE, "#cc5de8", "#22b8cf", AMBER, RED]

PLOT_LAYOUT = dict(
    paper_bgcolor=PANEL, plot_bgcolor=PANEL,
    font=dict(color=TEXT, family="Inter, system-ui, sans-serif", size=12),
    margin=dict(l=55, r=25, t=50, b=40),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.18),
    hovermode="x unified",
)


def _fig_html(fig: go.Figure, div_id: str) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id,
                       config={"displayModeBar": False, "responsive": True})


# --------------------------------------------------------------------------- #
# Individual figures
# --------------------------------------------------------------------------- #
def fig_term_structure(sig: pd.DataFrame) -> go.Figure:
    row = sig.dropna(subset=["VIX"]).iloc[-1]
    tenors, vals = [], []
    for label, days in [("VIX9D", 9), ("VIX", 30), ("VIX3M", 93), ("VIX6M", 186)]:
        if label in sig.columns and pd.notna(row.get(label)):
            tenors.append(days)
            vals.append(float(row[label]))
    fig = go.Figure()
    if vals:
        up = vals[-1] >= vals[0]
        fig.add_trace(go.Scatter(
            x=tenors, y=vals, mode="lines+markers+text",
            text=[f"{v:.1f}" for v in vals], textposition="top center",
            line=dict(color=GREEN if up else RED, width=3),
            marker=dict(size=10), name="Term structure",
        ))
    fig.update_layout(**PLOT_LAYOUT, title="Live VIX Term Structure",
                      height=300)
    fig.update_xaxes(title="Tenor (calendar days)")
    fig.update_yaxes(title="Implied vol")
    return fig


def fig_vix_history(sig: pd.DataFrame, years: int = 6) -> go.Figure:
    cut = sig.index.max() - pd.DateOffset(years=years)
    s = sig.loc[sig.index >= cut]
    fig = go.Figure()
    # regime shading
    if "regime" in s.columns:
        reg = s["regime"]
        change = (reg != reg.shift()).cumsum()
        for _, grp in s.groupby(change):
            r = grp["regime"].iloc[0]
            fig.add_vrect(x0=grp.index[0], x1=grp.index[-1],
                          fillcolor=REGIME_COLORS.get(r, "rgba(0,0,0,0)"),
                          line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=s.index, y=s["VIX"], name="VIX",
                             line=dict(color=ACCENT, width=1.4)))
    if "VIX3M" in s.columns:
        fig.add_trace(go.Scatter(x=s.index, y=s["VIX3M"], name="VIX3M",
                                 line=dict(color=MUTED, width=1, dash="dot")))
    fig.add_hline(y=20, line=dict(color=AMBER, width=1, dash="dash"),
                  annotation_text="20", annotation_position="right")
    fig.update_layout(**PLOT_LAYOUT, title="VIX vs VIX3M — shaded by regime",
                      height=340)
    fig.update_yaxes(title="Vol points")
    return fig


def fig_ratio(sig: pd.DataFrame, years: int = 6) -> go.Figure:
    cut = sig.index.max() - pd.DateOffset(years=years)
    s = sig.loc[sig.index >= cut]
    fig = go.Figure()
    fig.add_hrect(y0=0.6, y1=1.0, fillcolor="rgba(81,207,102,0.07)", line_width=0,
                  annotation_text="CONTANGO (short-vol carry)", annotation_position="bottom left",
                  annotation=dict(font=dict(color=GREEN, size=11)))
    fig.add_hrect(y0=1.0, y1=1.6, fillcolor="rgba(255,107,107,0.10)", line_width=0,
                  annotation_text="BACKWARDATION (stress)", annotation_position="top left",
                  annotation=dict(font=dict(color=RED, size=11)))
    fig.add_trace(go.Scatter(x=s.index, y=s["ts_ratio"], name="VIX / VIX3M",
                             line=dict(color=ACCENT, width=1.3)))
    fig.add_hline(y=1.0, line=dict(color=TEXT, width=1))
    fig.update_layout(**PLOT_LAYOUT, title="Term-Structure Ratio (VIX / VIX3M)",
                      height=300, showlegend=False)
    fig.update_yaxes(title="Ratio", range=[0.6, 1.6])
    return fig


def fig_vrp(sig: pd.DataFrame, years: int = 6) -> go.Figure:
    cut = sig.index.max() - pd.DateOffset(years=years)
    s = sig.loc[sig.index >= cut]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s["VIX"], name="Implied (VIX)",
                             line=dict(color=ACCENT, width=1.2)))
    fig.add_trace(go.Scatter(x=s.index, y=s["rv_20"], name="Realized (20d)",
                             line=dict(color=ORANGE, width=1.2)))
    fig.add_trace(go.Bar(x=s.index, y=s["vrp"], name="VRP (premium)",
                         marker=dict(color=np.where(s["vrp"] >= 0, GREEN, RED)),
                         opacity=0.35))
    fig.update_layout(**PLOT_LAYOUT, title="Variance Risk Premium — implied vs realized",
                      height=300, barmode="overlay")
    fig.update_yaxes(title="Vol points")
    return fig


def fig_equity(results: dict[str, BacktestResult]) -> go.Figure:
    fig = go.Figure()
    for i, r in enumerate(results.values()):
        is_bench = r.key.startswith("bench")
        fig.add_trace(go.Scatter(
            x=r.equity.index, y=r.equity.values, name=r.name,
            line=dict(color="#ffffff" if is_bench else STRAT_COLORS[i % len(STRAT_COLORS)],
                      width=2.4 if is_bench else 1.7,
                      dash="dash" if is_bench else "solid"),
        ))
    fig.update_layout(**PLOT_LAYOUT, title="Growth of $1 — strategies vs SPX (log scale)",
                      height=420)
    fig.update_yaxes(title="Equity ($, log)", type="log")
    return fig


def fig_drawdown(results: dict[str, BacktestResult]) -> go.Figure:
    fig = go.Figure()
    for i, r in enumerate(results.values()):
        is_bench = r.key.startswith("bench")
        fig.add_trace(go.Scatter(
            x=r.drawdown.index, y=r.drawdown.values * 100, name=r.name,
            line=dict(color="#ffffff" if is_bench else STRAT_COLORS[i % len(STRAT_COLORS)],
                      width=1.6, dash="dash" if is_bench else "solid"),
        ))
    fig.update_layout(**PLOT_LAYOUT, title="Drawdowns (%)", height=300)
    fig.update_yaxes(title="Drawdown %")
    return fig


def fig_allocation(result: BacktestResult) -> go.Figure:
    """Recent target-allocation timeline for the flagship strategy."""
    w = result.weights.tail(252 * 2)
    fig = go.Figure()
    palette = {"SVXY": GREEN, "VXX": RED, "SPX": ACCENT, "CASH": MUTED}
    for col in ["SVXY", "VXX", "SPX", "CASH"]:
        if col in w.columns and w[col].abs().sum() > 0:
            fig.add_trace(go.Scatter(
                x=w.index, y=w[col] * 100, name=col, stackgroup="one",
                line=dict(width=0.5, color=palette.get(col, MUTED)),
                fillcolor=palette.get(col, MUTED),
            ))
    fig.update_layout(**PLOT_LAYOUT, title=f"{result.name} — target allocation (last 2y)",
                      height=300)
    fig.update_yaxes(title="Weight %", range=[0, 100])
    return fig


# --------------------------------------------------------------------------- #
# HTML assembly
# --------------------------------------------------------------------------- #
def _kpi_cards(snap: dict, source: str) -> str:
    regime = snap.get("regime", "NA")
    rcolor = {"CALM": GREEN, "NORMAL": ACCENT, "STRESS": AMBER, "CRISIS": RED}.get(regime, MUTED)
    struct = snap.get("structure", "—")
    scolor = GREEN if struct == "CONTANGO" else RED
    cards = [
        ("Current VIX", f"{snap.get('VIX','—')}", f"1y pctile {snap.get('vix_1y_pctile','—')}%", ACCENT),
        ("Regime", regime, snap.get("date", ""), rcolor),
        ("Term Structure", struct, f"ratio {snap.get('ts_ratio','—')}", scolor),
        ("Roll Yield (ann)", f"{snap.get('roll_yield_ann_%','—')}%",
         "short-vol carry" if snap.get('roll_yield_ann_%', 0) > 0 else "negative carry", scolor),
        ("Variance Risk Prem", f"{snap.get('VRP','—')}", "implied − realized",
         GREEN if (snap.get('VRP') or 0) > 0 else RED),
        ("SPX Trend", "RISK-ON" if snap.get("spx_above_200dma") else "RISK-OFF",
         "vs 200d MA", GREEN if snap.get("spx_above_200dma") else RED),
    ]
    html = ""
    for label, value, sub, color in cards:
        html += f"""
        <div class="card">
          <div class="card-label">{label}</div>
          <div class="card-value" style="color:{color}">{value}</div>
          <div class="card-sub">{sub}</div>
        </div>"""
    return html


def _signal_call(snap: dict, target: pd.DataFrame) -> str:
    """The explicit BUY / SELL / HOLD signal — what the model says to do today.

    `target` is the flagship strategy's full target-weight DataFrame. Today's
    row is the desired position; the change vs yesterday is the actual trade.
    """
    today = target.iloc[-1]
    prev = target.iloc[-2] if len(target) > 1 else today * 0

    # ---- directional stance ----
    svxy, vxx = float(today.get("SVXY", 0)), float(today.get("VXX", 0))
    if svxy > 0:
        action, acolor = "SELL VOLATILITY", GREEN
        stance = f"Short vol — hold {svxy*100:.0f}% SVXY (harvest contango carry)"
    elif vxx > 0:
        action, acolor = "BUY VOLATILITY", RED
        stance = f"Long vol — hold {vxx*100:.0f}% VXX (tail protection)"
    else:
        action, acolor = "STAY IN CASH", AMBER
        stance = "Flat — no volatility exposure, preserve capital"

    # ---- today's trade (change vs yesterday) ----
    trades = []
    for k in ["SVXY", "VXX", "SPX"]:
        d = float(today.get(k, 0)) - float(prev.get(k, 0))
        if abs(d) > 0.01:
            verb = "BUY" if d > 0 else "SELL"
            trades.append(f"{verb} {abs(d)*100:.0f}% {k}")
    trade_txt = " · ".join(trades) if trades else "No change — HOLD current position"

    # ---- allocation breakdown ----
    parts = []
    for k in ["SVXY", "VXX", "SPX"]:
        if abs(float(today.get(k, 0))) > 1e-6:
            parts.append(f"{float(today[k])*100:.0f}% {k}")
    cash = float(today.get("CASH", 0)) * 100
    if cash > 1:
        parts.append(f"{cash:.0f}% cash")
    alloc = ", ".join(parts) if parts else "100% cash"

    regime = snap.get("regime", "NA")
    notes = {
        "CALM": "Steep contango — harvest short-vol carry at full size.",
        "NORMAL": "Ordinary contango — harvest carry at reduced size.",
        "STRESS": "Curve flattening / elevated VIX — preserve capital in cash.",
        "CRISIS": "Backwardation + high VIX — small tactical long-vol tail position.",
    }
    return f"""
    <div class="signal">
      <div class="signal-title">📟 Today's Signal — {snap.get('date','')} · regime {regime}</div>
      <div class="signal-action" style="color:{acolor}">{action}</div>
      <div class="signal-stance">{stance}</div>
      <div class="signal-row"><span class="lbl">Trade today</span> {trade_txt}</div>
      <div class="signal-row"><span class="lbl">Target book</span> {alloc}</div>
      <div class="signal-note">{notes.get(regime,'')}</div>
    </div>"""


def _metrics_table(results: dict[str, BacktestResult]) -> str:
    cols = ["CAGR_%", "Vol_%", "Sharpe", "Sortino", "MaxDD_%", "Calmar",
            "WinRate_%", "TimeInMkt_%", "Trades/yr"]
    head = "".join(f"<th>{c.replace('_',' ')}</th>" for c in cols)
    rows = ""
    for r in results.values():
        s = r.stats
        cells = ""
        for c in cols:
            v = s.get(c, "—")
            cls = ""
            if c in ("CAGR_%", "Sharpe", "Sortino", "Calmar") and isinstance(v, (int, float)):
                cls = "pos" if v > 0 else "neg"
            if c == "MaxDD_%" and isinstance(v, (int, float)):
                cls = "neg"
            cells += f'<td class="{cls}">{v}</td>'
        bench = " bench" if r.key.startswith("bench") else ""
        rows += f'<tr class="{bench}"><td class="name">{r.name}</td>{cells}</tr>'
    return f"""
    <table class="metrics">
      <thead><tr><th class="name">Strategy</th>{head}</tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def _strategy_legend(strategies) -> str:
    items = ""
    for s in strategies:
        items += f'<li><b>{s.name}</b> — {s.description}</li>'
    return f'<ul class="legend">{items}</ul>'


def _signal_guide(snap: dict) -> str:
    """One self-contained block: every signal explained + how to act on it,
    annotated with the *current* reading so it doubles as today's interpretation.
    """
    vix = snap.get("VIX", "—")
    ratio = snap.get("ts_ratio", "—")
    struct = snap.get("structure", "—")
    roll = snap.get("roll_yield_ann_%", "—")
    vrp = snap.get("VRP", "—")
    pct = snap.get("vix_1y_pctile", "—")
    regime = snap.get("regime", "—")
    trend = "RISK-ON (above 200d MA)" if snap.get("spx_above_200dma") else "RISK-OFF (below 200d MA)"

    # Signal -> meaning -> action -> current reading
    rows = [
        ("VIX level",
         "30-day implied volatility of the S&P 500 — the market's expected swing. High = fear, low = calm.",
         "Low VIX favours selling vol (carry); spikes are usually faded as VIX mean-reverts.",
         f"{vix}  ({pct}th percentile of the last year)"),
        ("Term structure (VIX / VIX3M)",
         "Slope of the vol curve. Ratio &lt;1 = <b>contango</b> (calm, curve upward); &ge;1 = <b>backwardation</b> (stress, curve inverted).",
         "Contango pays short-vol holders a roll yield → sell vol. Backwardation pays long-vol holders → buy vol or go to cash.",
         f"{ratio} → <b>{struct}</b>"),
        ("Roll yield (annualised)",
         "The carry a short-vol position earns purely from the curve rolling down, if nothing else moves.",
         "Positive & large → short-vol carry is attractive. Negative → the curve is paying you to be long vol / flat.",
         f"{roll}% per year"),
        ("Variance risk premium (VRP)",
         "Implied vol (VIX) minus 20-day realised vol. The premium option sellers harvest for bearing risk.",
         "Positive → implied is richer than realised → edge to selling vol. Negative → vol is cheap; don't short it.",
         f"{vrp} vol points"),
        ("VIX 1-year percentile",
         "Where today's VIX ranks vs the last 12 months — a mean-reversion gauge.",
         "Very high (&gt;80) → spikes tend to revert down (fade). Very low (&lt;20) → carry is thin and a spike is overdue.",
         f"{pct}th percentile"),
        ("SPX 200-day trend",
         "Is the S&P 500 above its 200-day moving average — the broad risk-on/off filter.",
         "Short-vol carry works best in uptrends; below the 200d MA, size down or stand aside.",
         trend),
    ]
    body = ""
    for name, meaning, action, now in rows:
        body += f"""<tr>
            <td class="g-name">{name}</td>
            <td>{meaning}</td>
            <td class="g-act">{action}</td>
            <td class="g-now">{now}</td>
        </tr>"""

    # Regime playbook
    playbook = [
        ("CALM", GREEN, "Low VIX, steep contango", "Harvest carry — full short-vol (SVXY) position."),
        ("NORMAL", ACCENT, "Ordinary contango", "Harvest carry at reduced (half) size."),
        ("STRESS", AMBER, "Elevated VIX or flattening curve", "Preserve capital — move to cash, stop selling vol."),
        ("CRISIS", RED, "Backwardation + high VIX", "Small tactical long-vol (VXX) for tail capture."),
    ]
    pb = ""
    for name, color, cond, action in playbook:
        here = " ◄ now" if name == regime else ""
        pb += f"""<tr>
            <td><span class="dot" style="background:{color}"></span><b style="color:{color}">{name}</b>{here}</td>
            <td>{cond}</td>
            <td>{action}</td>
        </tr>"""

    return f"""
    <p class="g-intro">Every panel above is driven by the signals below. The first table explains
    <b>what each signal means</b> and <b>how to act on it</b>, with today's reading in the last column.
    The second table is the regime playbook the flagship strategy follows. The headline call is the
    synthesis of all of them. <b>Today: VIX {vix}, {struct}, regime {regime}.</b></p>
    <table class="guide">
      <thead><tr><th>Signal</th><th>What it measures</th><th>Investment interpretation</th><th>Now</th></tr></thead>
      <tbody>{body}</tbody>
    </table>
    <h3 class="g-sub">Regime playbook</h3>
    <table class="guide">
      <thead><tr><th>Regime</th><th>Condition</th><th>Suggested positioning</th></tr></thead>
      <tbody>{pb}</tbody>
    </table>
    <p class="g-warn">⚠ Educational framework, <b>not investment advice.</b> Short-volatility carry
    earns small gains most of the time but suffers rare, severe losses when the curve inverts
    (e.g. SVXY −90% in a day, 5 Feb 2018). Size positions for that tail, not for the calm.</p>"""


TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>sVIX — Volatility Strategy Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root {{ --bg:{BG}; --panel:{PANEL}; --text:{TEXT}; --muted:{MUTED}; --accent:{ACCENT}; }}
  * {{ box-sizing:border-box; }}
  body {{ background:var(--bg); color:var(--text); font-family:Inter,system-ui,Segoe UI,sans-serif;
         margin:0; padding:0 0 60px; }}
  header {{ padding:26px 32px 14px; border-bottom:1px solid {GRID}; }}
  h1 {{ margin:0; font-size:24px; letter-spacing:.5px; }}
  h1 span {{ color:var(--accent); }}
  .subtitle {{ color:var(--muted); font-size:13px; margin-top:4px; }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:0 24px; }}
  .cards {{ display:grid; grid-template-columns:repeat(6,1fr); gap:12px; margin:22px 0; }}
  .card {{ background:var(--panel); border:1px solid {GRID}; border-radius:12px; padding:14px 16px; }}
  .card-label {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.6px; }}
  .card-value {{ font-size:25px; font-weight:700; margin:6px 0 2px; }}
  .card-sub {{ color:var(--muted); font-size:11px; }}
  .signal {{ background:linear-gradient(135deg,#161b26,#1b2233); border:1px solid {GRID};
             border-left:4px solid var(--accent); border-radius:12px; padding:16px 20px; margin:8px 0 24px; }}
  .signal-title {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.6px; }}
  .signal-action {{ font-size:30px; font-weight:800; letter-spacing:.5px; margin:8px 0 2px; }}
  .signal-stance {{ font-size:14px; color:var(--text); margin-bottom:10px; }}
  .signal-row {{ font-size:13.5px; color:var(--text); margin:3px 0; }}
  .signal-row .lbl {{ display:inline-block; min-width:96px; color:var(--muted);
                      text-transform:uppercase; font-size:10.5px; letter-spacing:.5px; }}
  .signal-note {{ color:var(--muted); font-size:13px; margin-top:8px; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
  .panel {{ background:var(--panel); border:1px solid {GRID}; border-radius:12px; padding:6px; margin-bottom:18px; }}
  h2 {{ font-size:16px; margin:30px 0 12px; padding-left:10px; border-left:3px solid var(--accent); }}
  table.metrics {{ width:100%; border-collapse:collapse; font-size:13px; }}
  table.metrics th, table.metrics td {{ padding:9px 10px; text-align:right; border-bottom:1px solid {GRID}; }}
  table.metrics th {{ color:var(--muted); font-weight:600; font-size:11px; text-transform:uppercase; }}
  table.metrics td.name, table.metrics th.name {{ text-align:left; }}
  table.metrics tr.bench {{ background:rgba(255,255,255,0.04); font-style:italic; }}
  table.metrics td.pos {{ color:{GREEN}; }} table.metrics td.neg {{ color:{RED}; }}
  ul.legend {{ list-style:none; padding:0; font-size:13px; color:var(--text); }}
  ul.legend li {{ padding:7px 0; border-bottom:1px solid {GRID}; }}
  ul.legend b {{ color:var(--accent); }}
  .g-intro {{ font-size:13.5px; line-height:1.7; color:var(--text); margin:2px 4px 18px; }}
  .g-sub {{ font-size:14px; margin:22px 4px 10px; color:var(--accent); }}
  .g-warn {{ font-size:12.5px; line-height:1.6; color:{AMBER}; background:rgba(255,212,59,0.06);
             border:1px solid rgba(255,212,59,0.2); border-radius:10px; padding:12px 16px; margin:18px 4px 4px; }}
  table.guide {{ width:100%; border-collapse:collapse; font-size:13px; line-height:1.55; }}
  table.guide th {{ color:var(--muted); font-weight:600; font-size:11px; text-transform:uppercase;
                    text-align:left; padding:8px 12px; border-bottom:1px solid {GRID}; }}
  table.guide td {{ padding:11px 12px; border-bottom:1px solid {GRID}; vertical-align:top; }}
  table.guide td.g-name {{ font-weight:600; color:var(--text); white-space:nowrap; }}
  table.guide td.g-act {{ color:var(--muted); }}
  table.guide td.g-now {{ color:{GREEN}; font-weight:600; white-space:nowrap; }}
  .dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:7px; }}
  footer {{ color:var(--muted); font-size:11px; text-align:center; margin-top:30px; padding:0 24px; line-height:1.6; }}
  @media (max-width:900px) {{ .cards{{grid-template-columns:repeat(2,1fr);}} .grid2{{grid-template-columns:1fr;}} }}
</style></head>
<body>
<header>
  <div class="wrap">
    <h1>s<span>VIX</span> · Volatility Strategy Dashboard</h1>
    <div class="subtitle">{period} &nbsp;·&nbsp; {n_obs} observations &nbsp;·&nbsp; data source: <b>{source}</b>
       &nbsp;·&nbsp; generated {generated}</div>
  </div>
</header>
<div class="wrap">
  <div class="cards">{cards}</div>
  {signal}

  <h2>Volatility Landscape — where we are now</h2>
  <div class="grid2">
    <div class="panel">{term}</div>
    <div class="panel">{ratio}</div>
  </div>
  <div class="panel">{vix_hist}</div>
  <div class="panel">{vrp}</div>

  <h2>Strategy Performance</h2>
  <div class="panel">{equity}</div>
  <div class="grid2">
    <div class="panel">{drawdown}</div>
    <div class="panel">{allocation}</div>
  </div>

  <h2>Performance & Risk Metrics</h2>
  <div class="panel" style="padding:14px 18px;">{table}</div>

  <h2>Strategy Playbook</h2>
  <div class="panel" style="padding:14px 22px;">{legend}</div>

  <h2>How to Read the Signals &amp; Investment Suggestions</h2>
  <div class="panel" style="padding:16px 22px;">{guide}</div>

  <footer>
    sVIX is a research & education tool, <b>not investment advice</b>. Backtests use real ETF
    returns (incl. the Feb-2018 SVXY collapse and 2020 COVID crash) net of {cost_bps}bps/trade costs;
    past performance does not predict future results. Short-volatility strategies carry severe tail risk.
  </footer>
</div>
</body></html>"""


def build(results: dict[str, BacktestResult], sig: pd.DataFrame, md,
          flagship_key: str = "regime_switch", cost_bps: float = 5.0,
          out_path: str | None = None) -> str:
    from .strategies import all_strategies

    from .strategies import get as get_strategy

    snap = latest_snapshot(sig)
    flagship = results.get(flagship_key) or next(iter(results.values()))
    # fresh target weights for today's explicit buy/sell call
    flagship_target = get_strategy(flagship_key).weights(sig)

    html = TEMPLATE.format(
        BG=BG, PANEL=PANEL, TEXT=TEXT, MUTED=MUTED, ACCENT=ACCENT,
        GRID=GRID, GREEN=GREEN, RED=RED, AMBER=AMBER,
        period=f"{md.meta['start']} → {md.meta['end']}",
        n_obs=md.meta["n_obs"], source=md.source.upper(),
        generated=sig.index.max().strftime("%Y-%m-%d"),
        cost_bps=int(cost_bps),
        cards=_kpi_cards(snap, md.source),
        signal=_signal_call(snap, flagship_target),
        term=_fig_html(fig_term_structure(sig), "term"),
        ratio=_fig_html(fig_ratio(sig), "ratio"),
        vix_hist=_fig_html(fig_vix_history(sig), "vixhist"),
        vrp=_fig_html(fig_vrp(sig), "vrp"),
        equity=_fig_html(fig_equity(results), "equity"),
        drawdown=_fig_html(fig_drawdown(results), "dd"),
        allocation=_fig_html(fig_allocation(flagship), "alloc"),
        table=_metrics_table(results),
        legend=_strategy_legend(all_strategies()),
        guide=_signal_guide(snap),
    )

    out_path = out_path or os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "output", "dashboard.html")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    return out_path
