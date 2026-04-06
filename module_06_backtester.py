"""
=============================================================================
TRADING ALGORITHM — MODULE 6: BACKTESTER
=============================================================================
Committee mandates incorporated:
  - 10-bar cooldown deduplication (resets on stop hit)
  - 2.5% maximum stop distance filter
  - Dec 2025 – Feb 2026 held as out-of-sample
  - Performance reported by: regime, zone_source, tf_conflict, depth bucket,
    score tier, leg quality, coiling status, coin
  - Staged exit: 50% at T1 (1R), stop to break-even, 50% at T2 (2R)

Trade simulation:
  - Entry: signal bar close price (conservative — real entry is next bar open)
  - Stop: structural (last 15m swing) or ATR fallback
  - Simulation: 5m bars scanned bar-by-bar
  - Max hold: 96 × 5m bars (8 hours) — committee-tunable
  - Outcomes: WIN_FULL (1.5R), WIN_PARTIAL (0.5R), LOSS (-1R),
              TIMEOUT (market close, ±varies), TIMEOUT_PARTIAL (0.5+varies)

Key finding from 2-coin test (BTC+BNB only):
  IS win rate: ~16-22% (hold-time dependent)
  This is below the ~53% break-even threshold for the 1R:1.5R avg structure.
  Full 20-coin backtest required for a representative verdict.
  Two coins (both BTC-correlated, both bear-biased Q4 2025) is not a
  sufficient sample for strategy evaluation.
=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from module_01_data_layer         import load_all
from module_02_trend_classifier   import classify_all, classify_symbol as _cs
from module_03_sr_zones           import detect_zones
from module_04_leg_compression    import compute_all, SWING_LOOKBACK
from module_05_signal_aggregator  import aggregate_signals, signal_summary


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

COOLDOWN_BARS    = 10      # 1h bars before same-direction signal fires again
MAX_STOP_PCT     = 2.5     # hard maximum stop distance (%)
MAX_HOLD_5M      = 1152    # 1152 × 5m = 96 hours — 1h setups need room to develop
OOS_START        = pd.Timestamp("2025-12-01", tz="UTC")  # Douglas mandate


# ---------------------------------------------------------------------------
# Trade dataclass
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    symbol          : str
    timestamp       : pd.Timestamp
    direction       : str
    entry_price     : float
    stop_price      : float
    target1_price   : float
    target2_price   : float
    stop_pct        : float
    # Signal metadata
    regime          : str
    zone_source     : str
    tf_conflict     : bool
    leg_quality     : str
    confluence      : float
    coiling         : bool
    pullback_depth  : float
    # Backtest result
    outcome         : str   = "PENDING"   # WIN_FULL | WIN_PARTIAL | LOSS | TIMEOUT | TIMEOUT_PARTIAL
    r_multiple      : float = 0.0
    split           : str   = "IS"        # IS | OOS


# ---------------------------------------------------------------------------
# Trade simulation
# ---------------------------------------------------------------------------

ROUND_TRIP_FEE_PCT = 0.0004    # 0.04% Binance taker × 2 sides

def simulate_trade(
    direction    : str,
    entry        : float,
    stop         : float,
    target1      : float,
    target2      : float,
    signal_open_ts: pd.Timestamp,          # open_time of the 15m signal bar
    df5m         : pd.DataFrame,
    max_bars     : int = MAX_HOLD_5M,
) -> tuple[str, float]:
    """
    Simulate a single trade on 5m bars.

    Exit structure: 50% partial at T1 (1R), breakeven stop, 50% at T2 (measured-move).
      • Timing fix : simulation starts at CLOSE of 1h signal bar (open_time + 60 min).
      • Fee fix    : round-trip 0.04% deducted from every realised R.

    Trail exit was tested and rejected (v4 → reverted):
      A 1×stop_dist chandelier trail fires before T2 on normal pullbacks because
      the minimum T1→T2 distance (0.5×stop at R:R=1.5) is narrower than the trail
      width. It converted 30 WIN_FULL → WIN_PARTIAL, IS PF 1.020 → 0.977. Reverted.

    Returns (outcome, r_multiple):
      WIN_FULL    : 0.5 + 0.5×T2R − fee_r
      WIN_PARTIAL : 0.5 − fee_r
      LOSS        : −1.0 − fee_r
      TIMEOUT     : market_r − fee_r  (±varies)
    """
    # ── Timing fix: enter at close of signal bar, not its open ────────────
    start_ts = signal_open_ts + pd.Timedelta(minutes=60)  # 1h bar closes 60min after open

    try:
        idx = df5m.index.searchsorted(start_ts)
    except Exception:
        return "TIMEOUT", 0.0

    if idx >= len(df5m):
        return "TIMEOUT", 0.0

    stop_dist = abs(entry - stop)
    if stop_dist <= 0:
        return "TIMEOUT", 0.0

    # ── Fee in R units ────────────────────────────────────────────────────
    stop_dist_frac = stop_dist / entry if entry > 0 else 0.001
    fee_r          = round(ROUND_TRIP_FEE_PCT / stop_dist_frac, 5)

    # Pre-compute WIN_FULL R from actual structural T2 distance
    t2_r       = abs(target2 - entry) / stop_dist
    win_full_r = round(0.5 + 0.5 * t2_r - fee_r, 4)

    hit_t1  = False
    be_stop = entry    # break-even stop after T1 hit

    for i in range(idx, min(idx + max_bars, len(df5m))):
        bar = df5m.iloc[i]
        high, low = float(bar["high"]), float(bar["low"])

        if not hit_t1:
            if direction == "LONG":
                if low  <= stop:    return "LOSS",        round(-1.0 - fee_r, 4)
                if high >= target1: hit_t1 = True
            else:
                if high >= stop:    return "LOSS",        round(-1.0 - fee_r, 4)
                if low  <= target1: hit_t1 = True
        else:
            # Phase 2: T1 hit — hold with breakeven stop, target T2
            # Trail exit was tested and rejected:
            #   1×stop_dist trail is 2× wider than the T1→T2_min gap (0.5×stop),
            #   so it fires before T2 on normal pullbacks, converting WIN_FULL
            #   trades to WIN_PARTIAL and degrading IS PF 1.020 → 0.977.
            if direction == "LONG":
                if low  <= be_stop: return "WIN_PARTIAL", round(0.5 - fee_r, 4)
                if high >= target2: return "WIN_FULL",    win_full_r
            else:
                if high >= be_stop: return "WIN_PARTIAL", round(0.5 - fee_r, 4)
                if low  <= target2: return "WIN_FULL",    win_full_r

    # Max hold reached
    last_close = float(df5m.iloc[min(idx + max_bars - 1, len(df5m) - 1)]["close"])
    r = (last_close - entry) / stop_dist if direction == "LONG" \
        else (entry - last_close) / stop_dist

    if hit_t1:
        return "TIMEOUT_PARTIAL", round(0.5 + 0.5 * r - fee_r, 4)
    return "TIMEOUT", round(r - fee_r, 4)


# ---------------------------------------------------------------------------
# Deduplication / cooldown
# ---------------------------------------------------------------------------

def _next_timestamp(df5m: pd.DataFrame, from_idx: int, n_bars: int) -> pd.Timestamp:
    """Return timestamp n_bars after from_idx, clamped to df5m length."""
    target_idx = min(from_idx + n_bars, len(df5m) - 1)
    return df5m.index[target_idx]


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------

def run_backtest(
    signal_frames : dict[str, pd.DataFrame],
    data          : dict[str, dict[str, pd.DataFrame]],
    cooldown      : int  = COOLDOWN_BARS,
    max_stop      : float= MAX_STOP_PCT,
    max_hold      : int  = MAX_HOLD_5M,
    oos_start     : pd.Timestamp = OOS_START,
) -> pd.DataFrame:
    """
    Simulate all signals and return a DataFrame of completed trades.
    Applies deduplication cooldown and stop-distance cap before simulating.
    """
    all_trades: list[dict] = []

    for symbol, sig_df in sorted(signal_frames.items()):
        fired = sig_df[sig_df["direction"] != "NO_SIGNAL"].copy()
        if fired.empty:
            continue

        df5m = data[symbol].get("5m")
        if df5m is None:
            continue

        # Per-direction cooldown: maps direction → "cooldown active until" timestamp
        cooldown_until: dict[str, pd.Timestamp] = {}

        for ts, row in fired.iterrows():
            direction = row["direction"]
            sdp       = float(row["stop_distance_pct"])

            # ── Filter 1: stop cap ──────────────────────────────────────────
            if sdp > max_stop:
                continue

            # ── Filter 2: cooldown ──────────────────────────────────────────
            if direction in cooldown_until and ts <= cooldown_until[direction]:
                continue

            entry   = float(row["entry_price"])
            stop    = float(row["stop_price"])
            target1 = float(row["target1_price"])
            target2 = float(row["target2_price"])

            # ── Simulate ────────────────────────────────────────────────────
            outcome, r = simulate_trade(
                direction, entry, stop, target1, target2,
                ts,          # signal_open_ts — simulate_trade adds the +15m offset
                df5m, max_hold
            )

            # Update cooldown: resets on loss (allow re-entry sooner)
            idx_entry = df5m.index.searchsorted(ts)
            if outcome != "LOSS":
                # 1h cooldown: COOLDOWN_BARS × 12 five-minute bars per 1h bar
                cooldown_until[direction] = _next_timestamp(df5m, idx_entry, cooldown * 12)
            else:
                # After loss: clear cooldown immediately to allow re-entry
                cooldown_until.pop(direction, None)

            split = "OOS" if ts >= oos_start else "IS"

            all_trades.append({
                "symbol"        : symbol,
                "timestamp"     : ts,
                "direction"     : direction,
                "entry_price"   : entry,
                "stop_price"    : stop,
                "target1_price" : target1,
                "target2_price" : target2,
                "stop_pct"      : sdp,
                "regime"        : row["regime"],
                "zone_source"   : row["zone_source"],
                "tf_conflict"   : row["tf_conflict"],
                "leg_quality"   : row["leg_quality"],
                "confluence"    : row["confluence_score"],
                "coiling"       : row["is_coiling"],
                "depth"         : row["pullback_depth"],
                "outcome"       : outcome,
                "r"             : r,
                "split"         : split,
            })

    df = pd.DataFrame(all_trades)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Metrics calculator
# ---------------------------------------------------------------------------

def metrics(df: pd.DataFrame) -> dict:
    """Calculate trading metrics from a trades DataFrame."""
    if df.empty:
        return {"n": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "avg_r": 0.0, "total_r": 0.0, "max_dd_r": 0.0,
                "gross_profit_r": 0.0, "gross_loss_r": 0.0,
                "pct_stops": 0.0, "pct_timeouts": 0.0}

    n     = len(df)
    wins  = df[df["outcome"].isin(["WIN_FULL", "WIN_PARTIAL", "TIMEOUT_PARTIAL"])]
    stops = df[df["outcome"] == "LOSS"]
    gp    = max(wins["r"].sum(), 0.0)
    gl    = abs(stops["r"].sum())

    cum_r     = df["r"].cumsum()
    run_max   = cum_r.cummax()
    max_dd    = (cum_r - run_max).min()

    return {
        "n"             : n,
        "win_rate"      : round(len(df[df["outcome"].isin(["WIN_FULL","WIN_PARTIAL"])]) / n, 4),
        "profit_factor" : round(gp / gl, 3) if gl > 0 else 9.99,
        "avg_r"         : round(df["r"].mean(), 4),
        "total_r"       : round(df["r"].sum(), 2),
        "max_dd_r"      : round(max_dd, 2),
        "gross_profit_r": round(gp, 2),
        "gross_loss_r"  : round(gl, 2),
        "pct_stops"     : round(len(stops) / n, 3),
        "pct_timeouts"  : round(len(df[df["outcome"].isin(["TIMEOUT","TIMEOUT_PARTIAL"])]) / n, 3),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _mline(label: str, df: pd.DataFrame, width: int = 30) -> str:
    m  = metrics(df)
    pf = m["profit_factor"]
    star = " ★" if pf >= 1.0 else ("  " if pf >= 0.5 else "")
    return (f"  {label:<{width}} n={m['n']:>5}  WR={m['win_rate']:.1%}  "
            f"PF={pf:.2f}  avgR={m['avg_r']:.3f}  "
            f"totalR={m['total_r']:>8.1f}  ddR={m['max_dd_r']:.1f}{star}")


def full_report(trades: pd.DataFrame, split: str = "IS") -> None:
    """Print comprehensive segmented performance report."""
    df = trades[trades["split"] == split].copy()
    label = f"IN-SAMPLE (Jan 2023 – Nov 2025)" if split == "IS" \
        else f"OUT-OF-SAMPLE (Dec 2025 – Feb 2026)"

    print("\n" + "=" * 78)
    print(f"  BACKTEST RESULTS — {label}")
    print(f"  Signals: {len(df):,} trades after dedup + stop cap ({MAX_STOP_PCT}% max stop)")
    print("=" * 78)

    print(f"\n  {'OVERALL':<30}", end="")
    m = metrics(df)
    print(f"n={m['n']:>5}  WR={m['win_rate']:.1%}  PF={m['profit_factor']:.2f}  "
          f"avgR={m['avg_r']:.3f}  totalR={m['total_r']:.1f}  ddR={m['max_dd_r']:.1f}")

    # Outcome distribution
    print(f"\n  Outcome distribution:")
    for oc, cnt in df["outcome"].value_counts().items():
        print(f"    {oc:<18} {cnt:>5}  ({cnt/len(df):.1%})")

    # By regime
    print(f"\n  By REGIME:")
    for rg, grp in df.groupby("regime"):
        print(_mline(f"  {rg}", grp))

    # By TF conflict
    print(f"\n  By TF CONFLICT:")
    for tfc, grp in df.groupby("tf_conflict"):
        lbl = "Counter-trend" if tfc else "Aligned      "
        print(_mline(f"  {lbl}", grp))

    # By zone source
    print(f"\n  By ZONE SOURCE:")
    for src, grp in df.groupby("zone_source"):
        print(_mline(f"  {src}", grp))

    # By depth bucket
    print(f"\n  By PULLBACK DEPTH:")
    df["depth_b"] = pd.cut(df["depth"],
        bins=[0, .20, .35, .50, .65, .75, 2.0],
        labels=["<0.20", "0.20-0.35", "0.35-0.50", "0.50-0.65", "0.65-0.75", ">0.75"])
    for b, grp in df.groupby("depth_b", observed=True):
        print(_mline(f"  depth {b}", grp))

    # By coiling
    print(f"\n  By COILING:")
    for c, grp in df.groupby("coiling"):
        print(_mline(f"  {'Coiling' if c else 'No coil'}", grp))

    # By leg quality
    print(f"\n  By LEG QUALITY:")
    for lq, grp in df.groupby("leg_quality"):
        print(_mline(f"  {lq}", grp))

    # By score tier
    print(f"\n  By CONFLUENCE SCORE:")
    df["score_t"] = pd.cut(df["confluence"],
        bins=[3.9, 5.0, 6.0, 7.0, 10.0],
        labels=["4-5", "5-6", "6-7", "7+"])
    for t, grp in df.groupby("score_t", observed=True):
        print(_mline(f"  score {t}", grp))

    # By symbol
    print(f"\n  By SYMBOL:")
    for sym, grp in df.groupby("symbol"):
        print(_mline(f"  {sym}", grp, width=14))

    # Monthly equity curve (in R)
    print(f"\n  Monthly R accumulation:")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df2 = df.copy()
        df2["month"] = df2["timestamp"].dt.to_period("M")
    monthly = df2.groupby("month")["r"].sum()
    cum_r = 0.0
    for m_period, m_r in monthly.items():
        cum_r += m_r
        bar_r = "█" * max(0, int(abs(m_r) / 2))
        sign = "+" if m_r >= 0 else "-"
        print(f"    {m_period}  {sign}{abs(m_r):>6.1f}R  cum={cum_r:>8.1f}R  {bar_r}")

    # Best filter combinations
    print(f"\n  FILTER COMBINATIONS (best edge candidates):")
    combos = [
        ("Aligned only",            df[~df["tf_conflict"]]),
        ("Depth 0.35-0.65",         df[(df["depth"]>=.35)&(df["depth"]<=.65)]),
        ("Score >= 5 + IDEAL",      df[(df["confluence"]>=5.0)&(df["leg_quality"]=="IDEAL")]),
        ("Score >= 6",              df[df["confluence"]>=6.0]),
        ("Aligned + score>=5",      df[~df["tf_conflict"]&(df["confluence"]>=5.0)]),
        ("Aligned + IDEAL + sc>=5", df[~df["tf_conflict"]&(df["leg_quality"]=="IDEAL")&(df["confluence"]>=5.0)]),
        ("Fib + coiling",           df[(df["depth"]>=.35)&(df["depth"]<=.65)&df["coiling"]]),
        ("Aligned + Fib + sc>=6",   df[~df["tf_conflict"]&(df["depth"]>=.35)&(df["depth"]<=.65)&(df["confluence"]>=6.0)]),
    ]
    for label, grp in combos:
        print(_mline(label, grp))

    print()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    print("Loading data...")
    data, _ = load_all(run_sanity_check=False, debug=False)

    print("Classifying trends...")
    classified = classify_all(data, timeframes=["1h", "4h"])
    for sym in data:
        for tf in ["30m", "15m"]:
            if tf in data[sym]:
                classified[sym][tf] = _cs(
                    data, sym, tf, lookback=SWING_LOOKBACK.get(tf, 5)
                )

    print("Detecting S/R zones...")
    all_zones = detect_zones(data, classified)

    print("Running leg counter + compression...")
    m4 = compute_all(data, classified)

    print("Generating signals...")
    signals = aggregate_signals(data, classified, m4, all_zones)

    # ── Signal summary ────────────────────────────────────────────────────
    summary = signal_summary(signals)
    total_signals = int(summary["signals"].sum())
    total_bars    = int(summary["total_bars"].sum())
    print(f"\n  Raw signals: {total_signals:,} ({total_signals/total_bars*100:.2f}% of {total_bars:,} bars)")

    print(f"\nRunning backtest (max_hold={MAX_HOLD_5M} bars × 5m = "
          f"{MAX_HOLD_5M*5//60}h, cooldown={COOLDOWN_BARS} bars × 15m, "
          f"max_stop={MAX_STOP_PCT}%)...")

    trades = run_backtest(signals, data)

    if trades.empty:
        print("No trades generated. Check signal generation.")
    else:
        is_n  = int((trades["split"] == "IS").sum())
        oos_n = int((trades["split"] == "OOS").sum())
        print(f"\n  Total trades: {len(trades):,}  (IS={is_n:,}  OOS={oos_n:,})")

        # ── In-sample report ─────────────────────────────────────────────
        full_report(trades, split="IS")

        # ── OOS report ───────────────────────────────────────────────────
        if oos_n > 0:
            full_report(trades, split="OOS")

        # ── Critical context note ────────────────────────────────────────
        print("=" * 78)
        print("  COMMITTEE NOTE — INTERPRETATION GUIDE")
        print("=" * 78)
        print("""
  Break-even thresholds for this exit structure:
    With avg win = 1.0R (mix of WIN_FULL 1.5R + WIN_PARTIAL 0.5R):
    Break-even win rate ≈ 53%

  If win rate is significantly below 53% across all coins and regimes:
    → The entry signal needs refinement (depth threshold, min score, etc.)
    → Consider tighter stop placement (ATR-based, not structural swing)
    → Consider removing low-quality filters (depth < 0.35, VALID quality)

  If win rate is above 40% in specific regimes/filters but below 53% overall:
    → A targeted filter exists — identify and apply it
    → Run module 6 again on the filtered signal subset

  Healthy profit factor target: PF >= 1.3 (profitable after fees)
  Current fee assumption: 0.04% round-trip (Binance taker)
        """)

        # ── Save trades CSV for offline analysis ─────────────────────────
        out_path = os.path.join(os.path.dirname(__file__), "backtest_trades.csv")
        trades.to_csv(out_path, index=False)
        print(f"  Full trades saved to: {out_path}")
        print()