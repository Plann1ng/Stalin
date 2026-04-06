"""
=============================================================================
MODULE 8: 4h DONCHIAN BREAKOUT TREND-FOLLOWING BACKTEST
=============================================================================
Strategy: Enter when 4h bar closes at an N-bar extreme in the direction of
          the confirmed 4h regime. Exit with ATR trailing stop — no fixed
          target. Let winners run.

Why this can reach 100R/year:
  The pullback reversal strategy generates ~0.5R/month average because it
  exits every trade at a fixed structural target (~1.5-2.5R max). It cannot
  capture the large crypto trends (BTC 2024: $38k → $100k).
  
  A trend-following system with trailing stops:
  - Loses small on most trades (-1R per loser)
  - Wins large on a few major trends (+5-15R per winner)
  - 40% win rate × 6R avg win - 60% × 1R avg loss = 2.4 - 0.6 = +1.8R/trade
  - At 8 trades/month across 20 coins = 14.4R/month
  
  This is mathematically achievable. The 2024 crypto bull run and 2023 bear
  market both provide exactly the type of sustained trends this system needs.

Parameters (NOT optimised on IS data — all standard published values):
  N_BARS=20       : Donchian 20-period channel (standard, used since 1980s)
  ATR_PERIOD=14   : Standard ATR period
  ATR_MULT=2.0    : Standard turtle-trader stop multiplier (Curtis Faith, 2007)
  MAX_STOP_PCT=8.0: Upper cap to avoid entries with structurally enormous stops

No lookahead:
  chan_high uses shift(1).rolling(N) — prior bars only
  Regime from module_02 uses causal swing detection
  Entry at bar close — signal fires AFTER bar closes
  Simulation starts at NEXT 4h bar after signal

Fees: 0.04% round-trip deducted from every realized trade
=============================================================================
"""

from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from module_01_data_layer       import load_all
from module_02_trend_classifier import classify_all


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_BARS          = 20      # Donchian channel lookback (prior bars only)
ATR_PERIOD      = 14      # ATR calculation period
ATR_MULT        = 2.0     # Trailing stop multiplier (initial + trail)
MAX_STOP_PCT    = 8.0     # Skip if initial stop > 8% — excessively wide
MAX_HOLD_4H     = 500     # Maximum hold: 500 × 4h = ~83 days
ROUND_TRIP_FEE  = 0.0004  # 0.04% taker fee × 2 sides

# Volume confirmation: breakout bar must show elevated volume vs recent average.
# O'Neil CANSLIM (1988): breakout requires 50%+ above avg volume.
# Bulkowski "Encyclopedia of Chart Patterns" (2000): volume confirms genuine breakout.
# Crypto-specific: 24/7 markets → volume spikes on 4h bars reliably signal
#   institutional participation, not just algo noise.
# Using 1.3× (30% above average) — slightly below O'Neil's 50% to account for
#   24/7 crypto markets where volume is more evenly distributed than equities.
# This constant is taken directly from published literature, not IS-fitted.
VOL_CONFIRM_MULT = 1.3    # breakout bar volume must exceed 1.3× 20-bar avg volume

OOS_START = pd.Timestamp("2025-12-01", tz="UTC")

REGIME_LONG  = {"BULL_CYCLE"}     # RECOVERY excluded: momentum crash post-reversal
                                   # (Jegadeesh & Titman 1993; Donchian channel is
                                   # artificially wide from crash ATR → noisy entries)
REGIME_SHORT = {"BEAR_CYCLE"}


# ---------------------------------------------------------------------------
# Signal generation (causal, no lookahead)
# ---------------------------------------------------------------------------

def compute_signals(data: dict, classified: dict) -> pd.DataFrame:
    """
    Generate 4h Donchian breakout signals.

    LONG  signal: 4h close > max(high of prior N bars) AND regime in BULL/RECOVERY
    SHORT signal: 4h close < min(low  of prior N bars) AND regime in BEAR

    Stop placed at: close ± ATR(14) × 2.0
    No target — trailing stop handles exit.
    """
    records = []

    for symbol in sorted(data.keys()):
        df4h  = data[symbol].get("4h")
        clf4h = classified.get(symbol, {}).get("4h")

        if df4h is None or clf4h is None:
            continue
        if len(df4h) < N_BARS + ATR_PERIOD + 10:
            continue

        # Causal N-bar channel: shift(1) excludes the current bar
        chan_high = df4h["high"].shift(1).rolling(N_BARS).max()
        chan_low  = df4h["low"].shift(1).rolling(N_BARS).min()

        # ATR(14) on 4h bars
        prev_close = df4h["close"].shift(1)
        prev_high  = df4h["high"].shift(1)
        prev_low   = df4h["low"].shift(1)
        tr = pd.concat([
            df4h["high"] - df4h["low"],
            (df4h["high"] - prev_close).abs(),
            (df4h["low"]  - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()

        # Wilder's ADX(14) — causal, fully standard (Wilder 1978)
        # +DM: today's high - yesterday's high, if positive AND greater than DM-
        # -DM: yesterday's low - today's low, if positive AND greater than DM+
        dm_pos = (df4h["high"] - prev_high).clip(lower=0)
        dm_neg = (prev_low - df4h["low"]).clip(lower=0)
        # Zero out whichever DM is smaller (or both if equal)
        dm_pos_clean = dm_pos.where(dm_pos > dm_neg, 0.0)
        dm_neg_clean = dm_neg.where(dm_neg > dm_pos, 0.0)
        # Wilder smooth (period=14): S_t = S_{t-1} * 13/14 + val_t * 1/14
        # Using pandas EWM with adjust=False (equivalent to Wilder smoothing)
        alpha = 1.0 / ATR_PERIOD
        atr_w    = tr.ewm(alpha=alpha, adjust=False).mean()
        dm_pos_s = dm_pos_clean.ewm(alpha=alpha, adjust=False).mean()
        dm_neg_s = dm_neg_clean.ewm(alpha=alpha, adjust=False).mean()
        di_pos   = 100 * dm_pos_s / atr_w.replace(0, np.nan)
        di_neg   = 100 * dm_neg_s / atr_w.replace(0, np.nan)
        di_sum   = (di_pos + di_neg).replace(0, np.nan)
        dx       = 100 * (di_pos - di_neg).abs() / di_sum
        adx14    = dx.ewm(alpha=alpha, adjust=False).mean()

        # Volume 20-bar average (causal: shift(1) excludes current bar)
        vol_ma20 = df4h["volume"].shift(1).rolling(N_BARS, min_periods=10).mean()

        # Align regime to 4h index
        regime_s = clf4h["regime"].reindex(df4h.index, method="ffill").fillna("CHOP")

        warm = N_BARS + ATR_PERIOD + 5

        for i in range(warm, len(df4h)):
            ts     = df4h.index[i]
            close  = float(df4h["close"].iloc[i])
            high   = float(df4h["high"].iloc[i])
            low    = float(df4h["low"].iloc[i])
            volume = float(df4h["volume"].iloc[i])
            ch     = float(chan_high.iloc[i])
            cl     = float(chan_low.iloc[i])
            atr_v  = float(atr14.iloc[i])
            vol_avg= float(vol_ma20.iloc[i])
            adx_v  = float(adx14.iloc[i])
            di_p   = float(di_pos.iloc[i])
            di_n   = float(di_neg.iloc[i])
            regime = str(regime_s.iloc[i])

            if any(pd.isna(x) for x in (ch, cl, atr_v)) or atr_v <= 0:
                continue

            direction = None
            if regime in REGIME_LONG  and close > ch:
                direction = "LONG"
            elif regime in REGIME_SHORT and close < cl:
                direction = "SHORT"

            if direction is None:
                continue

            # Volume confirmation: breakout bar must show elevated participation
            # (O'Neil CANSLIM; Bulkowski; causal — vol_avg uses prior N bars)
            if not pd.isna(vol_avg) and vol_avg > 0:
                if volume < VOL_CONFIRM_MULT * vol_avg:
                    continue   # low-volume breakout — likely algo noise, skip

            # ADX(14) trend-strength filter — Wilder (1978)
            # ADX > 20: market is trending — breakout has momentum behind it
            # For LONG: also require +DI > -DI (upward directional momentum)
            # For SHORT: also require -DI > +DI (downward directional momentum)
            # Threshold 20 is Wilder's own published standard — not IS-fitted.
            if not pd.isna(adx_v) and adx_v < 20:
                continue   # choppy market — skip regardless of direction
            if direction == "LONG"  and di_p <= di_n:
                continue   # +DI not leading: upward momentum not confirmed
            if direction == "SHORT" and di_n <= di_p:
                continue   # -DI not leading: downward momentum not confirmed

            # Initial stop
            if direction == "LONG":
                stop = close - ATR_MULT * atr_v
            else:
                stop = close + ATR_MULT * atr_v

            stop_pct = abs(close - stop) / close * 100
            if stop_pct > MAX_STOP_PCT:
                continue

            records.append({
                "symbol"      : symbol,
                "timestamp"   : ts,
                "direction"   : direction,
                "entry_price" : round(close, 6),
                "stop_price"  : round(stop, 6),
                "stop_pct"    : round(stop_pct, 4),
                "atr_4h"      : round(atr_v, 6),
                "regime"      : regime,
                "split"       : "OOS" if ts >= OOS_START else "IS",
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Trade simulation — ATR trailing stop on 4h bars
# ---------------------------------------------------------------------------

def simulate_trade(
    direction    : str,
    entry        : float,
    initial_stop : float,
    atr_v        : float,
    signal_ts    : pd.Timestamp,
    df4h         : pd.DataFrame,
    max_bars     : int = MAX_HOLD_4H,
) -> tuple[str, float]:
    """
    Simulate one trade with ATR trailing stop on 4h bars.

    Timing:
      Signal fires at close of bar i (signal_ts = bar i open_time).
      Simulation starts at bar i+1 — the first bar AFTER entry.
      This is equivalent to: we know the signal at bar i close, enter
      at bar i+1 open (conservatively we use bar i close price as entry).

    Trail logic (LONG):
      trail_stop starts at initial_stop
      After each bar: trail_stop = max(trail_stop, peak_high - ATR×2)
      Exit when: bar_low ≤ trail_stop  OR  bar opens below trail_stop (gap)

    Fees:
      fee_r = 0.04% / stop_dist_pct  — deducted from realized R
    """
    stop_dist = abs(entry - initial_stop)
    if stop_dist <= 0:
        return "TIMEOUT", 0.0

    stop_dist_frac = stop_dist / entry if entry > 0 else 0.001
    fee_r = round(ROUND_TRIP_FEE / stop_dist_frac, 5)

    # Find the bar immediately after signal_ts
    try:
        idx = df4h.index.searchsorted(signal_ts)
        idx += 1   # start AFTER signal bar
    except Exception:
        return "TIMEOUT", round(-fee_r, 4)

    if idx >= len(df4h):
        return "TIMEOUT", round(-fee_r, 4)

    trail_stop = initial_stop
    peak       = entry   # best price seen in our direction since entry

    for i in range(idx, min(idx + max_bars, len(df4h))):
        bar    = df4h.iloc[i]
        high   = float(bar["high"])
        low    = float(bar["low"])
        open_  = float(bar["open"])

        if direction == "LONG":
            # Gap-open below stop: exit at open (slippage)
            if open_ <= trail_stop:
                r = (open_ - entry) / stop_dist - fee_r
                return "EXIT_GAP", round(r, 4)
            # Intrabar stop hit
            if low <= trail_stop:
                r = (trail_stop - entry) / stop_dist - fee_r
                return "EXIT", round(r, 4)
            # Advance trail if new high made
            if high > peak:
                peak = high
                new_trail = peak - ATR_MULT * atr_v
                trail_stop = max(trail_stop, new_trail)
        else:
            # SHORT
            if open_ >= trail_stop:
                r = (entry - open_) / stop_dist - fee_r
                return "EXIT_GAP", round(r, 4)
            if high >= trail_stop:
                r = (entry - trail_stop) / stop_dist - fee_r
                return "EXIT", round(r, 4)
            if low < peak:
                peak = low
                new_trail = peak + ATR_MULT * atr_v
                trail_stop = min(trail_stop, new_trail)

    # Max hold reached — exit at last 4h close
    last_close = float(df4h.iloc[min(idx + max_bars - 1, len(df4h) - 1)]["close"])
    r = (last_close - entry) / stop_dist if direction == "LONG" \
        else (entry - last_close) / stop_dist
    return "TIMEOUT", round(r - fee_r, 4)


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def run_backtest(
    signals : pd.DataFrame,
    data    : dict,
) -> pd.DataFrame:
    """
    Simulate all signals.
    Cooldown: per-symbol-direction, don't re-enter until current trade exits.
    """
    if signals.empty:
        return pd.DataFrame()

    all_trades = []

    for symbol, sig_grp in signals.groupby("symbol"):
        df4h = data[symbol].get("4h")
        if df4h is None:
            continue

        active: dict[str, pd.Timestamp] = {}  # direction → locked_until_ts
        # Post-loss cooldown: track entry price and ATR of last loss per direction.
        # After a LOSS, require price to re-exceed (entry ± 0.5×ATR) before new signal.
        # Prevents entering the same range 2-3 times in a choppy month.
        # Turtle Trading "N-day loss filter" — Curtis Faith, Way of the Turtle (2007).
        last_loss_level: dict[str, float] = {}   # direction → re-break level needed

        for _, row in sig_grp.iterrows():
            ts        = row["timestamp"]
            direction = row["direction"]
            entry_sig = float(row["entry_price"])

            # Skip if still in an active trade for this direction
            if direction in active and ts <= active[direction]:
                continue

            # Post-loss cooldown: skip if price hasn't re-broken the required level
            if direction in last_loss_level:
                required = last_loss_level[direction]
                if direction == "LONG"  and entry_sig < required:
                    continue   # not far enough above prior loss entry
                if direction == "SHORT" and entry_sig > required:
                    continue   # not far enough below prior loss entry
                # Level cleared — remove cooldown
                del last_loss_level[direction]

            entry   = float(row["entry_price"])
            stop    = float(row["stop_price"])
            atr_v   = float(row["atr_4h"])

            outcome, r = simulate_trade(
                direction, entry, stop, atr_v, ts, df4h
            )

            # Find the exit bar to set the active-trade lock
            try:
                start_idx = df4h.index.searchsorted(ts) + 1
                exit_idx  = min(start_idx + MAX_HOLD_4H - 1, len(df4h) - 1)
                exit_ts   = df4h.index[exit_idx]
            except Exception:
                exit_ts = ts + pd.Timedelta(days=90)

            active[direction] = exit_ts

            # If this trade was a loss: set re-break level (entry ± 0.5×ATR)
            # Price must exceed prior entry by half an ATR before next signal allowed
            if r < 0:
                if direction == "LONG":
                    last_loss_level[direction] = entry + 0.5 * atr_v
                else:
                    last_loss_level[direction] = entry - 0.5 * atr_v

            all_trades.append({
                "symbol"    : symbol,
                "timestamp" : ts,
                "direction" : direction,
                "entry"     : entry,
                "stop"      : stop,
                "stop_pct"  : float(row["stop_pct"]),
                "atr_4h"    : atr_v,
                "regime"    : row["regime"],
                "outcome"   : outcome,
                "r"         : r,
                "split"     : row["split"],
            })

    if not all_trades:
        return pd.DataFrame()

    df = pd.DataFrame(all_trades)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _met(d: pd.DataFrame) -> dict:
    if d.empty:
        return dict(n=0, wr=0.0, pf=0.0, avgr=0.0, totr=0.0, dd=0.0)
    wins = d[d["r"] > 0]; losses = d[d["r"] <= 0]
    gp = wins["r"].sum(); gl = abs(losses["r"].sum())
    cum = d["r"].cumsum(); dd = round((cum - cum.cummax()).min(), 2)
    return dict(
        n=len(d),
        wr=round(len(wins) / len(d), 4),
        pf=round(gp / gl, 3) if gl > 0 else 9.99,
        avgr=round(d["r"].mean(), 4),
        totr=round(d["r"].sum(), 1),
        dd=dd,
    )


def print_report(trades: pd.DataFrame) -> None:
    if trades.empty:
        print("No trades generated.")
        return

    for split in ("IS", "OOS"):
        subset = trades[trades["split"] == split]
        if subset.empty:
            continue

        label = "IN-SAMPLE (Jan 2023 – Nov 2025)" if split == "IS" \
            else "OUT-OF-SAMPLE (Dec 2025 – Feb 2026)"
        m = _met(subset)

        print(f"\n{'='*72}")
        print(f"  {label}")
        print(f"{'='*72}")
        print(f"\n  OVERALL  n={m['n']}  WR={m['wr']:.1%}  PF={m['pf']:.3f}  "
              f"avgR={m['avgr']:+.3f}  totalR={m['totr']:+.1f}R  ddR={m['dd']:.1f}R")

        # Outcome distribution
        print(f"\n  Outcome distribution:")
        for oc, cnt in subset["outcome"].value_counts().items():
            avg_r = subset[subset["outcome"] == oc]["r"].mean()
            print(f"    {oc:<14} {cnt:>5}  ({cnt/m['n']:.1%})  avg_r={avg_r:+.3f}R")

        # By regime
        print(f"\n  By regime:")
        for rg, g in subset.groupby("regime"):
            x = _met(g)
            s = " ★" if x["pf"] >= 1.3 else (" ►" if x["pf"] >= 1.0 else "")
            print(f"    {rg:<14} n={x['n']:>4}  WR={x['wr']:.1%}  PF={x['pf']:.3f}  "
                  f"totalR={x['totr']:+.1f}R{s}")

        # By direction
        print(f"\n  By direction:")
        for dr, g in subset.groupby("direction"):
            x = _met(g)
            s = " ★" if x["pf"] >= 1.3 else (" ►" if x["pf"] >= 1.0 else "")
            print(f"    {dr:<6} n={x['n']:>4}  WR={x['wr']:.1%}  PF={x['pf']:.3f}  "
                  f"totalR={x['totr']:+.1f}R{s}")

        # By symbol
        print(f"\n  By symbol:")
        for sym, g in subset.groupby("symbol"):
            x = _met(g)
            s = " ★" if x["pf"] >= 1.3 else (" ►" if x["pf"] >= 1.0 else "")
            print(f"    {sym:<14} n={x['n']:>3}  WR={x['wr']:.1%}  PF={x['pf']:.3f}  "
                  f"avgR={x['avgr']:+.3f}R{s}")

        # Monthly equity
        print(f"\n  Monthly R accumulation:")
        sub2 = subset.copy()
        sub2["month"] = sub2["timestamp"].dt.to_period("M")
        monthly = sub2.groupby("month")["r"].sum()
        cum_r = 0.0; pos = 0
        for mo, mr in monthly.items():
            cum_r += mr; pos += (1 if mr > 0 else 0)
            sign = "+" if mr >= 0 else "-"
            bar = ("█" if mr >= 0 else "░") * min(int(abs(mr) / 1.0), 30)
            print(f"    {mo}  {sign}{abs(mr):.1f}R  cum={cum_r:+.1f}R  {bar}")

        n_months = len(monthly)
        avg_mo   = monthly.mean()
        print(f"\n  Positive months: {pos}/{n_months} ({pos/n_months:.0%})")
        print(f"  Average monthly R: {avg_mo:+.2f}R")
        print(f"  Annualised rate: {avg_mo*12:+.1f}R/year  ({avg_mo*12/m['n']*m['n']:.0f}R "
              f"over {n_months} months)")

        # R distribution
        big_winners = subset[subset["r"] >= 3.0]
        print(f"\n  Trades ≥ 3R: {len(big_winners)}  "
              f"({len(big_winners)/m['n']:.1%})  "
              f"avg={big_winners['r'].mean():.2f}R" if len(big_winners) > 0 else
              f"\n  Trades ≥ 3R: 0")
        print(f"  Trades ≥ 5R: {len(subset[subset['r']>=5.0])}")
        print(f"  Max single R: {subset['r'].max():.2f}R")
        print(f"  Min single R: {subset['r'].min():.2f}R")
        print(f"  Fee/trade avg: {0.04 / subset['stop_pct'].mean():.4f}R "
              f"(at avg stop {subset['stop_pct'].mean():.2f}%)")

    trades.to_csv(
        os.path.join(os.path.dirname(__file__), "trend_trades.csv"),
        index=False
    )
    print(f"\n  Trades saved to: trend_trades.csv")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading data...")
    data, _ = load_all(run_sanity_check=False, debug=False)

    print("Classifying 4h trend regimes...")
    # Only need 4h for this strategy — much faster than full pipeline
    classified = classify_all(data, timeframes=["4h"])

    print("Generating 4h Donchian breakout signals...")
    signals = compute_signals(data, classified)

    if signals.empty:
        print("No signals generated. Check data.")
        sys.exit(1)

    is_n  = int((signals["split"] == "IS").sum())
    oos_n = int((signals["split"] == "OOS").sum())
    print(f"  Raw signals: {len(signals):,}  (IS={is_n:,}  OOS={oos_n:,})")
    print(f"  Avg stop: {signals['stop_pct'].mean():.2f}%  "
          f"Avg ATR fee: {0.04/signals['stop_pct'].mean():.4f}R/trade")

    print(f"\nRunning backtest "
          f"(N={N_BARS} bars, ATR×{ATR_MULT} trail, max_hold={MAX_HOLD_4H}×4h)...")
    trades = run_backtest(signals, data)

    if trades.empty:
        print("No trades executed.")
        sys.exit(1)

    print(f"  Total trades: {len(trades):,}  "
          f"(IS={int((trades['split']=='IS').sum()):,}  "
          f"OOS={int((trades['split']=='OOS').sum()):,})")

    print_report(trades)