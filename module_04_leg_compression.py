"""
=============================================================================
TRADING ALGORITHM — MODULE 4: LEG COUNTER + COMPRESSION DETECTOR
=============================================================================
Two components, one module:

COMPONENT A — Leg Counter (Brooks mandate)
  Runs on: 15m and 30m (setup timeframes)
  Counts with-trend legs since the last significant pivot.
  Identifies whether price is currently in a pullback and how deep.
  Entry model: buy pullbacks after leg 1 or leg 2. Avoid leg 3+.

COMPONENT B — Compression Detector
  Runs on: 5m and 15m (entry timeframes)
  Detects range contraction + declining volume = coiling spring.
  Uses Binance taker_buy delta for directional pressure inside compression.

Output schema (Douglas condition):

  LegState(
      symbol             : str
      timeframe          : str
      timestamp          : pd.Timestamp
      confirmed_bias     : str        # from module 2
      leg_num            : int        # 1=first leg, 2=second, 3+=late/exhausted
      bar_context        : str        # WITH_TREND | PULLBACK | CHOP
      in_pullback        : bool
      pullback_depth_pct : float      # % retracement of prior leg (0–1+)
      prior_leg_size_pct : float      # size of prior with-trend leg (%)
      leg_quality        : str        # IDEAL | VALID | LATE | SKIP
  )

  CompressionState(
      symbol              : str
      timeframe           : str
      timestamp           : pd.Timestamp
      is_compressed       : bool
      range_ratio         : float     # rolling range / ATR14
      atr14               : float
      bars_in_compression : int
      volume_declining    : bool
      delta_bias          : str       # BUY_PRESSURE | SELL_PRESSURE | NEUTRAL
      coiling             : bool      # compressed + volume_declining
      breakout_watch      : bool      # coiling AND near a quality S/R zone
  )

Usage:
  from module_04_leg_compression import compute_all

  result = compute_all(data, classified)
  legs        = result["legs"]        # dict[symbol][tf] → pd.DataFrame
  compression = result["compression"] # dict[symbol][tf] → pd.DataFrame
=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# Leg counter
LEG_COUNTER_TFS  = ["15m", "30m", "1h"]
SWING_LOOKBACK   = {"15m": 5, "30m": 4, "1h": 3}

# A pullback is "deep" if it retraces more than this fraction of the prior leg
DEEP_PULLBACK_THRESHOLD = 0.618   # Fibonacci — beyond this the leg may be broken
# A pullback is "shallow" below this threshold — still early, safer entry
SHALLOW_PULLBACK_THRESHOLD = 0.382

# Compression detector
COMPRESSION_TFS       = ["5m", "15m"]
COMPRESSION_WINDOW    = 10     # bars to measure rolling range
ATR_PERIOD            = 14     # bars for ATR baseline
COMPRESSION_THRESHOLD = 0.40   # fallback static threshold (overridden by dynamic below)
COMPRESSION_PERCENTILE = 20    # rolling Nth percentile of range_ratio = dynamic threshold
COMPRESSION_PERCENTILE_WINDOW = 200  # bars of history for percentile calculation
VOLUME_SLOPE_WINDOW   = 8      # bars for volume trend (declining into compression)
DELTA_NEUTRAL_BAND    = 0.48   # buy_ratio between 0.48–0.52 = neutral pressure


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LegState:
    symbol             : str
    timeframe          : str
    timestamp          : pd.Timestamp
    confirmed_bias     : str
    leg_num            : int
    bar_context        : str        # WITH_TREND | PULLBACK | CHOP
    in_pullback        : bool
    pullback_depth_pct : float      # 0.0 = no retrace, 1.0 = full retrace
    prior_leg_size_pct : float      # size of prior with-trend leg as % of price
    leg_quality        : str        # IDEAL | VALID | LATE | SKIP


@dataclass
class CompressionState:
    symbol              : str
    timeframe           : str
    timestamp           : pd.Timestamp
    is_compressed       : bool
    range_ratio         : float
    atr14               : float
    bars_in_compression : int
    volume_declining    : bool
    delta_bias          : str       # BUY_PRESSURE | SELL_PRESSURE | NEUTRAL
    coiling             : bool
    breakout_watch      : bool


# ---------------------------------------------------------------------------
# Shared: swing detection (lightweight, same logic as module 2)
# ---------------------------------------------------------------------------

def _find_swings(df: pd.DataFrame, lookback: int) -> tuple[np.ndarray, np.ndarray,
                                                            np.ndarray, np.ndarray]:
    """
    Causal swing detection — no lookahead.
    Swing at bar i confirmed after lb bars; label placed at i+lb with
    the actual swing price stored at that detection slot.

    Returns: (is_sh, is_sl, causal_sh_price, causal_sl_price)
    """
    highs = df["high"].values
    lows  = df["low"].values
    n     = len(df)

    causal_sh = np.full(n, np.nan)
    causal_sl = np.full(n, np.nan)

    for i in range(lookback, n - lookback):
        wh = highs[i - lookback : i + lookback + 1]
        wl = lows [i - lookback : i + lookback + 1]
        if highs[i] == wh.max() and highs[i] > highs[i - 1]:
            det = min(i + lookback, n - 1)
            if np.isnan(causal_sh[det]):
                causal_sh[det] = highs[i]   # store ACTUAL swing high price
        if lows[i] == wl.min() and lows[i] < lows[i - 1]:
            det = min(i + lookback, n - 1)
            if np.isnan(causal_sl[det]):
                causal_sl[det] = lows[i]    # store ACTUAL swing low price

    return (~np.isnan(causal_sh), ~np.isnan(causal_sl), causal_sh, causal_sl)


# ---------------------------------------------------------------------------
# COMPONENT A — Leg Counter
# ---------------------------------------------------------------------------

def _leg_quality(leg_num: int, depth: float) -> str:
    """
    Classify the trade quality of a pullback entry.

    IDEAL : leg 1 or 2 pullback, depth 0.382–0.618 (healthy retrace)
    VALID : leg 1 or 2 pullback, depth outside ideal but not broken
    LATE  : leg 3+ pullback (move may be exhausted)
    SKIP  : pullback too deep (> 0.75 — structure likely broken) or no pullback
    """
    if depth > 0.75:
        return "SKIP"
    if leg_num >= 3:
        return "LATE"
    if SHALLOW_PULLBACK_THRESHOLD <= depth <= DEEP_PULLBACK_THRESHOLD:
        return "IDEAL"
    return "VALID"


def compute_legs(
    df             : pd.DataFrame,
    symbol         : str,
    timeframe      : str,
    confirmed_bias_series: pd.Series,
) -> pd.DataFrame:
    """
    Compute bar-by-bar LegState for one symbol + timeframe.

    Logic (BULL_TREND example — inverted for BEAR_TREND):
      - Each new swing high completes a with-trend leg
      - Between swing high and next swing low = pullback
      - Between swing low and next swing high = new leg
      - leg_num increments each time a new with-trend leg starts
      - pullback_depth = (last SH - current close) / (last SH - last SL)

    CHOP / RANGING / TRANSITION → leg_num=0, context=CHOP
    """
    lb  = SWING_LOOKBACK.get(timeframe, 5)
    df  = df.copy()
    is_sh, is_sl, causal_sh_price, causal_sl_price = _find_swings(df, lb)

    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    n      = len(df)

    # Output arrays
    leg_num_arr    = np.zeros(n, dtype=int)
    context_arr    = ["CHOP"] * n
    in_pb_arr      = np.zeros(n, dtype=bool)
    pb_depth_arr   = np.zeros(n, dtype=float)
    leg_size_arr   = np.zeros(n, dtype=float)
    quality_arr    = ["SKIP"] * n

    # Running state
    last_sh     : Optional[float] = None
    last_sl     : Optional[float] = None
    leg_start   : Optional[float] = None  # price at start of current with-trend leg
    leg_num     = 0
    in_pullback = False
    prev_active_bias = "TRANSITION"  # tracks last non-chop bias for reset detection

    for i in range(n):
        bias  = confirmed_bias_series.iloc[i]
        close = closes[i]

        # Reset leg counter when trend genuinely flips direction
        if bias in ("BULL_TREND", "BEAR_TREND"):
            if prev_active_bias in ("BULL_TREND", "BEAR_TREND") and bias != prev_active_bias:
                leg_num     = 0
                in_pullback = False
                last_sh     = None
                last_sl     = None
            prev_active_bias = bias

        if bias == "BULL_TREND":
            # Update swing points using CAUSAL stored prices
            if is_sl[i]:
                new_sl = float(causal_sl_price[i])   # actual swing low, not highs[i]
                if last_sl is None or new_sl < last_sl or not in_pullback:
                    # Pullback found its low — start of new leg
                    leg_start   = new_sl
                    in_pullback = False
                    if last_sh is not None:
                        leg_num += 1
                last_sl = new_sl

            if is_sh[i]:
                last_sh     = float(causal_sh_price[i])  # actual swing high
                in_pullback = True   # now watching for pullback

            # Compute pullback depth
            if in_pullback and last_sh is not None and last_sl is not None:
                leg_range = last_sh - last_sl
                if leg_range > 0:
                    depth = (last_sh - close) / leg_range
                    depth = max(0.0, min(depth, 1.5))
                else:
                    depth = 0.0
                prior_leg_pct = leg_range / last_sl if last_sl > 0 else 0.0
                context       = "PULLBACK"
            else:
                depth         = 0.0
                prior_leg_pct = 0.0
                context       = "WITH_TREND"

        elif bias == "BEAR_TREND":
            # Mirror logic for bear — using causal stored prices
            if is_sh[i]:
                new_sh = float(causal_sh_price[i])   # actual swing high
                if last_sh is None or new_sh > last_sh or not in_pullback:
                    leg_start   = new_sh
                    in_pullback = False
                    if last_sl is not None:
                        leg_num += 1
                last_sh = new_sh

            if is_sl[i]:
                last_sl     = float(causal_sl_price[i])  # actual swing low
                in_pullback = True

            if in_pullback and last_sl is not None and last_sh is not None:
                leg_range = last_sh - last_sl
                if leg_range > 0:
                    depth = (close - last_sl) / leg_range
                    depth = max(0.0, min(depth, 1.5))
                else:
                    depth = 0.0
                prior_leg_pct = leg_range / last_sh if last_sh > 0 else 0.0
                context       = "PULLBACK"
            else:
                depth         = 0.0
                prior_leg_pct = 0.0
                context       = "WITH_TREND"

        else:
            # RANGING / TRANSITION / CHOP — hold last confirmed leg_num,
            # do not reset. leg_num resets only when opposing bias confirms.
            in_pullback   = False
            depth         = 0.0
            prior_leg_pct = 0.0
            context       = "CHOP"

        quality = _leg_quality(leg_num, depth) if in_pullback else "SKIP"

        leg_num_arr[i]  = leg_num
        context_arr[i]  = context
        in_pb_arr[i]    = in_pullback
        pb_depth_arr[i] = round(depth, 4)
        leg_size_arr[i] = round(prior_leg_pct * 100, 4)
        quality_arr[i]  = quality

    df["confirmed_bias"]     = confirmed_bias_series.values
    df["leg_num"]            = leg_num_arr
    df["bar_context"]        = context_arr
    df["in_pullback"]        = in_pb_arr
    df["pullback_depth_pct"] = pb_depth_arr
    df["prior_leg_size_pct"] = leg_size_arr
    df["leg_quality"]        = quality_arr

    return df


# ---------------------------------------------------------------------------
# COMPONENT B — Compression Detector
# ---------------------------------------------------------------------------

def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    """True Range ATR — handles gaps between candles."""
    high  = df["high"]
    low   = df["low"]
    close = df["close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low  - close).abs(),
    ], axis=1).max(axis=1)

    return tr.rolling(period, min_periods=period).mean()


def _delta_bias(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Rolling buy_ratio over `window` bars.
    Returns: BUY_PRESSURE | SELL_PRESSURE | NEUTRAL
    Uses Binance taker_buy delta — the gold column from module 1.
    """
    if "buy_ratio" not in df.columns:
        return pd.Series("NEUTRAL", index=df.index)

    rolling_ratio = df["buy_ratio"].rolling(window, min_periods=3).mean()

    conditions = [
        rolling_ratio > (0.5 + DELTA_NEUTRAL_BAND / 2),
        rolling_ratio < (0.5 - DELTA_NEUTRAL_BAND / 2),
    ]
    choices    = ["BUY_PRESSURE", "SELL_PRESSURE"]
    return pd.Series(
        np.select(conditions, choices, default="NEUTRAL"),
        index=df.index,
    )


def compute_compression(
    df        : pd.DataFrame,
    symbol    : str,
    timeframe : str,
) -> pd.DataFrame:
    """
    Compute bar-by-bar CompressionState for one symbol + timeframe.
    """
    df = df.copy()

    # ── ATR14 baseline ─────────────────────────────────────────────────────
    df["_atr14"] = _atr(df, ATR_PERIOD)

    # ── Rolling candle range (high - low) ──────────────────────────────────
    df["_candle_range"] = df["high"] - df["low"]
    df["_rolling_range"] = (
        df["_candle_range"]
        .rolling(COMPRESSION_WINDOW, min_periods=COMPRESSION_WINDOW)
        .mean()
    )

    # ── Range ratio ────────────────────────────────────────────────────────
    df["range_ratio"] = df["_rolling_range"] / df["_atr14"].where(df["_atr14"] > 0)
    df["range_ratio"] = df["range_ratio"].fillna(1.0)

    # ── Dynamic compression threshold (rolling Nth percentile) ────────────
    # Self-calibrating per symbol/TF: "compressed" means bottom 20% of
    # recent range ratios for THIS instrument — not a fixed equity-tuned number.
    df["_dynamic_threshold"] = (
        df["range_ratio"]
        .rolling(COMPRESSION_PERCENTILE_WINDOW, min_periods=30)
        .quantile(COMPRESSION_PERCENTILE / 100)
        .fillna(COMPRESSION_THRESHOLD)
    )
    df["is_compressed"] = df["range_ratio"] < df["_dynamic_threshold"]

    # ── Consecutive bars in compression ────────────────────────────────────
    compressed_arr = df["is_compressed"].values
    bars_in        = np.zeros(len(df), dtype=int)
    count          = 0
    for i in range(len(df)):
        if compressed_arr[i]:
            count += 1
        else:
            count = 0
        bars_in[i] = count
    df["bars_in_compression"] = bars_in

    # ── Volume declining slope ─────────────────────────────────────────────
    if "volume" in df.columns:
        vol_ma       = df["volume"].rolling(VOLUME_SLOPE_WINDOW, min_periods=3).mean()
        vol_ma_lagged = vol_ma.shift(VOLUME_SLOPE_WINDOW // 2)
        df["volume_declining"] = vol_ma < vol_ma_lagged
    else:
        df["volume_declining"] = False

    # ── Delta bias (directional pressure) ─────────────────────────────────
    df["delta_bias"] = _delta_bias(df, COMPRESSION_WINDOW)

    # ── Coiling: compressed + volume declining ─────────────────────────────
    df["coiling"] = df["is_compressed"] & df["volume_declining"]

    # ── Rename ATR for output ──────────────────────────────────────────────
    df["atr14"] = df["_atr14"]

    # Drop internal columns
    df = df.drop(
        columns=["_atr14", "_candle_range", "_rolling_range", "_dynamic_threshold"],
        errors="ignore",
    )

    return df


# ---------------------------------------------------------------------------
# Combined runner
# ---------------------------------------------------------------------------

def compute_all(
    data       : dict[str, dict[str, pd.DataFrame]],
    classified : dict[str, dict[str, pd.DataFrame]],
) -> dict[str, dict[str, dict[str, pd.DataFrame]]]:
    """
    Run both components for all symbols.

    Returns:
      result["legs"][symbol][tf]        → DataFrame with LegState columns
      result["compression"][symbol][tf] → DataFrame with CompressionState columns
    """
    legs_out        : dict[str, dict[str, pd.DataFrame]] = {}
    compression_out : dict[str, dict[str, pd.DataFrame]] = {}

    for symbol in sorted(data.keys()):

        # ── Leg counter ───────────────────────────────────────────────────
        legs_out[symbol] = {}
        for tf in LEG_COUNTER_TFS:
            if tf not in data[symbol]:
                continue

            # Get confirmed_bias from module 2 classification
            # If this TF was classified, use it. Otherwise align from 30m.
            if tf in classified.get(symbol, {}):
                bias_series = classified[symbol][tf]["confirmed_bias"]
            elif "30m" in classified.get(symbol, {}):
                # Align 30m bias to this TF by forward-fill
                bias_30m = classified[symbol]["30m"]["confirmed_bias"]
                df_tf    = data[symbol][tf]
                bias_series = bias_30m.reindex(df_tf.index, method="ffill").fillna("TRANSITION")
            else:
                continue

            legs_out[symbol][tf] = compute_legs(
                data[symbol][tf].copy(),
                symbol,
                tf,
                bias_series,
            )

        # ── Compression detector ──────────────────────────────────────────
        compression_out[symbol] = {}
        for tf in COMPRESSION_TFS:
            if tf not in data[symbol]:
                continue
            compression_out[symbol][tf] = compute_compression(
                data[symbol][tf].copy(),
                symbol,
                tf,
            )

    return {"legs": legs_out, "compression": compression_out}


# ---------------------------------------------------------------------------
# Convenience: current bar snapshot
# ---------------------------------------------------------------------------

def current_leg(
    result : dict,
    symbol : str,
    tf     : str = "15m",
) -> Optional[LegState]:
    """Return the LegState for the most recent bar."""
    sym = symbol.upper()
    try:
        df  = result["legs"][sym][tf]
        row = df.iloc[-1]
        return LegState(
            symbol             = sym,
            timeframe          = tf,
            timestamp          = df.index[-1],
            confirmed_bias     = row["confirmed_bias"],
            leg_num            = int(row["leg_num"]),
            bar_context        = row["bar_context"],
            in_pullback        = bool(row["in_pullback"]),
            pullback_depth_pct = float(row["pullback_depth_pct"]),
            prior_leg_size_pct = float(row["prior_leg_size_pct"]),
            leg_quality        = row["leg_quality"],
        )
    except (KeyError, IndexError):
        return None


def current_compression(
    result : dict,
    symbol : str,
    tf     : str = "5m",
) -> Optional[CompressionState]:
    """Return the CompressionState for the most recent bar."""
    sym = symbol.upper()
    try:
        df  = result["compression"][sym][tf]
        row = df.iloc[-1]
        return CompressionState(
            symbol              = sym,
            timeframe           = tf,
            timestamp           = df.index[-1],
            is_compressed       = bool(row["is_compressed"]),
            range_ratio         = float(row["range_ratio"]),
            atr14               = float(row["atr14"]) if not pd.isna(row["atr14"]) else 0.0,
            bars_in_compression = int(row["bars_in_compression"]),
            volume_declining    = bool(row["volume_declining"]),
            delta_bias          = row["delta_bias"],
            coiling             = bool(row["coiling"]),
            breakout_watch      = False,   # set by module 5 when near a zone
        )
    except (KeyError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    from module_01_data_layer       import load_all
    from module_02_trend_classifier import classify_all, classify_symbol as _cs

    print("Loading data...")
    data, _ = load_all(run_sanity_check=False, debug=False)

    print("Classifying trends...")
    classified = classify_all(data, timeframes=["1h", "4h"])
    for sym in data:
        for tf in ["30m", "15m"]:
            if tf in data[sym]:
                classified[sym][tf] = _cs(
                    data, sym, tf,
                    lookback=SWING_LOOKBACK.get(tf, 5),
                )

    print("Running leg counter + compression detector...")
    result = compute_all(data, classified)

    # ── Leg quality distribution ──────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  MODULE 4 — LEG COUNTER")
    print("  Leg quality distribution — 15m, full dataset")
    print("=" * 68)

    quality_tally: dict[str, int] = {"IDEAL": 0, "VALID": 0, "LATE": 0, "SKIP": 0}
    context_tally: dict[str, int] = {"WITH_TREND": 0, "PULLBACK": 0, "CHOP": 0}

    for sym in result["legs"]:
        if "15m" not in result["legs"][sym]:
            continue
        df = result["legs"][sym]["15m"]
        for q in quality_tally:
            quality_tally[q] += int((df["leg_quality"] == q).sum())
        for c in context_tally:
            context_tally[c] += int((df["bar_context"] == c).sum())

    total_bars = sum(quality_tally.values())
    print(f"\n  Across all 20 symbols on 15m ({total_bars:,} total bars):\n")
    print(f"  {'Context':<14} {'Bars':>8}  {'%':>6}")
    print("  " + "-" * 32)
    for ctx, cnt in context_tally.items():
        pct = cnt / total_bars * 100 if total_bars else 0
        print(f"  {ctx:<14} {cnt:>8,}  {pct:>5.1f}%")

    print(f"\n  {'Leg quality':<14} {'Bars':>8}  {'%':>6}  (in-pullback bars only)")
    print("  " + "-" * 44)
    pb_total = quality_tally["IDEAL"] + quality_tally["VALID"] + quality_tally["LATE"] + quality_tally["SKIP"]
    for q, cnt in quality_tally.items():
        pct = cnt / pb_total * 100 if pb_total else 0
        print(f"  {q:<14} {cnt:>8,}  {pct:>5.1f}%")

    # ── BTCUSDT 15m current state ─────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  BTCUSDT — Current leg state (last 10 bars, 15m)")
    print("=" * 68)
    btc15 = result["legs"]["BTCUSDT"]["15m"]
    tail  = btc15.tail(10)
    print(f"\n  {'Timestamp':<22} {'Bias':<14} {'Leg':>3} {'Context':<12} "
          f"{'InPB':>5} {'Depth':>6} {'Quality'}")
    print("  " + "-" * 74)
    for ts, row in tail.iterrows():
        print(
            f"  {str(ts)[:19]:<22} {row['confirmed_bias']:<14} "
            f"{int(row['leg_num']):>3} {row['bar_context']:<12} "
            f"{'Y' if row['in_pullback'] else 'N':>5} "
            f"{row['pullback_depth_pct']:>6.3f} "
            f"{row['leg_quality']}"
        )

    # ── Compression stats ─────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  MODULE 4 — COMPRESSION DETECTOR")
    print("  Compression + coiling frequency — 5m, full dataset")
    print("=" * 68)

    comp_compressed = comp_coiling = comp_total = 0
    for sym in result["compression"]:
        if "5m" not in result["compression"][sym]:
            continue
        df = result["compression"][sym]["5m"]
        comp_total      += len(df)
        comp_compressed += int(df["is_compressed"].sum())
        comp_coiling    += int(df["coiling"].sum())

    print(f"\n  Total 5m bars across all symbols: {comp_total:,}")
    print(f"  Compressed bars:  {comp_compressed:,}  ({comp_compressed/comp_total*100:.1f}%)")
    print(f"  Coiling bars:     {comp_coiling:,}  ({comp_coiling/comp_total*100:.1f}%)")
    print(f"  (Coiling = compressed + volume declining = highest-quality setup precursor)")

    # ── BTCUSDT 5m current compression ───────────────────────────────────
    print("\n" + "=" * 68)
    print("  BTCUSDT — Current compression state (last 10 bars, 5m)")
    print("=" * 68)
    btc5 = result["compression"]["BTCUSDT"]["5m"]
    tail5 = btc5.tail(10)
    print(f"\n  {'Timestamp':<22} {'Compressed':>10} {'RatioR':>7} "
          f"{'ATR14':>8} {'BarsInC':>7} {'VolDown':>7} "
          f"{'Delta':>14} {'Coiling':>7}")
    print("  " + "-" * 90)
    for ts, row in tail5.iterrows():
        print(
            f"  {str(ts)[:19]:<22} "
            f"{'YES' if row['is_compressed'] else 'no':>10} "
            f"{row['range_ratio']:>7.3f} "
            f"{row['atr14']:>8.2f} "
            f"{int(row['bars_in_compression']):>7} "
            f"{'Y' if row['volume_declining'] else 'N':>7} "
            f"{row['delta_bias']:>14} "
            f"{'COIL' if row['coiling'] else 'no':>7}"
        )

    # ── All symbols current state snapshot ───────────────────────────────
    print("\n" + "=" * 68)
    print("  All symbols — current 15m leg + 5m compression (latest bar)")
    print("=" * 68)
    print(f"\n  {'Symbol':<14} {'Leg':>3} {'Quality':<8} {'Context':<12} "
          f"{'Depth':>6}  {'Coiling':>7}  {'Delta bias'}")
    print("  " + "-" * 68)

    for sym in sorted(result["legs"].keys()):
        ls = current_leg(result, sym, "15m")
        cs = current_compression(result, sym, "5m")
        if ls is None or cs is None:
            continue
        print(
            f"  {sym:<14} {ls.leg_num:>3} {ls.leg_quality:<8} "
            f"{ls.bar_context:<12} {ls.pullback_depth_pct:>6.3f}  "
            f"{'COIL' if cs.coiling else 'no':>7}  "
            f"{cs.delta_bias}"
        )
    print()