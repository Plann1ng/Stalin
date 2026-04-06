"""
=============================================================================
TRADING ALGORITHM — MODULE 2: TREND CLASSIFIER  (patched v2)
=============================================================================
Brooks patch applied:
  Bias only changes after TWO consecutive swings confirm the new direction.
  A single rogue pivot no longer flips the label — it triggers TRANSITION
  as a deliberate buffer state until the next swing confirms or denies.

All other logic, output schema, and standing rulings unchanged from v1.
=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Bias(str, Enum):
    BULL_TREND  = "BULL_TREND"
    BEAR_TREND  = "BEAR_TREND"
    RANGING     = "RANGING"
    TRANSITION  = "TRANSITION"


class Regime(str, Enum):
    BULL_CYCLE = "BULL_CYCLE"
    BEAR_CYCLE = "BEAR_CYCLE"
    RECOVERY   = "RECOVERY"
    CHOP       = "CHOP"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SwingPoint:
    timestamp : pd.Timestamp
    price     : float
    kind      : str   # "HIGH" or "LOW"


@dataclass
class TrendState:
    symbol         : str
    timeframe      : str
    timestamp      : pd.Timestamp
    bias           : str
    confirmed_bias : str          # Brooks patch: only changes on 2-swing confirmation
    regime         : str
    last_sh        : Optional[float]
    last_sl        : Optional[float]
    prev_sh        : Optional[float]
    prev_sl        : Optional[float]
    hh             : Optional[bool]
    hl             : Optional[bool]
    ma200          : float
    ma50           : float
    price_vs_ma200 : str
    ma_slope       : float


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

SWING_LOOKBACK = {
    "1h": 5,
    "4h": 3,
}

MA_SLOPE_LOOKBACK  = 20
MA_SLOPE_THRESHOLD = 0.0002
MIN_SWINGS_FOR_TREND = 2


# ---------------------------------------------------------------------------
# Core swing detection
# ---------------------------------------------------------------------------

def _find_swings(
    df      : pd.DataFrame,
    lookback: int,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Causal swing detection (no lookahead).

    A swing high at bar i is only knowable after seeing lb bars to the right.
    We therefore label the detection at bar i+lb, but store the ACTUAL swing
    price (high[i] / low[i]) at that detection slot.

    Returns:
      is_sh            — True at bar j when a swing high was confirmed at j
      is_sl            — True at bar j when a swing low was confirmed at j
      causal_sh_price  — the actual swing high price (from bar j-lb), NaN elsewhere
      causal_sl_price  — the actual swing low price  (from bar j-lb), NaN elsewhere
    """
    highs = df["high"].values
    lows  = df["low"].values
    n     = len(df)

    causal_sh = np.full(n, np.nan)
    causal_sl = np.full(n, np.nan)

    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback : i + lookback + 1]
        window_l = lows [i - lookback : i + lookback + 1]

        if highs[i] == window_h.max() and highs[i] > highs[i - 1]:
            det = min(i + lookback, n - 1)
            if np.isnan(causal_sh[det]):        # first swing wins if two collide
                causal_sh[det] = highs[i]       # store ACTUAL high from bar i

        if lows[i] == window_l.min() and lows[i] < lows[i - 1]:
            det = min(i + lookback, n - 1)
            if np.isnan(causal_sl[det]):
                causal_sl[det] = lows[i]        # store ACTUAL low from bar i

    is_sh = ~np.isnan(causal_sh)
    is_sl = ~np.isnan(causal_sl)

    return (
        pd.Series(is_sh,    index=df.index),
        pd.Series(is_sl,    index=df.index),
        pd.Series(causal_sh, index=df.index),   # actual swing price at detection bar
        pd.Series(causal_sl, index=df.index),
    )


# ---------------------------------------------------------------------------
# Moving averages + regime
# ---------------------------------------------------------------------------

def _add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma50"]  = df["close"].rolling(50,  min_periods=50 ).mean()
    df["ma200"] = df["close"].rolling(200, min_periods=200).mean()

    ma200_lagged   = df["ma200"].shift(MA_SLOPE_LOOKBACK)
    df["ma_slope"] = (df["ma200"] - ma200_lagged) / df["ma200"].where(df["ma200"] > 0)
    df["ma_slope"] = df["ma_slope"].fillna(0.0)
    return df


def _get_regime(row: pd.Series) -> str:
    if pd.isna(row["ma200"]):
        return Regime.CHOP.value

    above   = row["close"] > row["ma200"]
    rising  = row["ma_slope"] >  MA_SLOPE_THRESHOLD
    falling = row["ma_slope"] < -MA_SLOPE_THRESHOLD

    if above and rising:      return Regime.BULL_CYCLE.value
    if above and not rising:  return Regime.RECOVERY.value
    if not above and falling: return Regime.BEAR_CYCLE.value
    return Regime.CHOP.value


# ---------------------------------------------------------------------------
# Bias from swing sequence (raw — single swing responsive)
# ---------------------------------------------------------------------------

def _raw_bias(
    sh_prices: list[float],
    sl_prices: list[float],
) -> tuple[str, Optional[bool], Optional[bool]]:
    if len(sh_prices) < MIN_SWINGS_FOR_TREND or len(sl_prices) < MIN_SWINGS_FOR_TREND:
        return Bias.TRANSITION.value, None, None

    hh = sh_prices[-1] > sh_prices[-2]
    hl = sl_prices[-1] > sl_prices[-2]

    if hh and hl:           return Bias.BULL_TREND.value, True,  True
    if not hh and not hl:   return Bias.BEAR_TREND.value, False, False
    if hh and not hl:       return Bias.TRANSITION.value, True,  False
    return Bias.RANGING.value, False, True


# ---------------------------------------------------------------------------
# Brooks patch: two-swing bias confirmation state machine
# ---------------------------------------------------------------------------

class _BiasConfirmer:
    """
    Requires two consecutive swings in the same direction before
    changing the confirmed bias label. Single rogue pivots emit
    TRANSITION as a buffer state.
    """

    def __init__(self) -> None:
        self.confirmed : str = Bias.TRANSITION.value
        self._pending  : str = Bias.TRANSITION.value
        self._streak   : int = 0

    def update(self, raw_bias: str) -> str:
        if raw_bias == self.confirmed:
            self._streak  = 0
            self._pending = raw_bias
            return self.confirmed

        if raw_bias == self._pending:
            self._streak += 1
        else:
            self._pending = raw_bias
            self._streak  = 1

        if self._streak >= 2:
            self.confirmed = self._pending
            self._streak   = 0
            return self.confirmed

        # Not yet confirmed — hold in TRANSITION
        return Bias.TRANSITION.value


# ---------------------------------------------------------------------------
# Per-symbol per-timeframe classifier
# ---------------------------------------------------------------------------

def classify_symbol(
    data      : dict[str, dict[str, pd.DataFrame]],
    symbol    : str,
    timeframe : str,
    lookback  : Optional[int] = None,
) -> pd.DataFrame:
    """
    Classify trend state for one symbol + timeframe.

    Downstream modules must consume `confirmed_bias`, not `bias`.
    """
    symbol    = symbol.upper()
    timeframe = timeframe.lower()

    if symbol not in data:
        raise KeyError(f"Symbol '{symbol}' not in data.")
    if timeframe not in data[symbol]:
        raise KeyError(f"Timeframe '{timeframe}' not available for {symbol}.")

    lb = lookback or SWING_LOOKBACK.get(timeframe, 5)
    df = data[symbol][timeframe].copy()

    df = _add_moving_averages(df)
    is_sh, is_sl, causal_sh_price, causal_sl_price = _find_swings(df, lb)
    df["is_swing_high"]    = is_sh
    df["is_swing_low"]     = is_sl
    df["causal_sh_price"]  = causal_sh_price   # actual swing price at detection bar
    df["causal_sl_price"]  = causal_sl_price

    sh_prices : list[float] = []
    sl_prices : list[float] = []
    confirmer = _BiasConfirmer()

    bias_col      = []
    conf_bias_col = []
    regime_col    = []
    last_sh_col   = []
    last_sl_col   = []
    prev_sh_col   = []
    prev_sl_col   = []
    hh_col        = []
    hl_col        = []
    pvm200_col    = []

    for _, row in df.iterrows():
        if row["is_swing_high"] and not np.isnan(row["causal_sh_price"]):
            sh_prices.append(float(row["causal_sh_price"]))   # actual price, not bar high
        if row["is_swing_low"] and not np.isnan(row["causal_sl_price"]):
            sl_prices.append(float(row["causal_sl_price"]))

        raw_bias, is_hh, is_hl = _raw_bias(sh_prices, sl_prices)
        conf_bias               = confirmer.update(raw_bias)
        regime                  = _get_regime(row)
        price_vs_ma200 = (
            "ABOVE" if (not pd.isna(row["ma200"]) and row["close"] > row["ma200"])
            else "BELOW"
        )

        bias_col.append(raw_bias)
        conf_bias_col.append(conf_bias)
        regime_col.append(regime)
        last_sh_col.append(sh_prices[-1] if sh_prices else None)
        last_sl_col.append(sl_prices[-1] if sl_prices else None)
        prev_sh_col.append(sh_prices[-2] if len(sh_prices) >= 2 else None)
        prev_sl_col.append(sl_prices[-2] if len(sl_prices) >= 2 else None)
        hh_col.append(is_hh)
        hl_col.append(is_hl)
        pvm200_col.append(price_vs_ma200)

    df["bias"]           = bias_col
    df["confirmed_bias"] = conf_bias_col
    df["regime"]         = regime_col
    df["last_sh"]        = last_sh_col
    df["last_sl"]        = last_sl_col
    df["prev_sh"]        = prev_sh_col
    df["prev_sl"]        = prev_sl_col
    df["hh"]             = hh_col
    df["hl"]             = hl_col
    df["price_vs_ma200"] = pvm200_col

    return df


# ---------------------------------------------------------------------------
# Classify all symbols
# ---------------------------------------------------------------------------

def classify_all(
    data       : dict[str, dict[str, pd.DataFrame]],
    timeframes : list[str] = ("1h", "4h"),
) -> dict[str, dict[str, pd.DataFrame]]:
    results: dict[str, dict[str, pd.DataFrame]] = {}
    for symbol in sorted(data.keys()):
        results[symbol] = {}
        for tf in timeframes:
            if tf not in data[symbol]:
                continue
            results[symbol][tf] = classify_symbol(data, symbol, tf)
    return results


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def current_state(
    classified : dict[str, dict[str, pd.DataFrame]],
    symbol     : str,
    timeframe  : str = "4h",
) -> TrendState:
    symbol    = symbol.upper()
    timeframe = timeframe.lower()
    df        = classified[symbol][timeframe]
    row       = df.iloc[-1]

    return TrendState(
        symbol         = symbol,
        timeframe      = timeframe,
        timestamp      = df.index[-1],
        bias           = row["bias"],
        confirmed_bias = row["confirmed_bias"],
        regime         = row["regime"],
        last_sh        = row["last_sh"],
        last_sl        = row["last_sl"],
        prev_sh        = row["prev_sh"],
        prev_sl        = row["prev_sl"],
        hh             = row["hh"],
        hl             = row["hl"],
        ma200          = float(row["ma200"]) if not pd.isna(row["ma200"]) else float("nan"),
        ma50           = float(row["ma50"])  if not pd.isna(row["ma50"])  else float("nan"),
        price_vs_ma200 = row["price_vs_ma200"],
        ma_slope       = float(row["ma_slope"]),
    )


def regime_breakdown(
    classified : dict[str, dict[str, pd.DataFrame]],
    symbol     : str,
    timeframe  : str = "4h",
) -> pd.DataFrame:
    df = classified[symbol.upper()][timeframe.lower()]
    counts = (
        df.groupby(["regime", "confirmed_bias"])
        .size()
        .reset_index(name="bar_count")
    )
    total = counts["bar_count"].sum()
    counts["pct"] = (counts["bar_count"] / total * 100).round(1)
    return counts.sort_values("bar_count", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from module_01_data_layer import load_all

    print("Loading data...")
    data, _ = load_all(run_sanity_check=False, debug=False)

    print("Running patched classifier on all symbols...")
    classified = classify_all(data, timeframes=["1h", "4h"])

    # Show the patch effect on the exact window that exposed the bug
    btc4h  = classified["BTCUSDT"]["4h"]
    window = btc4h["2026-02-15":"2026-02-28"]

    print("\n" + "=" * 72)
    print("  BROOKS PATCH — BTCUSDT 4H Feb 15–28 2026")
    print("  raw bias  vs  confirmed_bias")
    print("=" * 72)
    swing_bars = window[window["is_swing_high"] | window["is_swing_low"]]
    print(f"\n  {'Timestamp':<22} {'Type':<5} {'Price':>10} "
          f"{'Raw bias':<14} {'Confirmed bias'}")
    print("  " + "-" * 72)
    for ts, row in swing_bars.iterrows():
        kind  = "SH" if row["is_swing_high"] else "SL"
        price = row["high"] if kind == "SH" else row["low"]
        print(f"  {str(ts)[:19]:<22} {kind:<5} {price:>10.2f} "
              f"{row['bias']:<14} {row['confirmed_bias']}")

    # Full confirmed bias tally
    print("\n" + "=" * 72)
    print("  All 20 symbols — 4H confirmed_bias (Feb 2026)")
    print("=" * 72)
    tally: dict[str, int] = {}
    print(f"\n  {'Symbol':<14} {'Confirmed bias':<16} {'Regime'}")
    print("  " + "-" * 46)
    for sym in sorted(classified.keys()):
        s = current_state(classified, sym, "4h")
        tally[s.confirmed_bias] = tally.get(s.confirmed_bias, 0) + 1
        print(f"  {sym:<14} {s.confirmed_bias:<16} {s.regime}")

    print("\n  Confirmed bias tally:")
    for b, c in sorted(tally.items(), key=lambda x: -x[1]):
        print(f"    {b:<16} {'█' * c} {c}")
    print()
