"""
=============================================================================
TRADING ALGORITHM — MODULE 1: DATA LAYER
=============================================================================
Committee-approved output schema (Douglas condition):
  data[symbol][timeframe] -> pd.DataFrame

Timeframes produced:
  "5m"  -> loaded directly from CSV
  "15m" -> loaded directly from CSV
  "30m" -> loaded directly from CSV
  "1h"  -> resampled from 30m (2 candles)
  "4h"  -> resampled from 30m (8 candles)

Core columns on every DataFrame:
  open_time (DatetimeIndex UTC)
  open, high, low, close, volume
  buy_vol, sell_vol, delta, buy_ratio

Quality flag columns (Volman condition):
  is_gap_after    -> True if expected next candle is missing
  is_zero_vol     -> True if volume == 0
  is_duplicate    -> True if timestamp appears more than once

Side output:
  QualityReport per symbol — printed to console on load
=============================================================================
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path("/Users/kasuya/binance_merged")

# Maps filename suffix → canonical timeframe key → pandas offset alias
TF_FILE_MAP = {
    "5m":  "5min",
    "15m": "15min",
    "30m": "30min",
}

# Timeframes derived by resampling 30m
TF_RESAMPLE_MAP = {
    "1h": "1h",
    "4h": "4h",
}

# Columns we keep from the raw Binance Kline CSV
KEEP_COLS = [
    "open_time",
    "open", "high", "low", "close",
    "volume",
    "taker_buy_base_asset_volume",
    "quote_asset_volume",
    "number_of_trades",
]

# Columns to drop (Binance placeholders)
DROP_COLS = ["close_time", "ignore"]

# Column rename map → clean internal names
RENAME_MAP = {
    "taker_buy_base_asset_volume": "taker_buy_vol",
    "quote_asset_volume":          "quote_vol",
    "number_of_trades":            "num_trades",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class QualityReport:
    symbol:         str
    timeframe:      str
    candle_count:   int
    date_range:     str
    gap_count:      int
    zero_vol_count: int
    dup_count:      int

    def __str__(self) -> str:
        status = "CLEAN" if (self.gap_count + self.zero_vol_count + self.dup_count) == 0 else "FLAGGED"
        return (
            f"  [{status}] {self.symbol} {self.timeframe:>4s} | "
            f"{self.candle_count:>6,} candles | "
            f"{self.date_range} | "
            f"gaps={self.gap_count}  zero_vol={self.zero_vol_count}  dups={self.dup_count}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_symbol_tf(filename: str) -> Optional[tuple[str, str]]:
    """
    Extract (symbol, timeframe) from filename.
    Expects format: BTCUSDT_30m.csv  or  BTCUSDT_15m.csv  etc.
    Returns None if the file doesn't match the expected pattern.
    """
    stem = Path(filename).stem          # e.g. "BTCUSDT_30m"
    match = re.fullmatch(r"([A-Z0-9]+)_(\d+[mh])", stem, re.IGNORECASE)
    if not match:
        return None
    symbol = match.group(1).upper()
    tf     = match.group(2).lower()
    if tf not in TF_FILE_MAP:
        return None
    return symbol, tf


def _load_raw_csv(path: Path) -> pd.DataFrame:
    """Load a single Binance Kline CSV and return a clean DataFrame."""
    df = pd.read_csv(path, usecols=lambda c: c not in DROP_COLS)

    # Keep only the columns we want (others silently ignored)
    available = [c for c in KEEP_COLS if c in df.columns]
    df = df[available].copy()

    # Parse timestamp
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.set_index("open_time").sort_index()

    # Rename to clean internal names
    df = df.rename(columns=RENAME_MAP)

    # Coerce numeric columns
    numeric_cols = ["open", "high", "low", "close", "volume",
                    "taker_buy_vol", "quote_vol", "num_trades"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _add_delta_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive delta volume columns from Binance taker_buy_vol.
      buy_vol   = taker_buy_vol  (aggressive buy orders)
      sell_vol  = volume - buy_vol  (aggressive sell orders)
      delta     = buy_vol - sell_vol
      buy_ratio = buy_vol / volume  (0..1; 0.5 = neutral)
    """
    if "taker_buy_vol" not in df.columns:
        df["buy_vol"]   = np.nan
        df["sell_vol"]  = np.nan
        df["delta"]     = np.nan
        df["buy_ratio"] = np.nan
        return df

    df["buy_vol"]  = df["taker_buy_vol"]
    df["sell_vol"] = df["volume"] - df["buy_vol"]
    df["delta"]    = df["buy_vol"] - df["sell_vol"]

    # Avoid division by zero on zero-volume candles
    df["buy_ratio"] = np.where(
        df["volume"] > 0,
        df["buy_vol"] / df["volume"],
        np.nan,
    )
    return df


def _add_quality_flags(df: pd.DataFrame, tf_offset: str) -> pd.DataFrame:
    """
    Add three boolean quality-flag columns (Volman condition).

    is_gap_after  : the NEXT expected candle is missing
    is_zero_vol   : volume == 0 on this candle
    is_duplicate  : this timestamp appears more than once
    """
    # Duplicates
    df["is_duplicate"] = df.index.duplicated(keep=False)

    # Zero volume
    df["is_zero_vol"] = df["volume"] == 0

    # Gaps: expected spacing vs actual next timestamp
    expected_delta = pd.tseries.frequencies.to_offset(tf_offset).nanos
    actual_delta   = df.index.to_series().diff().shift(-1)  # gap AFTER current bar
    df["is_gap_after"] = actual_delta.dt.total_seconds() * 1e9 > expected_delta * 1.5

    return df


def _build_quality_report(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
) -> QualityReport:
    date_range = (
        f"{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}"
    )
    return QualityReport(
        symbol       = symbol,
        timeframe    = timeframe,
        candle_count = len(df),
        date_range   = date_range,
        gap_count    = int(df["is_gap_after"].sum()),
        zero_vol_count = int(df["is_zero_vol"].sum()),
        dup_count    = int(df["is_duplicate"].sum()),
    )


def _resample_ohlcv(df_30m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample a 30m DataFrame to a higher timeframe using lossless OHLCV rules.
    Delta volume columns aggregate correctly (sum).
    Closed on the left, label on the left — standard Binance convention.
    """
    agg = {
        "open":      "first",
        "high":      "max",
        "low":       "min",
        "close":     "last",
        "volume":    "sum",
    }

    # Optional columns present only if source had them
    optional_sum_cols = [
        "taker_buy_vol", "quote_vol", "num_trades",
        "buy_vol", "sell_vol",
    ]
    for col in optional_sum_cols:
        if col in df_30m.columns:
            agg[col] = "sum"

    df_resampled = (
        df_30m
        .resample(rule, closed="left", label="left")
        .agg(agg)
        .dropna(subset=["close"])   # drop incomplete trailing candle
    )

    # Recompute delta from aggregated buy/sell (aggregated sums are already correct)
    df_resampled = _add_delta_volume(df_resampled)

    return df_resampled


def _resample_sanity_check(df_30m: pd.DataFrame, symbol: str) -> None:
    """
    Brooks condition: print a spot-check showing raw 30m candles and the
    derived 1H and 4H candles beside them.  Runs once on the first symbol.
    """
    print("\n" + "=" * 68)
    print("  BROOKS SANITY CHECK — Resampling verification")
    print("=" * 68)

    # Pick the first complete 1H window (first two 30m candles)
    first_ts  = df_30m.index[0]
    two_bars  = df_30m.iloc[:2]

    print(f"\n  Symbol : {symbol}")
    print(f"  Window : {first_ts.strftime('%Y-%m-%d %H:%M')} UTC\n")

    print("  ── Raw 30m candles ──────────────────────────────────")
    for ts, row in two_bars.iterrows():
        print(
            f"  {ts.strftime('%H:%M')}  "
            f"O={row['open']:.2f}  H={row['high']:.2f}  "
            f"L={row['low']:.2f}  C={row['close']:.2f}  "
            f"Vol={row['volume']:.4f}  "
            f"Δvol={row.get('delta', float('nan')):.4f}"
        )

    # Derive 1H manually so numbers are crystal clear
    h1_open   = two_bars["open"].iloc[0]
    h1_high   = two_bars["high"].max()
    h1_low    = two_bars["low"].min()
    h1_close  = two_bars["close"].iloc[-1]
    h1_vol    = two_bars["volume"].sum()
    h1_buy    = two_bars.get("buy_vol", pd.Series([0, 0])).sum()
    h1_sell   = two_bars.get("sell_vol", pd.Series([0, 0])).sum()
    h1_delta  = h1_buy - h1_sell

    print("\n  ── Derived 1H candle ────────────────────────────────")
    print(
        f"  {first_ts.strftime('%H:%M')}  "
        f"O={h1_open:.2f}  H={h1_high:.2f}  "
        f"L={h1_low:.2f}  C={h1_close:.2f}  "
        f"Vol={h1_vol:.4f}  Δvol={h1_delta:.4f}"
    )

    # Verify against pandas resample
    df_1h = _resample_ohlcv(df_30m.iloc[:2], "1h")
    if not df_1h.empty:
        r = df_1h.iloc[0]
        match = (
            abs(r["open"]  - h1_open)  < 0.01 and
            abs(r["high"]  - h1_high)  < 0.01 and
            abs(r["low"]   - h1_low)   < 0.01 and
            abs(r["close"] - h1_close) < 0.01 and
            abs(r["volume"] - h1_vol)  < 0.0001
        )
        print(f"\n  Pandas resample match: {'✓ CONFIRMED' if match else '✗ MISMATCH — investigate'}")

    # 4H check (show first 8 bars)
    eight_bars = df_30m.iloc[:8]
    h4_open   = eight_bars["open"].iloc[0]
    h4_high   = eight_bars["high"].max()
    h4_low    = eight_bars["low"].min()
    h4_close  = eight_bars["close"].iloc[-1]
    h4_vol    = eight_bars["volume"].sum()

    print("\n  ── Derived 4H candle (from 8×30m) ───────────────────")
    print(
        f"  {first_ts.strftime('%H:%M')}  "
        f"O={h4_open:.2f}  H={h4_high:.2f}  "
        f"L={h4_low:.2f}  C={h4_close:.2f}  "
        f"Vol={h4_vol:.4f}"
    )

    df_4h = _resample_ohlcv(df_30m.iloc[:8], "4h")
    if not df_4h.empty:
        r = df_4h.iloc[0]
        match4 = (
            abs(r["open"]  - h4_open)  < 0.01 and
            abs(r["high"]  - h4_high)  < 0.01 and
            abs(r["low"]   - h4_low)   < 0.01 and
            abs(r["close"] - h4_close) < 0.01 and
            abs(r["volume"] - h4_vol)  < 0.0001
        )
        print(f"\n  Pandas resample match: {'✓ CONFIRMED' if match4 else '✗ MISMATCH — investigate'}")

    print("\n" + "=" * 68 + "\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_all(
    data_dir: str | Path = DATA_DIR,
    debug: bool = False,
    run_sanity_check: bool = True,
) -> tuple[dict[str, dict[str, pd.DataFrame]], list[QualityReport]]:
    """
    Load all Binance Kline CSV files from data_dir.

    Returns
    -------
    data : dict[symbol][timeframe] -> pd.DataFrame
        All timeframes including resampled 1h and 4h.
    reports : list[QualityReport]
        One report per (symbol, timeframe) pair.

    Parameters
    ----------
    data_dir          : path to folder containing CSV files
    debug             : if True, print verbose column info
    run_sanity_check  : if True, print Brooks' resampling spot-check once
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    # ── Phase 1: load raw files ─────────────────────────────────────────────
    raw: dict[str, dict[str, pd.DataFrame]] = {}

    print("\n" + "=" * 68)
    print("  MODULE 1 — DATA LAYER")
    print("  Loading CSVs from:", data_dir)
    print("=" * 68)

    for csv_path in csv_files:
        parsed = _parse_symbol_tf(csv_path.name)
        if parsed is None:
            if debug:
                print(f"  [SKIP] {csv_path.name} — unrecognised pattern")
            continue

        symbol, tf = parsed
        tf_offset  = TF_FILE_MAP[tf]

        df = _load_raw_csv(csv_path)
        df = _add_delta_volume(df)
        df = _add_quality_flags(df, tf_offset)

        raw.setdefault(symbol, {})[tf] = df

        if debug:
            print(f"  [LOAD] {csv_path.name} → {symbol} {tf} ({len(df):,} rows)")

    # ── Phase 2: resample 30m → 1h, 4h ────────────────────────────────────
    sanity_done = False
    for symbol, tfs in raw.items():
        if "30m" not in tfs:
            continue

        df_30m = tfs["30m"]

        if run_sanity_check and not sanity_done:
            _resample_sanity_check(df_30m, symbol)
            sanity_done = True

        for tf_key, rule in TF_RESAMPLE_MAP.items():
            df_resampled = _resample_ohlcv(df_30m, rule)
            df_resampled = _add_quality_flags(df_resampled, rule)
            tfs[tf_key]  = df_resampled

    # ── Phase 3: quality reports ────────────────────────────────────────────
    reports: list[QualityReport] = []
    tf_order = ["5m", "15m", "30m", "1h", "4h"]

    print("\n  QUALITY REPORT")
    print("  " + "-" * 64)

    for symbol in sorted(raw.keys()):
        for tf in tf_order:
            if tf not in raw[symbol]:
                continue
            df = raw[symbol][tf]
            rpt = _build_quality_report(df, symbol, tf)
            reports.append(rpt)
            print(rpt)

    total_gaps     = sum(r.gap_count      for r in reports)
    total_zerovol  = sum(r.zero_vol_count for r in reports)
    total_dups     = sum(r.dup_count      for r in reports)
    total_candles  = sum(r.candle_count   for r in reports)

    print("\n  " + "-" * 64)
    print(f"  TOTALS | {total_candles:>10,} candles across all symbols & TFs")
    print(f"         | gaps={total_gaps}  zero_vol={total_zerovol}  dups={total_dups}")
    print("=" * 68 + "\n")

    return raw, reports


def load_symbol(
    symbol: str,
    data_dir: str | Path = DATA_DIR,
    run_sanity_check: bool = False,
) -> tuple[dict[str, pd.DataFrame], list[QualityReport]]:
    """
    Convenience wrapper — load a single symbol only.

    Returns
    -------
    tfs      : dict[timeframe] -> pd.DataFrame
    reports  : list[QualityReport] for this symbol
    """
    all_data, all_reports = load_all(
        data_dir         = data_dir,
        run_sanity_check = run_sanity_check,
    )
    symbol = symbol.upper()
    if symbol not in all_data:
        available = sorted(all_data.keys())
        raise KeyError(f"Symbol '{symbol}' not found. Available: {available}")

    symbol_reports = [r for r in all_reports if r.symbol == symbol]
    return all_data[symbol], symbol_reports


# ---------------------------------------------------------------------------
# Quick self-test  (python module_01_data_layer.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data, reports = load_all(run_sanity_check=True, debug=True)

    print("\n  SPOT-CHECK — BTCUSDT 4H (first 3 candles)")
    print("  " + "-" * 50)
    if "BTCUSDT" in data and "4h" in data["BTCUSDT"]:
        df4h = data["BTCUSDT"]["4h"]
        print(df4h[["open", "high", "low", "close", "volume", "delta", "buy_ratio"]].head(3).to_string())
    else:
        print("  BTCUSDT 4H not available — check CSV files.")

    print("\n  SPOT-CHECK — BTCUSDT 1H (first 3 candles)")
    print("  " + "-" * 50)
    if "BTCUSDT" in data and "1h" in data["BTCUSDT"]:
        df1h = data["BTCUSDT"]["1h"]
        print(df1h[["open", "high", "low", "close", "volume", "delta", "buy_ratio"]].head(3).to_string())

    print("\n  Available symbols:", sorted(data.keys()))
    print("  Available TFs per symbol:", {s: sorted(tfs.keys()) for s, tfs in data.items()})
    print()
