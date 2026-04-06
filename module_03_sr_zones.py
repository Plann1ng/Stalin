"""
=============================================================================
TRADING ALGORITHM — MODULE 3: S/R ZONE DETECTOR + SCORER  (patched v2)
=============================================================================
Four committee patches applied:
  1. MIN_ZONE_SCORE raised 3.0 → 6.0
  2. RECENCY_HALF_LIFE reduced 200 → 50 bars + hard expiry at 500 bars
  3. MAX_ZONES_PER_SYMBOL = 25  (Volman hard cap)
  4. Scale-aware round numbers — 3 tiers, no micro-increment flooding

Target output: 10–25 zones per symbol, all recent, all high-conviction.
=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parameters — committee-tunable, single source of truth
# ---------------------------------------------------------------------------

ZONE_TOLERANCE_PCT          = 0.0015   # 0.15% clustering band
MIN_ZONE_SCORE              = 6.0      # PATCH 1: raised from 3.0
ROUND_NUMBER_TOLERANCE_PCT  = 0.0025   # 0.25% proximity to round number
RECENCY_HALF_LIFE           = 50       # PATCH 2: reduced from 200 bars
ZONE_EXPIRY_BARS            = 500      # PATCH 2: hard expiry — drop if last touch > 500 bars ago
MAX_ZONES_PER_SYMBOL        = 25       # PATCH 3: Volman hard cap
FLAG_PROXIMITY_BARS         = 3        # Volman disqualification window

TF_SCORE = {
    "4h" : 4.0,
    "1h" : 3.0,
    "30m": 2.0,
    "15m": 1.0,
    "5m" : 0.5,
}

ZONE_DETECTION_TFS = ["4h", "1h", "30m"]


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class SRZone:
    symbol              : str
    zone_type           : str       # "RESISTANCE" | "SUPPORT"
    price_mid           : float
    price_low           : float
    price_high          : float
    score               : float
    touch_count         : int
    tf_origin           : str
    last_touch_ts       : pd.Timestamp
    first_touch_ts      : pd.Timestamp
    is_round_number     : bool
    is_quality_flagged  : bool

    def contains(self, price: float) -> bool:
        return self.price_low <= price <= self.price_high

    def distance_pct(self, price: float) -> float:
        return (self.price_mid - price) / price


# ---------------------------------------------------------------------------
# PATCH 4 — Scale-aware round number detection (3 tiers)
# ---------------------------------------------------------------------------

def _is_round_number(price: float, tolerance_pct: float = ROUND_NUMBER_TOLERANCE_PCT) -> bool:
    """
    Three price tiers, three meaningful increments each.
    Prevents every cent being flagged as round on a $0.25 coin.

      Large  (>= $100)  : 1000 / 100 / 10
      Mid    ($1–$100)  : 10   / 1   / 0.5
      Micro  (< $1)     : 0.1  / 0.05 / 0.01
    """
    tol = price * tolerance_pct

    if price >= 100:
        increments = [1_000, 100, 10]
    elif price >= 1:
        increments = [10, 1, 0.5]
    else:
        increments = [0.1, 0.05, 0.01]

    for inc in increments:
        nearest = round(price / inc) * inc
        if abs(price - nearest) <= tol:
            return True
    return False


# ---------------------------------------------------------------------------
# PATCH 2 — Recency score with hard expiry
# ---------------------------------------------------------------------------

def _bars_per_day(tf: str) -> float:
    return {"5m": 288, "15m": 96, "30m": 48, "1h": 24, "4h": 6}.get(tf, 24)


def _recency_score(
    last_touch_ts  : pd.Timestamp,
    latest_ts      : pd.Timestamp,
    bars_per_day   : float,
) -> Optional[float]:
    """
    Exponential decay with hard expiry.
    Returns None if the zone is older than ZONE_EXPIRY_BARS — caller drops it.
    """
    delta_seconds  = (latest_ts - last_touch_ts).total_seconds()
    bar_seconds    = 86_400 / bars_per_day
    bars_ago       = delta_seconds / bar_seconds

    if bars_ago > ZONE_EXPIRY_BARS:
        return None   # hard expiry

    return float(np.exp(-np.log(2) * bars_ago / RECENCY_HALF_LIFE))


# ---------------------------------------------------------------------------
# Quality flag check (Volman standing rule)
# ---------------------------------------------------------------------------

def _is_near_flag(
    ts     : pd.Timestamp,
    df     : pd.DataFrame,
    n_bars : int = FLAG_PROXIMITY_BARS,
) -> bool:
    if "is_gap_after" not in df.columns and "is_zero_vol" not in df.columns:
        return False
    try:
        idx = df.index.get_loc(ts)
    except KeyError:
        return False

    start  = max(0, idx - n_bars)
    end    = min(len(df), idx + n_bars + 1)
    window = df.iloc[start:end]

    flagged = pd.Series(False, index=window.index)
    if "is_gap_after" in window.columns:
        flagged |= window["is_gap_after"].fillna(False)
    if "is_zero_vol" in window.columns:
        flagged |= window["is_zero_vol"].fillna(False)
    return bool(flagged.any())


# ---------------------------------------------------------------------------
# Core zone detection — single TF
# ---------------------------------------------------------------------------

def _detect_zones_single_tf(
    df     : pd.DataFrame,
    symbol : str,
    tf     : str,
) -> list[SRZone]:
    if "is_swing_high" not in df.columns or "is_swing_low" not in df.columns:
        return []

    latest_ts    = df.index[-1]
    bpd          = _bars_per_day(tf)
    tf_weight    = TF_SCORE.get(tf, 1.0)
    zones: list[SRZone] = []

    for zone_type, swing_col, price_col in [
        ("RESISTANCE", "is_swing_high", "high"),
        ("SUPPORT",    "is_swing_low",  "low"),
    ]:
        swing_df = df[df[swing_col]].copy()
        if swing_df.empty:
            continue

        prices = swing_df[price_col].values
        ts_arr = swing_df.index
        used   = np.zeros(len(prices), dtype=bool)

        for i in range(len(prices)):
            if used[i]:
                continue

            anchor  = prices[i]
            tol     = anchor * ZONE_TOLERANCE_PCT
            mask    = np.abs(prices - anchor) <= tol
            cluster = np.where(mask)[0]
            used[cluster] = True

            c_prices = prices[cluster]
            c_ts     = [ts_arr[j] for j in cluster]

            price_mid   = float(c_prices.mean())
            last_touch  = max(c_ts)
            first_touch = min(c_ts)

            # PATCH 2: hard expiry check
            recency = _recency_score(last_touch, latest_ts, bpd)
            if recency is None:
                continue   # zone too old — drop

            touch_count   = len(cluster)
            touch_score   = float(np.log1p(touch_count))
            is_round      = _is_round_number(price_mid)   # PATCH 4
            round_bonus   = 1.0 if is_round else 0.0
            score         = touch_score + recency + tf_weight + round_bonus

            # PATCH 1: minimum score gate
            if score < MIN_ZONE_SCORE:
                continue

            flagged = _is_near_flag(last_touch, df, FLAG_PROXIMITY_BARS)

            zones.append(SRZone(
                symbol              = symbol,
                zone_type           = zone_type,
                price_mid           = price_mid,
                price_low           = price_mid * (1 - ZONE_TOLERANCE_PCT),
                price_high          = price_mid * (1 + ZONE_TOLERANCE_PCT),
                score               = round(score, 3),
                touch_count         = touch_count,
                tf_origin           = tf,
                last_touch_ts       = last_touch,
                first_touch_ts      = first_touch,
                is_round_number     = is_round,
                is_quality_flagged  = flagged,
            ))

    return zones


# ---------------------------------------------------------------------------
# Deduplication — merge overlapping zones, keep highest TF
# ---------------------------------------------------------------------------

def _deduplicate_zones(zones: list[SRZone]) -> list[SRZone]:
    tf_rank = {"4h": 4, "1h": 3, "30m": 2, "15m": 1, "5m": 0}
    merged  = []
    used    = [False] * len(zones)

    for i, z1 in enumerate(zones):
        if used[i]:
            continue
        cluster = [i]
        for j, z2 in enumerate(zones):
            if i == j or used[j] or z1.zone_type != z2.zone_type:
                continue
            if abs(z1.price_mid - z2.price_mid) <= z1.price_mid * ZONE_TOLERANCE_PCT * 2:
                cluster.append(j)

        for idx in cluster:
            used[idx] = True

        if len(cluster) == 1:
            merged.append(z1)
            continue

        c_zones  = [zones[k] for k in cluster]
        best     = max(c_zones, key=lambda z: tf_rank.get(z.tf_origin, 0))
        bonus    = (len(cluster) - 1) * 0.5

        merged.append(SRZone(
            symbol              = best.symbol,
            zone_type           = best.zone_type,
            price_mid           = best.price_mid,
            price_low           = best.price_low,
            price_high          = best.price_high,
            score               = round(best.score + bonus, 3),
            touch_count         = sum(z.touch_count for z in c_zones),
            tf_origin           = best.tf_origin,
            last_touch_ts       = max(z.last_touch_ts  for z in c_zones),
            first_touch_ts      = min(z.first_touch_ts for z in c_zones),
            is_round_number     = best.is_round_number,
            is_quality_flagged  = any(z.is_quality_flagged for z in c_zones),
        ))

    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_zones_symbol(
    data       : dict[str, dict[str, pd.DataFrame]],
    classified : dict[str, dict[str, pd.DataFrame]],
    symbol     : str,
    tfs        : list[str] = ZONE_DETECTION_TFS,
) -> list[SRZone]:
    """
    Detect, score, deduplicate, and cap zones for one symbol.
    Returns at most MAX_ZONES_PER_SYMBOL zones sorted by score desc.
    """
    symbol = symbol.upper()
    all_zones: list[SRZone] = []

    for tf in tfs:
        if tf in classified.get(symbol, {}):
            df = classified[symbol][tf]
        elif tf in data.get(symbol, {}):
            df = data[symbol][tf]
        else:
            continue
        all_zones.extend(_detect_zones_single_tf(df, symbol, tf))

    all_zones = _deduplicate_zones(all_zones)
    all_zones.sort(key=lambda z: z.score, reverse=True)

    # PATCH 3: hard cap
    return all_zones[:MAX_ZONES_PER_SYMBOL]


def detect_zones(
    data       : dict[str, dict[str, pd.DataFrame]],
    classified : dict[str, dict[str, pd.DataFrame]],
    tfs        : list[str] = ZONE_DETECTION_TFS,
) -> dict[str, list[SRZone]]:
    """Run zone detection for all symbols. Returns dict[symbol] → list[SRZone]."""
    return {
        symbol: detect_zones_symbol(data, classified, symbol, tfs)
        for symbol in sorted(data.keys())
    }


def get_active_zones(
    symbol         : str,
    current_price  : float,
    all_zones      : dict[str, list[SRZone]],
    lookback_pct   : float = 0.10,
    min_score      : float = MIN_ZONE_SCORE,
    include_flagged: bool  = False,
    current_ts     : Optional[pd.Timestamp] = None,
) -> dict[str, list[SRZone]]:
    """
    Return nearest resistance and support zones around current price.

    current_ts: if provided, only zones whose first_touch_ts <= current_ts
    are returned. This prevents future-formed zones from appearing in
    historical signal evaluation (lookahead fix).
    """
    symbol = symbol.upper()
    if symbol not in all_zones:
        return {"resistance": [], "support": []}

    lo = current_price * (1 - lookback_pct)
    hi = current_price * (1 + lookback_pct)

    resistance, support = [], []

    for z in all_zones[symbol]:
        if z.score < min_score:
            continue
        if not include_flagged and z.is_quality_flagged:
            continue
        if not (lo <= z.price_mid <= hi):
            continue
        # Lookahead fix: skip zones not yet formed at current_ts
        if current_ts is not None and z.first_touch_ts > current_ts:
            continue
        if z.zone_type == "RESISTANCE" and z.price_mid >= current_price:
            resistance.append(z)
        elif z.zone_type == "SUPPORT" and z.price_mid <= current_price:
            support.append(z)

    resistance.sort(key=lambda z: z.price_mid)
    support.sort(key=lambda z: -z.price_mid)
    return {"resistance": resistance, "support": support}


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

    # Add 30m swing columns for zone detection
    for sym in data:
        if "30m" in data[sym]:
            classified[sym]["30m"] = _cs(data, sym, "30m", lookback=4)

    print("Detecting S/R zones (patched v2)...")
    all_zones = detect_zones(data, classified)

    # ── Zone count summary ────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  MODULE 3 (patched) — Zone count per symbol")
    print(f"  Thresholds: min_score={MIN_ZONE_SCORE}  "
          f"expiry={ZONE_EXPIRY_BARS} bars  cap={MAX_ZONES_PER_SYMBOL}")
    print("=" * 68)

    total_zones = 0
    print(f"\n  {'Symbol':<14} {'Total':>5}  {'Resist':>6}  "
          f"{'Support':>7}  {'Flagged':>7}  {'Round#':>6}  {'Top score':>10}")
    print("  " + "-" * 62)

    for sym in sorted(all_zones.keys()):
        zs      = all_zones[sym]
        n_res   = sum(1 for z in zs if z.zone_type == "RESISTANCE")
        n_sup   = sum(1 for z in zs if z.zone_type == "SUPPORT")
        n_flag  = sum(1 for z in zs if z.is_quality_flagged)
        n_round = sum(1 for z in zs if z.is_round_number)
        top_sc  = zs[0].score if zs else 0.0
        total_zones += len(zs)
        print(f"  {sym:<14} {len(zs):>5}  {n_res:>6}  {n_sup:>7}  "
              f"{n_flag:>7}  {n_round:>6}  {top_sc:>10.2f}")

    print(f"\n  Total zones: {total_zones}  (target: 300–500)")

    # ── BTCUSDT top 10 ────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  BTCUSDT — Top zones by score (patched)")
    print("=" * 68)
    btc_zones = all_zones.get("BTCUSDT", [])
    print(f"\n  {'Type':<12} {'Mid price':>12} {'Score':>6} "
          f"{'Touches':>7} {'TF':>5} {'Round':>6} {'Last touch':<14} {'Flagged'}")
    print("  " + "-" * 74)
    for z in btc_zones[:15]:
        print(
            f"  {z.zone_type:<12} {z.price_mid:>12.2f} {z.score:>6.2f} "
            f"{z.touch_count:>7} {z.tf_origin:>5} "
            f"{'YES' if z.is_round_number else 'no':>6} "
            f"{str(z.last_touch_ts)[:16]:<14}  "
            f"{'FLAG' if z.is_quality_flagged else 'clean'}"
        )

    # ── BTCUSDT active zones near current price ───────────────────────────
    btc_last = data["BTCUSDT"]["4h"].iloc[-1]["close"]
    active   = get_active_zones("BTCUSDT", btc_last, all_zones)

    print(f"\n" + "=" * 68)
    print(f"  BTCUSDT active zones  (price ≈ {btc_last:,.2f})")
    print("=" * 68)
    print(f"\n  Resistance above:")
    for z in active["resistance"][:5]:
        dist = z.distance_pct(btc_last) * 100
        print(f"    {z.price_mid:>12,.2f}  score={z.score:.2f}  "
              f"+{dist:.2f}%  tf={z.tf_origin}  touches={z.touch_count}"
              f"{'  [ROUND]' if z.is_round_number else ''}")

    print(f"\n  Support below:")
    for z in active["support"][:5]:
        dist = z.distance_pct(btc_last) * 100
        print(f"    {z.price_mid:>12,.2f}  score={z.score:.2f}  "
              f"{dist:.2f}%  tf={z.tf_origin}  touches={z.touch_count}"
              f"{'  [ROUND]' if z.is_round_number else ''}")

    # ── Score distribution check ──────────────────────────────────────────
    print(f"\n" + "=" * 68)
    print("  Score distribution — all symbols (patched)")
    print("=" * 68)
    all_scores = [z.score for zlist in all_zones.values() for z in zlist]
    if all_scores:
        arr = np.array(all_scores)
        print(f"\n  Min={arr.min():.2f}  Max={arr.max():.2f}  "
              f"Mean={arr.mean():.2f}  Median={np.median(arr):.2f}")
        print(f"  Zones scoring >= 8.0: {(arr >= 8.0).sum()}")
        print(f"  Zones scoring >= 7.0: {(arr >= 7.0).sum()}")
        print(f"  Zones scoring >= 6.0: {(arr >= 6.0).sum()}")
    print()
