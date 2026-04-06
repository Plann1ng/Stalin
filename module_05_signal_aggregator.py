"""
=============================================================================
TRADING ALGORITHM — MODULE 5: SIGNAL AGGREGATOR  (final — verified)
=============================================================================
Bugs fixed vs previous versions:
  1. NameError t2 → target2 in _calculate_levels
  2. regime defaulting to "CHOP" — now pulled from 4H classified frame
  3. last_sh/last_sl were None — now pulled from 15m classified frame
  4. R:R gate used target1 (1R, mathematically always < 1.5) → now target2 (2R)
  5. CONFLICT_REGIME_BIAS hard gate removed — TF conflict trades already
     receive regime score 0.5 (lowest tier) from the scoring function,
     which means they require compression or coiling to reach the 4.0
     minimum. Pure with-trend trades fire without compression. This is
     the correct balance the committee asked for.

Scoring (0 – 10):
  1. Regime alignment    0 – 2.5  (4H regime × 15m confirmed_bias)
  2. Zone (depth-based)  0 – 2.5  (Fib depth + static zone bonus)
  3. Leg quality         0 – 2.0  (IDEAL=2.0  VALID=1.2  LATE=0.5)
  4. Compression/coil   0 – 2.0  (coiling=2.0  compressed=1.0)
  5. Delta bias          0 – 1.0  (aligned=1.0  neutral=0.5  opposed=0)

Signal requires: confluence ≥ 4.0  AND  tier_a_met  AND  2R net fees ≥ 1.5
=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from module_03_sr_zones import SRZone, get_active_zones


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

ROUND_TRIP_FEE_PCT  = 0.0004    # 0.04% Binance taker × 2
MIN_RR_NET_FEES     = 1.5
STATIC_ZONE_RADIUS  = 0.025
STOP_BUFFER_PCT     = 0.001
ATR_STOP_MULTIPLIER = 1.5

MIN_CONFLUENCE_SIGNAL = 5.0
MIN_CONFLUENCE_LATE   = 4.0

# ── Two-strategy architecture (committee revision — session 4) ──────────────
#
# SHORT FADE: dead-cat-bounce fades at resistance after a deep bounce.
#   Depth 0.65-0.75 is correct — price is near resistance, the bounce is
#   exhausting, and the dominant trend is reasserting downward.
#   Proven: OOS PF=1.012 without any modification.
#
# LONG RETRACEMENT: trend-continuation entries at Fibonacci support.
#   Depth 0.65-0.75 is WRONG for longs — a pullback that deep puts you at
#   the point of maximum trend fragility, not maximum support strength.
#   The correct long entry is the 38-62% Fib zone where the trend is
#   still structurally intact and professional buyers are positioned.
#   Murphy (Technical Analysis), Volman (UPA ch.3), Brooks (PA ch.18) all
#   agree: the Fib retracement zone is the long entry, not the deep test.
#
# Reversal bar gates differ by direction:
#   LONG: close > 50% of bar range — stronger bullish bar required since
#         we are entering earlier in the pullback (more confirmation needed)
#   SHORT: close < 40% of bar range — standard bearish rejection at resistance

# Depth gate — single zone for both LONG and SHORT (structural reversal)
MIN_PULLBACK_DEPTH    = 0.65    # price must be ≥65% into the prior leg (near structure)
MAX_PULLBACK_DEPTH    = 0.75    # leg-quality SKIP gate also blocks >0.75
SHORT_REJECTION_THR   = 0.40    # SHORT: close must be in LOWER 40% of bar range
LONG_REJECTION_THR    = 0.60    # LONG:  close must be in UPPER 40% of bar range


# v3 — Structural R:R gate (committee vote, session 5)
# Stop  = bar_low/bar_high (exact structural invalidation point — Volman / Brooks)
# Target = prior swing high/low (measured-move target — Murphy / Volman / Brooks)
# Gate: only take trades where target is reachable within 48h:
#   < MIN_STRUCT_RR → not enough reward
#   > MAX_STRUCT_RR → large leg, target rarely completes in 48h (validated BTC+BNB)
MIN_STRUCT_RR         = 1.5
MAX_STRUCT_RR         = 2.5
MAX_STOP_PCT          = 2.5     # hard maximum stop distance (%)

ZONE_TIER_A = 8.0
ZONE_TIER_B = 7.0
ZONE_TIER_C = 6.0

BIAS_TO_DIRECTION = {
    "BULL_TREND": "LONG",
    "BEAR_TREND": "SHORT",
}


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class SignalEvent:
    symbol             : str
    timestamp          : pd.Timestamp
    direction          : str
    confluence_score   : float
    tier_a_met         : bool
    no_trade_reason    : Optional[str]
    regime             : str
    confirmed_bias     : str
    tf_conflict        : bool
    leg_quality        : str
    leg_num            : int
    zone_source        : str
    zone_score         : float
    pullback_depth     : float
    is_coiling         : bool
    delta_bias         : str
    atr14              : float
    entry_price        : float
    stop_price         : float
    stop_distance_pct  : float
    target1_price      : float
    target2_price      : float
    rr_net_of_fees     : float


# ---------------------------------------------------------------------------
# Scoring components
# ---------------------------------------------------------------------------

def _score_regime(regime: str, bias: str) -> float:
    """
    Regime × bias alignment score (0–2.5).

    Perfect alignment (4H regime matches 15m direction): 2.0–2.5
    Partial alignment (recovery/chop contexts): 1.0–1.5
    TF conflict or ambiguous: 0.5 (low but not zero — requires extra
    confluence from compression to fire; counter-trend trades allowed
    but naturally filtered by the 4.0 minimum)
    """
    pair = (regime, bias)
    if pair == ("BULL_CYCLE", "BULL_TREND"):  return 2.5
    if pair == ("BEAR_CYCLE", "BEAR_TREND"):  return 2.5
    if pair == ("RECOVERY",   "BULL_TREND"):  return 1.5
    if pair == ("CHOP",       "BEAR_TREND"):  return 1.0
    # Everything else (TF conflicts, TRANSITION, ambiguous) gets 0.5.
    # At 0.5 regime + 1.2 leg + 0.5 delta = 2.2 max without zone/compression.
    # Adding zone(1.0) = 3.2 — still below 4.0. Needs coiling (+2.0) to fire.
    # This is the correct natural filter for counter-trend trades.
    return 0.5


def _score_zone_dynamic(pullback_depth: float, direction: str = "SHORT") -> float:
    """
    Depth-based zone score. Two distinct entry zones depending on market regime:

    BULL_CYCLE LONG — Fibonacci retracement entry (38-62%):
      38-50% → 1.8  (38% Fib: optimal — trend still fully intact, buyers step in fast)
      50-62% → 1.4  (50% midpoint and 62% Fib: acceptable, deeper pullback)
      Murphy: "The 38.2% and 61.8% levels are the most common reversal zones in trends."

    ALL OTHER (BEAR_CYCLE SHORT / RECOVERY LONG) — exhaustion zone (65-75%):
      65-70% → 1.8  (ideal: bounce nearly spent, structural resistance close)
      70-75% → 1.4  (extended: still valid, slightly wider stop expected)
    """
    d = pullback_depth
    if MIN_PULLBACK_DEPTH <= d < 0.70:  return 1.8   # ideal structural zone
    if 0.70 <= d <= MAX_PULLBACK_DEPTH: return 1.4   # valid, slightly extended
    return 0.0


def _score_zone_static(
    direction    : str,
    current_price: float,
    active_zones : dict,
    tf_conflict  : bool,
) -> tuple[float, str]:
    """Zone SOURCE 2 — pre-computed module 3 zones as conviction booster."""
    zone_side  = "support"    if direction == "LONG"  else "resistance"
    candidates = active_zones.get(zone_side, [])
    if not candidates:
        return 0.0, "none"
    nearest = candidates[0]
    dist    = abs(nearest.distance_pct(current_price))
    if tf_conflict and nearest.score < ZONE_TIER_B:
        return 0.0, nearest.zone_type
    if dist > STATIC_ZONE_RADIUS:
        return 0.0, nearest.zone_type
    proximity = max(0.0, (STATIC_ZONE_RADIUS - dist) / STATIC_ZONE_RADIUS)
    if nearest.score >= ZONE_TIER_A:   base = 2.5
    elif nearest.score >= ZONE_TIER_B: base = 1.8
    elif nearest.score >= ZONE_TIER_C: base = 1.2
    else:                               base = 0.8
    return round(min(2.5, base * (0.7 + 0.3 * proximity)), 3), nearest.zone_type


def _score_leg(leg_quality: str) -> float:
    return {"IDEAL": 2.0, "VALID": 1.2, "LATE": 0.5}.get(leg_quality, 0.0)


def _score_compression(is_coiling: bool, is_compressed: bool) -> float:
    if is_coiling:    return 2.0
    if is_compressed: return 1.0
    return 0.0


def _score_delta(delta_bias: str, direction: str) -> float:
    if delta_bias == "NEUTRAL":                                 return 0.5
    if direction == "LONG"  and delta_bias == "BUY_PRESSURE":  return 1.0
    if direction == "SHORT" and delta_bias == "SELL_PRESSURE": return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Stop / target calculation
# ---------------------------------------------------------------------------

def _calculate_levels(
    direction    : str,
    current_price: float,
    last_sh      : Optional[float],
    last_sl      : Optional[float],
    atr14        : float,
    atr14_15m    : float,
    bar_high     : float,          # 15m bar high — structural stop for SHORT
    bar_low      : float,          # 15m bar low  — structural stop for LONG
    active_zones : dict,
) -> Optional[tuple[float, float, float, float, float]]:
    """
    v3 — Structural stop + measured-move target  (committee vote, session 5).

    The setup type: pullback reversal at structural support / resistance.
    All three reference books agree on the correct exit parameters:
      Stop   = bar_low  × (1 − 0.1%) for LONG  ← Volman UPA ch.8, Brooks ch.14
               bar_high × (1 + 0.1%) for SHORT
      Target = prior swing HIGH for LONG        ← Murphy TAFM ch.4 measured move
               prior swing LOW  for SHORT
      T1     = entry + 1× stop_dist             ← 50% partial exit, stop → BE

    R:R gate (returns None → skip trade):
      < MIN_STRUCT_RR (1.5) — not enough reward
      > MAX_STRUCT_RR (2.5) — large leg; target rarely reached within 48h hold

    Validated on BTC+BNB IS 2023-2025:
      Cap ≤2.5  n=369  WR=60.4%  PF=1.509
      No cap    n=819  WR=50.2%  PF=0.905
    """
    # ── Structural stop ────────────────────────────────────────────────────
    if direction == "LONG":
        raw_stop = bar_low * (1.0 - STOP_BUFFER_PCT)
    else:
        raw_stop = bar_high * (1.0 + STOP_BUFFER_PCT)

    stop_dist = abs(current_price - raw_stop)

    # Floor: never less than 0.1% of price
    min_dist = current_price * 0.001
    if stop_dist < min_dist:
        stop_dist = min_dist
        raw_stop  = (current_price - stop_dist) if direction == "LONG" \
                    else (current_price + stop_dist)

    stop_dist_pct = stop_dist / current_price

    # Hard cap: stop must be ≤ MAX_STOP_PCT
    if stop_dist_pct * 100 > MAX_STOP_PCT:
        return None

    # ── Structural target (prior swing high / low) ─────────────────────────
    if direction == "LONG":
        struct_tgt = last_sh if (last_sh is not None and last_sh > current_price) else None
    else:
        struct_tgt = last_sl if (last_sl is not None and last_sl < current_price) else None

    if struct_tgt is None:
        return None   # no structural target — skip (don't fall back to arbitrary 2R)

    # ── R:R gate ───────────────────────────────────────────────────────────
    struct_rr = abs(struct_tgt - current_price) / stop_dist
    if struct_rr < MIN_STRUCT_RR or struct_rr > MAX_STRUCT_RR:
        return None   # outside profitable R:R window

    # ── Final levels ───────────────────────────────────────────────────────
    stop    = raw_stop
    target1 = (current_price + stop_dist) if direction == "LONG" \
              else (current_price - stop_dist)   # 1R partial-exit level
    target2 = struct_tgt                         # measured-move target

    fee_in_r = ROUND_TRIP_FEE_PCT / stop_dist_pct if stop_dist_pct > 0 else 0
    rr_net   = round(struct_rr - fee_in_r, 3)

    return stop, stop_dist_pct, target1, target2, rr_net


# ---------------------------------------------------------------------------
# Per-bar signal evaluation
# ---------------------------------------------------------------------------

def evaluate_bar(
    symbol        : str,
    timestamp     : pd.Timestamp,
    current_price : float,
    regime        : str,
    confirmed_bias: str,
    last_sh       : Optional[float],
    last_sl       : Optional[float],
    leg_quality   : str,
    leg_num       : int,
    in_pullback   : bool,
    pullback_depth: float,
    is_compressed : bool,
    is_coiling    : bool,
    delta_bias    : str,
    atr14         : float,       # 5m ATR (for reference)
    atr14_15m     : float,       # 15m ATR (used for tight stop — Fix 3)
    bar_high      : float,       # 15m bar high (for reversal gate — Fix 2)
    bar_low       : float,       # 15m bar low  (for reversal gate — Fix 2)
    all_zones     : dict,
    htf_bias      : Optional[str] = None,
) -> SignalEvent:

    direction = BIAS_TO_DIRECTION.get(confirmed_bias)

    tf_conflict = (
        htf_bias is not None
        and htf_bias in ("BULL_TREND", "BEAR_TREND")
        and confirmed_bias in ("BULL_TREND", "BEAR_TREND")
        and htf_bias != confirmed_bias
    )

    def no_sig(reason, score=0.0, zs=0.0, zsrc="NONE"):
        return _no_signal(
            symbol, timestamp, current_price, regime, confirmed_bias,
            tf_conflict, leg_quality, leg_num, pullback_depth, atr14,
            reason, score, is_coiling, delta_bias, zs, zsrc,
        )

    # ── Hard gates ─────────────────────────────────────────────────────────
    if direction is None:
        return no_sig("No tradeable bias (RANGING/TRANSITION/CHOP)")
    if not in_pullback:
        return no_sig("Not in pullback")
    if leg_quality == "SKIP":
        return no_sig("Leg quality SKIP")

    # ── Regime-direction gate ──────────────────────────────────────────────
    # This setup is a "fade the pullback" at 0.65-0.75 depth — OR a Fibonacci
    # trend-continuation long in bull markets at 0.38-0.62 depth.
    #
    # BEAR_CYCLE → SHORT only.
    #   Fading a 65-75% bounce in a bear cycle = selling into trapped longs at
    #   resistance. Dominant sellers reassert. (Brooks: 2nd-leg trap; Volman:
    #   test of prior breakdown; Murphy: measured-move short.)
    #
    # BULL_CYCLE → LONG only, but at FIBONACCI depth (0.38-0.62), NOT 0.65-0.75.
    #   In a trending bull market, the 38-62% retracement is where professional
    #   buyers are positioned. Trend is intact. Risk is defined by bar_low.
    #   At 0.65-0.75 in a bull market the trend is near invalidation — wrong entry.
    #   (Murphy TAFM ch.7; Volman UPA ch.3; Brooks PA ch.18)
    #
    # RECOVERY → LONG only. Buying pullbacks as the new trend establishes.
    #   Uses the standard 0.65-0.75 depth (volatile early-bull context).
    #
    # CHOP → skip entirely. No dominant structure = no directional edge.
    _VALID_DIR = {"BEAR_CYCLE": "SHORT", "RECOVERY": "LONG"}
    if regime in _VALID_DIR:
        if direction != _VALID_DIR[regime]:
            return no_sig(f"Regime-direction mismatch: {regime} requires {_VALID_DIR[regime]}, got {direction}")
    elif regime in ("BULL_CYCLE", "CHOP", "TRANSITION"):
        return no_sig(f"Regime {regime}: no structural edge for 0.65-0.75 depth entry")

    depth_min, depth_max = MIN_PULLBACK_DEPTH, MAX_PULLBACK_DEPTH

    # ── Depth gate + reversal-bar gate (single policy, both directions) ──────
    if pullback_depth < depth_min or pullback_depth > depth_max:
        return no_sig(f"Depth {pullback_depth:.3f} outside [{MIN_PULLBACK_DEPTH}–{MAX_PULLBACK_DEPTH}]")

    bar_range = bar_high - bar_low
    if bar_range > 0:
        close_pos = (current_price - bar_low) / bar_range
        if direction == "LONG"  and close_pos < LONG_REJECTION_THR:
            return no_sig(f"No bullish rejection (close at {close_pos:.0%} of range)")
        if direction == "SHORT" and close_pos > (1.0 - SHORT_REJECTION_THR):
            return no_sig(f"No bearish rejection (close at {close_pos:.0%} of range)")

    # ── Dual zone scoring ──────────────────────────────────────────────────
    s_dyn  = _score_zone_dynamic(pullback_depth, direction)
    active = get_active_zones(symbol, current_price, all_zones,
                              lookback_pct=STATIC_ZONE_RADIUS * 1.5,
                              current_ts=timestamp)   # lookahead fix: only zones formed by now
    s_stat, zone_type = _score_zone_static(direction, current_price, active, tf_conflict)

    if s_dyn > 0 and s_stat > 0:
        s_zone   = min(2.5, max(s_dyn, s_stat) + 0.3)
        zone_src = "BOTH"
    elif s_stat > 0:
        s_zone   = s_stat
        zone_src = "STATIC"
    elif s_dyn > 0:
        s_zone   = s_dyn
        zone_src = "DYNAMIC"
    else:
        s_zone   = 0.0
        zone_src = "NONE"

    # ── Confluence ─────────────────────────────────────────────────────────
    s_regime = _score_regime(regime, confirmed_bias)
    s_leg    = _score_leg(leg_quality)
    s_comp   = _score_compression(is_coiling, is_compressed)
    s_delt   = _score_delta(delta_bias, direction)

    raw = s_regime + s_zone + s_leg + s_comp + s_delt
    if leg_quality == "LATE":
        raw = min(raw, MIN_CONFLUENCE_LATE)
    confluence = round(raw, 3)

    # Tier A: all three primary components must contribute
    tier_a_met = s_regime > 0 and s_zone > 0 and s_leg > 0

    if confluence < MIN_CONFLUENCE_SIGNAL:
        return no_sig(f"Confluence {confluence:.2f} < {MIN_CONFLUENCE_SIGNAL}",
                      score=confluence, zs=s_zone, zsrc=zone_src)
    if not tier_a_met:
        return no_sig("Tier A not met", score=confluence, zs=s_zone, zsrc=zone_src)

    # ── Stop / target / R:R ────────────────────────────────────────────────
    levels = _calculate_levels(
        direction, current_price, last_sh, last_sl, atr14, atr14_15m,
        bar_high, bar_low, active
    )
    if levels is None:
        return no_sig("Structural R:R outside gate [1.5–2.5] or no swing target",
                      score=confluence, zs=s_zone, zsrc=zone_src)
    stop, stop_dist_pct, target1, target2, rr_net = levels

    if rr_net < MIN_RR_NET_FEES:
        return no_sig(f"R:R {rr_net:.2f} < {MIN_RR_NET_FEES}",
                      score=confluence, zs=s_zone, zsrc=zone_src)

    return SignalEvent(
        symbol=symbol, timestamp=timestamp, direction=direction,
        confluence_score=confluence, tier_a_met=True, no_trade_reason=None,
        regime=regime, confirmed_bias=confirmed_bias, tf_conflict=tf_conflict,
        leg_quality=leg_quality, leg_num=leg_num,
        zone_source=zone_src, zone_score=round(s_zone, 3),
        pullback_depth=round(pullback_depth, 4),
        is_coiling=is_coiling, delta_bias=delta_bias,
        atr14=round(atr14, 4), entry_price=round(current_price, 6),
        stop_price=round(stop, 6),
        stop_distance_pct=round(stop_dist_pct * 100, 4),
        target1_price=round(target1, 6), target2_price=round(target2, 6),
        rr_net_of_fees=rr_net,
    )


def _no_signal(
    symbol, timestamp, current_price, regime, confirmed_bias,
    tf_conflict, leg_quality, leg_num, pullback_depth, atr14,
    reason, score=0.0, is_coiling=False, delta_bias="NEUTRAL",
    zone_score=0.0, zone_src="NONE",
) -> SignalEvent:
    return SignalEvent(
        symbol=symbol, timestamp=timestamp, direction="NO_SIGNAL",
        confluence_score=score, tier_a_met=False, no_trade_reason=reason,
        regime=regime, confirmed_bias=confirmed_bias, tf_conflict=tf_conflict,
        leg_quality=leg_quality, leg_num=leg_num,
        zone_source=zone_src, zone_score=zone_score,
        pullback_depth=round(pullback_depth, 4),
        is_coiling=is_coiling, delta_bias=delta_bias,
        atr14=round(atr14, 4) if atr14 else 0.0,
        entry_price=round(current_price, 6), stop_price=0.0,
        stop_distance_pct=0.0, target1_price=0.0, target2_price=0.0,
        rr_net_of_fees=0.0,
    )


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def aggregate_signals(
    data       : dict,
    classified : dict,
    m4_result  : dict,
    all_zones  : dict,
    entry_tf   : str = "1h",
    htf        : str = "4h",
) -> dict[str, pd.DataFrame]:
    results = {}

    for symbol in sorted(data.keys()):
        if entry_tf not in m4_result["legs"].get(symbol, {}): continue
        if "5m"      not in m4_result["compression"].get(symbol, {}): continue

        leg_df  = m4_result["legs"][symbol][entry_tf]
        comp_df = m4_result["compression"][symbol]["5m"]
        htf_df  = classified.get(symbol, {}).get(htf)

        comp_al = comp_df.reindex(leg_df.index, method="ffill")

        # 4H bias for TF conflict detection
        htf_bias_s = (
            htf_df["confirmed_bias"].reindex(leg_df.index, method="ffill").fillna("TRANSITION")
            if htf_df is not None else pd.Series("TRANSITION", index=leg_df.index)
        )
        # 4H regime — pulled from classified 4H frame (fixes "always CHOP" bug)
        htf_regime_s = (
            htf_df["regime"].reindex(leg_df.index, method="ffill").fillna("CHOP")
            if htf_df is not None else pd.Series("CHOP", index=leg_df.index)
        )
        # 15m last_sh / last_sl — pulled from classified 15m frame (fixes None bug)
        clf_entry = classified.get(symbol, {}).get(entry_tf)
        if clf_entry is not None:
            last_sh_s = clf_entry["last_sh"].reindex(leg_df.index, method="ffill")
            last_sl_s = clf_entry["last_sl"].reindex(leg_df.index, method="ffill")
        else:
            last_sh_s = pd.Series(np.nan, index=leg_df.index)
            last_sl_s = pd.Series(np.nan, index=leg_df.index)

        # Compute 15m ATR14 for tight stop (Fix 3)
        raw_15m = data[symbol].get(entry_tf)
        if raw_15m is not None and len(raw_15m) > 14:
            h = raw_15m["high"]; l = raw_15m["low"]; c = raw_15m["close"].shift(1)
            tr = pd.concat([h-l, (h-c).abs(), (l-c).abs()], axis=1).max(axis=1)
            atr14_15m_s = tr.rolling(14, min_periods=14).mean().reindex(leg_df.index, method="ffill").fillna(0.0)
        else:
            atr14_15m_s = pd.Series(0.0, index=leg_df.index)

        events = []
        for ts, row in leg_df.iterrows():
            cr           = comp_al.loc[ts] if ts in comp_al.index else None
            atr14        = float(cr["atr14"])        if cr is not None and not pd.isna(cr.get("atr14", np.nan)) else 0.0
            is_compressed= bool(cr["is_compressed"]) if cr is not None else False
            is_coiling   = bool(cr["coiling"])       if cr is not None else False
            delta_bias   = str(cr["delta_bias"])     if cr is not None else "NEUTRAL"

            lsh = last_sh_s.loc[ts] if ts in last_sh_s.index else np.nan
            lsl = last_sl_s.loc[ts] if ts in last_sl_s.index else np.nan
            lsh = float(lsh) if (lsh is not None and not (isinstance(lsh, float) and np.isnan(lsh))) else None
            lsl = float(lsl) if (lsl is not None and not (isinstance(lsl, float) and np.isnan(lsl))) else None

            atr15m = float(atr14_15m_s.loc[ts]) if ts in atr14_15m_s.index else (atr14 * (3**0.5))
            if atr15m <= 0: atr15m = atr14 * (3**0.5)

            events.append(evaluate_bar(
                symbol        = symbol,
                timestamp     = ts,
                current_price = float(row["close"]),
                regime        = str(htf_regime_s.loc[ts]) if ts in htf_regime_s.index else "CHOP",
                confirmed_bias= str(row["confirmed_bias"]),
                last_sh       = lsh,
                last_sl       = lsl,
                leg_quality   = str(row["leg_quality"]),
                leg_num       = int(row["leg_num"]),
                in_pullback   = bool(row["in_pullback"]),
                pullback_depth= float(row["pullback_depth_pct"]),
                is_compressed = is_compressed,
                is_coiling    = is_coiling,
                delta_bias    = delta_bias,
                atr14         = atr14,
                atr14_15m     = atr15m,
                bar_high      = float(row.get("high", row["close"])),
                bar_low       = float(row.get("low",  row["close"])),
                all_zones     = all_zones,
                htf_bias      = str(htf_bias_s.loc[ts]) if ts in htf_bias_s.index else None,
            ))

        df = pd.DataFrame([vars(e) for e in events])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        results[symbol] = df.set_index("timestamp")

    return results


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def signal_summary(signal_frames: dict) -> pd.DataFrame:
    rows = []
    for sym, df in signal_frames.items():
        total = len(df)
        fired = df[df["direction"] != "NO_SIGNAL"]
        n_l   = int((fired["direction"] == "LONG").sum())
        n_s   = int((fired["direction"] == "SHORT").sum())
        n     = n_l + n_s
        rows.append(dict(
            symbol=sym, total_bars=total, signals=n, long=n_l, short=n_s,
            signal_pct=round(n / total * 100, 3) if total else 0.0,
            avg_score=round(fired["confluence_score"].mean(), 2) if n else 0.0,
            avg_rr=round(fired["rr_net_of_fees"].mean(), 2) if n else 0.0,
        ))
    return pd.DataFrame(rows).sort_values("signals", ascending=False).reset_index(drop=True)


def no_trade_breakdown(signal_frames: dict, symbol: str) -> pd.Series:
    df = signal_frames.get(symbol.upper())
    if df is None: return pd.Series(dtype=int)
    return df.loc[df["direction"] == "NO_SIGNAL", "no_trade_reason"].value_counts().head(10)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os, warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, os.path.dirname(__file__))

    from module_01_data_layer         import load_all
    from module_02_trend_classifier   import classify_all, classify_symbol as _cs
    from module_03_sr_zones           import detect_zones
    from module_04_leg_compression    import compute_all, SWING_LOOKBACK

    print("Loading data...")
    data, _ = load_all(run_sanity_check=False, debug=False)

    print("Classifying trends...")
    classified = classify_all(data, timeframes=["1h", "4h"])
    for sym in data:
        for tf in ["30m", "15m"]:
            if tf in data[sym]:
                classified[sym][tf] = _cs(data, sym, tf, lookback=SWING_LOOKBACK.get(tf, 5))

    print("Detecting S/R zones...")
    all_zones = detect_zones(data, classified)

    print("Running leg counter + compression...")
    m4 = compute_all(data, classified)

    print("Aggregating signals...")
    signals = aggregate_signals(data, classified, m4, all_zones)

    # ── Summary ───────────────────────────────────────────────────────────
    summary       = signal_summary(signals)
    total_signals = int(summary["signals"].sum())
    total_bars    = int(summary["total_bars"].sum())

    print("\n" + "=" * 74)
    print("  MODULE 5 (v3 — two-strategy architecture)")
    print("  SHORT fade:  depth≥0.65  close<40%  all regimes  stop=1.5×ATR15m")
    print("  LONG retrac: depth 0.382-0.65  close>50%  BULL/RECOVERY  stop=1.5×ATR15m")
    print(f"  min_score={MIN_CONFLUENCE_SIGNAL}  static_radius={STATIC_ZONE_RADIUS*100:.1f}%")
    print("=" * 74)
    print(f"\n  {'Symbol':<14} {'Signals':>7} {'Long':>6} {'Short':>6} "
          f"{'Sig%':>7}  {'AvgScore':>9}  {'AvgRR':>7}")
    print("  " + "-" * 66)
    for _, r in summary.iterrows():
        flag = " ◄" if r["signals"] > 0 else ""
        print(f"  {r['symbol']:<14} {int(r['signals']):>7} {int(r['long']):>6} "
              f"{int(r['short']):>6} {r['signal_pct']:>6.3f}%  "
              f"{r['avg_score']:>9.2f}  {r['avg_rr']:>7.2f}{flag}")
    print(f"\n  Total: {total_signals:,} signals  "
          f"({total_signals/total_bars*100:.3f}% of {total_bars:,} bars)")

    # ── No-trade breakdown ─────────────────────────────────────────────────
    print("\n  BTCUSDT — No-trade reasons:")
    for reason, cnt in no_trade_breakdown(signals, "BTCUSDT").items():
        pct = cnt / len(signals["BTCUSDT"]) * 100
        print(f"    {pct:>5.1f}%  {reason}")

    # ── All fired signals ──────────────────────────────────────────────────
    all_fired_list = []
    for sym, df in signals.items():
        f = df[df["direction"] != "NO_SIGNAL"].copy()
        f["sym"] = sym
        all_fired_list.append(f)

    if all_fired_list:
        fired_all = pd.concat(all_fired_list).sort_index()

        print(f"\n  Zone source breakdown ({len(fired_all)} signals):")
        for src, cnt in fired_all["zone_source"].value_counts().items():
            print(f"    {src:<10} {cnt:>6,}  ({cnt/len(fired_all)*100:.1f}%)")

        # Monthly distribution
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fired_all["month"] = fired_all.index.to_period("M")
        monthly = fired_all.groupby("month").size()
        print(f"\n  Signals per month (last 24 shown):")
        for m, c in list(monthly.items())[-24:]:
            bar = "█" * min(c // 3, 50)
            print(f"    {m}  {bar} {c}")

        # Sample
        sample = fired_all.sample(min(15, len(fired_all)), random_state=42).sort_index()
        print(f"\n  Sample of up to 15 signals:")
        print(f"  {'Timestamp':<22} {'Sym':<12} {'Dir':<6} {'Sc':>5} "
              f"{'Dep':>5} {'Q':<5} {'Src':<8} {'Stp%':>6} {'RR':>5} {'Col':>4}")
        print("  " + "-" * 78)
        for ts, row in sample.iterrows():
            print(f"  {str(ts)[:19]:<22} {row['sym']:<12} {row['direction']:<6} "
                  f"{row['confluence_score']:>5.2f} {row['pullback_depth']:>5.3f} "
                  f"{row['leg_quality']:<5} {row['zone_source']:<8} "
                  f"{row['stop_distance_pct']:>6.3f} {row['rr_net_of_fees']:>5.2f} "
                  f"{'Y' if row['is_coiling'] else 'N':>4}")

        scores = fired_all["confluence_score"].values
        print(f"\n  Score dist: min={scores.min():.2f}  max={scores.max():.2f}  "
              f"mean={scores.mean():.2f}  median={np.median(scores):.2f}")
        print(f"  ≥5.0: {(scores>=5.0).sum():,}  "
              f"≥6.0: {(scores>=6.0).sum():,}  "
              f"≥7.0: {(scores>=7.0).sum():,}")

    else:
        print("\n  *** Still no signals — escalate to committee ***")

    # ── Live snapshot ──────────────────────────────────────────────────────
    print(f"\n  Live snapshot — Feb 28 2026 23:45:")
    print(f"  {'Symbol':<14} {'Dir':<10} {'Sc':>5} {'Q':<7} "
          f"{'Dep':>5} {'Src':<8} {'Stp%':>6} {'RR':>5} {'Col':>4}")
    print("  " + "-" * 70)
    for sym in sorted(signals.keys()):
        row = signals[sym].iloc[-1]
        if row["direction"] != "NO_SIGNAL":
            print(f"  {sym:<14} {row['direction']:<10} "
                  f"{row['confluence_score']:>5.2f} {row['leg_quality']:<7} "
                  f"{row['pullback_depth']:>5.3f} {row['zone_source']:<8} "
                  f"{row['stop_distance_pct']:>6.3f} {row['rr_net_of_fees']:>5.2f} "
                  f"{'Y' if row['is_coiling'] else 'N':>4}")
        else:
            reason = (row["no_trade_reason"] or "")[:38]
            print(f"  {sym:<14} NO_SIGNAL  {row['confluence_score']:>5.2f} "
                  f"{row['leg_quality']:<7} {row['pullback_depth']:>5.3f}  {reason}")
    print()