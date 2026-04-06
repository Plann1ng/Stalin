"""
Stalin Trading System — Indicators & Structural Analysis
Pivot detection, ATR, regime classification, structural levels.
All indicators are computed WITHOUT lookahead bias.
"""

import pandas as pd
import numpy as np
from config import (
    REGIME_SMA_PERIOD, PIVOT_LEFT_BARS, PIVOT_RIGHT_BARS,
    PIVOT_CONFIRMATION_DELAY, ATR_PERIOD, VOLATILITY_LOOKBACK_4H_BARS,
    VOLATILITY_PERCENTILE_THRESHOLD, TRAILING_PIVOT_LEFT, TRAILING_PIVOT_RIGHT,
    MAX_STRUCTURAL_LEVELS
)


def compute_atr(df: pd.DataFrame, period: int = None) -> pd.Series:
    """Compute Average True Range."""
    period = period or ATR_PERIOD
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr


def compute_regime(df: pd.DataFrame) -> pd.Series:
    """
    Classify market regime as BULLCYCLE or BEARCYCLE.
    Uses SMA of close price. No lookahead — uses completed bar data only.
    """
    sma = df['close'].rolling(window=REGIME_SMA_PERIOD, min_periods=REGIME_SMA_PERIOD).mean()
    regime = pd.Series('NEUTRAL', index=df.index)
    regime[df['close'] > sma] = 'BULLCYCLE'
    regime[df['close'] <= sma] = 'BEARCYCLE'
    regime[sma.isna()] = 'NEUTRAL'
    return regime


def detect_pivots(df: pd.DataFrame, left: int = None, right: int = None) -> pd.DataFrame:
    """
    Detect pivot highs and pivot lows with confirmation delay.

    A pivot high at bar i is confirmed only after bar i + right_bars has completed.
    The pivot becomes USABLE only after an additional PIVOT_CONFIRMATION_DELAY bars.

    This means:
    - Pivot at bar i is detected at bar i + right (right bars must form lower highs)
    - Pivot is usable from bar i + right + PIVOT_CONFIRMATION_DELAY onward

    Returns DataFrame with columns:
    - pivot_high: price of pivot high (NaN if not a pivot)
    - pivot_low: price of pivot low (NaN if not a pivot)
    - pivot_high_confirmed_bar: bar index when this pivot high becomes usable
    - pivot_low_confirmed_bar: bar index when this pivot low becomes usable
    """
    left = left or PIVOT_LEFT_BARS
    right = right or PIVOT_RIGHT_BARS

    n = len(df)
    pivot_high = pd.Series(np.nan, index=df.index)
    pivot_low = pd.Series(np.nan, index=df.index)
    pivot_high_usable_bar = pd.Series(np.nan, index=df.index)
    pivot_low_usable_bar = pd.Series(np.nan, index=df.index)

    highs = df['high'].values
    lows = df['low'].values

    for i in range(left, n - right):
        # Check pivot high: bar i has highest high in window [i-left, i+right]
        is_pivot_high = True
        for j in range(i - left, i):
            if highs[j] >= highs[i]:
                is_pivot_high = False
                break
        if is_pivot_high:
            for j in range(i + 1, i + right + 1):
                if highs[j] >= highs[i]:
                    is_pivot_high = False
                    break

        if is_pivot_high:
            pivot_high.iloc[i] = highs[i]
            # Confirmed after right bars + confirmation delay
            usable_bar = i + right + PIVOT_CONFIRMATION_DELAY
            if usable_bar < n:
                pivot_high_usable_bar.iloc[i] = usable_bar

        # Check pivot low: bar i has lowest low in window [i-left, i+right]
        is_pivot_low = True
        for j in range(i - left, i):
            if lows[j] <= lows[i]:
                is_pivot_low = False
                break
        if is_pivot_low:
            for j in range(i + 1, i + right + 1):
                if lows[j] <= lows[i]:
                    is_pivot_low = False
                    break

        if is_pivot_low:
            pivot_low.iloc[i] = lows[i]
            usable_bar = i + right + PIVOT_CONFIRMATION_DELAY
            if usable_bar < n:
                pivot_low_usable_bar.iloc[i] = usable_bar

    return pd.DataFrame({
        'pivot_high': pivot_high,
        'pivot_low': pivot_low,
        'pivot_high_usable_bar': pivot_high_usable_bar,
        'pivot_low_usable_bar': pivot_low_usable_bar
    }, index=df.index)


def get_usable_structural_levels(pivots_df: pd.DataFrame, current_bar: int,
                                  level_type: str = 'both') -> dict:
    """
    Get all structural levels that are confirmed and usable at the current bar.
    No lookahead: only returns levels where usable_bar <= current_bar.

    Args:
        pivots_df: DataFrame from detect_pivots()
        current_bar: current bar index
        level_type: 'high', 'low', or 'both'

    Returns:
        dict with 'highs' and 'lows' lists of (bar_index, price) tuples,
        sorted by price descending for highs, ascending for lows.
    """
    result = {'highs': [], 'lows': []}

    if level_type in ('high', 'both'):
        mask = (
            pivots_df['pivot_high'].notna() &
            (pivots_df['pivot_high_usable_bar'] <= current_bar)
        )
        highs = pivots_df.loc[mask, 'pivot_high']
        result['highs'] = sorted(
            [(idx, price) for idx, price in highs.items()],
            key=lambda x: x[1], reverse=True
        )[-MAX_STRUCTURAL_LEVELS:]

    if level_type in ('low', 'both'):
        mask = (
            pivots_df['pivot_low'].notna() &
            (pivots_df['pivot_low_usable_bar'] <= current_bar)
        )
        lows = pivots_df.loc[mask, 'pivot_low']
        result['lows'] = sorted(
            [(idx, price) for idx, price in lows.items()],
            key=lambda x: x[1]
        )[-MAX_STRUCTURAL_LEVELS:]

    return result


def compute_volatility_filter(df: pd.DataFrame) -> pd.Series:
    """
    Volatility regime filter.
    ATR(14) must be above the 30th percentile of its 90-day (540 bar) rolling range.
    Returns boolean Series: True = volatility sufficient for trading.
    """
    atr = compute_atr(df, ATR_PERIOD)
    lookback = VOLATILITY_LOOKBACK_4H_BARS

    vol_pass = pd.Series(False, index=df.index)

    for i in range(lookback, len(df)):
        atr_window = atr.iloc[max(0, i - lookback):i + 1].dropna()
        if len(atr_window) < 20:
            continue
        threshold = np.percentile(atr_window.values, VOLATILITY_PERCENTILE_THRESHOLD)
        if atr.iloc[i] >= threshold:
            vol_pass.iloc[i] = True

    return vol_pass


def compute_momentum_confirmation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Momentum confirmation filter.
    For each bar, compute where the close falls within the bar's range.

    Returns DataFrame with:
    - close_position: 0.0 (close at low) to 1.0 (close at high)
    - short_momentum_ok: True if close in bottom 30% of range (bearish)
    - long_momentum_ok: True if close in top 30% of range (bullish)
    """
    from config import MOMENTUM_THRESHOLD

    bar_range = df['high'] - df['low']
    # Avoid division by zero for doji bars
    bar_range = bar_range.replace(0, np.nan)

    close_position = (df['close'] - df['low']) / bar_range
    close_position = close_position.fillna(0.5)  # Doji bars = neutral

    return pd.DataFrame({
        'close_position': close_position,
        'short_momentum_ok': close_position <= MOMENTUM_THRESHOLD,
        'long_momentum_ok': close_position >= (1.0 - MOMENTUM_THRESHOLD)
    }, index=df.index)


def find_next_structural_level(levels: list, price: float, direction: str) -> float:
    """
    Find the next structural level in the trade's direction from the given price.

    For shorts (direction='short'): find the highest support level BELOW price
    For longs (direction='long'): find the lowest resistance level ABOVE price

    Args:
        levels: list of (bar_index, price) tuples
        price: current price
        direction: 'short' or 'long'

    Returns:
        Price of next structural level, or None if no level found
    """
    if direction == 'short':
        # Find support levels below current price
        below = [p for _, p in levels if p < price]
        if below:
            return max(below)  # Nearest support below
        return None
    else:
        # Find resistance levels above current price
        above = [p for _, p in levels if p > price]
        if above:
            return min(above)  # Nearest resistance above
        return None


def detect_trailing_pivots(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect pivots specifically for the trailing stop mechanism.
    Uses potentially different left/right parameters than entry pivots.
    """
    return detect_pivots(df, left=TRAILING_PIVOT_LEFT, right=TRAILING_PIVOT_RIGHT)


def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all indicators and attach to the DataFrame.
    Master function called once before backtesting.
    """
    df = df.copy()

    # ATR
    df['atr'] = compute_atr(df)

    # Regime
    df['regime'] = compute_regime(df)

    # Pivots for entry signals
    pivots = detect_pivots(df)
    df['pivot_high'] = pivots['pivot_high']
    df['pivot_low'] = pivots['pivot_low']
    df['pivot_high_usable_bar'] = pivots['pivot_high_usable_bar']
    df['pivot_low_usable_bar'] = pivots['pivot_low_usable_bar']

    # Trailing stop pivots
    trailing_pivots = detect_trailing_pivots(df)
    df['trail_pivot_high'] = trailing_pivots['pivot_high']
    df['trail_pivot_low'] = trailing_pivots['pivot_low']
    df['trail_pivot_high_usable_bar'] = trailing_pivots['pivot_high_usable_bar']
    df['trail_pivot_low_usable_bar'] = trailing_pivots['pivot_low_usable_bar']

    # Volatility filter
    df['volatility_ok'] = compute_volatility_filter(df)

    # Momentum confirmation
    momentum = compute_momentum_confirmation(df)
    df['close_position'] = momentum['close_position']
    df['short_momentum_ok'] = momentum['short_momentum_ok']
    df['long_momentum_ok'] = momentum['long_momentum_ok']

    return df

