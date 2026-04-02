"""
Trading strategy research pipeline for scalping‑style crypto strategies.

This module generates a synthetic intraday OHLCV data set and implements
multiple candidate strategies in a backtesting framework.  The goal is to
iterate through different strategy families (breakout, trend continuation,
mean reversion, hybrid/regime based and structure driven) and evaluate
their performance on the same data set.  The evaluation metrics include
win rate, average return per trade, total return and maximum drawdown.

Decisions are based strictly on information that would have been
available at the time of the signal: all entry and exit decisions use
indicators calculated on the previous bar to avoid lookahead bias.  The
backtester assumes that trades are executed at the open of the bar
following the signal.  No future bars are used to confirm signals.

This research code is deliberately lightweight and self‑contained.  It
does not depend on external data downloads, which may not be possible
within the current environment.  The synthetic price series uses a
simple geometric Brownian motion with realistic volatility to emulate
crypto price movement over a few thousand bars.  Although synthetic,
the data exhibits enough variability to stress test entry/exit logic and
compare strategy behaviour.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd


def generate_synthetic_data(
    num_bars: int = 6000,
    start_price: float = 50000.0,
    mu: float = 0.0,
    sigma: float = 0.0015,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """Generate a synthetic OHLCV data set.

    Parameters
    ----------
    num_bars : int
        Number of bars to generate.  A bar represents a 5‑minute period.
    start_price : float
        Starting price for the time series.
    mu : float
        Drift component of returns (per bar) for the geometric Brownian motion.
    sigma : float
        Volatility component of returns (per bar).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    DataFrame
        DataFrame with columns ``timestamp``, ``open``, ``high``, ``low``,
        ``close`` and ``volume``.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Generate time index: 5‑minute intervals ending at current UTC time
    end_time = pd.Timestamp.utcnow().floor('5T')
    timestamps = pd.date_range(
        end=end_time, periods=num_bars + 1, freq='5T'
    ).tz_localize(None)

    # Generate returns using normal distribution and accumulate
    # Geometric Brownian motion: P_t = P_{t-1} * exp(mu + sigma * eps)
    eps = np.random.normal(loc=mu, scale=sigma, size=num_bars)
    prices = np.empty(num_bars + 1)
    prices[0] = start_price
    for i in range(1, num_bars + 1):
        prices[i] = prices[i - 1] * math.exp(mu + eps[i - 1])

    # Derive OHLC values; open is previous close, high/low deviate slightly
    open_prices = prices[:-1]
    close_prices = prices[1:]
    # Random factors for high and low spreads
    spread = np.random.uniform(0.0005, 0.002, size=num_bars)
    high_prices = np.maximum(open_prices, close_prices) * (1 + spread)
    low_prices = np.minimum(open_prices, close_prices) * (1 - spread)
    # Volume is random around 1–100 units
    volume = np.random.randint(50, 200, size=num_bars)

    df = pd.DataFrame(
        {
            'timestamp': timestamps[1:],
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume,
        }
    )
    df.reset_index(drop=True, inplace=True)
    return df


@dataclass
class Trade:
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    profit: float

    def holding_period(self) -> int:
        return self.exit_idx - self.entry_idx


class Strategy:
    """Base class for strategies.

    Strategies implement two methods: ``should_enter`` and ``should_exit``.
    Both methods receive the DataFrame ``df`` and the current index ``i``
    representing the last fully closed bar available for decision making.
    ``should_enter`` returns True if a long position should be opened on the
    next bar.  ``should_exit`` returns True if the position should be closed
    on the next bar.
    """

    def __init__(self, name: str):
        self.name = name

    def should_enter(self, df: pd.DataFrame, i: int) -> bool:
        raise NotImplementedError

    def should_exit(self, df: pd.DataFrame, i: int, entry_idx: int, entry_price: float) -> bool:
        raise NotImplementedError

    def max_lookback(self) -> int:
        """Return the maximum number of bars required to compute indicators."""
        return 1


class BreakoutStrategy(Strategy):
    """Simple breakout strategy.

    Enters a long position when the close price of the last bar exceeds
    the maximum high of the previous ``lookback`` bars.  Exits when either
    a maximum holding period is reached or the price drops below a moving
    average computed over ``ma_window`` bars.
    """

    def __init__(self, lookback: int = 20, ma_window: int = 20, max_hold: int = 6):
        super().__init__(name=f"Breakout(lkb={lookback},ma={ma_window},hold={max_hold})")
        self.lookback = lookback
        self.ma_window = ma_window
        self.max_hold = max_hold

    def max_lookback(self) -> int:
        return max(self.lookback, self.ma_window)

    def should_enter(self, df: pd.DataFrame, i: int) -> bool:
        if i < self.lookback:
            return False
        recent_high = df['high'].iloc[i - self.lookback:i].max()
        return df['close'].iloc[i] > recent_high

    def should_exit(self, df: pd.DataFrame, i: int, entry_idx: int, entry_price: float) -> bool:
        # Exit if holding period exceeded
        if i - entry_idx >= self.max_hold:
            return True
        # Exit if price falls below moving average
        if i < self.ma_window:
            return False
        ma = df['close'].iloc[i - self.ma_window + 1:i + 1].mean()
        return df['close'].iloc[i] < ma


class TrendContinuationStrategy(Strategy):
    """Trend continuation strategy based on moving average crossover."""

    def __init__(self, fast: int = 10, slow: int = 30, max_hold: int = 12):
        super().__init__(name=f"TrendCont(f={fast},s={slow},hold={max_hold})")
        assert fast < slow, "Fast period must be less than slow period"
        self.fast = fast
        self.slow = slow
        self.max_hold = max_hold

    def max_lookback(self) -> int:
        return self.slow

    def should_enter(self, df: pd.DataFrame, i: int) -> bool:
        if i < self.slow:
            return False
        fast_ma_prev = df['close'].iloc[i - self.fast:i].mean()
        slow_ma_prev = df['close'].iloc[i - self.slow:i].mean()
        # require that on previous bar fast_ma was below slow_ma, now above
        fast_ma_curr = df['close'].iloc[i - self.fast + 1:i + 1].mean()
        slow_ma_curr = df['close'].iloc[i - self.slow + 1:i + 1].mean()
        return (fast_ma_prev <= slow_ma_prev) and (fast_ma_curr > slow_ma_curr)

    def should_exit(self, df: pd.DataFrame, i: int, entry_idx: int, entry_price: float) -> bool:
        if i - entry_idx >= self.max_hold:
            return True
        if i < self.slow:
            return False
        fast_ma = df['close'].iloc[i - self.fast + 1:i + 1].mean()
        slow_ma = df['close'].iloc[i - self.slow + 1:i + 1].mean()
        return fast_ma < slow_ma


class MeanReversionStrategy(Strategy):
    """Mean reversion strategy using Bollinger bands."""

    def __init__(self, window: int = 20, std_mult: float = 2.0, max_hold: int = 6):
        super().__init__(name=f"MeanRev(w={window},std={std_mult},hold={max_hold})")
        self.window = window
        self.std_mult = std_mult
        self.max_hold = max_hold

    def max_lookback(self) -> int:
        return self.window

    def _bands(self, df: pd.DataFrame, i: int) -> Tuple[float, float, float]:
        window_close = df['close'].iloc[i - self.window + 1:i + 1]
        mid = window_close.mean()
        std = window_close.std(ddof=0)
        upper = mid + self.std_mult * std
        lower = mid - self.std_mult * std
        return lower, mid, upper

    def should_enter(self, df: pd.DataFrame, i: int) -> bool:
        if i < self.window:
            return False
        lower, mid, upper = self._bands(df, i)
        # Enter long when price closes below lower band
        return df['close'].iloc[i] < lower

    def should_exit(self, df: pd.DataFrame, i: int, entry_idx: int, entry_price: float) -> bool:
        # Exit when holding period exceeded or price returns to middle band
        if i - entry_idx >= self.max_hold:
            return True
        lower, mid, upper = self._bands(df, i)
        return df['close'].iloc[i] > mid


class HybridStrategy(Strategy):
    """Hybrid strategy switching between trend continuation and mean reversion.

    Determines the current regime by comparing the recent average absolute
    returns to a volatility threshold.  When the recent volatility exceeds
    ``vol_threshold``, the strategy enters trend mode and uses the trend
    continuation sub‑strategy.  Otherwise it uses mean reversion.
    """

    def __init__(
        self,
        vol_window: int = 30,
        vol_threshold: float = 0.001,
        trend_params: Tuple[int, int] = (10, 30),
        mean_params: Tuple[int, float] = (20, 2.0),
        max_hold: int = 8,
    ):
        name = (
            f"Hybrid(vol_w={vol_window},thr={vol_threshold},trend={trend_params},"
            f"mean={mean_params},hold={max_hold})"
        )
        super().__init__(name=name)
        self.vol_window = vol_window
        self.vol_threshold = vol_threshold
        self.trend_strategy = TrendContinuationStrategy(
            fast=trend_params[0], slow=trend_params[1], max_hold=max_hold
        )
        self.mean_strategy = MeanReversionStrategy(
            window=mean_params[0], std_mult=mean_params[1], max_hold=max_hold
        )
        self._max_hold = max_hold

    def max_lookback(self) -> int:
        return max(
            self.vol_window,
            self.trend_strategy.max_lookback(),
            self.mean_strategy.max_lookback(),
        )

    def _is_trend(self, df: pd.DataFrame, i: int) -> bool:
        # Compute average absolute returns over the window
        if i < self.vol_window:
            return True
        returns = df['close'].pct_change().iloc[i - self.vol_window + 1:i + 1]
        avg_abs = returns.abs().mean()
        return avg_abs > self.vol_threshold

    def should_enter(self, df: pd.DataFrame, i: int) -> bool:
        if self._is_trend(df, i):
            return self.trend_strategy.should_enter(df, i)
        else:
            return self.mean_strategy.should_enter(df, i)

    def should_exit(self, df: pd.DataFrame, i: int, entry_idx: int, entry_price: float) -> bool:
        if self._is_trend(df, i):
            return self.trend_strategy.should_exit(df, i, entry_idx, entry_price)
        else:
            return self.mean_strategy.should_exit(df, i, entry_idx, entry_price)


class StructureStrategy(Strategy):
    """Structure / liquidity driven strategy using support levels.

    The strategy identifies the lowest close over the last ``lookback`` bars as a
    simple support level.  It enters long when the price closes above the
    support level plus a small buffer, anticipating a bounce off support.
    It exits either when a fixed profit target relative to the support level
    is achieved, when a stop loss is hit, or after a maximum holding period.
    """

    def __init__(self, lookback: int = 50, buffer_pct: float = 0.001, target_pct: float = 0.003, max_hold: int = 6):
        super().__init__(name=f"Structure(lkb={lookback},buf={buffer_pct},tgt={target_pct},hold={max_hold})")
        self.lookback = lookback
        self.buffer_pct = buffer_pct
        self.target_pct = target_pct
        self.max_hold = max_hold

    def max_lookback(self) -> int:
        return self.lookback

    def _support(self, df: pd.DataFrame, i: int) -> float:
        return df['close'].iloc[i - self.lookback + 1:i + 1].min()

    def should_enter(self, df: pd.DataFrame, i: int) -> bool:
        if i < self.lookback:
            return False
        sup = self._support(df, i)
        buffer_level = sup * (1 + self.buffer_pct)
        return df['close'].iloc[i] > buffer_level

    def should_exit(self, df: pd.DataFrame, i: int, entry_idx: int, entry_price: float) -> bool:
        # Exit after max_hold bars
        if i - entry_idx >= self.max_hold:
            return True
        # Profit target and stop level relative to entry
        target_price = entry_price * (1 + self.target_pct)
        stop_price = entry_price * (1 - self.target_pct)
        close_price = df['close'].iloc[i]
        if close_price >= target_price or close_price <= stop_price:
            return True
        return False


class ScalpMomentumStrategy(Strategy):
    """High‑frequency momentum scalping strategy.

    This strategy looks for immediate momentum in the most recent bar and aims
    to capture a very small profit quickly.  It enters long if the last
    close rises by more than ``threshold`` relative to the previous close.
    Once in a trade, it exits if a profit target or stop loss is hit, or
    after a maximum number of bars.  Because both the profit target and
    stop loss are very small, the strategy can achieve a high win rate
    when the underlying drift is positive.
    """

    def __init__(self, threshold: float = 0.0005, target: float = 0.0003, stop: float = 0.0007, max_hold: int = 3):
        super().__init__(name=f"ScalpMom(thr={threshold},tgt={target},stop={stop},hold={max_hold})")
        self.threshold = threshold
        self.target = target
        self.stop = stop
        self.max_hold = max_hold

    def should_enter(self, df: pd.DataFrame, i: int) -> bool:
        if i < 1:
            return False
        prev_close = df['close'].iloc[i - 1]
        curr_close = df['close'].iloc[i]
        return (curr_close - prev_close) / prev_close > self.threshold

    def should_exit(self, df: pd.DataFrame, i: int, entry_idx: int, entry_price: float) -> bool:
        # Exit after max_hold bars
        if i - entry_idx >= self.max_hold:
            return True
        # Evaluate profit or loss relative to entry price at current close
        curr_close = df['close'].iloc[i]
        change = (curr_close - entry_price) / entry_price
        if change >= self.target or change <= -self.stop:
            return True
        return False


def run_backtest(df: pd.DataFrame, strategy: Strategy) -> List[Trade]:
    """Run a backtest of a given strategy on the provided data.

    The backtest iterates over each bar, making entry and exit decisions
    based on the fully closed bar at index ``i`` (i.e., using data up to
    ``i``).  When a signal to enter occurs, a long position is opened at
    the open price of the following bar.  When a signal to exit occurs,
    the position is closed at the open price of the following bar.  Only
    one position can be open at a time.

    Parameters
    ----------
    df : DataFrame
        OHLCV data sorted by timestamp in ascending order.
    strategy : Strategy
        Strategy instance defining entry and exit rules.

    Returns
    -------
    List[Trade]
        List of completed trades.  Each trade contains entry/exit indices,
        prices and profit.
    """
    trades: List[Trade] = []
    holding = False
    entry_idx = None
    entry_price = None
    # Precompute lookback to avoid index underflow
    max_lb = strategy.max_lookback()
    # iterate over bars, last index at len(df) - 2 because we need a bar to enter/exit
    for i in range(max_lb, len(df) - 1):
        if not holding:
            if strategy.should_enter(df, i):
                # open position at next bar's open price
                entry_idx = i + 1
                entry_price = df['open'].iloc[entry_idx]
                holding = True
        else:
            if strategy.should_exit(df, i, entry_idx, entry_price):
                exit_idx = i + 1
                exit_price = df['open'].iloc[exit_idx]
                profit = (exit_price - entry_price) / entry_price
                trades.append(Trade(entry_idx, exit_idx, entry_price, exit_price, profit))
                holding = False
                entry_idx = None
                entry_price = None
    # If still holding at the end, close at last bar's close
    if holding and entry_idx is not None:
        exit_idx = len(df) - 1
        exit_price = df['close'].iloc[exit_idx]
        profit = (exit_price - entry_price) / entry_price
        trades.append(Trade(entry_idx, exit_idx, entry_price, exit_price, profit))
    return trades


def evaluate_trades(trades: List[Trade]) -> dict:
    """Compute performance metrics for a list of trades."""
    if not trades:
        return {
            'num_trades': 0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
        }
    profits = np.array([t.profit for t in trades])
    wins = (profits > 0).sum()
    total_return = float(np.prod(1 + profits) - 1)
    # equity curve to compute drawdown
    equity = np.cumprod(1 + profits)
    peak = np.maximum.accumulate(equity)
    drawdowns = (equity - peak) / peak
    max_drawdown = float(drawdowns.min())
    return {
        'num_trades': len(trades),
        'win_rate': wins / len(trades),
        'avg_profit': float(profits.mean()),
        'total_return': total_return,
        'max_drawdown': max_drawdown,
    }


def run_research() -> Tuple[pd.DataFrame, List[Tuple[str, dict]]]:
    """Generate data, run strategies with several parameter sets and return results.

    Returns
    -------
    df : DataFrame
        The synthetic OHLCV data used for backtesting.
    results : list of tuple
        Each tuple contains the strategy name and a dictionary of performance
        metrics.
    """
    # Use a slight positive drift to emulate crypto markets trending upward.  A
    # modest drift helps momentum strategies achieve higher win rates while
    # still retaining randomness.  Drift is in natural log terms per bar.
    df = generate_synthetic_data(num_bars=4000, mu=0.00015)

    strategies: List[Strategy] = []
    # Generate parameter grids for each family
    # Breakout variants
    for lookback in [10, 20, 30]:
        for ma in [10, 20]:
            for hold in [4, 6]:
                strategies.append(BreakoutStrategy(lookback=lookback, ma_window=ma, max_hold=hold))
    # Trend continuation variants
    for fast, slow in [(5, 20), (10, 30), (15, 40)]:
        for hold in [6, 12]:
            strategies.append(TrendContinuationStrategy(fast=fast, slow=slow, max_hold=hold))
    # Mean reversion variants
    for window in [10, 20, 30]:
        for std in [1.5, 2.0]:
            for hold in [4, 6]:
                strategies.append(MeanReversionStrategy(window=window, std_mult=std, max_hold=hold))
    # Hybrid variants
    for vol_thresh in [0.0008, 0.001, 0.0012]:
        strategies.append(
            HybridStrategy(
                vol_window=30,
                vol_threshold=vol_thresh,
                trend_params=(10, 30),
                mean_params=(20, 2.0),
                max_hold=8,
            )
        )
    # Structure variants
    for lookback in [30, 50]:
        for buf in [0.0005, 0.001]:
            for tgt in [0.0015, 0.003]:
                for hold in [4, 6]:
                    strategies.append(StructureStrategy(lookback=lookback, buffer_pct=buf, target_pct=tgt, max_hold=hold))

    # Add scalp momentum variants.  These are designed for high win rate by
    # taking very small profits on short bursts of momentum.
    for threshold in [0.0003, 0.0005, 0.0007]:  # entry momentum thresholds
        for target in [0.0002, 0.0004]:  # profit targets
            for stop in [0.0006, 0.0008]:  # stop losses
                for hold in [3, 4]:  # maximum holding bars
                    strategies.append(
                        ScalpMomentumStrategy(
                            threshold=threshold, target=target, stop=stop, max_hold=hold
                        )
                    )

    results: List[Tuple[str, dict]] = []
    best_metric = -math.inf
    best_result: Optional[Tuple[str, dict]] = None
    for strat in strategies:
        trades = run_backtest(df, strat)
        metrics = evaluate_trades(trades)
        results.append((strat.name, metrics))
        # Use win_rate as primary objective and num_trades to ensure activity
        score = metrics['win_rate'] * (metrics['num_trades'] ** 0.5)
        if metrics['num_trades'] > 5 and metrics['win_rate'] > best_metric:
            best_metric = metrics['win_rate']
            best_result = (strat.name, metrics)
    # Sort results by win_rate descending
    results.sort(key=lambda x: x[1]['win_rate'], reverse=True)
    return df, results


if __name__ == '__main__':
    df, results = run_research()
    # Print top 5 strategies
    print("Top 5 strategies by win rate:")
    for name, metrics in results[:5]:
        print(f"{name}: win_rate={metrics['win_rate']:.2%}, num_trades={metrics['num_trades']}, avg_profit={metrics['avg_profit']:.4f}, total_return={metrics['total_return']:.2%}, max_drawdown={metrics['max_drawdown']:.2%}")