"""
Stalin Trading System — Core Strategy Logic
Entry signals, exit logic, position management.
All decisions use only confirmed past data — no lookahead.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from config import (
    ALLOW_LONGS, ALLOW_SHORTS,
    STOP_LOSS_ATR_MULTIPLE, STOP_LOSS_FIXED_PCT,
    MIN_STRUCTURAL_SPACING_R, TRAILING_ACTIVATION_R,
    EXITGAP_ATR_THRESHOLD, EXITGAP_USE_PRIOR_CLOSE,
    MAX_TRADE_DURATION_BARS, MAX_CONSECUTIVE_LOSSES,
    MAX_DRAWDOWN_FRACTION, CIRCUIT_BREAKER_COOLDOWN_BARS,
    FEE_ROUND_TRIP, SLIPPAGE_PER_SIDE, RISK_PER_TRADE,
    MAX_STRUCTURAL_LEVELS
)
from indicators import get_usable_structural_levels, find_next_structural_level


@dataclass
class Trade:
    """Represents a single trade."""
    entry_bar: int
    entry_time: pd.Timestamp
    entry_price: float
    direction: str  # 'short' or 'long'
    stop_loss: float
    risk_per_unit: float  # |entry - stop| per unit
    position_size: float  # in base currency units
    regime: str
    broken_level: float  # the structural level that was broken

    exit_bar: Optional[int] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0
    r_multiple: float = 0.0
    fees: float = 0.0

    # Trailing stop state
    trailing_stop: Optional[float] = None
    trailing_activated: bool = False
    max_favorable_excursion: float = 0.0
    bars_in_trade: int = 0


@dataclass
class CircuitBreaker:
    """Tracks drawdown and consecutive losses for circuit breaker."""
    consecutive_losses: int = 0
    peak_equity: float = 0.0
    is_halted: bool = False
    halt_bar: int = 0
    cooldown_remaining: int = 0

    def update_after_trade(self, trade: Trade, current_equity: float, current_bar: int):
        """Update circuit breaker state after a trade closes."""
        # Track consecutive losses
        if trade.r_multiple < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Track peak equity and drawdown
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        drawdown = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0

        # Check halt conditions
        if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            self.is_halted = True
            self.halt_bar = current_bar
            self.cooldown_remaining = CIRCUIT_BREAKER_COOLDOWN_BARS

        if drawdown >= MAX_DRAWDOWN_FRACTION:
            self.is_halted = True
            self.halt_bar = current_bar
            self.cooldown_remaining = CIRCUIT_BREAKER_COOLDOWN_BARS

    def update_bar(self):
        """Called each bar to decrement cooldown."""
        if self.is_halted and self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            if self.cooldown_remaining <= 0:
                self.is_halted = False
                self.consecutive_losses = 0

    def can_trade(self) -> bool:
        return not self.is_halted


class StalinStrategy:
    """
    Stalin Trading System — Structural Breakout/Breakdown Strategy

    Key changes from original:
    1. BULLCYCLE longs eliminated (ALLOW_LONGS = False)
    2. Structural trailing stop replaces fixed/breakeven trailing
    3. Momentum confirmation filter on breakout candle
    4. Volatility regime filter (ATR percentile)
    5. Structural spacing filter (room to profit)
    6. EXITGAP logic fixed for lookahead bias
    7. Pivot confirmation delay enforced
    8. Drawdown circuit breaker
    9. Realistic fee model (taker fills)
    """

    def __init__(self, df: pd.DataFrame, initial_capital: float):
        self.df = df
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.peak_equity = initial_capital

        self.trades: List[Trade] = []
        self.active_trade: Optional[Trade] = None
        self.circuit_breaker = CircuitBreaker(peak_equity=initial_capital)

        # Track which structural levels have been broken (to avoid re-entry)
        self.broken_levels: List[float] = []

        # Equity curve
        self.equity_curve = []

    def run(self) -> List[Trade]:
        """
        Run the strategy bar by bar.
        Entry at next candle open after signal on completed candle.
        """
        n = len(self.df)

        for i in range(1, n):
            bar = self.df.iloc[i]
            prev_bar = self.df.iloc[i - 1]

            # Update circuit breaker cooldown
            self.circuit_breaker.update_bar()

            # If in a trade, check exits first (using current bar's OHLC)
            if self.active_trade is not None:
                self._check_exits(i)

            # If no active trade, check for entry signals
            # Signal is based on prev_bar (completed), entry at bar i's open
            if self.active_trade is None and self.circuit_breaker.can_trade():
                self._check_entries(i)

            # Record equity
            self.equity_curve.append({
                'bar': i,
                'timestamp': bar['timestamp'],
                'equity': self.equity,
                'in_trade': self.active_trade is not None
            })

        # Close any remaining open trade at last bar's close
        if self.active_trade is not None:
            self._close_trade(len(self.df) - 1, self.df.iloc[-1]['close'], 'END_OF_DATA')

        return self.trades

    def _check_entries(self, bar_idx: int):
        """
        Check for entry signals.
        Signal is generated from the PREVIOUS bar (bar_idx - 1).
        Entry executes at bar_idx's open price.
        """
        prev_idx = bar_idx - 1
        prev = self.df.iloc[prev_idx]
        current = self.df.iloc[bar_idx]

        # Must have valid ATR
        if pd.isna(prev['atr']) or prev['atr'] <= 0:
            return

        # Volatility filter must pass on the signal bar
        if not prev['volatility_ok']:
            return

        # Get usable structural levels at the signal bar (prev_idx)
        levels = self._get_usable_levels_at_bar(prev_idx)

        # Check SHORT entries (BEARCYCLE only)
        if ALLOW_SHORTS and prev['regime'] == 'BEARCYCLE':
            self._check_short_entry(bar_idx, prev_idx, prev, current, levels)

        # Check LONG entries (disabled by default)
        if ALLOW_LONGS and prev['regime'] == 'BULLCYCLE':
            self._check_long_entry(bar_idx, prev_idx, prev, current, levels)

    def _check_short_entry(self, bar_idx: int, prev_idx: int,
                            prev: pd.Series, current: pd.Series, levels: dict):
        """
        Check for short entry: price breaks below a structural support level.
        """
        # Need support levels (pivot lows) to break below
        support_levels = levels['lows']
        if not support_levels:
            return

        # Momentum confirmation: prev candle must close in bottom 30% of range
        if not prev['short_momentum_ok']:
            return

        # Check each support level for a breakdown
        for level_bar, level_price in support_levels:
            # Skip if this level was already broken
            if self._is_level_broken(level_price, tolerance=prev['atr'] * 0.1):
                continue

            # Breakdown condition: previous bar CLOSED below the level
            if prev['close'] >= level_price:
                continue

            # The bar before the signal bar should have been above or at the level
            # (ensures this is a fresh break, not continuation)
            if prev_idx >= 2:
                two_bars_ago = self.df.iloc[prev_idx - 1]
                if two_bars_ago['close'] < level_price:
                    continue  # Already broken before

            # Calculate stop loss
            stop_loss = self._calculate_stop_loss(level_price, 'short', prev['atr'])
            entry_price = current['open']  # Enter at next bar's open

            # Stop must be above entry
            if stop_loss <= entry_price:
                continue

            risk_per_unit = stop_loss - entry_price

            # Structural spacing filter: check room to profit
            if not self._check_structural_spacing(entry_price, risk_per_unit, 'short', levels):
                continue

            # Calculate position size
            position_size = self._calculate_position_size(risk_per_unit)
            if position_size <= 0:
                continue

            # Create and record the trade
            self.active_trade = Trade(
                entry_bar=bar_idx,
                entry_time=current['timestamp'],
                entry_price=entry_price,
                direction='short',
                stop_loss=stop_loss,
                risk_per_unit=risk_per_unit,
                position_size=position_size,
                regime=prev['regime'],
                broken_level=level_price
            )

            self.broken_levels.append(level_price)
            return  # Only take one trade at a time

    def _check_long_entry(self, bar_idx: int, prev_idx: int,
                           prev: pd.Series, current: pd.Series, levels: dict):
        """
        Check for long entry: price breaks above a structural resistance level.
        Currently disabled (ALLOW_LONGS = False).
        """
        resistance_levels = levels['highs']
        if not resistance_levels:
            return

        if not prev['long_momentum_ok']:
            return

        for level_bar, level_price in resistance_levels:
            if self._is_level_broken(level_price, tolerance=prev['atr'] * 0.1):
                continue

            if prev['close'] <= level_price:
                continue

            if prev_idx >= 2:
                two_bars_ago = self.df.iloc[prev_idx - 1]
                if two_bars_ago['close'] > level_price:
                    continue

            stop_loss = self._calculate_stop_loss(level_price, 'long', prev['atr'])
            entry_price = current['open']

            if stop_loss >= entry_price:
                continue

            risk_per_unit = entry_price - stop_loss

            if not self._check_structural_spacing(entry_price, risk_per_unit, 'long', levels):
                continue

            position_size = self._calculate_position_size(risk_per_unit)
            if position_size <= 0:
                continue

            self.active_trade = Trade(
                entry_bar=bar_idx,
                entry_time=current['timestamp'],
                entry_price=entry_price,
                direction='long',
                stop_loss=stop_loss,
                risk_per_unit=risk_per_unit,
                position_size=position_size,
                regime=prev['regime'],
                broken_level=level_price
            )

            self.broken_levels.append(level_price)
            return

    def _check_exits(self, bar_idx: int):
        """
        Check all exit conditions for the active trade.
        Priority: Stop Loss > EXITGAP > Structural Trailing Stop > Max Duration
        """
        trade = self.active_trade
        bar = self.df.iloc[bar_idx]
        trade.bars_in_trade += 1

        # Update max favorable excursion
        if trade.direction == 'short':
            mfe_price = bar['low']
            current_r = (trade.entry_price - mfe_price) / trade.risk_per_unit
        else:
            mfe_price = bar['high']
            current_r = (mfe_price - trade.entry_price) / trade.risk_per_unit

        trade.max_favorable_excursion = max(trade.max_favorable_excursion, current_r)

        # 1. STOP LOSS CHECK (intra-bar)
        if self._check_stop_loss_hit(bar_idx):
            return

        # 2. EXITGAP CHECK (based on prior bar close, no lookahead)
        if self._check_exitgap(bar_idx):
            return

        # 3. STRUCTURAL TRAILING STOP
        self._update_trailing_stop(bar_idx)
        if self._check_trailing_stop_hit(bar_idx):
            return

        # 4. MAX DURATION
        if trade.bars_in_trade >= MAX_TRADE_DURATION_BARS:
            self._close_trade(bar_idx, bar['close'], 'MAX_DURATION')
            return

    def _check_stop_loss_hit(self, bar_idx: int) -> bool:
        """Check if stop loss was hit during this bar."""
        trade = self.active_trade
        bar = self.df.iloc[bar_idx]

        if trade.direction == 'short':
            if bar['high'] >= trade.stop_loss:
                # Stop hit — use stop price (or open if gapped above)
                exit_price = max(trade.stop_loss, bar['open'])
                self._close_trade(bar_idx, exit_price, 'STOP_LOSS')
                return True
        else:  # long
            if bar['low'] <= trade.stop_loss:
                exit_price = min(trade.stop_loss, bar['open'])
                self._close_trade(bar_idx, exit_price, 'STOP_LOSS')
                return True

        return False

    def _check_exitgap(self, bar_idx: int) -> bool:
        """
        EXITGAP logic — FIXED for lookahead bias.
        Decision based on PRIOR bar's close. Execution at current bar's open.

        An EXITGAP occurs when price gaps significantly against the position
        between the prior bar's close and the current bar's open.
        """
        trade = self.active_trade
        bar = self.df.iloc[bar_idx]

        if bar_idx < 1:
            return False

        prev_bar = self.df.iloc[bar_idx - 1]
        atr = prev_bar['atr'] if not pd.isna(prev_bar['atr']) else bar['atr']

        if pd.isna(atr) or atr <= 0:
            return False

        gap = abs(bar['open'] - prev_bar['close'])
        gap_threshold = atr * EXITGAP_ATR_THRESHOLD

        if gap < gap_threshold:
            return False

        # Check if gap is against the position
        if trade.direction == 'short':
            # Gap up against short
            if bar['open'] > prev_bar['close']:
                self._close_trade(bar_idx, bar['open'], 'EXITGAP')
                return True
        else:  # long
            # Gap down against long
            if bar['open'] < prev_bar['close']:
                self._close_trade(bar_idx, bar['open'], 'EXITGAP')
                return True

        return False

    def _update_trailing_stop(self, bar_idx: int):
        """
        Update structural trailing stop.
        For shorts: trail behind confirmed swing highs
        For longs: trail behind confirmed swing lows

        Only activates after trade reaches TRAILING_ACTIVATION_R in profit.
        """
        trade = self.active_trade
        bar = self.df.iloc[bar_idx]

        # Check if trailing should activate
        if trade.direction == 'short':
            current_r = (trade.entry_price - bar['close']) / trade.risk_per_unit
        else:
            current_r = (bar['close'] - trade.entry_price) / trade.risk_per_unit

        if current_r < TRAILING_ACTIVATION_R and not trade.trailing_activated:
            return

        trade.trailing_activated = True

        # Get confirmed trailing pivots at this bar
        if trade.direction == 'short':
            # Trail behind swing highs (resistance above)
            mask = (
                self.df['trail_pivot_high'].notna() &
                (self.df['trail_pivot_high_usable_bar'] <= bar_idx) &
                (self.df.index >= trade.entry_bar)  # Only pivots formed during trade
            )
            pivots = self.df.loc[mask, 'trail_pivot_high'].sort_index()

            if len(pivots) > 0:
                # Use the most recent confirmed swing high as trailing stop
                latest_pivot_price = pivots.iloc[-1]
                # Add small buffer above the pivot
                atr = bar['atr'] if not pd.isna(bar['atr']) else trade.risk_per_unit
                new_trail = latest_pivot_price + atr * 0.1

                # Trailing stop can only move DOWN for shorts (tighter)
                if trade.trailing_stop is None or new_trail < trade.trailing_stop:
                    # Only update if it's tighter than initial stop
                    if new_trail < trade.stop_loss:
                        trade.trailing_stop = new_trail

        else:  # long
            mask = (
                self.df['trail_pivot_low'].notna() &
                (self.df['trail_pivot_low_usable_bar'] <= bar_idx) &
                (self.df.index >= trade.entry_bar)
            )
            pivots = self.df.loc[mask, 'trail_pivot_low'].sort_index()

            if len(pivots) > 0:
                latest_pivot_price = pivots.iloc[-1]
                atr = bar['atr'] if not pd.isna(bar['atr']) else trade.risk_per_unit
                new_trail = latest_pivot_price - atr * 0.1

                # Trailing stop can only move UP for longs (tighter)
                if trade.trailing_stop is None or new_trail > trade.trailing_stop:
                    if new_trail > trade.stop_loss:
                        trade.trailing_stop = new_trail

    def _check_trailing_stop_hit(self, bar_idx: int) -> bool:
        """Check if the structural trailing stop was hit."""
        trade = self.active_trade
        if trade.trailing_stop is None:
            return False

        bar = self.df.iloc[bar_idx]

        if trade.direction == 'short':
            if bar['high'] >= trade.trailing_stop:
                exit_price = max(trade.trailing_stop, bar['open'])
                self._close_trade(bar_idx, exit_price, 'TRAILING_STOP')
                return True
        else:
            if bar['low'] <= trade.trailing_stop:
                exit_price = min(trade.trailing_stop, bar['open'])
                self._close_trade(bar_idx, exit_price, 'TRAILING_STOP')
                return True

        return False

    def _close_trade(self, bar_idx: int, exit_price: float, reason: str):
        """Close the active trade and record results."""
        trade = self.active_trade
        bar = self.df.iloc[bar_idx]

        trade.exit_bar = bar_idx
        trade.exit_time = bar['timestamp']
        trade.exit_price = exit_price
        trade.exit_reason = reason

        # Calculate P&L
        if trade.direction == 'short':
            raw_pnl = (trade.entry_price - exit_price) * trade.position_size
        else:
            raw_pnl = (exit_price - trade.entry_price) * trade.position_size

        # Calculate fees
        entry_value = trade.entry_price * trade.position_size
        exit_value = exit_price * trade.position_size
        trade.fees = (entry_value + exit_value) * (FEE_ROUND_TRIP / 2)
        # Add slippage
        trade.fees += (entry_value + exit_value) * SLIPPAGE_PER_SIDE

        trade.pnl = raw_pnl - trade.fees

        # R-multiple
        risk_amount = trade.risk_per_unit * trade.position_size
        trade.r_multiple = trade.pnl / risk_amount if risk_amount > 0 else 0

        # Update equity
        self.equity += trade.pnl

        # Record trade
        self.trades.append(trade)
        self.active_trade = None

        # Update circuit breaker
        self.circuit_breaker.update_after_trade(trade, self.equity, bar_idx)

    def _calculate_stop_loss(self, broken_level: float, direction: str, atr: float) -> float:
        """
        Calculate stop loss price.
        Stop is placed beyond the broken structural level.
        """
        if STOP_LOSS_FIXED_PCT is not None:
            buffer = broken_level * STOP_LOSS_FIXED_PCT
        else:
            buffer = atr * STOP_LOSS_ATR_MULTIPLE

        if direction == 'short':
            # Stop above the broken support level
            return broken_level + buffer
        else:
            # Stop below the broken resistance level
            return broken_level - buffer

    def _calculate_position_size(self, risk_per_unit: float) -> float:
        """
        Calculate position size based on risk per trade.
        Risk amount = equity * RISK_PER_TRADE
        Position size = risk_amount / risk_per_unit
        """
        risk_amount = self.equity * RISK_PER_TRADE
        if risk_per_unit <= 0:
            return 0.0
        position_size = risk_amount / risk_per_unit
        return position_size

    def _check_structural_spacing(self, entry_price: float, risk_per_unit: float,
                                    direction: str, levels: dict) -> bool:
        """
        Structural spacing filter.
        Ensures minimum 1.5R of clear space between entry and next structural level
        in the trade's direction.
        """
        if direction == 'short':
            # Find nearest support below entry
            next_level = find_next_structural_level(levels['lows'], entry_price, 'short')
            if next_level is not None:
                space = entry_price - next_level
                if space < risk_per_unit * MIN_STRUCTURAL_SPACING_R:
                    return False
        else:
            # Find nearest resistance above entry
            next_level = find_next_structural_level(levels['highs'], entry_price, 'long')
            if next_level is not None:
                space = next_level - entry_price
                if space < risk_per_unit * MIN_STRUCTURAL_SPACING_R:
                    return False

        return True

    def _get_usable_levels_at_bar(self, bar_idx: int) -> dict:
        """Get structural levels usable at a specific bar index."""
        pivots_df = self.df[['pivot_high', 'pivot_low',
                             'pivot_high_usable_bar', 'pivot_low_usable_bar']]
        return get_usable_structural_levels(pivots_df, bar_idx)

    def _is_level_broken(self, level_price: float, tolerance: float = 0) -> bool:
        """Check if a level has already been broken in a previous trade."""
        for bl in self.broken_levels:
            if abs(bl - level_price) <= tolerance:
                return True
        return False

