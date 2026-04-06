"""
Stalin Trading System — Backtest Engine
Runs the strategy and produces comprehensive analytics.
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from typing import List

from config import (
    INITIAL_CAPITAL, RESULTS_DIR, IS_OOS_SPLIT_DATE,
    get_timeframe_overrides
)
from data_loader import load_csv
from indicators import prepare_indicators
from strategy import StalinStrategy, Trade


def run_backtest(data_file: str = None, initial_capital: float = None, timeframe: str = None,
                 save: bool = True, output_tag: str = None) -> dict:
    """
    Run full backtest and return results dictionary.
    """
    initial_capital = initial_capital or INITIAL_CAPITAL

    # Apply conservative timeframe overrides for bar-count parameters
    if timeframe:
        overrides = get_timeframe_overrides(timeframe)
        if overrides:
            import indicators
            import strategy
            if "VOLATILITY_LOOKBACK_4H_BARS" in overrides:
                indicators.VOLATILITY_LOOKBACK_4H_BARS = overrides["VOLATILITY_LOOKBACK_4H_BARS"]
            if "MAX_TRADE_DURATION_BARS" in overrides:
                strategy.MAX_TRADE_DURATION_BARS = overrides["MAX_TRADE_DURATION_BARS"]
            if "CIRCUIT_BREAKER_COOLDOWN_BARS" in overrides:
                strategy.CIRCUIT_BREAKER_COOLDOWN_BARS = overrides["CIRCUIT_BREAKER_COOLDOWN_BARS"]
            print(f"Applying timeframe overrides for {timeframe}: {overrides}")

    # Load and prepare data
    print("=" * 70)
    print("STALIN TRADING SYSTEM — BACKTEST")
    print("=" * 70)

    df = load_csv(data_file)
    print(f"\nComputing indicators...")
    df = prepare_indicators(df)
    print(f"Indicators computed. {len(df)} bars ready.")

    # Run strategy
    print(f"\nRunning strategy...")
    strategy = StalinStrategy(df, initial_capital)
    trades = strategy.run()
    print(f"Strategy complete. {len(trades)} trades executed.")

    # Analyze results
    results = analyze_trades(trades, strategy.equity_curve, df, initial_capital)
    results['bars'] = len(df)

    # Print report
    print_report(results)

    # Save results
    if save:
        save_results(results, trades, strategy.equity_curve, output_tag=output_tag)

    return results


def analyze_trades(trades: List[Trade], equity_curve: list,
                    df: pd.DataFrame, initial_capital: float) -> dict:
    """Comprehensive trade analysis."""
    results = {
        'total_trades': len(trades),
        'initial_capital': initial_capital,
    }

    if len(trades) == 0:
        results['error'] = 'No trades generated'
        return results

    # Convert trades to DataFrame for analysis
    trade_data = []
    for t in trades:
        trade_data.append({
            'entry_bar': t.entry_bar,
            'entry_time': t.entry_time,
            'entry_price': t.entry_price,
            'exit_bar': t.exit_bar,
            'exit_time': t.exit_time,
            'exit_price': t.exit_price,
            'direction': t.direction,
            'regime': t.regime,
            'exit_reason': t.exit_reason,
            'pnl': t.pnl,
            'r_multiple': t.r_multiple,
            'fees': t.fees,
            'risk_per_unit': t.risk_per_unit,
            'position_size': t.position_size,
            'bars_in_trade': t.bars_in_trade,
            'max_favorable_excursion': t.max_favorable_excursion,
            'trailing_activated': t.trailing_activated,
            'broken_level': t.broken_level,
        })

    trades_df = pd.DataFrame(trade_data)

    # Basic stats
    results['total_trades'] = len(trades_df)
    results['winning_trades'] = len(trades_df[trades_df['r_multiple'] > 0])
    results['losing_trades'] = len(trades_df[trades_df['r_multiple'] <= 0])
    results['win_rate'] = results['winning_trades'] / results['total_trades'] if results['total_trades'] > 0 else 0

    # R-multiple stats
    results['total_r'] = trades_df['r_multiple'].sum()
    results['avg_r'] = trades_df['r_multiple'].mean()
    results['median_r'] = trades_df['r_multiple'].median()
    results['std_r'] = trades_df['r_multiple'].std()
    results['max_r'] = trades_df['r_multiple'].max()
    results['min_r'] = trades_df['r_multiple'].min()
    results['avg_winner_r'] = trades_df.loc[trades_df['r_multiple'] > 0, 'r_multiple'].mean() if results['winning_trades'] > 0 else 0
    results['avg_loser_r'] = trades_df.loc[trades_df['r_multiple'] <= 0, 'r_multiple'].mean() if results['losing_trades'] > 0 else 0

    # Expectancy
    results['expectancy_r'] = results['avg_r']
    results['profit_factor'] = (
        trades_df.loc[trades_df['r_multiple'] > 0, 'r_multiple'].sum() /
        abs(trades_df.loc[trades_df['r_multiple'] <= 0, 'r_multiple'].sum())
        if abs(trades_df.loc[trades_df['r_multiple'] <= 0, 'r_multiple'].sum()) > 0 else float('inf')
    )

    # P&L stats
    results['total_pnl'] = trades_df['pnl'].sum()
    results['total_fees'] = trades_df['fees'].sum()
    results['final_equity'] = initial_capital + results['total_pnl']
    results['total_return_pct'] = (results['final_equity'] / initial_capital - 1) * 100

    # Time-based stats
    if trades_df['entry_time'].iloc[0] is not None and trades_df['exit_time'].iloc[-1] is not None:
        first_trade = trades_df['entry_time'].iloc[0]
        last_trade = trades_df['exit_time'].iloc[-1]
        duration_days = (last_trade - first_trade).days
        duration_years = duration_days / 365.25 if duration_days > 0 else 1
        results['duration_days'] = duration_days
        results['duration_years'] = duration_years
        results['r_per_year'] = results['total_r'] / duration_years
        results['trades_per_year'] = results['total_trades'] / duration_years

    # Drawdown analysis
    equity_series = pd.Series([initial_capital] + [initial_capital + trades_df['pnl'].iloc[:i+1].sum()
                                                     for i in range(len(trades_df))])
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    results['max_drawdown_pct'] = abs(drawdown.min()) * 100
    results['max_drawdown_r'] = _compute_r_drawdown(trades_df['r_multiple'].values)

    # By exit reason
    results['by_exit_reason'] = {}
    for reason in trades_df['exit_reason'].unique():
        subset = trades_df[trades_df['exit_reason'] == reason]
        results['by_exit_reason'][reason] = {
            'count': len(subset),
            'pct_of_trades': len(subset) / len(trades_df) * 100,
            'avg_r': subset['r_multiple'].mean(),
            'total_r': subset['r_multiple'].sum(),
            'win_rate': len(subset[subset['r_multiple'] > 0]) / len(subset) * 100 if len(subset) > 0 else 0
        }

    # By direction
    results['by_direction'] = {}
    for direction in trades_df['direction'].unique():
        subset = trades_df[trades_df['direction'] == direction]
        results['by_direction'][direction] = {
            'count': len(subset),
            'avg_r': subset['r_multiple'].mean(),
            'total_r': subset['r_multiple'].sum(),
            'win_rate': len(subset[subset['r_multiple'] > 0]) / len(subset) * 100 if len(subset) > 0 else 0
        }

    # By regime
    results['by_regime'] = {}
    for regime in trades_df['regime'].unique():
        subset = trades_df[trades_df['regime'] == regime]
        results['by_regime'][regime] = {
            'count': len(subset),
            'avg_r': subset['r_multiple'].mean(),
            'total_r': subset['r_multiple'].sum(),
            'win_rate': len(subset[subset['r_multiple'] > 0]) / len(subset) * 100 if len(subset) > 0 else 0
        }

    # Trade duration stats
    results['avg_bars_in_trade'] = trades_df['bars_in_trade'].mean()
    results['median_bars_in_trade'] = trades_df['bars_in_trade'].median()
    results['max_bars_in_trade'] = trades_df['bars_in_trade'].max()

    # MFE analysis (how much profit was left on the table)
    results['avg_mfe_r'] = trades_df['max_favorable_excursion'].mean()
    results['mfe_vs_exit'] = (trades_df['max_favorable_excursion'] - trades_df['r_multiple']).mean()

    # Trailing stop stats
    results['trailing_activated_pct'] = trades_df['trailing_activated'].mean() * 100

    # Consecutive losses
    r_values = trades_df['r_multiple'].values
    results['max_consecutive_losses'] = _max_consecutive(r_values, lambda x: x <= 0)
    results['max_consecutive_wins'] = _max_consecutive(r_values, lambda x: x > 0)

    # In-sample / Out-of-sample split
    if IS_OOS_SPLIT_DATE:
        split_date = pd.to_datetime(IS_OOS_SPLIT_DATE)
        entry_tz = getattr(trades_df['entry_time'].dt, 'tz', None)
        if entry_tz is not None and split_date.tzinfo is None:
            split_date = split_date.tz_localize(entry_tz)
        elif entry_tz is None and split_date.tzinfo is not None:
            split_date = split_date.tz_localize(None)

        is_trades = trades_df[trades_df['entry_time'] < split_date]
        oos_trades = trades_df[trades_df['entry_time'] >= split_date]

        results['in_sample'] = {
            'trades': len(is_trades),
            'total_r': is_trades['r_multiple'].sum() if len(is_trades) > 0 else 0,
            'avg_r': is_trades['r_multiple'].mean() if len(is_trades) > 0 else 0,
            'win_rate': len(is_trades[is_trades['r_multiple'] > 0]) / len(is_trades) * 100 if len(is_trades) > 0 else 0
        }
        results['out_of_sample'] = {
            'trades': len(oos_trades),
            'total_r': oos_trades['r_multiple'].sum() if len(oos_trades) > 0 else 0,
            'avg_r': oos_trades['r_multiple'].mean() if len(oos_trades) > 0 else 0,
            'win_rate': len(oos_trades[oos_trades['r_multiple'] > 0]) / len(oos_trades) * 100 if len(oos_trades) > 0 else 0
        }

    # Store trades DataFrame for further analysis
    results['trades_df'] = trades_df

    return results


def _compute_r_drawdown(r_values: np.ndarray) -> float:
    """Compute maximum drawdown in R-multiples."""
    cumulative = np.cumsum(r_values)
    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative - peak
    return abs(drawdown.min()) if len(drawdown) > 0 else 0


def _max_consecutive(values: np.ndarray, condition_fn) -> int:
    """Find maximum consecutive elements satisfying a condition."""
    max_count = 0
    current_count = 0
    for v in values:
        if condition_fn(v):
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0
    return max_count


def print_report(results: dict):
    """Print formatted backtest report."""
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    if 'error' in results:
        print(f"\nERROR: {results['error']}")
        return

    print(f"\n{'OVERVIEW':=^50}")
    print(f"  Total Trades:          {results['total_trades']}")
    print(f"  Winning Trades:        {results['winning_trades']} ({results['win_rate']*100:.1f}%)")
    print(f"  Losing Trades:         {results['losing_trades']}")
    if 'duration_years' in results:
        print(f"  Duration:              {results['duration_days']} days ({results['duration_years']:.1f} years)")
        print(f"  Trades/Year:           {results['trades_per_year']:.1f}")

    print(f"\n{'R-MULTIPLE ANALYSIS':=^50}")
    print(f"  Total R:               {results['total_r']:+.2f}")
    print(f"  Average R:             {results['avg_r']:+.3f}")
    print(f"  Median R:              {results['median_r']:+.3f}")
    print(f"  Std Dev R:             {results['std_r']:.3f}")
    print(f"  Best Trade:            {results['max_r']:+.2f}R")
    print(f"  Worst Trade:           {results['min_r']:+.2f}R")
    print(f"  Avg Winner:            {results['avg_winner_r']:+.2f}R")
    print(f"  Avg Loser:             {results['avg_loser_r']:+.2f}R")
    print(f"  Expectancy:            {results['expectancy_r']:+.3f}R")
    print(f"  Profit Factor:         {results['profit_factor']:.2f}")
    if 'r_per_year' in results:
        print(f"  R/Year:                {results['r_per_year']:+.1f}")

    print(f"\n{'P&L':=^50}")
    print(f"  Initial Capital:       ${results['initial_capital']:,.0f}")
    print(f"  Final Equity:          ${results['final_equity']:,.0f}")
    print(f"  Total P&L:             ${results['total_pnl']:+,.0f}")
    print(f"  Total Fees:            ${results['total_fees']:,.0f}")
    print(f"  Total Return:          {results['total_return_pct']:+.1f}%")

    print(f"\n{'RISK METRICS':=^50}")
    print(f"  Max Drawdown (%):      {results['max_drawdown_pct']:.1f}%")
    print(f"  Max Drawdown (R):      {results['max_drawdown_r']:.1f}R")
    print(f"  Max Consec. Losses:    {results['max_consecutive_losses']}")
    print(f"  Max Consec. Wins:      {results['max_consecutive_wins']}")

    print(f"\n{'TRADE DURATION':=^50}")
    print(f"  Avg Bars in Trade:     {results['avg_bars_in_trade']:.1f}")
    print(f"  Median Bars:           {results['median_bars_in_trade']:.0f}")
    print(f"  Max Bars:              {results['max_bars_in_trade']}")

    print(f"\n{'EXIT ANALYSIS':=^50}")
    print(f"  Avg MFE:               {results['avg_mfe_r']:.2f}R")
    print(f"  Avg Profit Left:       {results['mfe_vs_exit']:.2f}R (MFE - Exit)")
    print(f"  Trailing Activated:    {results['trailing_activated_pct']:.1f}%")

    print(f"\n  {'Exit Reason':<20} {'Count':>6} {'%':>7} {'Avg R':>8} {'Total R':>9} {'Win%':>7}")
    print(f"  {'-'*58}")
    for reason, stats in results['by_exit_reason'].items():
        print(f"  {reason:<20} {stats['count']:>6} {stats['pct_of_trades']:>6.1f}% {stats['avg_r']:>+7.3f} {stats['total_r']:>+8.2f} {stats['win_rate']:>6.1f}%")

    if results.get('by_direction'):
        print(f"\n{'BY DIRECTION':=^50}")
        for direction, stats in results['by_direction'].items():
            print(f"  {direction.upper():<10} Trades: {stats['count']:>4}  Avg R: {stats['avg_r']:>+.3f}  Total R: {stats['total_r']:>+.1f}  Win: {stats['win_rate']:.1f}%")

    if results.get('by_regime'):
        print(f"\n{'BY REGIME':=^50}")
        for regime, stats in results['by_regime'].items():
            print(f"  {regime:<12} Trades: {stats['count']:>4}  Avg R: {stats['avg_r']:>+.3f}  Total R: {stats['total_r']:>+.1f}  Win: {stats['win_rate']:.1f}%")

    if results.get('in_sample'):
        print(f"\n{'IN-SAMPLE vs OUT-OF-SAMPLE':=^50}")
        is_r = results['in_sample']
        oos_r = results['out_of_sample']
        print(f"  In-Sample:    Trades: {is_r['trades']:>4}  Avg R: {is_r['avg_r']:>+.3f}  Total R: {is_r['total_r']:>+.1f}  Win: {is_r['win_rate']:.1f}%")
        print(f"  Out-of-Sample: Trades: {oos_r['trades']:>4}  Avg R: {oos_r['avg_r']:>+.3f}  Total R: {oos_r['total_r']:>+.1f}  Win: {oos_r['win_rate']:.1f}%")

    print("\n" + "=" * 70)


def save_results(results: dict, trades: List[Trade], equity_curve: list, output_tag: str = None):
    """Save results to files."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{output_tag}" if output_tag else ""

    # Save trades CSV
    if 'trades_df' in results:
        trades_path = os.path.join(RESULTS_DIR, f"trades_{timestamp}{suffix}.csv")
        results['trades_df'].to_csv(trades_path, index=False)
        print(f"\nTrades saved to: {trades_path}")

    # Save equity curve
    if equity_curve:
        eq_df = pd.DataFrame(equity_curve)
        eq_path = os.path.join(RESULTS_DIR, f"equity_{timestamp}{suffix}.csv")
        eq_df.to_csv(eq_path, index=False)
        print(f"Equity curve saved to: {eq_path}")

    # Save summary JSON (exclude non-serializable items)
    summary = {k: v for k, v in results.items()
               if k != 'trades_df' and not isinstance(v, pd.DataFrame)}
    # Convert numpy types
    summary = _convert_numpy(summary)
    summary_path = os.path.join(RESULTS_DIR, f"summary_{timestamp}{suffix}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved to: {summary_path}")


def _convert_numpy(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


if __name__ == "__main__":
    run_backtest()

