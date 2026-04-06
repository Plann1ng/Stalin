"""
Stalin Trading System — Main Entry Point
Run this file to execute the full backtest.

Usage:
    python main.py                          # Run with default settings
    python main.py --data path/to/data.csv  # Specify data file
    python main.py --data-dir path/to/dir   # Batch run all SYMBOL_TIMEFRAME.csv files
    python main.py --fetch                  # Fetch data from exchange first
    python main.py --visualize              # Generate charts after backtest
"""

import argparse
import sys
import os
import re
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_FILE, INITIAL_CAPITAL, RESULTS_DIR
from backtest import run_backtest


def _infer_symbol_timeframe(path: str):
    name = Path(path).name
    m = re.match(r'^(.+)_([0-9]+[mhdw])\.csv$', name, re.IGNORECASE)
    if not m:
        return None, None
    return m.group(1).upper(), m.group(2).lower()


def _build_summary_row(file_path: str, symbol: str, timeframe: str, results: dict) -> dict:
    trades = results.get('total_trades', 0)
    return {
        'file': file_path,
        'symbol': symbol,
        'timeframe': timeframe,
        'bars': results.get('bars', 0),
        'total_trades': trades,
        'total_pnl': results.get('total_pnl', 0.0),
        'total_r': results.get('total_r', 0.0),
        'max_drawdown_pct': results.get('max_drawdown_pct', 0.0),
        'max_drawdown_r': results.get('max_drawdown_r', 0.0),
        'win_rate': results.get('win_rate', 0.0),
        'profit_factor': results.get('profit_factor', 0.0 if trades else 0.0),
        'final_equity': results.get('final_equity', results.get('initial_capital', 0.0)),
        'duration_days': results.get('duration_days', 0),
        'r_per_year': results.get('r_per_year', 0.0),
        'status': 'ok' if 'error' not in results else 'error',
        'error': results.get('error', ''),
    }


def _run_directory_mode(args):
    data_dir = Path(args.data_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    csv_files = sorted(data_dir.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {data_dir}")

    summary_rows = []
    capital = args.capital or INITIAL_CAPITAL

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for f in csv_files:
        symbol, timeframe = _infer_symbol_timeframe(str(f))
        if symbol is None or timeframe is None:
            print(f"[SKIP] {f.name} (filename must match SYMBOL_TIMEFRAME.csv)")
            continue

        if args.symbol and symbol != args.symbol.upper():
            continue
        if args.timeframe and timeframe != args.timeframe.lower():
            continue

        print(f"[RUN ] {f.name} | symbol={symbol} timeframe={timeframe}")

        try:
            results = run_backtest(
                data_file=str(f),
                initial_capital=capital,
                timeframe=timeframe,
                save=not args.no_save,
                output_tag=f"{symbol}_{timeframe}"
            )
            summary_rows.append(_build_summary_row(str(f), symbol, timeframe, results))
            print(f"[DONE] {f.name} | trades={results.get('total_trades', 0)} pnl={results.get('total_pnl', 0):.2f}")
        except Exception as exc:
            print(f"[FAIL] {f.name} | {exc}")
            summary_rows.append({
                'file': str(f),
                'symbol': symbol,
                'timeframe': timeframe,
                'bars': 0,
                'total_trades': 0,
                'total_pnl': 0.0,
                'total_r': 0.0,
                'max_drawdown_pct': 0.0,
                'max_drawdown_r': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'final_equity': capital,
                'duration_days': 0,
                'r_per_year': 0.0,
                'status': 'error',
                'error': str(exc),
            })
            continue

    summary_df = pd.DataFrame(summary_rows)
    aggregate_path = Path(RESULTS_DIR) / 'aggregate_summary.csv'
    summary_df.to_csv(aggregate_path, index=False)
    print(f"Aggregate summary saved to: {aggregate_path}")

    return {'summary_file': str(aggregate_path), 'rows': len(summary_df)}


def main():
    parser = argparse.ArgumentParser(description='Stalin Trading System — Backtest')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to OHLCV CSV data file')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to directory of SYMBOL_TIMEFRAME.csv files')
    parser.add_argument('--symbol', type=str, default=None,
                        help='Optional symbol filter for --data-dir mode (e.g., BTCUSDT)')
    parser.add_argument('--timeframe', type=str, default=None,
                        help='Optional timeframe filter for --data-dir mode (e.g., 5m)')
    parser.add_argument('--capital', type=float, default=None,
                        help='Initial capital for backtest')
    parser.add_argument('--fetch', action='store_true',
                        help='Fetch data from exchange before backtesting')
    parser.add_argument('--exchange', type=str, default='binance',
                        help='Exchange to fetch data from (default: binance)')
    parser.add_argument('--since', type=str, default='2019-01-01',
                        help='Start date for data fetch (default: 2019-01-01)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization charts')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save per-run result files')

    args = parser.parse_args()

    # Backward-compatible convenience: if a directory is passed to --data,
    # treat it as --data-dir to avoid IsADirectoryError.
    if args.data and os.path.isdir(args.data) and not args.data_dir:
        args.data_dir = args.data
        args.data = None

    # Fetch data if requested
    if args.fetch:
        from data_loader import fetch_from_exchange
        fetch_from_exchange(
            exchange_id=args.exchange,
            since=args.since,
            save_path=args.data or DATA_FILE
        )

    # Directory mode
    if args.data_dir:
        return _run_directory_mode(args)

    # Single-file mode
    data_file = args.data or DATA_FILE
    capital = args.capital or INITIAL_CAPITAL
    _, inferred_tf = _infer_symbol_timeframe(data_file)
    timeframe = args.timeframe or inferred_tf

    results = run_backtest(
        data_file=data_file,
        initial_capital=capital,
        timeframe=timeframe,
        save=not args.no_save
    )

    # Visualize if requested
    if args.visualize:
        from visualize import plot_all

        # Load equity curve from results directory
        import glob
        eq_files = sorted(glob.glob(os.path.join(RESULTS_DIR, 'equity_*.csv')))
        equity_data = None
        if eq_files:
            equity_data = pd.read_csv(eq_files[-1]).to_dict('records')

        plot_all(results, equity_data)

    return results


if __name__ == "__main__":
    main()
