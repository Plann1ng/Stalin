"""
Stalin Trading System — Main Entry Point
Run this file to execute the full backtest.

Usage:
    python main.py                          # Run with default settings
    python main.py --data path/to/data.csv  # Specify data file
    python main.py --fetch                  # Fetch data from exchange first
    python main.py --visualize              # Generate charts after backtest
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_FILE, INITIAL_CAPITAL, RESULTS_DIR
from backtest import run_backtest


def main():
    parser = argparse.ArgumentParser(description='Stalin Trading System — Backtest')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to OHLCV CSV data file')
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
                        help='Do not save results to files')

    args = parser.parse_args()

    # Fetch data if requested
    if args.fetch:
        from data_loader import fetch_from_exchange
        fetch_from_exchange(
            exchange_id=args.exchange,
            since=args.since,
            save_path=args.data or DATA_FILE
        )

    # Run backtest
    data_file = args.data or DATA_FILE
    capital = args.capital or INITIAL_CAPITAL

    results = run_backtest(data_file=data_file, initial_capital=capital)

    # Visualize if requested
    if args.visualize:
        from visualize import plot_all
        from data_loader import load_csv
        from indicators import prepare_indicators

        # Load equity curve from results directory
        import glob
        eq_files = sorted(glob.glob(os.path.join(RESULTS_DIR, 'equity_*.csv')))
        equity_data = None
        if eq_files:
            import pandas as pd
            equity_data = pd.read_csv(eq_files[-1]).to_dict('records')

        plot_all(results, equity_data)

    return results


if __name__ == "__main__":
    main()

