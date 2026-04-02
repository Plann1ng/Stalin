"""
Run the mean‑reversion scalping strategy on all CSV files in a given
directory.

This script is designed for the user to run locally on their own
machine where a directory of Binance OHLCV CSV files is available.
Each CSV file must follow the format:

```
open_time,open,high,low,close,volume,close_time,quote_asset_volume,number_of_trades,taker_buy_base_volume,taker_buy_quote_volume,ignore
```

Example row:

```
2023-01-10 00:00:00.000+00:00,0.2458,0.2459,0.2454,0.2456,274333.9,2023-01-10 00:04:59.999000+00:00,67383.19196,150,153592.8,37719.49392,0
```

The script loads each CSV, converts the timestamps to ``pandas``
``datetime`` objects, selects the Open/High/Low/Close/Volume columns,
and runs the mean‑reversion strategy discovered in the research (10‑bar
window, 1.5× standard deviation for the Bollinger bands, maximum hold
of 6 bars).  It prints a summary of the number of trades, win rate,
average profit per trade, total return and maximum drawdown for each
file.

Users should adjust ``DATA_DIR`` to point to the directory containing
their CSV files.
"""

import os
import glob
import math
from typing import Tuple, List

import pandas as pd

# Adjust the import path so that this script can locate the sibling
# research module when run standalone.  When you run this script on
# your own machine, the ``research.py`` file should be located in the
# same directory as this script.
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from research import MeanReversionStrategy, run_backtest, evaluate_trades


# Path to the directory containing your CSV files.  Update this path
# before running the script on your machine.
DATA_DIR = "/Users/kasuya/binance_merged"


def load_csv(path: str) -> pd.DataFrame:
    """Load a Binance OHLCV CSV into a DataFrame.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    DataFrame
        DataFrame with columns ``timestamp``, ``open``, ``high``, ``low``,
        ``close`` and ``volume`` sorted by timestamp.
    """
    # Read the CSV using pandas.  Some files include a header row with
    # column names; others may omit the header and store numeric values
    # only.  Attempt to read with header=0; if 'open_time' is not in
    # columns, fall back to specifying column names explicitly.
    df = pd.read_csv(path, low_memory=False)
    expected_cols = [
        'open_time',
        'open',
        'high',
        'low',
        'close',
        'volume',
        'close_time',
        'quote_asset_volume',
        'number_of_trades',
        'taker_buy_base_volume',
        'taker_buy_quote_volume',
        'ignore',
    ]
    if 'open_time' not in df.columns:
        # No header present; treat first row as data and assign names
        df = pd.read_csv(path, header=None, names=expected_cols, low_memory=False)
    # Convert open_time to datetime.  It may be milliseconds since epoch
    # (as int) or ISO format string.  pandas.to_datetime can handle both.
    df['timestamp'] = pd.to_datetime(df['open_time'], utc=True)
    # Keep only the OHLCV columns and cast numeric types
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df[['open', 'high', 'low', 'close', 'volume']] = df[
        ['open', 'high', 'low', 'close', 'volume']
    ].astype(float)
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def process_file(path: str) -> Tuple[str, dict]:
    """Process a single CSV file and return metrics.

    Returns
    -------
    tuple
        Tuple of (filename, metrics dict).
    """
    df = load_csv(path)
    strategy = MeanReversionStrategy(window=10, std_mult=1.5, max_hold=6)
    trades = run_backtest(df, strategy)
    metrics = evaluate_trades(trades)
    return os.path.basename(path), metrics


def main():
    pattern = os.path.join(DATA_DIR, '*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No CSV files found in {DATA_DIR}")
        return
    print(f"Processing {len(files)} files in {DATA_DIR}\n")
    header = (
        f"{'File':<25} {'Trades':>8} {'WinRate':>8} {'AvgProfit':>10} "
        f"{'TotalRet':>10} {'MaxDD':>10}"
    )
    print(header)
    print('-' * len(header))
    for path in files:
        fname, metrics = process_file(path)
        if metrics['num_trades'] == 0:
            print(f"{fname:<25} {0:>8} {'N/A':>8} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
            continue
        print(
            f"{fname:<25} {metrics['num_trades']:>8} "
            f"{metrics['win_rate']*100:>7.2f}% "
            f"{metrics['avg_profit']*100:>9.2f}% "
            f"{metrics['total_return']*100:>9.2f}% "
            f"{metrics['max_drawdown']*100:>9.2f}%"
        )


if __name__ == '__main__':
    main()