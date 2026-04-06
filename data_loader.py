"""
Stalin Trading System — Data Loading & Preparation
Handles loading OHLCV data from CSV or exchange, with validation.
"""

import pandas as pd
import numpy as np
import os
from config import (
    DATA_FILE, TIMEFRAME, SYMBOL,
    BACKTEST_START_DATE, BACKTEST_END_DATE
)


def load_csv(filepath: str = None) -> pd.DataFrame:
    """
    Load OHLCV data from CSV file.
    Expected columns: timestamp/date, open, high, low, close, volume
    """
    filepath = filepath or DATA_FILE

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            f"Please provide a CSV with columns: timestamp, open, high, low, close, volume"
        )

    df = pd.read_csv(filepath)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Handle various timestamp column names
    time_col = None
    for candidate in ['timestamp', 'date', 'datetime', 'time', 'open_time']:
        if candidate in df.columns:
            time_col = candidate
            break

    if time_col is None:
        raise ValueError("No timestamp column found. Expected one of: timestamp, date, datetime, time, open_time")

    df['timestamp'] = pd.to_datetime(df[time_col])
    if time_col != 'timestamp':
        df = df.drop(columns=[time_col])

    # Ensure required columns exist
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Remove duplicates
    df = df.drop_duplicates(subset='timestamp', keep='last').reset_index(drop=True)

    # Filter date range
    if BACKTEST_START_DATE:
        df = df[df['timestamp'] >= pd.to_datetime(BACKTEST_START_DATE)]
    if BACKTEST_END_DATE:
        df = df[df['timestamp'] <= pd.to_datetime(BACKTEST_END_DATE)]

    df = df.reset_index(drop=True)

    # Validate data integrity
    _validate_ohlcv(df)

    return df


def _validate_ohlcv(df: pd.DataFrame):
    """Validate OHLCV data integrity."""
    # Check for NaN values
    nan_counts = df[['open', 'high', 'low', 'close', 'volume']].isna().sum()
    if nan_counts.any():
        print(f"WARNING: NaN values found:\n{nan_counts[nan_counts > 0]}")
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

    # Check high >= low
    invalid = df[df['high'] < df['low']]
    if len(invalid) > 0:
        print(f"WARNING: {len(invalid)} bars with high < low detected, fixing...")
        df.loc[df['high'] < df['low'], ['high', 'low']] = df.loc[
            df['high'] < df['low'], ['low', 'high']
        ].values

    # Check high >= open, close and low <= open, close
    df['high'] = df[['high', 'open', 'close']].max(axis=1)
    df['low'] = df[['low', 'open', 'close']].min(axis=1)

    print(f"Data loaded: {len(df)} bars from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")


def fetch_from_exchange(exchange_id: str = 'binance', symbol: str = None,
                         timeframe: str = None, since: str = '2019-01-01',
                         save_path: str = None) -> pd.DataFrame:
    """
    Fetch OHLCV data from exchange using ccxt.
    Optional — requires ccxt installed.
    """
    try:
        import ccxt
    except ImportError:
        raise ImportError("ccxt not installed. Run: pip install ccxt")

    symbol = symbol or SYMBOL
    timeframe = timeframe or TIMEFRAME
    save_path = save_path or DATA_FILE

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({'enableRateLimit': True})

    since_ts = exchange.parse8601(f"{since}T00:00:00Z")
    all_ohlcv = []

    print(f"Fetching {symbol} {timeframe} data from {exchange_id}...")

    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_ts, limit=1000)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since_ts = ohlcv[-1][0] + 1
        if len(ohlcv) < 1000:
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset='timestamp', keep='last').reset_index(drop=True)

    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved {len(df)} bars to {save_path}")

    return df

