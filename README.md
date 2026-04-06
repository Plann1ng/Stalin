# Stalin Trading System — v2.0

Structural breakout/breakdown strategy on BTC perpetual futures (4H timeframe).

## Key Changes from v1.0

1. **BULLCYCLE longs eliminated** — removes net-negative subsystem
2. **Structural trailing stop** — trails behind confirmed swing highs/lows instead of fixed/breakeven
3. **Momentum confirmation filter** — breakout candle must close in directional 30% of range
4. **Volatility regime filter** — ATR(14) must be above 30th percentile of 90-day range
5. **Structural spacing filter** — minimum 1.5R clear space to next level
6. **EXITGAP lookahead fix** — decisions based on prior bar close, execution at next open
7. **Pivot confirmation delay** — trades cannot be taken on the pivot bar itself
8. **Drawdown circuit breaker** — halts after 5 consecutive losses or 15% drawdown
9. **Realistic fees** — 0.08% round-trip taker fees + 0.02% slippage per side

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your BTC 4H OHLCV data in data/btc_4h.csv
#    Required columns: timestamp, open, high, low, close, volume

# 3. Run backtest
python main.py

# 4. Run with charts
python main.py --visualize

# 5. Fetch data from Binance (requires ccxt: pip install ccxt)
python main.py --fetch --visualize

