"""
Stalin Trading System — Configuration
All parameters centralized here for easy modification and optimization.
"""

# ============================================================================
# MARKET & TIMEFRAME
# ============================================================================
SYMBOL = "BTC/USDT"
TIMEFRAME = "4h"

# ============================================================================
# REGIME CLASSIFICATION
# ============================================================================
# SMA period for bull/bear cycle determination
REGIME_SMA_PERIOD = 200
# Only allow shorts (BEARCYCLE). Longs eliminated per analysis.
ALLOW_LONGS = False
ALLOW_SHORTS = True

# ============================================================================
# PIVOT DETECTION
# ============================================================================
# Number of bars left/right to confirm a pivot high/low
PIVOT_LEFT_BARS = 5
PIVOT_RIGHT_BARS = 5
# Confirmation delay: minimum bars AFTER pivot is confirmed before it can be used
PIVOT_CONFIRMATION_DELAY = 2
# Maximum number of structural levels to track
MAX_STRUCTURAL_LEVELS = 20

# ============================================================================
# ENTRY FILTERS
# ============================================================================
# Momentum confirmation: breakout candle must close in directional X% of its range
# For shorts: close must be in bottom 30% of candle range
# For longs: close must be in top 30% of candle range
MOMENTUM_THRESHOLD = 0.30

# Volatility regime filter: ATR(14) must be above this percentile of its 90-day range
ATR_PERIOD = 14
VOLATILITY_LOOKBACK_BARS = 90  # ~90 days on 4H = 540 bars, but we use 90-day calendar
VOLATILITY_LOOKBACK_4H_BARS = 540  # 90 days * 6 bars/day
VOLATILITY_PERCENTILE_THRESHOLD = 30  # 30th percentile minimum

# Structural spacing filter: minimum R-multiples of clear space to next level
MIN_STRUCTURAL_SPACING_R = 1.5

# ============================================================================
# RISK MANAGEMENT
# ============================================================================
# Risk per trade as fraction of equity
RISK_PER_TRADE = 0.01  # 1% risk per trade

# Stop loss: distance above broken level for shorts (in ATR multiples)
STOP_LOSS_ATR_MULTIPLE = 1.0
# Alternative: fixed percentage above/below entry
STOP_LOSS_FIXED_PCT = None  # Set to e.g. 0.02 for 2%, or None to use ATR-based

# ============================================================================
# EXIT LOGIC
# ============================================================================
# Structural trailing stop: trail behind Nth confirmed swing high (for shorts)
# or Nth confirmed swing low (for longs)
TRAILING_PIVOT_LEFT = 3
TRAILING_PIVOT_RIGHT = 3
# Number of confirmed pivots to trail behind (1 = most recent confirmed pivot)
TRAILING_PIVOT_COUNT = 1
# Minimum R-multiple profit before trailing stop activates
TRAILING_ACTIVATION_R = 0.5

# EXITGAP logic: gap threshold as fraction of ATR
EXITGAP_ATR_THRESHOLD = 1.5
# EXITGAP: require prior candle close confirmation (no lookahead)
EXITGAP_USE_PRIOR_CLOSE = True

# Maximum trade duration in bars (safety exit)
MAX_TRADE_DURATION_BARS = 200

# ============================================================================
# DRAWDOWN CIRCUIT BREAKER
# ============================================================================
# Halt trading after N consecutive losses
MAX_CONSECUTIVE_LOSSES = 5
# Halt trading after drawdown exceeds this fraction of peak equity
MAX_DRAWDOWN_FRACTION = 0.15  # 15% drawdown from peak
# Cooldown period after circuit breaker triggers (in bars)
CIRCUIT_BREAKER_COOLDOWN_BARS = 30  # ~5 days on 4H

# ============================================================================
# FEE MODEL
# ============================================================================
# Round-trip fees as fraction (taker fills on breakout entries)
FEE_ROUND_TRIP = 0.0008  # 0.08% round-trip (0.04% each way, taker)
# Slippage estimate per side as fraction
SLIPPAGE_PER_SIDE = 0.0002  # 0.02% per side

# ============================================================================
# BACKTEST SETTINGS
# ============================================================================
INITIAL_CAPITAL = 100000.0
DATA_FILE = "data/btc_4h.csv"  # Path to OHLCV data
RESULTS_DIR = "results/"

# Date range for backtest (None = use all available data)
BACKTEST_START_DATE = None  # e.g., "2020-01-01"
BACKTEST_END_DATE = None    # e.g., "2024-01-01"

# In-sample / Out-of-sample split date
IS_OOS_SPLIT_DATE = "2023-01-01"

