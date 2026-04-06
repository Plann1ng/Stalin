"""
Stalin Trading System — Visualization
Charts for equity curve, trade distribution, and analysis.
"""

import pandas as pd
import numpy as np
import os

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not installed. Visualization disabled.")

from config import RESULTS_DIR


def plot_all(results: dict, equity_curve_data: list = None, df: pd.DataFrame = None):
    """Generate all charts."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available. Skipping visualization.")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if 'trades_df' not in results or len(results['trades_df']) == 0:
        print("No trades to visualize.")
        return

    trades_df = results['trades_df']

    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    fig.suptitle('Stalin Trading System — Backtest Results', fontsize=16, fontweight='bold')

    # 1. Equity Curve
    _plot_equity_curve(axes[0, 0], trades_df, results['initial_capital'])

    # 2. R-Multiple Distribution
    _plot_r_distribution(axes[0, 1], trades_df)

    # 3. Cumulative R
    _plot_cumulative_r(axes[1, 0], trades_df)

    # 4. By Exit Reason
    _plot_by_exit_reason(axes[1, 1], results)

    # 5. Trade Duration vs R
    _plot_duration_vs_r(axes[2, 0], trades_df)

    # 6. Monthly R Heatmap (simplified as bar chart)
    _plot_monthly_r(axes[2, 1], trades_df)

    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, 'backtest_charts.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nCharts saved to: {chart_path}")


def _plot_equity_curve(ax, trades_df, initial_capital):
    """Plot equity curve over time."""
    equity = [initial_capital]
    for _, trade in trades_df.iterrows():
        equity.append(equity[-1] + trade['pnl'])

    times = [trades_df['entry_time'].iloc[0]] + list(trades_df['exit_time'])

    ax.plot(times, equity, 'b-', linewidth=1.5)
    ax.fill_between(times, initial_capital, equity, alpha=0.1,
                     where=[e >= initial_capital for e in equity], color='green')
    ax.fill_between(times, initial_capital, equity, alpha=0.1,
                     where=[e < initial_capital for e in equity], color='red')
    ax.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Equity Curve')
    ax.set_ylabel('Equity ($)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)


def _plot_r_distribution(ax, trades_df):
    """Plot histogram of R-multiples."""
    r_values = trades_df['r_multiple'].values
    bins = np.arange(
        max(r_values.min() - 0.5, -5),
        min(r_values.max() + 0.5, 15),
        0.25
    )
    colors = ['green' if r > 0 else 'red' for r in r_values]

    ax.hist(r_values, bins=bins, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.axvline(x=r_values.mean(), color='orange', linestyle='--',
               label=f'Mean: {r_values.mean():.2f}R')
    ax.set_title('R-Multiple Distribution')
    ax.set_xlabel('R-Multiple')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_cumulative_r(ax, trades_df):
    """Plot cumulative R over trades."""
    cum_r = trades_df['r_multiple'].cumsum()
    trade_nums = range(1, len(trades_df) + 1)

    ax.plot(trade_nums, cum_r, 'b-', linewidth=1.5)
    ax.fill_between(trade_nums, 0, cum_r,
                     where=cum_r >= 0, alpha=0.1, color='green')
    ax.fill_between(trade_nums, 0, cum_r,
                     where=cum_r < 0, alpha=0.1, color='red')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Cumulative R-Multiple')
    ax.set_xlabel('Trade Number')
    ax.set_ylabel('Cumulative R')
    ax.grid(True, alpha=0.3)


def _plot_by_exit_reason(ax, results):
    """Plot performance by exit reason."""
    reasons = results['by_exit_reason']
    names = list(reasons.keys())
    avg_rs = [reasons[r]['avg_r'] for r in names]
    counts = [reasons[r]['count'] for r in names]

    colors = ['green' if r > 0 else 'red' for r in avg_rs]
    bars = ax.bar(names, avg_rs, color=colors, alpha=0.7, edgecolor='white')

    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'n={count}', ha='center', va='bottom', fontsize=9)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title('Average R by Exit Reason')
    ax.set_ylabel('Average R-Multiple')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')


def _plot_duration_vs_r(ax, trades_df):
    """Scatter plot of trade duration vs R-multiple."""
    colors = ['green' if r > 0 else 'red' for r in trades_df['r_multiple']]
    ax.scatter(trades_df['bars_in_trade'], trades_df['r_multiple'],
               c=colors, alpha=0.5, s=30, edgecolors='white', linewidth=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title('Trade Duration vs R-Multiple')
    ax.set_xlabel('Bars in Trade')
    ax.set_ylabel('R-Multiple')
    ax.grid(True, alpha=0.3)


def _plot_monthly_r(ax, trades_df):
    """Plot monthly R performance."""
    trades_df = trades_df.copy()
    trades_df['month'] = trades_df['entry_time'].dt.to_period('M')
    monthly = trades_df.groupby('month')['r_multiple'].sum()

    colors = ['green' if r > 0 else 'red' for r in monthly.values]
    ax.bar(range(len(monthly)), monthly.values, color=colors, alpha=0.7, edgecolor='white')
    ax.set_title('Monthly R Performance')
    ax.set_ylabel('Total R')

    # Show every Nth label to avoid crowding
    n_labels = min(12, len(monthly))
    step = max(1, len(monthly) // n_labels)
    ax.set_xticks(range(0, len(monthly), step))
    ax.set_xticklabels([str(monthly.index[i]) for i in range(0, len(monthly), step)],
                        rotation=45, fontsize=8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

