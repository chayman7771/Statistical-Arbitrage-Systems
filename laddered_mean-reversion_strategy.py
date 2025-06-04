"""
Statistical Arbitrage Laddered Pair Trading Engine
Author: Cameron Hayman

Description:
    - Cointegration-based pair selection
    - Laddered mean-reversion strategy with rolling z-score entry/exit
    - All trades logged (open/close/forced close)
    - Final positions force-closed at end of backtest
    - Summary table and trade logs always match
    - PnL in dollars (not percent return) per unit traded

Assumptions & Limitations:
    - No commissions, slippage, or borrow fees included. Real-world results will be lower.
    - No FX impact: All tickers S&P 500 (USD). For global, FX must be modeled.
    - PnL is in dollars per trade (not as a percent of capital).
    - Position sizing: always 1 share Stock 1 and β shares Stock 2 per entry.
    - No explicit constraints for hard-to-borrow names or short sale limitations.
    - Prices are Yahoo! Adjusted Close (splits/dividends accounted for).
    - Strategy is for research/educational use only.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.stattools import coint
from datetime import datetime, timedelta

# ============================================================================
# PARAMETERS & SETTINGS
# ============================================================================
rolling_window = 60    # Rolling window for z-score calculation (days)
top_n_pairs = 10       # Number of top cointegrated pairs to trade
plot_top = 3           # Number of top pairs to visualize
entry_step = 0.5       # Step size between ladder rungs (z-score intervals)
entry_min = 1.0        # Minimum z-score for ladder entries
entry_max = 4.0        # Maximum z-score for ladder entries
do_plot = True         # Toggle visualization

# ============================================================================
# TICKER SELECTION
# ============================================================================
# For your sanity, I have commented out the full recall of all SP500 tickers
# sp500_tickers = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]['Symbol'].tolist()
# sp500_tickers = [x.replace('.', '-') for x in sp500_tickers] # For Yahoo compatibility

sp500_tickers = ['LII', 'MSI']  # Replace or expand for more pairs

# ============================================================================
# DATE RANGE SETUP (18 months lookback)
# ============================================================================
today = datetime.today()
end_date = today.strftime("%Y-%m-%d")
start_18m = (today - timedelta(days=540)).strftime("%Y-%m-%d")  # ~18 months

# ============================================================================
# DATA DOWNLOAD FROM YAHOO FINANCE
# ============================================================================
print("Downloading S&P 500 price data (18m window)...")
prices_18m = yf.download(sp500_tickers, start=start_18m, end=end_date)['Close'].dropna(axis=1, how='any')
print(f"Final tickers used (after cleaning):\n{list(prices_18m.columns)[:10]} ... total: {len(prices_18m.columns)}")

# ============================================================================
# PAIR SELECTION: COINTEGRATION TESTING (ENGLE-GRANGER)
# ============================================================================
print("Testing all pairs for cointegration (may take a few minutes)...")
columns = list(prices_18m.columns)
cointegration_scores = []
checked = set()
for i, stock1 in enumerate(columns):
    for j, stock2 in enumerate(columns):
        if i < j and (stock1, stock2) not in checked:
            s1, s2 = prices_18m[stock1], prices_18m[stock2]
            score, pval, _ = coint(s1, s2)
            cointegration_scores.append((stock1, stock2, pval))
            checked.add((stock1, stock2))
cointegration_scores.sort(key=lambda x: x[2])  # Lower p-value = stronger cointegration
top_pairs = [x for x in cointegration_scores if x[2] < 0.05][:top_n_pairs]
print(f"Top {len(top_pairs)} cointegrated pairs (p-value < 0.05):")
for (a, b, p) in top_pairs:
    print(f"{a}-{b}: p-value={p:.4f}")

# ============================================================================
# STRATEGY CORE: LADDERED MEAN-REVERTING PAIR TRADING FUNCTION
# ============================================================================
def laddered_stat_arb_trades(s1, s2, rolling_window=60, entry_step=0.5, entry_min=1.0, entry_max=4.0):
    """
    Main execution engine for laddered statistical arbitrage.
    - Runs rolling z-score mean-reversion with multiple ladder levels.
    - Opens and closes trades for every threshold breach.
    - At end, force-closes all open positions for precise PnL.
    - Returns all trade logs, PnL, Sharpe, rolling metrics, etc.
    """
    beta = np.polyfit(s2, s1, 1)[0]  # Hedge ratio via linear regression
    spread = s1 - beta * s2
    spread_mean = spread.rolling(rolling_window).mean()
    spread_std = spread.rolling(rolling_window).std()
    zscore = (spread - spread_mean) / spread_std
    zscore = zscore.fillna(0)
    idx = s1.index

    short_entries, long_entries = {}, {}
    trade_log = []
    positions = pd.Series(0, index=idx, dtype=float)
    pos1 = pd.Series(0, index=idx, dtype=float)
    pos2 = pd.Series(0, index=idx, dtype=float)
    total_pos = 0  # Net open position count

    for i in range(len(zscore)):
        z = zscore.iloc[i]
        date = idx[i]

        # ENTRY SHORT
        t = entry_min
        while t <= entry_max:
            if (z >= t) and (t not in short_entries):
                short_entries[t] = i
                total_pos -= 1
                trade_log.append({
                    'datetime': date, 'side': 'Open Short',
                    'stock1': s1.name, 'qty1': -1, 'price1': s1.iloc[i],
                    'stock2': s2.name, 'qty2': beta, 'price2': s2.iloc[i],
                    'zscore': z, 'action': 'Open Short', 'leg': t
                })
            t += entry_step

        # ENTRY LONG
        t = -entry_min
        while t >= -entry_max:
            if (z <= t) and (t not in long_entries):
                long_entries[t] = i
                total_pos += 1
                trade_log.append({
                    'datetime': date, 'side': 'Open Long',
                    'stock1': s1.name, 'qty1': 1, 'price1': s1.iloc[i],
                    'stock2': s2.name, 'qty2': -beta, 'price2': s2.iloc[i],
                    'zscore': z, 'action': 'Open Long', 'leg': t
                })
            t -= entry_step

        # EXIT SHORTS
        exit_shorts = []
        for t in list(short_entries.keys()):
            exit_z = t - 1.0
            if z <= exit_z:
                trade_log.append({
                    'datetime': date, 'side': 'Close Short',
                    'stock1': s1.name, 'qty1': 1, 'price1': s1.iloc[i],
                    'stock2': s2.name, 'qty2': -beta, 'price2': s2.iloc[i],
                    'zscore': z, 'action': 'Close Short', 'leg': t
                })
                total_pos += 1
                exit_shorts.append(t)
        for t in exit_shorts:
            del short_entries[t]

        # EXIT LONGS
        exit_longs = []
        for t in list(long_entries.keys()):
            exit_z = t + 1.0
            if z >= exit_z:
                trade_log.append({
                    'datetime': date, 'side': 'Close Long',
                    'stock1': s1.name, 'qty1': -1, 'price1': s1.iloc[i],
                    'stock2': s2.name, 'qty2': beta, 'price2': s2.iloc[i],
                    'zscore': z, 'action': 'Close Long', 'leg': t
                })
                total_pos -= 1
                exit_longs.append(t)
        for t in exit_longs:
            del long_entries[t]

        # Bookkeeping: Track current net position for stats and PnL
        positions.iloc[i] = total_pos
        pos1.iloc[i] = pos1.iloc[i-1] if i > 0 else 0
        pos2.iloc[i] = pos2.iloc[i-1] if i > 0 else 0
        for log in [x for x in trade_log if x['datetime'] == date]:
            pos1.iloc[i] += log['qty1']
            pos2.iloc[i] += log['qty2']

    # FORCE CLOSE: At last date, forcibly close all open trades at final price
    final_date = idx[-1]
    for t in list(short_entries.keys()):
        trade_log.append({
            'datetime': final_date, 'side': 'Force Close Short',
            'stock1': s1.name, 'qty1': 1, 'price1': s1.iloc[-1],
            'stock2': s2.name, 'qty2': -beta, 'price2': s2.iloc[-1],
            'zscore': zscore.iloc[-1], 'action': 'Force Close Short', 'leg': t
        })
    for t in list(long_entries.keys()):
        trade_log.append({
            'datetime': final_date, 'side': 'Force Close Long',
            'stock1': s1.name, 'qty1': -1, 'price1': s1.iloc[-1],
            'stock2': s2.name, 'qty2': beta, 'price2': s2.iloc[-1],
            'zscore': zscore.iloc[-1], 'action': 'Force Close Long', 'leg': t
        })

    # PERFORMANCE METRICS
    spread_change = spread.diff().fillna(0)
    trade_pnl = positions.shift(1).fillna(0) * spread_change  # PnL per step
    cum_pnl = trade_pnl.cumsum()
    rolling_sharpe = trade_pnl.rolling(rolling_window).mean() / (trade_pnl.rolling(rolling_window).std() + 1e-8) * np.sqrt(252)
    rolling_drawdown = (cum_pnl - cum_pnl.rolling(rolling_window).max()).abs()
    rolling_pnl = cum_pnl.rolling(rolling_window).mean()
    rolling_corr = s1.rolling(rolling_window).corr(s2)
    sharpe = trade_pnl.mean() / (trade_pnl.std() + 1e-8) * np.sqrt(252)
    max_drawdown = rolling_drawdown.max()
    final_pnl = cum_pnl.iloc[-1]

    # Return everything needed for logging and plotting
    return {
        'positions': positions,
        'cum_pnl': cum_pnl,
        'zscore': zscore,
        'trade_log': pd.DataFrame(trade_log),
        's1': s1, 's2': s2,
        'trade_pnl': trade_pnl,
        'rolling_sharpe': rolling_sharpe,
        'rolling_drawdown': rolling_drawdown,
        'rolling_pnl': rolling_pnl,
        'rolling_corr': rolling_corr,
        'sharpe': sharpe,
        'final_pnl': final_pnl,
        'max_drawdown': max_drawdown,
        'pos1': pos1,
        'pos2': pos2
    }

# ============================================================================
# RUN STRATEGY ON ALL SELECTED PAIRS AND COLLECT METRICS
# ============================================================================
metrics = []
results = {}
for stock1, stock2, _ in top_pairs:
    s1, s2 = prices_18m[stock1], prices_18m[stock2]
    res = laddered_stat_arb_trades(s1, s2)
    metrics.append({
        'Pair': f"{stock1}-{stock2}",
        'Final_PnL': res['final_pnl'],
        'Sharpe': res['sharpe'],
        'Max_Drawdown': res['max_drawdown'],
        'Trade_Count': len(res['trade_log']),
        'RollingSharpe_now': res['rolling_sharpe'].iloc[-1],
        'RollingPnL_now': res['rolling_pnl'].iloc[-1]
    })
    results[f"{stock1}-{stock2}"] = res
metrics_df = pd.DataFrame(metrics).sort_values('Final_PnL', ascending=False)

# ============================================================================
# PLOTTING FUNCTION FOR STRATEGY OUTPUT
# ============================================================================
def plot_all_metrics(res, stock1, stock2, title=None):
    """
    Multi-panel visualization of:
    - Trade entries/exits (on both legs)
    - Z-score dynamics and entry/exit/force-close markers
    - PnL evolution and rolling stats
    """
    idx = res['s1'].index
    trade_log = res['trade_log']

    # Extract all types of trades for color-coded markers
    open_long = trade_log[trade_log['action']=='Open Long']
    close_long = trade_log[trade_log['action']=='Close Long']
    open_short = trade_log[trade_log['action']=='Open Short']
    close_short = trade_log[trade_log['action']=='Close Short']
    force_close_long = trade_log[trade_log['action']=='Force Close Long']
    force_close_short = trade_log[trade_log['action']=='Force Close Short']

    fig, axs = plt.subplots(4, 1, figsize=(17, 18), sharex=True)

    # 1. Prices + entry/exit/force markers
    axs[0].plot(idx, res['s1'], label=f"{stock1} Price", alpha=0.9)
    axs[0].plot(idx, res['s2'], label=f"{stock2} Price", alpha=0.9)
    if len(open_long): axs[0].scatter(open_long['datetime'], res['s1'].loc[open_long['datetime']], marker='^', color='lime', s=70, label='Long Entry S1')
    if len(open_short): axs[0].scatter(open_short['datetime'], res['s1'].loc[open_short['datetime']], marker='v', color='purple', s=70, label='Short Entry S1')
    if len(close_long): axs[0].scatter(close_long['datetime'], res['s1'].loc[close_long['datetime']], marker='x', color='orange', s=70, label='Long Exit S1')
    if len(close_short): axs[0].scatter(close_short['datetime'], res['s1'].loc[close_short['datetime']], marker='x', color='red', s=70, label='Short Exit S1')
    if len(force_close_long): axs[0].scatter(force_close_long['datetime'], res['s1'].loc[force_close_long['datetime']], marker='P', color='gold', s=110, label='Force Close Long S1')
    if len(force_close_short): axs[0].scatter(force_close_short['datetime'], res['s1'].loc[force_close_short['datetime']], marker='P', color='cyan', s=110, label='Force Close Short S1')
    # Second leg markers
    if len(open_long): axs[0].scatter(open_long['datetime'], res['s2'].loc[open_long['datetime']], marker='^', color='deepskyblue', s=50, label='Short Entry S2')
    if len(open_short): axs[0].scatter(open_short['datetime'], res['s2'].loc[open_short['datetime']], marker='v', color='brown', s=50, label='Long Entry S2')
    if len(close_long): axs[0].scatter(close_long['datetime'], res['s2'].loc[close_long['datetime']], marker='x', color='black', s=40, label='Short Exit S2')
    if len(close_short): axs[0].scatter(close_short['datetime'], res['s2'].loc[close_short['datetime']], marker='x', color='hotpink', s=40, label='Long Exit S2')
    if len(force_close_long): axs[0].scatter(force_close_long['datetime'], res['s2'].loc[force_close_long['datetime']], marker='P', color='navy', s=110, label='Force Close Long S2')
    if len(force_close_short): axs[0].scatter(force_close_short['datetime'], res['s2'].loc[force_close_short['datetime']], marker='P', color='orange', s=110, label='Force Close Short S2')
    axs[0].set_ylabel("Price")
    axs[0].legend(loc='best', ncol=2)
    axs[0].set_title(f"{stock1} / {stock2} - All Buy/Sell Markers (Both Legs)")

    # 2. Z-score panel
    axs[1].plot(idx, res['zscore'], label='Spread Z-score', color='gray')
    for z in np.arange(-4, 4.5, 0.5):
        if z != 0:
            axs[1].axhline(z, color='blue', alpha=0.2, linestyle='--', linewidth=0.5)
    axs[1].axhline(0, color='black', linewidth=1)
    axs[1].set_ylabel("Z-score")
    axs[1].set_title("Spread Z-score (with thresholds)")
    axs[1].legend()

    # 3. PnL panel
    axs[2].plot(idx, res['cum_pnl'], label='Cumulative PnL', linewidth=2)
    axs[2].plot(idx, res['rolling_pnl'], label='Rolling PnL', linestyle='--')
    axs[2].set_ylabel("PnL")
    axs[2].set_title("Cumulative and Rolling PnL")
    axs[2].legend()

    # 4. Sharpe/Drawdown panel
    axs[3].plot(idx, res['rolling_sharpe'], label='Rolling Sharpe')
    axs[3].plot(idx, res['rolling_drawdown'], label='Rolling Max Drawdown')
    axs[3].set_ylabel("Stats")
    axs[3].set_title("Rolling Sharpe and Drawdown")
    axs[3].legend()

    if title: plt.suptitle(title, fontsize=17)
    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()

# ============================================================================
# PLOT TOP PAIRS' RESULTS
# ============================================================================
if do_plot:
    for _, row in metrics_df.head(plot_top).iterrows():
        pid = row['Pair']
        stock1, stock2 = pid.split('-')
        res = results[pid]
        plot_all_metrics(res, stock1, stock2, title=f"Laddered Stat Arb: {pid}")

# ============================================================================
# PRINT METRICS AND FULL TRADE LOG
# ============================================================================
print("\n=== Top Cointegrated Pairs: Summary ===\n")
print(metrics_df.head(10).to_string(index=False, float_format="%.2f"))

top_pair = metrics_df.iloc[0]['Pair']
print(f"\n--- TRADE LOG FOR {top_pair} ---\n")
print(results[top_pair]['trade_log'].to_string(index=False))

# ============================================================================
# TRADE-BY-TRADE PERCENT RETURN ANALYSIS (ADVANCED SECTION)
# ============================================================================
# This section analyzes each round-trip trade (open/close/force-close) for percent return

trade_log = results[top_pair]['trade_log']
completed_trades = trade_log[trade_log['action'].str.contains('Close')].copy()

percent_returns = []
for idx, close_row in completed_trades.iterrows():
    # Figure out which "open" this "close" belongs to
    if 'Short' in close_row['side']:
        entry_action = 'Open Short'
    else:
        entry_action = 'Open Long'

    # Find most recent open trade for the same leg
    entry_candidates = trade_log[
        (trade_log['action'] == entry_action) &
        (trade_log['leg'] == close_row['leg']) &
        (trade_log['datetime'] <= close_row['datetime'])
    ]
    if len(entry_candidates) == 0:
        percent_returns.append(np.nan)
        continue
    entry_row = entry_candidates.iloc[-1]

    # Compute PnL for this round-trip (both legs)
    pnl = (close_row['price1'] - entry_row['price1']) * entry_row['qty1'] + \
          (close_row['price2'] - entry_row['price2']) * entry_row['qty2']
    notional = abs(entry_row['qty1'] * entry_row['price1']) + abs(entry_row['qty2'] * entry_row['price2'])
    percent_returns.append(pnl / notional if notional > 0 else np.nan)

# Add percent return column to the DataFrame
completed_trades['pct_return'] = percent_returns

# Display sample of percent returns
print("\n--- SAMPLE PERCENT RETURN ANALYSIS (TOP PAIR) ---\n")
print(completed_trades[['datetime', 'side', 'action', 'leg', 'pct_return']].tail(10).to_string(index=False))


# ============================================================================
# ADDITIONAL SUMMARY STATISTICS (Percent Return Table)
# ============================================================================

pct_ret = completed_trades['pct_return'].dropna()
summary_stats = {
    'Mean % Return': f"{100*pct_ret.mean():.3f}%",
    'Median % Return': f"{100*pct_ret.median():.3f}%",
    'StdDev % Return': f"{100*pct_ret.std():.3f}%",
    'Win Rate': f"{(pct_ret > 0).mean() * 100:.1f}%",
    'Largest Win': f"{100*pct_ret.max():.3f}%",
    'Largest Loss': f"{100*pct_ret.min():.3f}%",
    'Total Trades': len(pct_ret)
}

print("\n=== TRADE-BY-TRADE PERCENT RETURN SUMMARY (TOP PAIR) ===")
for k, v in summary_stats.items():
    print(f"{k:>18}: {v}")

# --- OPTIONAL: Export trade log, metrics, or percent return analysis ---
# completed_trades.to_csv('trade_log_percent_return.csv')
# metrics_df.to_csv('pair_metrics_summary.csv')
