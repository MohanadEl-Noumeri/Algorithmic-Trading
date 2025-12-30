import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, GetPortfolioHistoryRequest
from alpaca.trading.enums import OrderSide, QueryOrderStatus
from alpaca.data.historical import CryptoHistoricalDataClient
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIG ---
keys = yaml.safe_load(open("../../conf/keys.yaml"))
API_KEY = keys['KEYS']['APCA-API-KEY-ID']
SECRET_KEY = keys['KEYS']['APCA-API-SECRET-KEY']

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = CryptoHistoricalDataClient(API_KEY, SECRET_KEY)

OUTPUT_DIR = Path("../../analysis/live_performance")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("LIVE PERFORMANCE ANALYSIS")
print("=" * 60)

# --- 1. Get Account Info ---
print("\n[1/5] Fetching Account Info...")
account = trading_client.get_account()

print(f"  Account Status: {account.status}")
print(f"  Equity: ${float(account.equity):,.2f}")
print(f"  Cash: ${float(account.cash):,.2f}")
print(f"  Buying Power: ${float(account.buying_power):,.2f}")

initial_capital = 100000  # Alpaca Paper starts with $100k
current_equity = float(account.equity)
total_return_pct = ((current_equity - initial_capital) / initial_capital) * 100

print(f"\n  Initial Capital: ${initial_capital:,.2f}")
print(f"  Total Return: {total_return_pct:+.2f}%")

# --- 2. Get All Closed Orders ---
print("\n[2/5] Fetching Trade History...")

end_date = datetime.now()
start_date = end_date - timedelta(days=7)

# FIXED: Use GetOrdersRequest with proper enum
request = GetOrdersRequest(
    status=QueryOrderStatus.CLOSED,
    after=start_date,
    limit=500
)

orders = trading_client.get_orders(filter=request)

if not orders:
    print("  ⚠️  No closed orders found!")
    exit()

print(f"  Found {len(orders)} closed orders")

# --- 3. Process Orders into Trades ---
print("\n[3/5] Processing Orders...")

trades = []
for order in orders:
    trades.append({
        'order_id': order.id,
        'symbol': order.symbol,
        'side': order.side,
        'qty': float(order.filled_qty) if order.filled_qty else 0,
        'filled_price': float(order.filled_avg_price) if order.filled_avg_price else 0,
        'notional': float(order.notional) if order.notional else 0,
        'submitted_at': order.submitted_at,
        'filled_at': order.filled_at,
        'status': order.status,
        'type': order.type
    })

df_orders = pd.DataFrame(trades)
df_orders['submitted_at'] = pd.to_datetime(df_orders['submitted_at'])
df_orders['filled_at'] = pd.to_datetime(df_orders['filled_at'])

df_orders = df_orders.sort_values('submitted_at').reset_index(drop=True)

print(f"  Processed {len(df_orders)} orders")
print(f"  Buy Orders: {len(df_orders[df_orders['side'] == 'buy'])}")
print(f"  Sell Orders: {len(df_orders[df_orders['side'] == 'sell'])}")

df_orders.to_csv(OUTPUT_DIR / "live_orders_raw.csv", index=False)
print(f"\n  ✓ Saved: {OUTPUT_DIR / 'live_orders_raw.csv'}")

# --- 4. Match Buy/Sell Pairs ---
print("\n[4/5] Matching Trade Pairs...")

completed_trades = []
buys = df_orders[df_orders['side'] == 'buy'].copy()
sells = df_orders[df_orders['side'] == 'sell'].copy()

for _, buy in buys.iterrows():
    matching_sells = sells[sells['submitted_at'] > buy['submitted_at']]

    if len(matching_sells) > 0:
        sell = matching_sells.iloc[0]

        entry_price = buy['filled_price']
        exit_price = sell['filled_price']
        qty = buy['qty']

        pnl = (exit_price - entry_price) * qty
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100

        duration = (sell['filled_at'] - buy['filled_at']).total_seconds() / 60

        completed_trades.append({
            'entry_time': buy['submitted_at'],
            'exit_time': sell['submitted_at'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'qty': qty,
            'duration_min': duration,
            'pnl_usd': pnl,
            'pnl_pct': pnl_pct,
            'position_size': buy['notional']
        })

if not completed_trades:
    print("  ⚠️  No completed trade pairs found!")
    print("  (This is normal if the bot is still holding a position)")
    df_trades = None
else:
    df_trades = pd.DataFrame(completed_trades)

    winners = df_trades[df_trades['pnl_pct'] > 0]
    losers = df_trades[df_trades['pnl_pct'] <= 0]

    print(f"\n  Completed Trades: {len(df_trades)}")
    print(f"  Winners: {len(winners)} ({len(winners) / len(df_trades) * 100:.1f}%)")
    print(f"  Losers: {len(losers)} ({len(losers) / len(df_trades) * 100:.1f}%)")

    if len(winners) > 0:
        print(f"  Avg Win: {winners['pnl_pct'].mean():.3f}%")
    if len(losers) > 0:
        print(f"  Avg Loss: {losers['pnl_pct'].mean():.3f}%")

    print(f"  Total PnL: ${df_trades['pnl_usd'].sum():.2f}")
    print(f"  Avg Trade Duration: {df_trades['duration_min'].mean():.1f} min")

    df_trades.to_csv(OUTPUT_DIR / "live_trades_matched.csv", index=False)
    print(f"\n  ✓ Saved: {OUTPUT_DIR / 'live_trades_matched.csv'}")

# --- 5. Load Trade Journal ---
print("\n[5/5] Loading Bot Trade Journal...")

# Check for different journal filenames
journal_files = [
    Path("trade_journal.csv"),
    Path("trade_journal_v2.csv"),
    Path("../../trade_journal.csv"),
    Path("../../trade_journal_v2.csv")
]

df_journal = None
for journal_path in journal_files:
    if journal_path.exists():
        df_journal = pd.read_csv(journal_path)
        df_journal['timestamp'] = pd.to_datetime(df_journal['timestamp'])

        print(f"  ✓ Found journal: {journal_path}")
        print(f"  Journal Entries: {len(df_journal)}")

        if 'action' in df_journal.columns:
            print(f"  BUY signals: {len(df_journal[df_journal['action'] == 'BUY'])}")
            print(f"  SELL signals: {len(df_journal[df_journal['action'] == 'SELL'])}")

        if 'probability' in df_journal.columns:
            print(f"\n  Probability Distribution:")
            print(f"    Mean: {df_journal['probability'].mean():.3f}")
            print(f"    Min:  {df_journal['probability'].min():.3f}")
            print(f"    Max:  {df_journal['probability'].max():.3f}")
        break

if df_journal is None:
    print("  ⚠️  No trade journal found")
    print(f"     Searched: {[str(p) for p in journal_files]}")

# --- 6. Visualizations ---
print("\n[6/6] Creating Visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Account Equity Over Time
ax1 = axes[0, 0]
try:
    # Try to get portfolio history
    portfolio_request = GetPortfolioHistoryRequest(
        period="7D",
        timeframe="5Min"  # Use string instead of enum
    )

    portfolio_history = trading_client.get_portfolio_history(filter=portfolio_request)

    if portfolio_history.equity and len(portfolio_history.equity) > 0:
        timestamps = [datetime.fromtimestamp(ts) for ts in portfolio_history.timestamp]
        equity = portfolio_history.equity

        ax1.plot(timestamps, equity, color='green', linewidth=2, label='Equity')
        ax1.axhline(y=initial_capital, color='gray', linestyle='--', label='Initial Capital')
        ax1.set_title('Account Equity Over Time (Last 7 Days)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Equity ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    else:
        # Fallback: Create equity curve from trades
        if df_trades is not None:
            trade_equity = [initial_capital]
            trade_times = [df_trades['exit_time'].min()]

            for _, trade in df_trades.iterrows():
                trade_equity.append(trade_equity[-1] + trade['pnl_usd'])
                trade_times.append(trade['exit_time'])

            ax1.plot(trade_times, trade_equity, color='green', linewidth=2, marker='o', label='Equity')
            ax1.axhline(y=initial_capital, color='gray', linestyle='--', label='Initial Capital')
            ax1.set_title('Account Equity from Completed Trades')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Equity ($)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'No portfolio history or trade data',
                     ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Account Equity Over Time')

except Exception as e:
    print(f"  ⚠️  Portfolio history error: {e}")
    ax1.text(0.5, 0.5, f'Portfolio history unavailable\nError: {str(e)[:50]}...',
             ha='center', va='center', transform=ax1.transAxes, fontsize=10)
    ax1.set_title('Account Equity Over Time')

# Plot 2: Trade PnL Distribution
ax2 = axes[0, 1]
if df_trades is not None and len(df_trades) > 0:
    pnl_values = df_trades['pnl_pct'].values

    # Create histogram
    ax2.hist(pnl_values, bins=20, color='blue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Breakeven')

    # Add statistics
    mean_pnl = pnl_values.mean()
    ax2.axvline(x=mean_pnl, color='green', linestyle=':', linewidth=2, label=f'Mean: {mean_pnl:.2f}%')

    ax2.set_title('Trade PnL Distribution')
    ax2.set_xlabel('PnL (%)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
else:
    ax2.text(0.5, 0.5, 'No completed trades yet',
             ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    ax2.set_title('Trade PnL Distribution')

# Plot 3: Model Probability Over Time
ax3 = axes[1, 0]
if df_journal is not None and 'probability' in df_journal.columns:
    ax3.plot(df_journal['timestamp'], df_journal['probability'],
             color='purple', linewidth=1, alpha=0.7, label='Model Probability')
    ax3.axhline(y=0.515, color='green', linestyle='--', label='Entry Threshold', linewidth=1.5)
    ax3.axhline(y=0.49, color='red', linestyle='--', label='Exit Threshold', linewidth=1.5)
    ax3.axhline(y=0.5, color='gray', linestyle=':', label='Random (50%)', linewidth=1, alpha=0.5)
    ax3.set_title('Model Confidence Over Time')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Probability')
    ax3.set_ylim(0.4, 0.6)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
else:
    ax3.text(0.5, 0.5, 'No journal data found\nBot needs to create trade_journal.csv',
             ha='center', va='center', transform=ax3.transAxes, fontsize=11)
    ax3.set_title('Model Confidence Over Time')

# Plot 4: Trade Duration
ax4 = axes[1, 1]
if df_trades is not None and len(df_trades) > 0:
    durations = df_trades['duration_min'].values

    ax4.hist(durations, bins=20, color='orange', alpha=0.7, edgecolor='black')

    # Add statistics
    mean_duration = durations.mean()
    median_duration = np.median(durations)

    ax4.axvline(x=mean_duration, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_duration:.0f} min')
    ax4.axvline(x=median_duration, color='blue', linestyle=':', linewidth=2,
                label=f'Median: {median_duration:.0f} min')

    ax4.set_title('Trade Duration Distribution')
    ax4.set_xlabel('Duration (minutes)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
else:
    ax4.text(0.5, 0.5, 'No completed trades yet',
             ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    ax4.set_title('Trade Duration Distribution')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'live_performance_dashboard.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {OUTPUT_DIR / 'live_performance_dashboard.png'}")

# --- Summary Report ---
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"\n📊 Account Metrics:")
print(f"  Current Equity: ${current_equity:,.2f}")
print(f"  Total Return: {total_return_pct:+.2f}%")
print(f"  Days Running: ~7")

if df_trades is not None and len(df_trades) > 0:
    winners = df_trades[df_trades['pnl_pct'] > 0]
    losers = df_trades[df_trades['pnl_pct'] <= 0]

    print(f"\n📈 Trading Metrics:")
    print(f"  Completed Trades: {len(df_trades)}")
    print(f"  Win Rate: {len(winners) / len(df_trades) * 100:.1f}%")

    if len(winners) > 0 and len(losers) > 0:
        print(f"  Avg Win/Loss: {winners['pnl_pct'].mean():.3f}% / {losers['pnl_pct'].mean():.3f}%")
        rr_ratio = abs(winners['pnl_pct'].mean() / losers['pnl_pct'].mean())
        print(f"  Risk/Reward Ratio: {rr_ratio:.2f}:1")

    print(f"  Best Trade: {df_trades['pnl_pct'].max():.3f}%")
    print(f"  Worst Trade: {df_trades['pnl_pct'].min():.3f}%")
    print(f"  Total PnL: ${df_trades['pnl_usd'].sum():.2f}")

print("\n" + "=" * 60)
print("NEXT STEP: Run backtest validation")
print("  python backtest_validation.py")
print("=" * 60)