import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
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

initial_capital = 100000  # Alpaca Paper startet mit $100k
current_equity = float(account.equity)
total_return_pct = ((current_equity - initial_capital) / initial_capital) * 100

print(f"\n  Initial Capital: ${initial_capital:,.2f}")
print(f"  Total Return: {total_return_pct:+.2f}%")

# --- 2. Get All Closed Orders ---
print("\n[2/5] Fetching Trade History...")

# Get orders from the last 7 days
end_date = datetime.now()
start_date = end_date - timedelta(days=7)

orders = trading_client.get_orders(
    status='closed',
    after=start_date,
    limit=500
)

if not orders:
    print("    No closed orders found!")
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

# Sort by time
df_orders = df_orders.sort_values('submitted_at').reset_index(drop=True)

print(f"  Processed {len(df_orders)} orders")
print(f"  Buy Orders: {len(df_orders[df_orders['side'] == 'buy'])}")
print(f"  Sell Orders: {len(df_orders[df_orders['side'] == 'sell'])}")

# Save raw orders
df_orders.to_csv(OUTPUT_DIR / "live_orders_raw.csv", index=False)
print(f"\n  ✓ Saved: {OUTPUT_DIR / 'live_orders_raw.csv'}")

# --- 4. Match Buy/Sell Pairs ---
print("\n[4/5] Matching Trade Pairs...")

completed_trades = []
buys = df_orders[df_orders['side'] == 'buy'].copy()
sells = df_orders[df_orders['side'] == 'sell'].copy()

for _, buy in buys.iterrows():
    # Find next sell after this buy
    matching_sells = sells[sells['submitted_at'] > buy['submitted_at']]

    if len(matching_sells) > 0:
        sell = matching_sells.iloc[0]

        # Calculate PnL
        entry_price = buy['filled_price']
        exit_price = sell['filled_price']
        qty = buy['qty']

        pnl = (exit_price - entry_price) * qty
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100

        # Duration
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
    print("    No completed trade pairs found!")
    print("  (This is normal if the bot is still holding a position)")
else:
    df_trades = pd.DataFrame(completed_trades)

    # Stats
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

    # Save
    df_trades.to_csv(OUTPUT_DIR / "live_trades_matched.csv", index=False)
    print(f"\n  ✓ Saved: {OUTPUT_DIR / 'live_trades_matched.csv'}")

# --- 5. Load Trade Journal (from bot) ---
print("\n[5/5] Loading Bot Trade Journal...")

journal_path = Path("trade_journal.csv")
if journal_path.exists():
    df_journal = pd.read_csv(journal_path)
    df_journal['timestamp'] = pd.to_datetime(df_journal['timestamp'])

    print(f"  Journal Entries: {len(df_journal)}")
    print(f"  BUY signals: {len(df_journal[df_journal['action'] == 'BUY'])}")
    print(f"  SELL signals: {len(df_journal[df_journal['action'] == 'SELL'])}")

    # Stats
    print(f"\n  Probability Distribution:")
    print(f"    Mean: {df_journal['probability'].mean():.3f}")
    print(f"    Min:  {df_journal['probability'].min():.3f}")
    print(f"    Max:  {df_journal['probability'].max():.3f}")
else:
    print("    trade_journal.csv not found")
    df_journal = None

# --- 6. Visualizations ---
print("\n[6/6] Creating Visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Account Equity Over Time (reconstructed from trades)
ax1 = axes[0, 0]
try:
    # Try Alpaca API first
    from alpaca.trading.requests import GetPortfolioHistoryRequest
    from alpaca.trading.enums import TimeFrame as TradingTimeFrame

    portfolio_request = GetPortfolioHistoryRequest(
        period="1W",
        timeframe=TradingTimeFrame.Hour
    )
    portfolio_history = trading_client.get_portfolio_history(portfolio_request)

    if hasattr(portfolio_history, 'equity') and portfolio_history.equity:
        timestamps = [datetime.fromtimestamp(ts) for ts in portfolio_history.timestamp]
        equity = portfolio_history.equity

        ax1.plot(timestamps, equity, color='green', linewidth=2, label='Account Equity')
        ax1.axhline(y=initial_capital, color='gray', linestyle='--', label='Initial Capital')
        ax1.set_title('Account Equity Over Time (Hourly)')
    else:
        raise ValueError("No equity data")

except Exception as e:
    # Fallback: Reconstruct from trades
    if completed_trades and len(completed_trades) > 0:
        df_equity = pd.DataFrame(completed_trades)
        df_equity = df_equity.sort_values('exit_time')
        df_equity['cumulative_pnl'] = df_equity['pnl_usd'].cumsum()
        df_equity['equity'] = initial_capital + df_equity['cumulative_pnl']

        ax1.plot(df_equity['exit_time'], df_equity['equity'],
                 color='green', linewidth=2, marker='o', markersize=4, label='Account Equity')
        ax1.axhline(y=initial_capital, color='gray', linestyle='--', label='Initial Capital')
        ax1.set_title('Account Equity Over Time (Reconstructed from Trades)')
    else:
        ax1.text(0.5, 0.5, f'No equity data available\n(No completed trades yet)',
                 ha='center', va='center', transform=ax1.transAxes, fontsize=10)
        ax1.set_title('Account Equity Over Time')

ax1.set_xlabel('Time')
ax1.set_ylabel('Equity ($)')
ax1.grid(True, alpha=0.3)
ax1.legend()
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 2: Trade PnL Distribution
ax2 = axes[0, 1]
if completed_trades:
    pnl_values = df_trades['pnl_pct'].values
    ax2.hist(pnl_values, bins=20, color='blue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Breakeven')
    ax2.set_title('Trade PnL Distribution')
    ax2.set_xlabel('PnL (%)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
else:
    ax2.text(0.5, 0.5, 'No completed trades yet',
             ha='center', va='center', transform=ax2.transAxes)

# Plot 3: Model Probability Over Time
ax3 = axes[1, 0]
if df_journal is not None and len(df_journal) > 0:
    # Sample data if too many points (for better visualization)
    if len(df_journal) > 1000:
        df_plot = df_journal.sample(n=1000, random_state=42).sort_values('timestamp')
    else:
        df_plot = df_journal

    ax3.plot(df_plot['timestamp'], df_plot['probability'],
             color='purple', linewidth=1, alpha=0.7, marker='o', markersize=2)
    ax3.axhline(y=0.515, color='green', linestyle='--', label='Entry Threshold', linewidth=2)
    ax3.axhline(y=0.45, color='red', linestyle='--', label='Exit Threshold', linewidth=2)
    ax3.axhline(y=0.5, color='gray', linestyle=':', label='Random', alpha=0.5)
    ax3.set_title(f'Model Confidence Over Time ({len(df_journal)} predictions)')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Probability')
    ax3.set_ylim(0.4, 0.6)
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Rotate x-axis labels
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
else:
    ax3.text(0.5, 0.5, 'No journal data\n(trade_journal.csv not found)',
             ha='center', va='center', transform=ax3.transAxes, fontsize=10)
    ax3.set_title('Model Confidence Over Time')
    ax3.axis('off')

# Plot 4: Trade Duration
ax4 = axes[1, 1]
if completed_trades:
    ax4.hist(df_trades['duration_min'], bins=20, color='orange',
             alpha=0.7, edgecolor='black')
    ax4.set_title('Trade Duration Distribution')
    ax4.set_xlabel('Duration (minutes)')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
else:
    ax4.text(0.5, 0.5, 'No completed trades yet',
             ha='center', va='center', transform=ax4.transAxes)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'live_performance_dashboard.png', dpi=150)
print(f"\n✓ Saved: {OUTPUT_DIR / 'live_performance_dashboard.png'}")

# --- Summary Report ---
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"\n Account Metrics:")
print(f"  Current Equity: ${current_equity:,.2f}")
print(f"  Total Return: {total_return_pct:+.2f}%")
print(f"  Days Running: ~7")

if completed_trades:
    print(f"\n Trading Metrics:")
    print(f"  Completed Trades: {len(df_trades)}")
    print(f"  Win Rate: {len(winners) / len(df_trades) * 100:.1f}%")
    print(f"  Avg Win/Loss: {winners['pnl_pct'].mean():.3f}% / {losers['pnl_pct'].mean():.3f}%")
    print(f"  Best Trade: {df_trades['pnl_pct'].max():.3f}%")
    print(f"  Worst Trade: {df_trades['pnl_pct'].min():.3f}%")

print("\n" + "=" * 60)
print("NEXT STEP: Run backtest validation")
print("  python scripts/10_monitoring/backtest_validation.py")
print("=" * 60)