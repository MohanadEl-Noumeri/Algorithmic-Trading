import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
from pathlib import Path
import pickle
from sklearn.metrics import classification_report, confusion_matrix

# --- Konfiguration laden ---
params = yaml.safe_load(open("../../conf/params.yaml"))
DATA_PATH = params["DATA_ACQUISITON"]["DATA_PATH"]
MODEL_PATH = params["MODELING"]["MODEL_PATH"]
test_file = Path(DATA_PATH) / "test.parquet"


print("BACKTESTING")

# --- 1. Modell und Daten laden ---
scaler = pickle.load(open(os.path.join(MODEL_PATH, "feature_scaler.pkl"), "rb"))


# Modell Definition
class MLP(torch.nn.Module):
    def __init__(self, in_dim, h1, h2, dropout_p):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, h1),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_p),
            torch.nn.Linear(h1, h2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_p),
            torch.nn.Linear(h2, 1)
        )

    def forward(self, x): return self.net(x)


checkpoint = torch.load(os.path.join(MODEL_PATH, "best_model.pt"))
config = checkpoint['config']
model = MLP(config['in_dim'], config['h1'], config['h2'], config['dropout'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("✓ Model loaded successfully")

# --- 2. Load Data ---
print("\n--- Loading Test Data ---")
df = pd.read_parquet(test_file)
print(f"Initial shape: {df.shape}")

# Load BTC raw data for price and returns
btc_raw_path = Path(DATA_PATH) / "BTCUSD_1m_raw.parquet"
df_raw = pd.read_parquet(btc_raw_path)
df_raw['log_return'] = np.log(df_raw['close'] / df_raw['close'].shift(1))

# Check if close/log_return already exist in test data
cols_to_merge = []
if 'close' not in df.columns:
    cols_to_merge.append('close')
if 'log_return' not in df.columns:
    cols_to_merge.append('log_return')

if cols_to_merge:
    print(f"Merging: {cols_to_merge}")
    df = pd.merge(
        df,
        df_raw[['timestamp'] + cols_to_merge],
        on='timestamp',
        how='left'
    )
else:
    print("✓ close and log_return already present")

# Clean NaNs
initial_len = len(df)
df = df.dropna(subset=['log_return', 'close']).reset_index(drop=True)
print(f"Removed {initial_len - len(df)} rows with NaN")
print(f"Final shape: {df.shape}")

# Features List
with open(params["DATA_PREP"]["FEATURE_PATH"], "r") as f:
    features = [line.strip() for line in f.readlines()]

print(f"✓ Loaded {len(features)} features")

# --- 3. CRITICAL FIX: Shift Features to Prevent Look-Ahead Bias ---
print("\n--- Feature Lag Implementation ---")
print("⚠️  IMPORTANT: Using features from t-1 to predict at t")

# Create lagged features
X_raw = df[features].shift(1).values  # Use YESTERDAY's features
df = df.iloc[1:].reset_index(drop=True)  # Drop first row (no lagged features)
X_raw = X_raw[1:]  # Match array length

print(f"After lag: {len(df)} samples")

# --- 4. Inference ---
print("\n--- Running Inference ---")
X_scaled = scaler.transform(X_raw)

with torch.no_grad():
    logits = model(torch.tensor(X_scaled).float())
    probs = torch.sigmoid(logits).numpy().flatten()

df['prob_up'] = probs
print(f"Prediction range: [{probs.min():.4f}, {probs.max():.4f}]")

# --- 5. Trading Algorithm with Realistic Costs ---
print("\n--- Trading Strategy Setup ---")

# Configuration
CONFIDENCE_THRESHOLD = 0.515  # Only trade when model is confident
EXIT_THRESHOLD = 0.495  # Exit early if confidence drops
BROKER_FEE = 0.001  # 0.1% per trade
SPREAD_BPS = 5  # 5 basis points
SLIPPAGE_BPS = 3  # 3 basis points

print(f"Entry Threshold: {CONFIDENCE_THRESHOLD}")
print(f"Exit Threshold: {EXIT_THRESHOLD}")
print(f"Total Cost per Trade: {(BROKER_FEE + (SPREAD_BPS + SLIPPAGE_BPS) / 10000) * 100:.3f}%")

# Generate Signals
df['signal'] = 0
df.loc[df['prob_up'] > CONFIDENCE_THRESHOLD, 'signal'] = 1
df.loc[df['prob_up'] < EXIT_THRESHOLD, 'signal'] = 0

# Position with Hysteresis (hold position unless clear signal)
df['position'] = 0
current_pos = 0

for i in range(len(df)):
    if df.loc[i, 'signal'] == 1:
        current_pos = 1
    elif df.loc[i, 'signal'] == 0 and df.loc[i, 'prob_up'] < EXIT_THRESHOLD:
        current_pos = 0
    df.loc[i, 'position'] = current_pos

# Calculate Costs (only on position changes)
df['trade_flag'] = df['position'].diff().abs().fillna(0)
df['total_costs'] = df['trade_flag'] * (
        BROKER_FEE + (SPREAD_BPS + SLIPPAGE_BPS) / 10000
)

# --- 6. Performance Calculation ---
print("\n--- Performance Calculation ---")

# Strategy Returns
df['strategy_log_return'] = (df['log_return'] * df['position']) - df['total_costs']
df['strategy_cum_return'] = df['strategy_log_return'].cumsum().apply(np.exp)
df['market_cum_return'] = df['log_return'].cumsum().apply(np.exp)

# Metrics
total_return = (df['strategy_cum_return'].iloc[-1] - 1) * 100
market_return = (df['market_cum_return'].iloc[-1] - 1) * 100
trades = df['trade_flag'].sum()

# Win rate (only when invested)
invested_periods = df[df['position'] == 1]
if len(invested_periods) > 0:
    win_rate = (invested_periods['log_return'] > 0).mean() * 100
    avg_return_per_trade = invested_periods['log_return'].mean() * 100
else:
    win_rate = 0
    avg_return_per_trade = 0

# Sharpe Ratio (annualized for 1-min data)
strategy_std = df['strategy_log_return'].std()
sharpe = (df['strategy_log_return'].mean() / strategy_std) * np.sqrt(525600) if strategy_std > 0 else 0

# Max Drawdown
cummax = df['strategy_cum_return'].cummax()
drawdown = (df['strategy_cum_return'] - cummax) / cummax
max_drawdown = drawdown.min() * 100

print(f"\n{'=' * 60}")
print("BACKTEST RESULTS")
print(f"{'=' * 60}")
print(f"Strategy Return:     {total_return:>10.2f}%")
print(f"Buy & Hold Return:   {market_return:>10.2f}%")
print(f"Alpha:               {total_return - market_return:>10.2f}%")
print(f"\nNumber of Trades:    {trades:>10.0f}")
print(f"Win Rate (invested): {win_rate:>10.2f}%")
print(f"Avg Return/Trade:    {avg_return_per_trade:>10.4f}%")
print(f"\nSharpe Ratio:        {sharpe:>10.2f}")
print(f"Max Drawdown:        {max_drawdown:>10.2f}%")
print(f"Total Costs:         {df['total_costs'].sum() * 100:>10.2f}%")

# Time in Market
time_invested = (df['position'] == 1).sum() / len(df) * 100
print(f"Time in Market:      {time_invested:>10.2f}%")

# --- 7. Detailed Trade Analysis ---
print(f"\n{'=' * 60}")
print("TRADE ANALYSIS")
print(f"{'=' * 60}")

# Find all trades
trade_entries = df[df['trade_flag'] == 1].copy()
trade_entries['entry_price'] = trade_entries['close']
trade_entries['exit_idx'] = np.nan

trades_list = []
in_position = False
entry_idx = None

for i in range(len(df)):
    if df.loc[i, 'trade_flag'] == 1:
        if df.loc[i, 'position'] == 1 and not in_position:
            # Entry
            entry_idx = i
            in_position = True
        elif df.loc[i, 'position'] == 0 and in_position:
            # Exit
            trades_list.append({
                'entry_time': df.loc[entry_idx, 'timestamp'],
                'exit_time': df.loc[i, 'timestamp'],
                'entry_price': df.loc[entry_idx, 'close'],
                'exit_price': df.loc[i, 'close'],
                'duration_min': i - entry_idx,
                'return': df.loc[entry_idx:i, 'log_return'].sum(),
                'entry_prob': df.loc[entry_idx, 'prob_up']
            })
            in_position = False

if trades_list:
    trades_df = pd.DataFrame(trades_list)
    trades_df['return_pct'] = trades_df['return'] * 100

    print(f"Completed Trades: {len(trades_df)}")
    print(f"\nTrade Duration Stats (minutes):")
    print(trades_df['duration_min'].describe())

    print(f"\nReturn Distribution:")
    print(trades_df['return_pct'].describe())

    winners = trades_df[trades_df['return_pct'] > 0]
    losers = trades_df[trades_df['return_pct'] <= 0]

    print(f"\nWinning Trades: {len(winners)} ({len(winners) / len(trades_df) * 100:.1f}%)")
    print(f"Losing Trades:  {len(losers)} ({len(losers) / len(trades_df) * 100:.1f}%)")

    if len(winners) > 0 and len(losers) > 0:
        avg_win = winners['return_pct'].mean()
        avg_loss = losers['return_pct'].mean()
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        print(f"\nAvg Win:  {avg_win:.4f}%")
        print(f"Avg Loss: {avg_loss:.4f}%")
        print(f"Profit Factor: {profit_factor:.2f}")

# --- 8. Plots ---
print("\n--- Generating Plots ---")

# Plot 1: Equity Curve
plt.figure(figsize=(14, 7))
plt.plot(df['timestamp'], df['market_cum_return'], label='Buy & Hold (BTC)', alpha=0.6, linewidth=2)
plt.plot(df['timestamp'], df['strategy_cum_return'], label='ML Strategy', color='green', linewidth=2)
plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
plt.title('Equity Curve: ML Model vs Buy & Hold', fontsize=14, fontweight='bold')
plt.xlabel('Time')
plt.ylabel('Cumulative Return (1.0 = start)')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('equity_curve.png', dpi=150)
print("✓ Saved: equity_curve.png")

# Plot 2: Trade Entries & Exits (First 500 minutes)
subset = df.iloc[:500].copy()
buys = subset[subset['trade_flag'] == 1]
buys = buys[buys['position'] == 1]
sells = subset[subset['trade_flag'] == 1]
sells = sells[sells['position'] == 0]

plt.figure(figsize=(14, 7))
plt.plot(subset['timestamp'], subset['close'], label='BTC Price', alpha=0.7, linewidth=1.5)
plt.scatter(buys['timestamp'], buys['close'], marker='^', color='green', s=150, label='Buy', zorder=5)
plt.scatter(sells['timestamp'], sells['close'], marker='v', color='red', s=150, label='Sell', zorder=5)
plt.title('Trade Entries & Exits (First 500 Minutes)', fontsize=14, fontweight='bold')
plt.xlabel('Time')
plt.ylabel('BTC Price ($)')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('trade_entries.png', dpi=150)
print("✓ Saved: trade_entries.png")

# Plot 3: Probability Distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['prob_up'], bins=50, kde=True, color='purple')
plt.axvline(x=0.5, color='red', linestyle='--', label='Random (0.5)', linewidth=2)
plt.axvline(x=CONFIDENCE_THRESHOLD, color='green', linestyle='--', label=f'Entry ({CONFIDENCE_THRESHOLD})', linewidth=2)
plt.axvline(x=EXIT_THRESHOLD, color='orange', linestyle='--', label=f'Exit ({EXIT_THRESHOLD})', linewidth=2)
plt.title('Distribution of Model Probabilities', fontsize=14, fontweight='bold')
plt.xlabel('Probability (Up)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('probability_distribution.png', dpi=150)
print("✓ Saved: probability_distribution.png")

# Plot 4: Drawdown
plt.figure(figsize=(14, 7))
plt.fill_between(df['timestamp'], 0, drawdown * 100, color='red', alpha=0.3)
plt.plot(df['timestamp'], drawdown * 100, color='red', linewidth=1)
plt.title('Strategy Drawdown Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Time')
plt.ylabel('Drawdown (%)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('drawdown.png', dpi=150)
print("✓ Saved: drawdown.png")


print("BACKTESTING COMPLETED!")