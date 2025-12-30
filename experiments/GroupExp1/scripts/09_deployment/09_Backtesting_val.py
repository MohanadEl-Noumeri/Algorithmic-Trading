import pandas as pd
import numpy as np
import torch
import yaml
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
import matplotlib.pyplot as plt

# --- CONFIG ---
keys = yaml.safe_load(open("../../conf/keys.yaml"))
params = yaml.safe_load(open("../../conf/params.yaml"))

API_KEY = keys['KEYS']['APCA-API-KEY-ID']
SECRET_KEY = keys['KEYS']['APCA-API-SECRET-KEY']

MODEL_PATH = Path(params["MODELING"]["MODEL_PATH"])
FEATURE_PATH = params["DATA_PREP"]["FEATURE_PATH"]

data_client = CryptoHistoricalDataClient(API_KEY, SECRET_KEY)

OUTPUT_DIR = Path("../../analysis/backtest_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("BACKTEST VALIDATION")
print("Comparing Live Performance vs. Backtest")
print("=" * 60)

# --- 1. Determine Time Period (last 7 days) ---
end_dt = datetime.now().replace(second=0, microsecond=0)
start_dt = end_dt - timedelta(days=7)

print(f"\n[1/7] Time Period:")
print(f"  Start: {start_dt}")
print(f"  End:   {end_dt}")

# --- 2. Fetch Historical Data ---
print(f"\n[2/7] Fetching Historical Data...")

req = CryptoBarsRequest(
    symbol_or_symbols=["BTC/USD", "ETH/USD"],
    timeframe=TimeFrame.Minute,
    start=start_dt,
    end=end_dt
)

bars = data_client.get_crypto_bars(req).df

df_btc = bars.loc["BTC/USD"].reset_index().sort_values('timestamp')
df_eth = bars.loc["ETH/USD"].reset_index().sort_values('timestamp')

print(f"  BTC bars: {len(df_btc)}")
print(f"  ETH bars: {len(df_eth)}")

# --- 3. Feature Engineering (same as live bot) ---
print(f"\n[3/7] Engineering Features...")


def add_log_return(df):
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


def add_emas(df, periods=[10, 30, 60]):
    for w in periods:
        df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()
    return df


def add_ema_differences(df):
    if 'ema_30' in df.columns and 'ema_10' in df.columns:
        df["ema_30_10"] = df["ema_30"] - df["ema_10"]
    return df


def add_ema_slope(df, window=5):
    if 'ema_10' in df.columns:
        df["ema10_slope"] = df["ema_10"].diff(window) / window
    return df


def add_volatility(df, window=30):
    df[f"volatility_{window}"] = df["log_return"].rolling(window=window).std()
    return df


def add_rsi(df, window=14):
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, 0.000001)
    df[f"rsi_{window}_norm"] = (100 - (100 / (1 + rs))) / 100.0
    return df


def add_macd(df):
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd - signal
    return df


def add_bollinger_bands(df, window=20):
    rm = df["close"].rolling(window).mean()
    rstd = df["close"].rolling(window).std()
    upper = rm + (2 * rstd)
    lower = rm - (2 * rstd)
    df["bb_position"] = (df["close"] - lower) / (upper - lower + 1e-10)
    return df


def add_atr(df, window=14):
    h_l = df["high"] - df["low"]
    h_c = (df["high"] - df["close"].shift()).abs()
    l_c = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
    df["atr"] = tr.rolling(window).mean()
    df["atr_pct"] = df["atr"] / df["close"]
    return df


def add_roc(df, periods=[5, 10]):
    for p in periods:
        df[f"roc_{p}"] = ((df["close"] - df["close"].shift(p)) / df["close"].shift(p)) * 100
    return df


def add_stochastic(df, window=14):
    l_min = df["low"].rolling(window).min()
    h_max = df["high"].rolling(window).max()
    df["stoch_k"] = 100 * (df["close"] - l_min) / (h_max - l_min + 1e-10)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    return df


# Process ETH
df_eth = (df_eth
          .pipe(add_log_return)
          .pipe(add_emas)
          .pipe(add_ema_differences)
          .pipe(add_ema_slope)
          .pipe(add_volatility)
          .pipe(add_rsi)
          .pipe(add_macd)
          .pipe(add_bollinger_bands)
          .pipe(add_atr)
          .pipe(add_roc)
          .pipe(add_stochastic))

# Rename with eth_ prefix
rename_map = {col: f"eth_{col}" for col in df_eth.columns if col != 'timestamp'}
df_eth = df_eth.rename(columns=rename_map)

# Process BTC (minimal)
df_btc = add_log_return(df_btc)

# Join
df = pd.merge(
    df_btc[['timestamp', 'close', 'log_return', 'high', 'low']],
    df_eth,
    on='timestamp',
    how='inner'
)

# Cross-asset features
df['btc_eth_ratio'] = df['close'] / df['eth_close']
df['btc_eth_return_diff'] = df['log_return'] - df['eth_log_return']
df['btc_eth_corr_60'] = df['log_return'].rolling(60).corr(df['eth_log_return'])

# Add ATR for stops
df = add_atr(df, window=14)

# Clean
df = df.dropna().reset_index(drop=True)

print(f"  Final shape: {df.shape}")
print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# --- 4. Load Model ---
print(f"\n[4/7] Loading Model...")


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


checkpoint = torch.load(MODEL_PATH / "best_model.pt", map_location='cpu')
config = checkpoint['config']
model = MLP(config['in_dim'], config['h1'], config['h2'], config['dropout'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

scaler = pickle.load(open(MODEL_PATH / "feature_scaler.pkl", "rb"))

with open(FEATURE_PATH, "r") as f:
    FEATURES = [line.strip() for line in f.readlines()]

print(f"  Features: {len(FEATURES)}")

# --- 5. Generate Predictions ---
print(f"\n[5/7] Generating Predictions...")

# CRITICAL: Lag features by 1 (like in live trading)
X_raw = df[FEATURES].shift(1).values
df = df.iloc[1:].reset_index(drop=True)
X_raw = X_raw[1:]

X_scaled = scaler.transform(X_raw)

with torch.no_grad():
    logits = model(torch.tensor(X_scaled, dtype=torch.float32))
    probs = torch.sigmoid(logits).numpy().flatten()

df['prob_up'] = probs

print(f"  Predictions generated: {len(probs)}")
print(f"  Prob range: [{probs.min():.4f}, {probs.max():.4f}]")

# --- 6. Backtest Strategy (EXACT SAME LOGIC AS LIVE BOT) ---
print(f"\n[6/7] Running Backtest...")

CONFIDENCE_THRESHOLD = 0.515
EXIT_THRESHOLD = 0.45
BROKER_FEE = 0.001
SPREAD_BPS = 5
SLIPPAGE_BPS = 3

# Signals
df['signal'] = 0
df.loc[df['prob_up'] > CONFIDENCE_THRESHOLD, 'signal'] = 1
df.loc[df['prob_up'] < EXIT_THRESHOLD, 'signal'] = 0

# Position with hysteresis
df['position'] = 0
current_pos = 0

for i in range(len(df)):
    if df.loc[i, 'signal'] == 1:
        current_pos = 1
    elif df.loc[i, 'signal'] == 0 and df.loc[i, 'prob_up'] < EXIT_THRESHOLD:
        current_pos = 0
    df.loc[i, 'position'] = current_pos

# Costs
df['trade_flag'] = df['position'].diff().abs().fillna(0)
df['total_costs'] = df['trade_flag'] * (BROKER_FEE + (SPREAD_BPS + SLIPPAGE_BPS) / 10000)

# Returns
df['strategy_log_return'] = (df['log_return'] * df['position']) - df['total_costs']
df['strategy_cum_return'] = df['strategy_log_return'].cumsum().apply(np.exp)
df['market_cum_return'] = df['log_return'].cumsum().apply(np.exp)

# Metrics
total_return = (df['strategy_cum_return'].iloc[-1] - 1) * 100
market_return = (df['market_cum_return'].iloc[-1] - 1) * 100
trades = df['trade_flag'].sum()

invested_periods = df[df['position'] == 1]
win_rate = (invested_periods['log_return'] > 0).mean() * 100 if len(invested_periods) > 0 else 0

print(f"\n  Backtest Results:")
print(f"    Strategy Return: {total_return:+.2f}%")
print(f"    Market Return:   {market_return:+.2f}%")
print(f"    Alpha:           {total_return - market_return:+.2f}%")
print(f"    Number of Trades: {trades}")
print(f"    Win Rate: {win_rate:.1f}%")

# --- 7. Compare with Live Trading ---
print(f"\n[7/7] Comparison with Live Trading...")

live_trades_file = Path("../../analysis/live_performance/live_trades_matched.csv")
if live_trades_file.exists():
    df_live = pd.read_csv(live_trades_file)

    live_return = df_live['pnl_pct'].sum()
    live_trades_count = len(df_live)
    live_win_rate = (df_live['pnl_pct'] > 0).mean() * 100

    print(f"\n  📊 Side-by-Side Comparison:")
    print(f"  {'Metric':<25} {'Backtest':>12} {'Live':>12} {'Delta':>12}")
    print(f"  {'-' * 65}")
    print(f"  {'Total Return':<25} {total_return:>11.2f}% {live_return:>11.2f}% {live_return - total_return:>11.2f}%")
    print(f"  {'Number of Trades':<25} {trades:>12.0f} {live_trades_count:>12.0f} {live_trades_count - trades:>12.0f}")
    print(f"  {'Win Rate':<25} {win_rate:>11.1f}% {live_win_rate:>11.1f}% {live_win_rate - win_rate:>11.1f}%")

    # Calculate discrepancy
    return_diff = abs(live_return - total_return)
    trades_diff = abs(live_trades_count - trades)

    print(f"\n  🔍 Discrepancy Analysis:")
    if return_diff < 1.0 and trades_diff <= 2:
        print(f"    ✅ EXCELLENT: Backtest matches live performance!")
        print(f"       Return diff: {return_diff:.2f}% (< 1%)")
        print(f"       Trade diff: {trades_diff} (< 3)")
    elif return_diff < 3.0 and trades_diff <= 5:
        print(f"    ⚠️  ACCEPTABLE: Minor differences detected")
        print(f"       Return diff: {return_diff:.2f}%")
        print(f"       Trade diff: {trades_diff}")
        print(f"       Possible causes: slippage, timing delays")
    else:
        print(f"    ❌ WARNING: Significant discrepancy!")
        print(f"       Return diff: {return_diff:.2f}%")
        print(f"       Trade diff: {trades_diff}")

else:
    print(f"  ⚠️  No live trades file found")
    print(f"     Run live_performance_analyzer.py first")

# --- 8. Visualization ---
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Equity Curves
ax1 = axes[0]
ax1.plot(df['timestamp'], df['market_cum_return'], label='Buy & Hold', alpha=0.7)
ax1.plot(df['timestamp'], df['strategy_cum_return'], label='Backtest Strategy', linewidth=2)
ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax1.set_title('Backtest: Equity Curve (Last 7 Days)')
ax1.set_ylabel('Cumulative Return')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Probability & Positions
ax2 = axes[1]
ax2_twin = ax2.twinx()

ax2.plot(df['timestamp'], df['prob_up'], color='purple', alpha=0.6, label='Model Probability')
ax2.axhline(y=CONFIDENCE_THRESHOLD, color='green', linestyle='--', label='Entry Threshold')
ax2.axhline(y=EXIT_THRESHOLD, color='red', linestyle='--', label='Exit Threshold')
ax2.set_ylabel('Probability', color='purple')
ax2.set_xlabel('Time')
ax2.legend(loc='upper left')

ax2_twin.fill_between(df['timestamp'], 0, df['position'], alpha=0.2, color='blue', label='Position')
ax2_twin.set_ylabel('Position (0/1)', color='blue')
ax2_twin.set_ylim(-0.1, 1.1)
ax2_twin.legend(loc='upper right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'backtest_vs_live.png', dpi=150)
print(f"\n✓ Saved: {OUTPUT_DIR / 'backtest_vs_live.png'}")

# Save results
df.to_csv(OUTPUT_DIR / 'backtest_detailed.csv', index=False)
print(f"✓ Saved: {OUTPUT_DIR / 'backtest_detailed.csv'}")

print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)