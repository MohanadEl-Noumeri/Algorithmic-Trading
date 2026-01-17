import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml
import seaborn as sns

print("="*60)
print("INDIVIDUAL FEATURE ANALYSIS")
print("="*60)

# Config laden
params = yaml.safe_load(open("../../conf/params.yaml"))
data_path = Path(params['DATA_ACQUISITON']['DATA_PATH'])
IMG_PATH = Path("../../images/feature_analysis")
IMG_PATH.mkdir(exist_ok=True, parents=True)

# Raw files laden
btc_file = data_path / "BTCUSD_1m_raw.parquet"
eth_file = data_path / "ETHUSD_1m_raw.parquet"

print(f"\nLoading raw data...")
df_btc = pd.read_parquet(btc_file)
df_eth = pd.read_parquet(eth_file)

# Nur ersten Teil der Daten für schnellere Visualisierung (ca. 2 Wochen)
SAMPLE_SIZE = 20000  # ~14 Tage
df_btc = df_btc.iloc[:SAMPLE_SIZE].copy()
df_eth = df_eth.iloc[:SAMPLE_SIZE].copy()

print(f"Using {SAMPLE_SIZE:,} rows ({SAMPLE_SIZE/1440:.1f} days)")

# Helper Functions
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
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/window, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/window, adjust=False).mean()
    rs = gain / loss.replace(0, 0.000001)
    df[f"rsi_{window}"] = 100 - (100 / (1 + rs))
    df[f"rsi_{window}_norm"] = df[f"rsi_{window}"] / 100.0
    return df

def add_macd(df):
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = macd - signal
    return df

def add_bollinger_bands(df, window=20):
    rm = df["close"].rolling(window).mean()
    rstd = df["close"].rolling(window).std()
    df["bb_upper"] = rm + (2 * rstd)
    df["bb_lower"] = rm - (2 * rstd)
    df["bb_middle"] = rm
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / rm
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)
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

# Process ETH with all features
print("\nProcessing ETH features...")
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

df_eth = df_eth.dropna()
print(f"✓ ETH processed: {df_eth.shape}")

# =============================================================================
# INDIVIDUAL FEATURE PLOTS
# =============================================================================

print("\n" + "="*60)
print("GENERATING INDIVIDUAL FEATURE PLOTS")
print("="*60)

# --- 1. LOG RETURNS ---
print("\n1. Log Returns...")
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Time series
axes[0].plot(df_eth['timestamp'], df_eth['log_return'],
            linewidth=0.5, alpha=0.7, color='#2C3E50')
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1)
axes[0].set_title('ETH Log Returns Over Time', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Log Return')
axes[0].grid(True, alpha=0.3)

# Distribution with statistics
axes[1].hist(df_eth['log_return'], bins=100, alpha=0.7, color='#3498DB', edgecolor='black')
axes[1].axvline(df_eth['log_return'].mean(), color='red', linestyle='--',
               linewidth=2, label=f"Mean: {df_eth['log_return'].mean():.6f}")
axes[1].axvline(df_eth['log_return'].std(), color='green', linestyle='--',
               linewidth=2, label=f"Std: {df_eth['log_return'].std():.6f}")
axes[1].axvline(-df_eth['log_return'].std(), color='green', linestyle='--', linewidth=2)
axes[1].set_title('Distribution of Log Returns', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Log Return')
axes[1].set_ylabel('Frequency')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(IMG_PATH / "042_log_returns.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 01_log_returns.png")
plt.close()

# --- 2. EMAs (Trend) ---
print("\n2. EMAs (Trend Indicators)...")
fig, ax = plt.subplots(figsize=(16, 8))

ax.plot(df_eth['timestamp'], df_eth['close'], label='ETH Price',
       linewidth=2, color='black', alpha=0.7)
ax.plot(df_eth['timestamp'], df_eth['ema_10'], label='EMA 10',
       linewidth=1.5, color='#E74C3C', alpha=0.8)
ax.plot(df_eth['timestamp'], df_eth['ema_30'], label='EMA 30',
       linewidth=1.5, color='#F39C12', alpha=0.8)
ax.plot(df_eth['timestamp'], df_eth['ema_60'], label='EMA 60',
       linewidth=1.5, color='#2ECC71', alpha=0.8)

ax.set_title('ETH Price with EMAs (Multi-Timeframe Trends)', fontsize=14, fontweight='bold')
ax.set_xlabel('Time')
ax.set_ylabel('Price ($)')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(IMG_PATH / "042_emas_trend.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 02_emas_trend.png")
plt.close()

# --- 3. EMA Differences (Trend Strength) ---
print("\n3. EMA Differences (Trend Strength)...")
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Time series
axes[0].plot(df_eth['timestamp'], df_eth['ema_30_10'],
            linewidth=1, color='#9B59B6', alpha=0.8)
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
axes[0].fill_between(df_eth['timestamp'], 0, df_eth['ema_30_10'],
                     where=(df_eth['ema_30_10'] >= 0), color='green', alpha=0.3, label='Uptrend')
axes[0].fill_between(df_eth['timestamp'], 0, df_eth['ema_30_10'],
                     where=(df_eth['ema_30_10'] < 0), color='red', alpha=0.3, label='Downtrend')
axes[0].set_title('EMA Difference (30-10): Trend Direction & Strength', fontsize=14, fontweight='bold')
axes[0].set_ylabel('EMA 30 - EMA 10')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Distribution
axes[1].hist(df_eth['ema_30_10'], bins=100, alpha=0.7, color='#9B59B6', edgecolor='black')
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1].set_title('Distribution of EMA Difference', fontsize=14, fontweight='bold')
axes[1].set_xlabel('EMA 30 - EMA 10')
axes[1].set_ylabel('Frequency')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(IMG_PATH / "042_ema_difference.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 03_ema_difference.png")
plt.close()

# --- 4. EMA Slope (Trend Acceleration) ---
print("\n4. EMA Slope (Trend Acceleration)...")
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Time series
axes[0].plot(df_eth['timestamp'], df_eth['ema10_slope'],
            linewidth=1, color='#E67E22', alpha=0.8)
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
axes[0].fill_between(df_eth['timestamp'], 0, df_eth['ema10_slope'],
                     where=(df_eth['ema10_slope'] >= 0), color='green', alpha=0.3, label='Accelerating Up')
axes[0].fill_between(df_eth['timestamp'], 0, df_eth['ema10_slope'],
                     where=(df_eth['ema10_slope'] < 0), color='red', alpha=0.3, label='Accelerating Down')
axes[0].set_title('EMA 10 Slope: Trend Acceleration', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Slope (Price/Minute)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Distribution
axes[1].hist(df_eth['ema10_slope'], bins=100, alpha=0.7, color='#E67E22', edgecolor='black')
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1].set_title('Distribution of EMA Slope', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Slope')
axes[1].set_ylabel('Frequency')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(IMG_PATH / "042_ema_slope.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 04_ema_slope.png")
plt.close()

# --- 5. Volatility ---
print("\n5. Volatility (Market Risk)...")
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Time series with price
ax1 = axes[0]
ax2 = ax1.twinx()

ax1.plot(df_eth['timestamp'], df_eth['volatility_30'],
        linewidth=1.5, color='#E74C3C', label='Volatility (30min)')
ax2.plot(df_eth['timestamp'], df_eth['close'],
        linewidth=1, color='#95A5A6', alpha=0.5, label='ETH Price')

ax1.set_title('Volatility vs Price: High Vol = High Risk/Opportunity', fontsize=14, fontweight='bold')
ax1.set_ylabel('Volatility (30min Rolling Std)', color='#E74C3C')
ax2.set_ylabel('Price ($)', color='#95A5A6')
ax1.tick_params(axis='y', labelcolor='#E74C3C')
ax2.tick_params(axis='y', labelcolor='#95A5A6')
ax1.grid(True, alpha=0.3)

# Distribution (log scale for better visualization)
axes[1].hist(df_eth['volatility_30'], bins=100, alpha=0.7, color='#E74C3C', edgecolor='black')
axes[1].axvline(df_eth['volatility_30'].median(), color='blue', linestyle='--',
               linewidth=2, label=f"Median: {df_eth['volatility_30'].median():.6f}")
axes[1].set_title('Distribution of Volatility (Note: Right-Skewed)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Volatility')
axes[1].set_ylabel('Frequency')
axes[1].set_yscale('log')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(IMG_PATH / "042_volatility.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 05_volatility.png")
plt.close()

# --- 6. RSI (Overbought/Oversold) ---
print("\n6. RSI (Overbought/Oversold)...")
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Time series with zones
axes[0].plot(df_eth['timestamp'], df_eth['rsi_14'],
            linewidth=1, color='#8E44AD', alpha=0.8)
axes[0].axhline(y=70, color='red', linestyle='--', linewidth=2, label='Overbought (70)')
axes[0].axhline(y=30, color='green', linestyle='--', linewidth=2, label='Oversold (30)')
axes[0].axhline(y=50, color='gray', linestyle='-', linewidth=1, alpha=0.5, label='Neutral (50)')
axes[0].fill_between(df_eth['timestamp'], 70, 100, color='red', alpha=0.1)
axes[0].fill_between(df_eth['timestamp'], 0, 30, color='green', alpha=0.1)
axes[0].set_title('RSI (14): Momentum Indicator', fontsize=14, fontweight='bold')
axes[0].set_ylabel('RSI')
axes[0].set_ylim(0, 100)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Distribution
axes[1].hist(df_eth['rsi_14'], bins=50, alpha=0.7, color='#8E44AD', edgecolor='black')
axes[1].axvline(x=70, color='red', linestyle='--', linewidth=2, label='Overbought')
axes[1].axvline(x=30, color='green', linestyle='--', linewidth=2, label='Oversold')
axes[1].axvline(x=50, color='gray', linestyle='-', linewidth=1, label='Neutral')
axes[1].set_title('RSI Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('RSI')
axes[1].set_ylabel('Frequency')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(IMG_PATH / "042_rsi.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 06_rsi.png")
plt.close()

# --- 7. MACD (Trend + Momentum) ---
print("\n7. MACD (Trend + Momentum)...")
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Price with MACD signals
axes[0].plot(df_eth['timestamp'], df_eth['close'],
            linewidth=1.5, color='black', alpha=0.7, label='ETH Price')
# Mark crossovers
macd_cross_up = (df_eth['macd'] > df_eth['macd_signal']) & (df_eth['macd'].shift(1) <= df_eth['macd_signal'].shift(1))
macd_cross_down = (df_eth['macd'] < df_eth['macd_signal']) & (df_eth['macd'].shift(1) >= df_eth['macd_signal'].shift(1))
axes[0].scatter(df_eth.loc[macd_cross_up, 'timestamp'], df_eth.loc[macd_cross_up, 'close'],
               marker='^', color='green', s=100, label='MACD Cross Up', zorder=5)
axes[0].scatter(df_eth.loc[macd_cross_down, 'timestamp'], df_eth.loc[macd_cross_down, 'close'],
               marker='v', color='red', s=100, label='MACD Cross Down', zorder=5)
axes[0].set_title('ETH Price with MACD Crossover Signals', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Price ($)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# MACD Histogram
axes[1].bar(df_eth['timestamp'], df_eth['macd_hist'],
           color=np.where(df_eth['macd_hist'] >= 0, 'green', 'red'),
           alpha=0.6, width=0.0007)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[1].set_title('MACD Histogram: Momentum Strength', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('MACD Histogram')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(IMG_PATH / "042_macd.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 07_macd.png")
plt.close()

# --- 8. Bollinger Bands ---
print("\n8. Bollinger Bands (Volatility Bands)...")
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Price with bands
axes[0].plot(df_eth['timestamp'], df_eth['close'],
            linewidth=1.5, color='black', label='ETH Price')
axes[0].plot(df_eth['timestamp'], df_eth['bb_upper'],
            linewidth=1, color='red', linestyle='--', alpha=0.7, label='Upper Band')
axes[0].plot(df_eth['timestamp'], df_eth['bb_middle'],
            linewidth=1, color='blue', linestyle='--', alpha=0.7, label='Middle (SMA)')
axes[0].plot(df_eth['timestamp'], df_eth['bb_lower'],
            linewidth=1, color='green', linestyle='--', alpha=0.7, label='Lower Band')
axes[0].fill_between(df_eth['timestamp'], df_eth['bb_lower'], df_eth['bb_upper'],
                     color='gray', alpha=0.1)
axes[0].set_title('Bollinger Bands: Price Channels', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Price ($)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# BB Position
axes[1].plot(df_eth['timestamp'], df_eth['bb_position'],
            linewidth=1, color='#16A085', alpha=0.8)
axes[1].axhline(y=1, color='red', linestyle='--', linewidth=2, label='Upper Band (1.0)')
axes[1].axhline(y=0, color='green', linestyle='--', linewidth=2, label='Lower Band (0.0)')
axes[1].axhline(y=0.5, color='gray', linestyle='-', linewidth=1, alpha=0.5, label='Middle (0.5)')
axes[1].fill_between(df_eth['timestamp'], 0.8, 1.0, color='red', alpha=0.1, label='Overbought Zone')
axes[1].fill_between(df_eth['timestamp'], 0.0, 0.2, color='green', alpha=0.1, label='Oversold Zone')
axes[1].set_title('Bollinger Band Position: Relative Price Location', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('BB Position (0=Lower, 1=Upper)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(IMG_PATH / "042_bollinger_bands.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 08_bollinger_bands.png")
plt.close()

# --- 9. ATR (Average True Range) ---
print("\n9. ATR (Volatility Measure)...")
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Time series with price
ax1 = axes[0]
ax2 = ax1.twinx()

ax1.plot(df_eth['timestamp'], df_eth['atr_pct'] * 100,
        linewidth=1.5, color='#D35400', label='ATR %')
ax2.plot(df_eth['timestamp'], df_eth['close'],
        linewidth=1, color='#95A5A6', alpha=0.5, label='ETH Price')

ax1.set_title('ATR (% of Price): Volatility & Risk Measure', fontsize=14, fontweight='bold')
ax1.set_ylabel('ATR as % of Price', color='#D35400')
ax2.set_ylabel('Price ($)', color='#95A5A6')
ax1.tick_params(axis='y', labelcolor='#D35400')
ax2.tick_params(axis='y', labelcolor='#95A5A6')
ax1.grid(True, alpha=0.3)

# Distribution
axes[1].hist(df_eth['atr_pct'] * 100, bins=100, alpha=0.7, color='#D35400', edgecolor='black')
axes[1].axvline(df_eth['atr_pct'].median() * 100, color='blue', linestyle='--',
               linewidth=2, label=f"Median: {df_eth['atr_pct'].median()*100:.3f}%")
axes[1].set_title('ATR Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('ATR (% of Price)')
axes[1].set_ylabel('Frequency')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(IMG_PATH / "042_atr.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 09_atr.png")
plt.close()

# --- 10. ROC (Rate of Change) ---
print("\n10. ROC (Rate of Change)...")
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# ROC 5 vs ROC 10
axes[0].plot(df_eth['timestamp'], df_eth['roc_5'],
            linewidth=1, color='#E74C3C', alpha=0.8, label='ROC 5min')
axes[0].plot(df_eth['timestamp'], df_eth['roc_10'],
            linewidth=1, color='#3498DB', alpha=0.8, label='ROC 10min')
axes[0].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[0].fill_between(df_eth['timestamp'], 0, df_eth['roc_5'],
                     where=(df_eth['roc_5'] >= 0), color='green', alpha=0.2)
axes[0].fill_between(df_eth['timestamp'], 0, df_eth['roc_5'],
                     where=(df_eth['roc_5'] < 0), color='red', alpha=0.2)
axes[0].set_title('Rate of Change: Short-term Momentum', fontsize=14, fontweight='bold')
axes[0].set_ylabel('ROC (%)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Distribution comparison
axes[1].hist(df_eth['roc_5'], bins=100, alpha=0.6, color='#E74C3C',
            edgecolor='black', label='ROC 5min')
axes[1].hist(df_eth['roc_10'], bins=100, alpha=0.6, color='#3498DB',
            edgecolor='black', label='ROC 10min')
axes[1].axvline(x=0, color='black', linestyle='-', linewidth=2)
axes[1].set_title('ROC Distribution: Comparison', fontsize=14, fontweight='bold')
axes[1].set_xlabel('ROC (%)')
axes[1].set_ylabel('Frequency')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(IMG_PATH / "042_roc.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 10_roc.png")
plt.close()

# --- 11. Stochastic Oscillator ---
print("\n11. Stochastic Oscillator...")
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Time series
axes[0].plot(df_eth['timestamp'], df_eth['stoch_k'],
            linewidth=1, color='#1ABC9C', alpha=0.8, label='Stoch K (Fast)')
axes[0].plot(df_eth['timestamp'], df_eth['stoch_d'],
            linewidth=1.5, color='#E74C3C', alpha=0.8, label='Stoch D (Slow)')
axes[0].axhline(y=80, color='red', linestyle='--', linewidth=2, label='Overbought (80)')
axes[0].axhline(y=20, color='green', linestyle='--', linewidth=2, label='Oversold (20)')
axes[0].fill_between(df_eth['timestamp'], 80, 100, color='red', alpha=0.1)
axes[0].fill_between(df_eth['timestamp'], 0, 20, color='green', alpha=0.1)
axes[0].set_title('Stochastic Oscillator: Momentum', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Stochastic Value')
axes[0].set_ylim(0, 100)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Distribution
axes[1].hist(df_eth['stoch_k'], bins=50, alpha=0.7, color='#1ABC9C', edgecolor='black')
axes[1].axvline(x=80, color='red', linestyle='--', linewidth=2, label='Overbought')
axes[1].axvline(x=20, color='green', linestyle='--', linewidth=2, label='Oversold')
axes[1].set_title('Stochastic K Distribution (Note: U-Shape)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Stochastic K')
axes[1].set_ylabel('Frequency')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(IMG_PATH / "042_stochastic.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 11_stochastic.png")
plt.close()

# --- SUMMARY ---
print("\n" + "="*60)
print("FEATURE ANALYSIS COMPLETED!")
print("="*60)
print(f"\nGenerated 11 detailed feature plots in: {IMG_PATH}/")
print("\nFiles created:")
print("  01. log_returns.png          - Basic price movements")
print("  02. emas_trend.png           - Multi-timeframe trends")
print("  03. ema_difference.png       - Trend strength")
print("  04. ema_slope.png            - Trend acceleration")
print("  05. volatility.png           - Market risk")
print("  06. rsi.png                  - Overbought/oversold")
print("  07. macd.png                 - Trend + momentum crossovers")
print("  08. bollinger_bands.png      - Volatility channels")
print("  09. atr.png                  - True volatility measure")
print("  10. roc.png                  - Rate of change momentum")
print("  11. stochastic.png           - Momentum oscillator")
print("\n✅ All plots ready for presentation!")