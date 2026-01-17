import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import numpy as np

print("=" * 60)
print("DATA PREPARATION INSPECTION (Raw Files → Feature Engineering)")
print("=" * 60)

# 1. Config laden
params = yaml.safe_load(open("../../conf/params.yaml"))
data_path = Path(params['DATA_ACQUISITON']['DATA_PATH'])
IMG_PATH = Path("../../images")
IMG_PATH.mkdir(exist_ok=True)

# 2. Raw files laden
btc_file = data_path / "BTCUSD_1m_raw.parquet"
eth_file = data_path / "ETHUSD_1m_raw.parquet"

print(f"\nLoading raw data:")
print(f"  BTC: {btc_file}")
print(f"  ETH: {eth_file}")

if not btc_file.exists() or not eth_file.exists():
    print("❌ Error: Raw files not found!")
    print("Please run crypto_data_acquisition.py first.")
    exit(1)

df_btc_raw = pd.read_parquet(btc_file)
df_eth_raw = pd.read_parquet(eth_file)

print(f"✓ BTC loaded: {df_btc_raw.shape}")
print(f"✓ ETH loaded: {df_eth_raw.shape}")

# --- FEATURE ENGINEERING (subset for visualization) ---
print("\n" + "=" * 60)
print("FEATURE ENGINEERING (Simulated)")
print("=" * 60)


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
    df["atr_pct"] = tr.rolling(window).mean() / df["close"]
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


def compute_target_normalized(df, minutes):
    prices = df["close"].values
    windows = np.lib.stride_tricks.sliding_window_view(prices, window_shape=minutes)

    x = np.arange(minutes)
    x_mean = x.mean()
    denominator = np.sum((x - x_mean) ** 2)
    y_mean = np.mean(windows, axis=1, keepdims=True)
    numerator = np.sum((windows - y_mean) * (x - x_mean), axis=1)

    raw_slopes = numerator / denominator
    current_prices = prices[:len(raw_slopes)]
    norm_slopes = raw_slopes / current_prices

    pad = np.full(minutes - 1, np.nan)
    full_slopes = np.concatenate([norm_slopes, pad])

    df[f"target_{minutes}m"] = (full_slopes > 0).astype(int)
    return df


# Process ETH (Full Features)
print("\nProcessing ETH features...")
df_eth = (df_eth_raw.copy()
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

print(f"✓ ETH features: {len([c for c in df_eth.columns if c.startswith('eth_')])} features")

# Process BTC (Targets only)
print("\nProcessing BTC targets...")
df_btc = df_btc_raw.copy()
df_btc = add_log_return(df_btc)

# Compute targets for multiple windows
TARGET_WINDOWS = [5, 10, 15, 20, 30, 60, 120, 240]
for m in TARGET_WINDOWS:
    df_btc = compute_target_normalized(df_btc, m)

target_cols = [c for c in df_btc.columns if c.startswith('target_')]
print(f"✓ BTC targets: {len(target_cols)} target windows")

# Join ETH features + BTC targets
print("\nJoining ETH features + BTC targets...")
df_combined = pd.merge(
    df_btc[['timestamp', 'close', 'log_return'] + target_cols],
    df_eth,
    on='timestamp',
    how='inner'
)

# Cross-Asset Features
print("Adding cross-asset features...")
df_combined['btc_eth_ratio'] = df_combined['close'] / df_combined['eth_close']
df_combined['btc_eth_return_diff'] = df_combined['log_return'] - df_combined['eth_log_return']
df_combined['btc_eth_corr_60'] = (
    df_combined['log_return']
    .rolling(60)
    .corr(df_combined['eth_log_return'])
)

# Drop temporary columns
df_combined = df_combined.drop(columns=['close', 'log_return'])

# Clean NaNs
print("\nCleaning NaNs...")
before = len(df_combined)
df_combined = df_combined.dropna().reset_index(drop=True)
after = len(df_combined)
print(f"  Removed {before - after:,} rows ({(before - after) / before * 100:.2f}%)")
print(f"  Final shape: {df_combined.shape}")

# Feature categories
eth_features = [c for c in df_combined.columns if c.startswith('eth_')]
cross_features = [c for c in df_combined.columns if 'btc_eth' in c]
target_cols = [c for c in df_combined.columns if c.startswith('target_')]

print("\n" + "=" * 60)
print("FEATURE SUMMARY")
print("=" * 60)
print(f"  ETH Features: {len(eth_features)}")
print(f"  Cross-Asset Features: {len(cross_features)}")
print(f"  BTC Targets: {len(target_cols)}")
print(f"  Total: {len(df_combined.columns)}")

# --- VISUALIZATION ---
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

# Plot 1: Feature Correlation Matrix
print("\n1. Correlation Matrix...")
important_features = [
    'eth_log_return',
    'eth_ema_10',
    'eth_ema_30',
    'eth_ema_60',
    'eth_ema_30_10',
    'eth_ema10_slope',
    'eth_volatility_30',
    'eth_rsi_14_norm',
    'eth_macd_hist',
    'eth_bb_position',
    'eth_atr_pct',
    'eth_roc_5',
    'eth_roc_10',
    'eth_stoch_k',
    'btc_eth_ratio',
    'btc_eth_return_diff',
    'btc_eth_corr_60'
]

plt.figure(figsize=(14, 12))
corr = df_combined[important_features].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f",
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            annot_kws={"size": 8})
plt.title("Feature Correlation Matrix\n(ETH Features + Cross-Asset Features)",
          fontsize=14, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(IMG_PATH / "04_correlation_matrix_updated.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 04_correlation_matrix_updated.png")
plt.close()

# Plot 2: Class Balance
print("\n2. Class Balance for all targets...")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, target in enumerate(target_cols):
    vc = df_combined[target].value_counts()
    axes[idx].bar(['Down (0)', 'Up (1)'],
                  [vc.get(0, 0), vc.get(1, 0)],
                  color=['#E74C3C', '#2ECC71'])
    axes[idx].set_title(f'{target}', fontweight='bold', fontsize=11)
    axes[idx].set_ylabel('Count')
    axes[idx].grid(axis='y', alpha=0.3)

    total = vc.sum()
    down_pct = vc.get(0, 0) / total * 100
    up_pct = vc.get(1, 0) / total * 100
    axes[idx].text(0, vc.get(0, 0) / 2, f'{down_pct:.1f}%',
                   ha='center', fontweight='bold', color='white', fontsize=10)
    axes[idx].text(1, vc.get(1, 0) / 2, f'{up_pct:.1f}%',
                   ha='center', fontweight='bold', color='white', fontsize=10)

plt.suptitle('BTC Target Distribution (All Time Windows)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(IMG_PATH / "04_class_balance_updated.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 04_class_balance_updated.png")
plt.close()

# Plot 3: ETH Features Distribution
print("\n3. ETH Features Distribution...")
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

eth_plot_features = [
    'eth_log_return',
    'eth_ema_10',
    'eth_volatility_30',
    'eth_rsi_14_norm',
    'eth_macd_hist',
    'eth_bb_position',
    'eth_atr_pct',
    'eth_roc_5',
    'eth_stoch_k'
]

for idx, feature in enumerate(eth_plot_features):
    if feature in df_combined.columns:
        data = df_combined[feature].dropna()
        axes[idx].hist(data, bins=50, alpha=0.7,
                       color='#3498DB', edgecolor='black')
        axes[idx].set_title(feature, fontweight='bold', fontsize=10)
        axes[idx].set_xlabel('Value', fontsize=9)
        axes[idx].set_ylabel('Frequency', fontsize=9)
        axes[idx].grid(axis='y', alpha=0.3)

        mean_val = data.mean()
        axes[idx].axvline(mean_val, color='red', linestyle='--',
                          linewidth=2, label=f'μ={mean_val:.4f}')
        axes[idx].legend(fontsize=8)

plt.suptitle('Distribution of ETH Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(IMG_PATH / "04_eth_features_distribution.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 04_eth_features_distribution.png")
plt.close()

# Plot 4: Cross-Asset Features Over Time
print("\n4. Cross-Asset Features over Time...")
sample_df = df_combined.iloc[:10000].copy()

fig, axes = plt.subplots(3, 1, figsize=(16, 10))

axes[0].plot(sample_df['timestamp'], sample_df['btc_eth_ratio'],
             linewidth=1, alpha=0.8, color='#9B59B6')
axes[0].set_title('BTC/ETH Price Ratio', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Ratio')
axes[0].grid(True, alpha=0.3)

axes[1].plot(sample_df['timestamp'], sample_df['btc_eth_return_diff'],
             linewidth=1, alpha=0.8, color='#E67E22')
axes[1].set_title('BTC/ETH Return Difference', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Return Diff')
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
axes[1].grid(True, alpha=0.3)

axes[2].plot(sample_df['timestamp'], sample_df['btc_eth_corr_60'],
             linewidth=1, alpha=0.8, color='#16A085')
axes[2].set_title('BTC/ETH Correlation (60min Rolling)', fontweight='bold', fontsize=12)
axes[2].set_ylabel('Correlation')
axes[2].set_xlabel('Time')
axes[2].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
axes[2].grid(True, alpha=0.3)

plt.suptitle('Cross-Asset Features Over Time (First 10k Minutes)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(IMG_PATH / "04_cross_asset_features.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 04_cross_asset_features.png")
plt.close()

# Plot 5: Target Correlation
print("\n5. Target Correlation Matrix...")
target_corr = df_combined[target_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(target_corr, annot=True, cmap="YlOrRd", fmt=".2f",
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Correlation Between BTC Target Windows",
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(IMG_PATH / "04_target_correlation.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 04_target_correlation.png")
plt.close()

# --- SUMMARY ---
print("\n" + "=" * 60)
print("ANALYSIS SUMMARY")
print("=" * 60)

print(f"\nDataset Information:")
print(f"  Total Rows: {len(df_combined):,}")
print(f"  Date Range: {df_combined['timestamp'].min()} to {df_combined['timestamp'].max()}")
print(f"  Duration: {(df_combined['timestamp'].max() - df_combined['timestamp'].min()).days} days")

print(f"\nFeature Summary:")
print(f"  ETH Features: {len(eth_features)}")
print(f"  Cross-Asset Features: {len(cross_features)}")
print(f"  Total Input Features: {len(eth_features) + len(cross_features)}")
print(f"  BTC Target Variables: {len(target_cols)}")

print(f"\nClass Balance (target_15m):")
vc = df_combined['target_15m'].value_counts(normalize=True)
print(f"  Down (0): {vc.get(0, 0) * 100:.2f}%")
print(f"  Up (1): {vc.get(1, 0) * 100:.2f}%")

print("\n" + "=" * 60)
print("VISUALIZATION COMPLETED!")
print("=" * 60)
print(f"\nGenerated 5 plots in: {IMG_PATH}/")
print("\nFiles created:")
print("  1. 04_correlation_matrix_updated.png")
print("  2. 04_class_balance_updated.png")
print("  3. 04_eth_features_distribution.png")
print("  4. 04_cross_asset_features.png")
print("  5. 04_target_correlation.png")
print("\n✅ Ready for presentation!")