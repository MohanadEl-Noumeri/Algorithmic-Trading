import pandas as pd
import numpy as np
import yaml
from pathlib import Path

# --- 1. Parameter laden ---
params = yaml.safe_load(open("../../conf/params.yaml"))
data_path = Path(params['DATA_ACQUISITON']['DATA_PATH'])

symbols = ["BTCUSD_1m_raw", "ETHUSD_1m_raw"]
EMAS = [5, 10, 15, 20, 30, 60, 120, 240]
TARGET_WINDOWS = [5, 10, 15, 20, 30, 60, 120, 240]


# --- 2. Hilfsfunktion: Log-Returns ---
def add_log_return(df):
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


# --- 3. Hilfsfunktion: EMAs berechnen ---
def add_emas(df):
    for w in EMAS:
        df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()
    return df


# --- 4. EMA-Differenzen ---
def add_ema_differences(df):
    df["ema_10_5"] = df["ema_10"] - df["ema_5"]
    df["ema_30_10"] = df["ema_30"] - df["ema_10"]
    return df


# --- 5. EMA-Slope ---
def add_ema_slope(df, window=5):
    df["ema10_slope"] = df["ema_10"].diff(window) / window
    return df


# --- 6. Normalisierung ---
def normalize(df):
    df["close_norm"] = (df["close"] - df["close"].mean()) / df["close"].std()
    return df


# --- 7. Zielvariable: Steigung per Linear Regression ---
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


# --- 8. Volatilität ---
def add_volatility(df, window=30):
    df[f"volatility_{window}"] = df["log_return"].rolling(window=window).std()
    return df


# --- 9. RSI (Relative Strength Index) ---
def add_rsi(df, window=14):
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, 0.000001)
    df[f"rsi_{window}"] = 100 - (100 / (1 + rs))
    df[f"rsi_{window}_norm"] = df[f"rsi_{window}"] / 100.0
    df = df.drop(columns=[f"rsi_{window}"])

    return df


# --- 10. MACD ---
def add_macd(df):
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


# --- 11. Bollinger Bands ---
def add_bollinger_bands(df, window=20):
    rolling_mean = df["close"].rolling(window).mean()
    rolling_std = df["close"].rolling(window).std()
    df["bb_upper"] = rolling_mean + (2 * rolling_std)
    df["bb_lower"] = rolling_mean - (2 * rolling_std)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / rolling_mean
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)
    return df


# --- 12. ATR ---
def add_atr(df, window=14):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(window).mean()
    df["atr_pct"] = df["atr"] / df["close"]
    return df


# --- 13. ROC ---
def add_roc(df, periods=[5, 10, 20, 30]):
    for p in periods:
        df[f"roc_{p}"] = ((df["close"] - df["close"].shift(p)) / df["close"].shift(p)) * 100
    return df


# --- 14. Stochastic ---
def add_stochastic(df, window=14):
    low_min = df["low"].rolling(window).min()
    high_max = df["high"].rolling(window).max()
    df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-10)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    return df


# ============================================================
# MAIN PIPELINE
# ============================================================

print("=" * 60)
print("DATA PREPARATION")
print("Strategy: ETH Features → BTC Targets")
print("=" * 60)

# --- Process BTC (for targets only) ---
print(f"\n--- Processing BTC (targets only) ---")
btc_file = data_path / "BTCUSD_1m_raw.parquet"
df_btc_raw = pd.read_parquet(btc_file)

print(f"BTC shape: {df_btc_raw.shape}")

# Only compute targets for BTC
df_btc = df_btc_raw.copy()
for m in TARGET_WINDOWS:
    df_btc = compute_target_normalized(df_btc, m)

# Keep only timestamp + targets
btc_keep_cols = ['timestamp'] + [c for c in df_btc.columns if c.startswith('target_')]
df_btc_targets = df_btc[btc_keep_cols].copy()

print(f"BTC targets: {len(btc_keep_cols) - 1}")

# --- Process ETH (full feature engineering) ---
print(f"\n--- Processing ETH (full features) ---")
eth_file = data_path / "ETHUSD_1m_raw.parquet"
df_eth = pd.read_parquet(eth_file)

print(f"ETH shape: {df_eth.shape}")

# Full pipeline for ETH
df_eth = (
    df_eth.pipe(add_log_return)
    .pipe(add_emas)
    .pipe(add_ema_differences)
    .pipe(add_ema_slope)
    .pipe(add_volatility, window=30)
    .pipe(add_rsi, window=14)
    .pipe(add_macd)
    .pipe(add_bollinger_bands)
    .pipe(add_atr)
    .pipe(add_roc)
    .pipe(add_stochastic)
    .pipe(normalize)
)

# Select only the most important 15 features
# Based on typical feature importance for crypto
selected_features = [
    'timestamp',
    'log_return',  # 1. Momentum
    'ema_10',  # 2. Short-term trend
    'ema_30',  # 3. Medium-term trend
    'ema_60',  # 4. Longer-term trend
    'ema_30_10',  # 5. Trend strength
    'ema10_slope',  # 6. Trend direction
    'volatility_30',  # 7. Risk/nervousness
    'rsi_14_norm',  # 8. Overbought/oversold
    'macd_hist',  # 9. Momentum indicator
    'bb_position',  # 10. Relative price position
    'atr_pct',  # 11. Volatility measure
    'roc_5',  # 12. Short-term rate of change
    'roc_10',  # 13. Medium-term rate of change
    'stoch_k',  # 14. Stochastic momentum
    'stoch_d'  # 15. Smoothed stochastic
]

df_eth_features = df_eth[selected_features].copy()

# Rename with eth_ prefix
rename_dict = {c: f'eth_{c}' if c != 'timestamp' else c for c in df_eth_features.columns}
df_eth_features = df_eth_features.rename(columns=rename_dict)

print(f"ETH features (selected): {len(selected_features) - 1}")

# --- Join BTC targets + ETH features ---
print(f"\n--- Joining BTC targets + ETH features ---")

# Use index-based join (better for time series)
df_combined = (
    df_btc_targets.set_index('timestamp')
    .join(df_eth_features.set_index('timestamp'), how='inner')
    .reset_index()
)

print(f"Combined shape: {df_combined.shape}")
print(f"Overlap: {len(df_combined) / len(df_btc_targets) * 100:.1f}%")

# --- Add Cross-Asset Features ---
print(f"\n--- Adding cross-asset features ---")

# Use raw BTC and ETH data for cross-asset calculations
df_btc_prices = df_btc_raw[['timestamp', 'close']].set_index('timestamp')
df_eth_prices = df_eth[['timestamp', 'close', 'log_return']].set_index('timestamp')

# Calculate BTC log return for cross-asset
df_btc_raw['log_return'] = np.log(df_btc_raw['close'] / df_btc_raw['close'].shift(1))
df_btc_log_return = df_btc_raw[['timestamp', 'log_return']].set_index('timestamp')

# Join prices and returns
df_combined = df_combined.set_index('timestamp')
df_combined = df_combined.join(df_btc_prices, how='left')
df_combined = df_combined.join(df_btc_log_return, rsuffix='_btc', how='left')
df_combined = df_combined.join(df_eth_prices[['close', 'log_return']], rsuffix='_eth', how='left')
df_combined = df_combined.reset_index()

# 1. BTC/ETH Ratio
df_combined['btc_eth_ratio'] = df_combined['close'] / df_combined['close_eth']

# 2. Return Difference
df_combined['btc_eth_return_diff'] = df_combined['log_return'] - df_combined['log_return_eth']

# 3. Correlation (60min)
df_combined['btc_eth_corr_60'] = (
    df_combined['log_return']
    .rolling(60)
    .corr(df_combined['log_return_eth'])
)

# Drop temporary columns
df_combined = df_combined.drop(columns=['close', 'log_return', 'close_eth', 'log_return_eth'])

print(f"Added 3 cross-asset features")

# --- Clean NaNs ---
print(f"\n--- Cleaning ---")
before = len(df_combined)
df_combined = df_combined.dropna().reset_index(drop=True)
after = len(df_combined)

print(f"Removed {before - after} rows with NaN ({(before - after) / before * 100:.2f}%)")

# --- Save ---
output_file = data_path / "btc_eth_combined.parquet"
df_combined.to_parquet(output_file, index=False)

print(f"\n✓ Saved: {output_file}")
print(f"✓ Final shape: {df_combined.shape}")

# --- Summary ---
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

feature_cols = [c for c in df_combined.columns if c not in ['timestamp'] and not c.startswith('target_')]
target_cols = [c for c in df_combined.columns if c.startswith('target_')]

eth_base = [f for f in feature_cols if f.startswith('eth_')]
cross_asset = [f for f in feature_cols if 'btc_eth' in f]

print(f"\nColumns:")
print(f"  ETH features: {len(eth_base)}")
print(f"  Cross-asset features: {len(cross_asset)}")
print(f"  Total features: {len(feature_cols)}")
print(f"  BTC targets: {len(target_cols)}")

print(f"\nData:")
print(f"  Rows: {len(df_combined):,}")
print(f"  Date range: {df_combined['timestamp'].min()} to {df_combined['timestamp'].max()}")

print("\n" + "=" * 60)
print("DATA PREPARATION COMPLETED!")
print("=" * 60)
print(f"\nNext: Run split")
print(f"  python scripts/05_train_test_split.py")