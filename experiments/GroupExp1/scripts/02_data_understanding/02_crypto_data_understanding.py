
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path
import numpy as np

sns.set(style="whitegrid")

# --- 1. Parameter laden ---
params = yaml.safe_load(open("../../conf/params.yaml"))

data_path = Path(params['DATA_ACQUISITON']['DATA_PATH'])
symbols = ["BTCUSD_1m_raw", "ETHUSD_1m_raw"]


# --- 2. Hilfsfunktion: Log-Returns berechnen ---
def add_log_returns(df, price_col="close"):
    df['log_return'] = np.log(df[price_col] / df[price_col].shift(1))
    return df


# --- 3. Data Understanding ---
for symbol_file in symbols:
    file_path = data_path / f"{symbol_file}.parquet"
    if not file_path.exists():
        print(f"File {file_path} does not exist. Skipping.")
        continue

    print(f"\n--- Data Understanding: {symbol_file} ---")

    # 3a: Daten laden
    df = pd.read_parquet(file_path)

    # 3b: Spalten anzeigen
    print("Columns in dataset:", df.columns.tolist())

    # 3c: Datentypen und Nullwerte
    print("\nInfo:")
    print(df.info())

    # 3d: Deskriptive Statistik für numerische Spalten
    print("\nDescriptive statistics:")
    print(df.describe())

    # 3e: Log-Returns hinzufügen
    df = add_log_returns(df)

    # --- 4. Plots ---

    # 4a: Zeitreihe Close-Price
    plt.figure(figsize=(16, 6))
    plt.plot(df['timestamp'], df['close'], label='Close Price', color='blue', linewidth=1)
    plt.title(f"{symbol_file} - Close Price over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Close Price")
    plt.legend()
    plt.show()

    # 4b: Histogramm der Log-Returns
    plt.figure(figsize=(10, 5))
    sns.histplot(df['log_return'].dropna(), bins=100, kde=True, color='purple')
    plt.title(f"{symbol_file} - Distribution of 1-min Log-Returns")
    plt.xlabel("Log-Return")
    plt.ylabel("Frequency")
    plt.show()

    '''
    # 4c: Volumen über Zeit
    plt.figure(figsize=(16, 6))
    plt.plot(df['timestamp'], df['trade_count'], label='trade_count', color='green', linewidth=1)
    plt.title(f"{symbol_file} - VWAP over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Trade Count")
    plt.legend()
    plt.show()
    '''

    # 4d: Absolute Returns (sehr stabil)
    df["abs_return"] = df["close"].pct_change().abs()

    plt.figure(figsize=(16, 6))
    plt.plot(df["timestamp"], df["abs_return"], label="Absolute Returns", linewidth=1)
    plt.title(f"{symbol_file} - Absolute Returns over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("|Return|")
    plt.legend()
    plt.show()

    # 4e: Candle Range (High-Low)
    df["range"] = df["high"] - df["low"]

    plt.figure(figsize=(16, 6))
    plt.plot(df["timestamp"], df["range"], label="Candle Range (High-Low)", linewidth=1)
    plt.title(f"{symbol_file} - Candle Range over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Range")
    plt.legend()
    plt.show()

    # 4f: Normalisierte Range (Range / Close)
    df["range_norm"] = df["range"] / df["close"]

    plt.figure(figsize=(16, 6))
    plt.plot(df["timestamp"], df["range_norm"], label="Normalized Range", linewidth=1)
    plt.title(f"{symbol_file} - Normalized Range over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Range / Close")
    plt.legend()
    plt.show()

    # 4g: Rolling Volatility (10 Bars)
    df["vol10"] = df["log_return"].rolling(10).std()

    plt.figure(figsize=(16, 6))
    plt.plot(df["timestamp"], df["vol10"], label="Rolling Volatility (10 bars)", linewidth=1)
    plt.title(f"{symbol_file} - Rolling 10-Bar Volatility")
    plt.xlabel("Timestamp")
    plt.ylabel("Volatility")
    plt.legend()
    plt.show()

    # --- 5. Erste Findings ---
    print(f"\nFindings for {symbol_file}:")
    print(f"- Total rows: {len(df)}")
    print(f"- Start date: {df['timestamp'].min()}, End date: {df['timestamp'].max()}")
    print(f"- Mean close price: {df['close'].mean():.2f}, Std dev: {df['close'].std():.2f}")
    print(f"- Mean volume: {df['volume'].mean():.2f}, Std dev: {df['volume'].std():.2f}")
    print(f"- Log-Return mean: {df['log_return'].mean():.6f}, std: {df['log_return'].std():.6f}")
