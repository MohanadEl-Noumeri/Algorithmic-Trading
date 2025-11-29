
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

    # 4c: Volumen über Zeit
    plt.figure(figsize=(16, 6))
    plt.plot(df['timestamp'], df['volume'], label='Volume', color='green', linewidth=1)
    plt.title(f"{symbol_file} - Trading Volume over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Volume")
    plt.legend()
    plt.show()


    # --- 5. Erste Findings ---
    print(f"\nFindings for {symbol_file}:")
    print(f"- Total rows: {len(df)}")
    print(f"- Start date: {df['timestamp'].min()}, End date: {df['timestamp'].max()}")
    print(f"- Mean close price: {df['close'].mean():.2f}, Std dev: {df['close'].std():.2f}")
    print(f"- Mean volume: {df['volume'].mean():.2f}, Std dev: {df['volume'].std():.2f}")
    print(f"- Log-Return mean: {df['log_return'].mean():.6f}, std: {df['log_return'].std():.6f}")
