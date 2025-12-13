# Python - Data Acquisition für Krypto (1-Minute OHLCV) mit NaN-Filtering

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd
from datetime import datetime
import yaml
from pathlib import Path
import numpy as np

# --- 1. API Credentials laden ---
keys = yaml.safe_load(open("../../conf/keys.yaml"))
API_KEY = keys['KEYS']['APCA-API-KEY-ID']
SECRET_KEY = keys['KEYS']['APCA-API-SECRET-KEY']

# --- 2. Parameter für Datenabruf ---
params = yaml.safe_load(open("../../conf/params.yaml"))
PATH_BARS = params['DATA_ACQUISITON']['DATA_PATH']
START_DATE = datetime.strptime(params['DATA_ACQUISITON']['START_DATE'], "%Y-%m-%d")
END_DATE = datetime.strptime(params['DATA_ACQUISITON']['END_DATE'], "%Y-%m-%d")

# sicherstellen, dass das Zielverzeichnis existiert
base_path = Path(PATH_BARS)
base_path.mkdir(parents=True, exist_ok=True)

# --- 3. Alpaca Crypto Client initialisieren ---
client = CryptoHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)

# --- 4. Liste der Symbole für Krypto ---
symbols = ["BTC/USD", "ETH/USD"]

# --- 5. Daten abrufen und speichern ---
for symbol in symbols:
    print(f"\n{'=' * 60}")
    print(f"Fetching 1-Minute bars for {symbol}")
    print(f"From: {START_DATE.date()} to {END_DATE.date()}")
    print(f"{'=' * 60}")

    request = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=START_DATE,
        end=END_DATE
    )

    bars = client.get_crypto_bars(request)
    df = bars.df
    df = df[df.index.get_level_values("symbol") == symbol]
    df.reset_index(inplace=True)
    if 'symbol' in df.columns:
        df.drop(columns=['symbol'], inplace=True)

    print(f"\nInitial shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # --- DATA QUALITY CHECKS ---

    # 1. Check for NaN values
    print(f"\n--- NaN Check ---")
    nan_counts = df.isnull().sum()
    total_nans = nan_counts.sum()

    if total_nans > 0:
        print(f"⚠️  Found {total_nans} NaN values:")
        for col in nan_counts[nan_counts > 0].index:
            print(f"  {col}: {nan_counts[col]} NaNs ({nan_counts[col] / len(df) * 100:.2f}%)")
    else:
        print("✓ No NaN values found")

    # 2. Check for Inf values
    print(f"\n--- Inf Check ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = inf_count

    if inf_counts:
        print(f"⚠️  Found Inf values:")
        for col, count in inf_counts.items():
            print(f"  {col}: {count} Infs ({count / len(df) * 100:.2f}%)")
    else:
        print("✓ No Inf values found")

    # 3. Check for duplicate timestamps
    print(f"\n--- Duplicate Check ---")
    duplicates = df['timestamp'].duplicated().sum()
    if duplicates > 0:
        print(f"⚠️  Found {duplicates} duplicate timestamps")
    else:
        print("✓ No duplicate timestamps")

    # 4. Check for zero/negative prices
    print(f"\n--- Price Validity Check ---")
    zero_close = (df['close'] <= 0).sum()
    zero_volume = (df['volume'] <= 0).sum()

    if zero_close > 0:
        print(f"⚠️  Found {zero_close} rows with close <= 0")
    if zero_volume > 0:
        print(f"⚠️  Found {zero_volume} rows with volume <= 0")

    if zero_close == 0 and zero_volume == 0:
        print("✓ All prices and volumes are positive")

    # 5. Check for gaps in timestamps
    print(f"\n--- Timestamp Gap Check ---")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    time_diffs = df['timestamp'].diff().dt.total_seconds() / 60  # in minutes

    # Expected: 1 minute between bars
    gaps = time_diffs[time_diffs > 1.5]  # Allow 50% tolerance
    if len(gaps) > 0:
        print(f"⚠️  Found {len(gaps)} gaps in timestamps (>1.5 min)")
        print(f"  Max gap: {gaps.max():.1f} minutes")
        print(f"  Mean gap: {gaps.mean():.1f} minutes")
    else:
        print("✓ No significant gaps in timestamps")

    # --- CLEANING ---
    print(f"\n--- Cleaning Data ---")
    initial_rows = len(df)

    # 1. Remove NaNs
    if total_nans > 0:
        df = df.dropna()
        print(f"✓ Removed {initial_rows - len(df)} rows with NaN")

    # 2. Remove Infs
    if inf_counts:
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        print(f"✓ Removed rows with Inf values")

    # 3. Remove duplicates
    if duplicates > 0:
        df = df.drop_duplicates(subset=['timestamp'])
        print(f"✓ Removed {duplicates} duplicate timestamps")

    # 4. Remove invalid prices
    if zero_close > 0 or zero_volume > 0:
        df = df[(df['close'] > 0) & (df['volume'] >= 0)]
        print(f"✓ Removed rows with invalid prices/volumes")

    # 5. Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    final_rows = len(df)
    removed = initial_rows - final_rows

    print(f"\n--- Summary ---")
    print(f"Initial rows: {initial_rows:,}")
    print(f"Final rows:   {final_rows:,}")
    print(f"Removed:      {removed:,} ({removed / initial_rows * 100:.2f}%)")
    print(f"Date range:   {df['timestamp'].min()} to {df['timestamp'].max()}")

    # --- STATISTICS ---
    print(f"\n--- Statistics ---")
    print(f"Close price:")
    print(f"  Min:  ${df['close'].min():.2f}")
    print(f"  Max:  ${df['close'].max():.2f}")
    print(f"  Mean: ${df['close'].mean():.2f}")
    print(f"  Std:  ${df['close'].std():.2f}")

    print(f"Volume:")
    print(f"  Min:  {df['volume'].min():.2f}")
    print(f"  Max:  {df['volume'].max():.2f}")
    print(f"  Mean: {df['volume'].mean():.2f}")

    # --- SAVE ---
    print(f"\n--- Saving ---")

    csv_path = f"{PATH_BARS}/{symbol.replace('/', '')}_1m_raw.csv"
    parquet_path = f"{PATH_BARS}/{symbol.replace('/', '')}_1m_raw.parquet"

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    print(f"✓ Saved CSV:     {csv_path}")
    print(f"✓ Saved Parquet: {parquet_path}")
    print(f"✓ Final shape:   {df.shape}")

print(f"\n{'=' * 60}")
print("DATA ACQUISITION COMPLETED!")
print(f"{'=' * 60}")