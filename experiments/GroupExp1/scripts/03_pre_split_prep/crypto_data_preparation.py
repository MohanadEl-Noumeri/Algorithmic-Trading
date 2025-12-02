import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.linear_model import LinearRegression

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
        df[f"ema_{w}"] = df["close"].ewm(span=w).mean()
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
  #  df["volume_norm"] = (df["volume"] - df["volume"].mean()) / df["volume"].std()
    return df


# --- 7. Zielvariable: Steigung per Linear Regression ---
def compute_target_normalized(df, minutes):
    # 1. Daten vorbereiten
    prices = df["close"].values

    # 2. Sliding Window View (Vektorisierung)
    windows = np.lib.stride_tricks.sliding_window_view(prices, window_shape=minutes)

    # 3. x-Achse vorbereiten (0, 1, ..., minutes-1)
    x = np.arange(minutes)
    x_mean = x.mean()

    # 4. Steigung (Slope) berechnen (Vektorisiert)
    # Formel: Sum((x - x_mean) * (y - y_mean)) / Sum((x - x_mean)^2)

    denominator = np.sum((x - x_mean) ** 2)
    y_mean = np.mean(windows, axis=1, keepdims=True)
    numerator = np.sum((windows - y_mean) * (x - x_mean), axis=1)

    raw_slopes = numerator / denominator

    # --- Normalisierung ---
    # Teilen des Slopes durch den Preis am Anfang des Fensters.

    current_prices = prices[:len(raw_slopes)]

    norm_slopes = raw_slopes / current_prices

    # 5. Padding für das DataFrame
    pad = np.full(minutes - 1, np.nan)
    full_slopes = np.concatenate([norm_slopes, pad])

    # --- SPEICHERN ---

    # Target 1: Binär (Bleibt gleich, da das Vorzeichen sich durch /Preis nicht ändert)
    df[f"target_{minutes}m"] = (full_slopes > 0).astype(int)

    # Target 2 (TBC): Die tatsächliche Trendstärke
    # df[f"target_{minutes}m_strength"] = full_slopes

    return df


# --- 8. Data Preparation Pipeline ---
for symbol_file in symbols:
    print(f"\n--- Data Preparation: {symbol_file} ---")

    file_path = data_path / f"{symbol_file}.parquet"
    df = pd.read_parquet(file_path)

    # --- 8a: Feature Engineering ---
    df = (
        df.pipe(add_log_return)
          .pipe(add_emas)
          .pipe(add_ema_differences)
          .pipe(add_ema_slope)
          .pipe(normalize)
    )

    # --- 8b: Targets generieren ---
    for m in TARGET_WINDOWS:
        df = compute_target_normalized(df, m)

    # --- 8c: Cleaning ---
    df = df.dropna().reset_index(drop=True)

    # --- 8d: Speichern ---
    out_file = data_path / f"{symbol_file}_prepared.parquet"
    df.to_parquet(out_file, index=False)

    print(f"- Saved: {out_file}")
    print(f"- Shape: {df.shape}")
