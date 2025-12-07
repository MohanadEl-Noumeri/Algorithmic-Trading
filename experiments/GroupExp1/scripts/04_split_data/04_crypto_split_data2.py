import numpy as np
import pandas as pd
from pathlib import Path
import yaml

# ----------------------------------------------------
# Step 04 — Safe Multi-Asset Feature Dataset
# ----------------------------------------------------

params = yaml.safe_load(open("../../conf/params.yaml"))
data_path = Path(params["DATA_ACQUISITON"]["DATA_PATH"])

symbols = {
    "BTC": "BTCUSD_1m_raw_prepared",
    "ETH": "ETHUSD_1m_raw_prepared",
}

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # Rest = Test


def time_split(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    train = df.iloc[:n_train].reset_index(drop=True)
    val = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test = df.iloc[n_train + n_val:].reset_index(drop=True)

    return train, val, test


# --- Load both symbols ---
df_btc = pd.read_parquet(data_path / f"{symbols['BTC']}.parquet")
df_eth = pd.read_parquet(data_path / f"{symbols['ETH']}.parquet")

# --- Synchronize once ---
df = (
    df_btc.set_index("timestamp")
          .join(df_eth.set_index("timestamp").add_prefix("eth_"), how="left")
          .reset_index()
)

# --- Split cleanly ---
train, val, test = time_split(df)

# --- Save ---
train.to_parquet(data_path / "train.parquet", index=False)
val.to_parquet(data_path / "val.parquet", index=False)
test.to_parquet(data_path / "test.parquet", index=False)

print("Train:", train.shape)
print("Val:", val.shape)
print("Test:", test.shape)
