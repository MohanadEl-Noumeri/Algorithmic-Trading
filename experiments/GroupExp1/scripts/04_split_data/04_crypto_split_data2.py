import numpy as np
import pandas as pd
from pathlib import Path
import yaml

# ----------------------------------------------------
# Step 05 — Train/Val/Test Split
# Simple chronological split
# ----------------------------------------------------

params = yaml.safe_load(open("../../conf/params.yaml"))
data_path = Path(params["DATA_ACQUISITON"]["DATA_PATH"])

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # Rest = 0.15 Test

print("=" * 60)
print("TRAIN/VAL/TEST SPLIT")
print("=" * 60)


def time_split(df):
    """Split dataframe chronologically into train/val/test."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    train = df.iloc[:n_train].reset_index(drop=True)
    val = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test = df.iloc[n_train + n_val:].reset_index(drop=True)

    print(f"\nSplit Summary:")
    print(f"  Train: {len(train):,} rows ({TRAIN_FRAC * 100:.0f}%)")
    print(f"    Date: {train['timestamp'].min()} to {train['timestamp'].max()}")
    print(f"  Val:   {len(val):,} rows ({VAL_FRAC * 100:.0f}%)")
    print(f"    Date: {val['timestamp'].min()} to {val['timestamp'].max()}")
    print(f"  Test:  {len(test):,} rows ({(1 - TRAIN_FRAC - VAL_FRAC) * 100:.0f}%)")
    print(f"    Date: {test['timestamp'].min()} to {test['timestamp'].max()}")

    return train, val, test


# --- Load combined data ---
combined_file = data_path / "btc_eth_combined.parquet"
df = pd.read_parquet(combined_file)

print(f"\nLoaded: {df.shape}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# --- Split ---
train, val, test = time_split(df)

# --- Check Target Balance ---
print("\n" + "=" * 60)
print("TARGET BALANCE CHECK")
print("=" * 60)

target_cols = [c for c in train.columns if c.startswith('target_')]

if target_cols:
    example_target = target_cols[0]
    print(f"\nBalance for {example_target}:")

    for split_name, split_df in [("Train", train), ("Val", val), ("Test", test)]:
        balance = split_df[example_target].value_counts(normalize=True)
        print(f"  {split_name:5s}: Down={balance.get(0, 0):.1%}, Up={balance.get(1, 0):.1%}")

# --- Save Splits ---
print("\n" + "=" * 60)
print("SAVING SPLITS")
print("=" * 60)

train.to_parquet(data_path / "train.parquet", index=False)
val.to_parquet(data_path / "val.parquet", index=False)
test.to_parquet(data_path / "test.parquet", index=False)

print(f"✓ train.parquet: {train.shape}")
print(f"✓ val.parquet:   {val.shape}")
print(f"✓ test.parquet:  {test.shape}")

# --- Summary ---
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

feature_cols = [c for c in train.columns if c not in ['timestamp'] and not c.startswith('target_')]

print(f"\nDataset:")
print(f"  Total rows: {len(df):,}")
print(f"  Features: {len(feature_cols)}")
print(f"  Targets: {len(target_cols)}")

print("SPLIT COMPLETED!")
