import numpy as np
import pandas as pd
from pathlib import Path
import yaml

# ----------------------------------------------------
# Step 04 — Split & Shuffle Data (Crypto, 1m OHLCV)
# ----------------------------------------------------

# 1) Config laden
params = yaml.safe_load(open("../../conf/params.yaml"))
data_path = Path(params["DATA_ACQUISITON"]["DATA_PATH"])

# Prepared-Dateien aus Step 03
symbols_prepared = [
    "BTCUSD_1m_raw_prepared",
    "ETHUSD_1m_raw_prepared",
]

# Split-Verhältnisse
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # Rest wird Test


def time_based_split(df: pd.DataFrame):
    """
    Zeitbasierter Split für eine einzelne Symbol-Zeitreihe.
    Kein Shuffling hier, nur chronologische Trennung.
    """
    df = df.copy()

    # Timestamp sicherstellen + sortieren
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)
    else:
        # Fallback: nach Index sortiert (sollte bei dir nicht nötig sein)
        df = df.reset_index(drop=True)

    n = len(df)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val:].reset_index(drop=True)

    return train_df, val_df, test_df


def main():
    all_train = []
    all_val = []
    all_test = []

    for symbol_name in symbols_prepared:
        in_path = data_path / f"{symbol_name}.parquet"

        if not in_path.exists():
            print(f"[WARN] Datei {in_path} existiert nicht – wird übersprungen.")
            continue

        print(f"\n--- Split für {symbol_name} ---")
        df = pd.read_parquet(in_path)

        # Symbol-Info ergänzen, falls sie später gebraucht wird
        if "symbol" not in df.columns:
            df["symbol"] = symbol_name

        train_df, val_df, test_df = time_based_split(df)

        print(
            f"{symbol_name}: total={len(df)}, "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

        all_train.append(train_df)
        all_val.append(val_df)
        all_test.append(test_df)

    # Wenn gar nichts geladen wurde
    if not all_train:
        print("\n[ERROR] Keine Daten geladen – bitte Pfade / Filenamen prüfen.")
        return

    # 2) Globales Zusammenführen & Shuffling (symbol-übergreifend)
    train_global = pd.concat(all_train, ignore_index=True)
    val_global = pd.concat(all_val, ignore_index=True)
    test_global = pd.concat(all_test, ignore_index=True)

    # Shuffling mit fester Seed für Reproduzierbarkeit
    train_shuffled = train_global.sample(frac=1.0, random_state=42).reset_index(drop=True)
    val_shuffled = val_global.sample(frac=1.0, random_state=42).reset_index(drop=True)
    test_shuffled = test_global.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # 3) Speichern – globale, geshuffelte Splits
    train_path_out = data_path / "crypto_train_shuffled.parquet"
    val_path_out = data_path / "crypto_val_shuffled.parquet"
    test_path_out = data_path / "crypto_test_shuffled.parquet"

    train_shuffled.to_parquet(train_path_out, index=False)
    val_shuffled.to_parquet(val_path_out, index=False)
    test_shuffled.to_parquet(test_path_out, index=False)

    print("\n--- Fertig ---")
    print("Train:", train_shuffled.shape, "->", train_path_out)
    print("Val:  ", val_shuffled.shape, "->", val_path_out)
    print("Test: ", test_shuffled.shape, "->", test_path_out)


if __name__ == "__main__":
    main()
