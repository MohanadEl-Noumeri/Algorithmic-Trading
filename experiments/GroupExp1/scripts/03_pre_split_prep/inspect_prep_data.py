import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

# 1. Config laden
params = yaml.safe_load(open("../../conf/params.yaml"))
data_path = Path(params['DATA_ACQUISITON']['DATA_PATH'])
IMG_PATH = Path("../../images") # Ordner für Readme-Bilder
IMG_PATH.mkdir(exist_ok=True)

# Wähle eine Datei zum Checken
symbol = "BTCUSD_1m_raw_prepared"
file_path = data_path / f"{symbol}.parquet"

print(f"--- Inspiziere: {file_path} ---")
df = pd.read_parquet(file_path)

# --- A. DESCRIPTIVE STATISTICS (für die Readme Tabelle) ---
print("\n1. Data Head (Beispielzeilen):")
print(df[["log_return", "ema_10", "ema10_slope", "target_15m"]].head())

print("\n2. Class Balance (Wie oft geht es hoch/runter?):")
target_cols = [c for c in df.columns if "target" in c]
for t in target_cols:
    vc = df[t].value_counts(normalize=True)
    print(f"{t}: Down={vc[0]:.1%}, Up={vc[1]:.1%}")

# --- B. PLOTS (für die Readme Bilder) ---

# Plot 1: Correlation Heatmap
plt.figure(figsize=(10, 8))
# Wir nehmen nur Features, keine Targets für die Korrelation
feature_cols = ["log_return", "ema_10", "ema_60", "ema_240", "ema_30_10", "ema10_slope", "volatility_30", "rsi_14_norm",
                #"volume_norm"
                ]
corr = df[feature_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(IMG_PATH / "04_correlation_matrix.png")
print(f"\n- Bild gespeichert: {IMG_PATH / '04_correlation_matrix.png'}")

# Plot 2: Class Balance Visualisierung
plt.figure(figsize=(8, 5))
# Wir plotten nur das 15m Target als Beispiel
sns.countplot(x="target_15m", data=df, palette="viridis")
plt.title("Target Distribution (15m Window)")
plt.xlabel("0 = Down, 1 = Up")
plt.ylabel("Anzahl")
plt.savefig(IMG_PATH / "04_class_balance.png")
print(f"- Bild gespeichert: {IMG_PATH / '04_class_balance.png'}")

plt.show()