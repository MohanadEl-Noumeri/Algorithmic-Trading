import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
from pathlib import Path
import pickle

# --- Konfiguration laden ---
params = yaml.safe_load(open("../../conf/params.yaml"))
DATA_PATH = params["DATA_ACQUISITON"]["DATA_PATH"]
MODEL_PATH = params["MODELING"]["MODEL_PATH"]
test_file = Path(DATA_PATH) / "test.parquet"

# --- 1. Modell und Daten laden ---
scaler = pickle.load(open(os.path.join(MODEL_PATH, "feature_scaler.pkl"), "rb"))

# Modell Definition (unverändert)
class MLP(torch.nn.Module):
    def __init__(self, in_dim, h1, h2, dropout_p):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, h1),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_p),
            torch.nn.Linear(h1, h2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_p),
            torch.nn.Linear(h2, 1)
        )
    def forward(self, x): return self.net(x)

checkpoint = torch.load(os.path.join(MODEL_PATH, "best_model.pt"))
config = checkpoint['config']
model = MLP(config['in_dim'], config['h1'], config['h2'], config['dropout'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ---------------------------------------------------------
# KORREKTUR: Raw Data dazu laden für 'log_return' und 'close'
# ---------------------------------------------------------
print("Loading Test Data...")
df = pd.read_parquet(test_file)

# Wir laden die originale BTC Raw Datei, um 'close' und 'log_return' wiederzubekommen
btc_raw_path = Path(DATA_PATH) / "BTCUSD_1m_raw.parquet"
df_raw = pd.read_parquet(btc_raw_path)

# Log Returns im Raw-File berechnen (falls noch nicht drin oder um sicher zu gehen)
df_raw['log_return'] = np.log(df_raw['close'] / df_raw['close'].shift(1))

# Wir mergen 'close' und 'log_return' anhand des Timestamps zurück in unser Test-Set
# Wichtig: how='inner' oder 'left', damit die Zeilen zum Test-Set passen
df = pd.merge(df, df_raw[['timestamp', 'close', 'log_return']], on='timestamp', how='left')

# NaNs entfernen (die durch den Shift im Raw File entstanden sein könnten, falls Test am Anfang liegt)
df = df.dropna(subset=['log_return', 'close']).reset_index(drop=True)

# Features Liste laden (oder definieren)
with open(params["DATA_PREP"]["FEATURE_PATH"], "r") as f:
    features = [line.strip() for line in f.readlines()]

print(f"Loaded {len(features)} features from text file.")
# ---------------------------------------------------------

print(f"Data ready for backtest. Shape: {df.shape}")

# --- 2. Inference (Vorhersage) ---
X_raw = df[features].values
X_scaled = scaler.transform(X_raw)
with torch.no_grad():
    logits = model(torch.tensor(X_scaled).float())
    probs = torch.sigmoid(logits).numpy().flatten()

df['prob_up'] = probs

# --- 3. Trading Algorithmus (Die Logik ableiten) ---
# Frage: How to specify entry and exit points?
# Antwort: Wir nutzen Schwellenwerte (Thresholds).

CONFIDENCE_THRESHOLD = 0.60  # Nur kaufen, wenn Modell > 60% sicher ist
EXIT_THRESHOLD = 0.50        # Verkaufen, wenn Wahrscheinlichkeit sinkt

# Signal generieren (1 = Long, 0 = Cash)
df['signal'] = 0
df.loc[df['prob_up'] > CONFIDENCE_THRESHOLD, 'signal'] = 1

# Optional: Position halten, solange prob > EXIT_THRESHOLD (Hystere-Effekt)
# Das verhindert ständiges Rein/Raus bei 0.59 -> 0.61 -> 0.59
df['position'] = df['signal'] # Simple version: Position = Signal

# Shift position by 1: Wir handeln HEUTE basierend auf dem Signal von GESTERN/VORHIN
df['position_delayed'] = df['position'].shift(1)

# --- 4. Performance Berechnung ---
# Kosten simulieren (z.B. 0.1% pro Trade)
cost_per_trade = 0.001
trades = df['position'].diff().abs() # Wann ändert sich die Position?
costs = trades * cost_per_trade

# Strategie Return: Markt-Return * Position - Kosten
# Wir nutzen die BTC log returns, da wir BTC traden
df['strategy_log_return'] = (df['log_return'] * df['position_delayed']) - costs
df['strategy_cum_return'] = df['strategy_log_return'].cumsum().apply(np.exp)
df['market_cum_return'] = df['log_return'].cumsum().apply(np.exp)

# --- 5. Metriken & Analyse ---
print(f"{'='*60}\nBACKTEST RESULTS\n{'='*60}")
total_return = df['strategy_cum_return'].iloc[-1] - 1
market_return = df['market_cum_return'].iloc[-1] - 1
win_rate = (df[df['position_delayed']==1]['log_return'] > 0).mean()

print(f"Total Return Strategy: {total_return*100:.2f}%")
print(f"Total Return Market:   {market_return*100:.2f}%")
print(f"Win Rate (when invested): {win_rate*100:.2f}%")
print(f"Number of Trades: {trades.sum()}")

# --- 6. Plots (Visualisierung) ---
#
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['market_cum_return'], label='Buy & Hold (BTC)', alpha=0.5)
plt.plot(df['timestamp'], df['strategy_cum_return'], label='ML Strategy', color='green')
plt.title('Equity Curve: ML Model vs. Buy & Hold')
plt.legend()
plt.show()

#
# die ersten 500 Minuten
subset = df.iloc[:500]
plt.figure(figsize=(12, 6))
plt.plot(subset['timestamp'], subset['close'], label='Price', alpha=0.5)
# Buy Signals (Green Triangles)
buys = subset[subset['position'].diff() == 1]
sells = subset[subset['position'].diff() == -1]
plt.scatter(buys['timestamp'], buys['close'], marker='^', color='green', s=100, label='Buy')
plt.scatter(sells['timestamp'], sells['close'], marker='v', color='red', s=100, label='Sell')
plt.title('Trade Entries & Exits (Zoomed In)')
plt.legend()
plt.show()

# Verteilung der Wahrscheinlichkeiten
plt.figure(figsize=(10, 4))
sns.histplot(df['prob_up'], bins=50, kde=True)
plt.axvline(x=0.5, color='red', linestyle='--')
plt.title('Distribution of Model Probabilities')
plt.xlabel('Probability Up')
plt.show()