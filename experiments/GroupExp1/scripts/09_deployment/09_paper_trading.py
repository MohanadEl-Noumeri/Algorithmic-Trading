import time
import os
import yaml
import logging
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Alpaca Libraries
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, CryptoLatestTradeRequest
from alpaca.data.timeframe import TimeFrame

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("paper_trading_bracket.log")
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# Pfade anpassen (Geht 3 Ordner hoch zum Root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONF_DIR = BASE_DIR / "conf"

# Config laden
try:
    params = yaml.safe_load(open(CONF_DIR / "params.yaml"))
    keys = yaml.safe_load(open(CONF_DIR / "keys.yaml"))
except FileNotFoundError:
    logger.error("Configuration files not found! Check path to keys.yaml/params.yaml")
    exit(1)

# API Keys (PAPER TRADING!)
# Wir nutzen hier die Keys, die du in keys.yaml definiert hast.
# Achte darauf, dass sie mit 'PK' anfangen!
API_KEY = keys['KEYS']['APCA-API-KEY-ID']
SECRET_KEY = keys['KEYS']['APCA-API-SECRET-KEY']

# Paths
DATA_PATH = Path(params["DATA_ACQUISITON"]["DATA_PATH"])
MODEL_PATH = Path(params["MODELING"]["MODEL_PATH"])
FEATURE_PATH = params["DATA_PREP"]["FEATURE_PATH"]

# --- STRATEGY SETTINGS ---
TRADING_PAIR = "BTC/USD"  # Was wir handeln
FEATURE_PAIR = "ETH/USD"  # Woraus wir Features ableiten
TIMEFRAME = TimeFrame.Minute
LOOKBACK_MIN = 300  # Historie für Indikatoren
TRADE_AMOUNT = 1000  # Dollar pro Trade

# --- TEST THRESHOLDS (NIEDRIG EINGESTELLT) ---
# Normal wäre 0.60. Wir nehmen 0.505, damit du HEUTE Action siehst.
CONFIDENCE_THRESHOLD = 0.0
EXIT_THRESHOLD = 0.500

# --- RISK MANAGEMENT (BRACKET ORDER) ---
TAKE_PROFIT_PCT = 1.015  # +1.5% Gewinnziel
STOP_LOSS_PCT = 0.9925  # -0.75% Stop Loss (Risk/Reward 2:1)


# --- MODEL CLASS (Muss identisch zum Training sein) ---
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


# --- INITIALIZATION ---
logger.info("Initializing Smart Bracket Bot...")

# 1. Clients
if not API_KEY.startswith("PK"):
    logger.warning("WARNUNG: Dein Key fängt nicht mit 'PK' an. Sicher, dass es Paper Trading ist?")

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = CryptoHistoricalDataClient(API_KEY, SECRET_KEY)

# 2. Features laden (aus Textdatei)
with open(FEATURE_PATH, "r") as f:
    FEATURE_LIST = [line.strip() for line in f.readlines()]
logger.info(f"Loaded {len(FEATURE_LIST)} features configuration.")

# 3. Scaler laden
scaler = pickle.load(open(MODEL_PATH / "feature_scaler.pkl", "rb"))

# 4. Modell laden
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MODEL_PATH / "best_model.pt", map_location=device)
config = checkpoint['config']

model = MLP(config['in_dim'], config['h1'], config['h2'], config['dropout'])
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

logger.info("System Ready.")


# --- FEATURE ENGINEERING (Identisch zum Training) ---
def add_log_return(df):
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


def add_emas(df, periods=[10, 30, 60]):
    for w in periods:
        df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()
    return df


def add_ema_differences(df):
    df["ema_30_10"] = df["ema_30"] - df["ema_10"]
    return df


def add_ema_slope(df, window=5):
    df["ema10_slope"] = df["ema_10"].diff(window) / window
    return df


def add_volatility(df, window=30):
    df[f"volatility_{window}"] = df["log_return"].rolling(window=window).std()
    return df


def add_rsi(df, window=14):
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, 0.000001)
    # WICHTIG: Hier normalisieren, falls im Training geschehen (0.0 bis 1.0)
    df[f"rsi_{window}_norm"] = (100 - (100 / (1 + rs))) / 100.0
    return df



def add_macd(df):
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd - signal
    return df


def add_bollinger_bands(df, window=20):
    rm = df["close"].rolling(window).mean()
    rstd = df["close"].rolling(window).std()
    upper = rm + (2 * rstd)
    lower = rm - (2 * rstd)
    df["bb_position"] = (df["close"] - lower) / (upper - lower + 1e-10)
    return df


def add_atr(df, window=14):
    h_l = df["high"] - df["low"]
    h_c = (df["high"] - df["close"].shift()).abs()
    l_c = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
    df["atr_pct"] = tr.rolling(window).mean() / df["close"]
    return df


def add_roc(df, periods=[5, 10]):
    for p in periods:
        df[f"roc_{p}"] = ((df["close"] - df["close"].shift(p)) / df["close"].shift(p)) * 100
    return df


def add_stochastic(df, window=14):
    l_min = df["low"].rolling(window).min()
    h_max = df["high"].rolling(window).max()
    df["stoch_k"] = 100 * (df["close"] - l_min) / (h_max - l_min + 1e-10)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    return df


def prepare_latest_features():
    """Holt Daten, berechnet Features und gibt den LETZTEN Vektor zurück."""

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(minutes=LOOKBACK_MIN + 30)  # Puffer erhöhen

    # 1. Fetch Data
    req = CryptoBarsRequest(
        symbol_or_symbols=[TRADING_PAIR, FEATURE_PAIR],
        timeframe=TIMEFRAME,
        start=start_dt,
        end=end_dt
    )
    bars = data_client.get_crypto_bars(req).df

    # Check if data is empty
    if bars.empty:
        logger.warning("Alpaca returned empty dataframe.")
        return None

    # Trennen & Sortieren
    df_btc = bars.loc[TRADING_PAIR].reset_index().sort_values('timestamp')
    df_eth = bars.loc[FEATURE_PAIR].reset_index().sort_values('timestamp')

    # 2. Process ETH (Features)
    df_eth = add_log_return(df_eth)
    df_eth = add_emas(df_eth)
    df_eth = add_ema_differences(df_eth)
    df_eth = add_ema_slope(df_eth)
    df_eth = add_volatility(df_eth)
    df_eth = add_rsi(df_eth)
    df_eth = add_macd(df_eth)
    df_eth = add_bollinger_bands(df_eth)
    df_eth = add_atr(df_eth)
    df_eth = add_roc(df_eth)
    df_eth = add_stochastic(df_eth)

    # Umbenennen
    rename_map = {}
    for col in df_eth.columns:
        if col != 'timestamp':
            rename_map[col] = f"eth_{col}"
    df_eth_renamed = df_eth.rename(columns=rename_map)

    # 3. Process BTC (Target Context)
    df_btc = add_log_return(df_btc)

    # Join
    df_combined = pd.merge(df_btc[['timestamp', 'close', 'log_return']], df_eth_renamed, on='timestamp', how='inner')

    # Cross Asset
    df_combined['btc_eth_ratio'] = df_combined['close'] / df_combined['eth_close']
    df_combined['btc_eth_return_diff'] = df_combined['log_return'] - df_combined['eth_log_return']
    df_combined['btc_eth_corr_60'] = df_combined['log_return'].rolling(60).corr(df_combined['eth_log_return'])

    # 4. Final Selection
    last_row = df_combined.iloc[-1:].copy()

    if last_row.isnull().values.any():
        logger.warning("NaN in latest features (history too short?).")
        return None

    try:
        X_raw = last_row[FEATURE_LIST].values
    except KeyError as e:
        logger.error(f"Missing feature columns: {e}")
        return None

    return X_raw


# --- MAIN TRADING CYCLE ---
def execute_trade_cycle():
    now_str = datetime.now().strftime('%H:%M:%S')
    logger.info(f"\n--- Cycle {now_str} ---")

    # 1. Daten & Model
    try:
        X_raw = prepare_latest_features()
        if X_raw is None:
            logger.info("Skipping (No Data).")
            return

        X_scaled = scaler.transform(X_raw)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

        with torch.no_grad():
            prob = torch.sigmoid(model(X_tensor)).item()

    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return

    logger.info(f"Model Prob (UP): {prob:.4f} (Threshold: {CONFIDENCE_THRESHOLD})")

    # 2. Position Check
    try:
        try:
            pos = trading_client.get_open_position(TRADING_PAIR.replace("/", ""))
            has_position = True
            logger.info(f"Current Position: {pos.qty} BTC (PnL: {pos.unrealized_pl})")
        except:
            has_position = False
            logger.info("Current Position: NONE")

        # Preis für Bracket Order holen
        latest_trade = data_client.get_crypto_latest_trade(
            CryptoLatestTradeRequest(symbol_or_symbols=TRADING_PAIR)
        )
        current_price = latest_trade[TRADING_PAIR].price
        logger.info(f"Market Price: ${current_price:.2f}")

    except Exception as e:
        logger.error(f"Alpaca API Error: {e}")
        return

    # ----------------------------------------------------
    # 3. DECISION LOGIC (BRACKET ORDER)
    # ----------------------------------------------------

    # --- BUY LOGIC ---
    if prob > CONFIDENCE_THRESHOLD and not has_position:
        logger.info(f"BUY SIGNAL DETECTED!")

        # Berechne Smart Exits
        take_profit_price = round(current_price * TAKE_PROFIT_PCT, 2)
        stop_loss_price = round(current_price * STOP_LOSS_PCT, 2)

        logger.info(f"   Entry: ~${current_price}")
        logger.info(f"   Target: ${take_profit_price} (+1.5%)")
        logger.info(f"   Stop:   ${stop_loss_price} (-0.75%)")

        try:
            # Bracket Order
            req = MarketOrderRequest(
                symbol=TRADING_PAIR,
                notional=TRADE_AMOUNT,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,
                take_profit={'limit_price': take_profit_price},
                stop_loss={'stop_price': stop_loss_price}
            )
            trading_client.submit_order(req)
            logger.info("Bracket Order successfully submitted!")

        except Exception as e:
            logger.error(f"Order failed: {e}")

    # --- SELL LOGIC (Nur Fallback) ---
    # Normalerweise schließt die Bracket Order (TP oder SL) die Position.
    # Wir verkaufen hier manuell nur, wenn das Modell EXTREM bearish wird (< Exit Threshold).
    elif prob < EXIT_THRESHOLD and has_position:
        logger.info(f"MANUAL SELL SIGNAL (Model turned bearish).")
        try:
            trading_client.close_position(TRADING_PAIR.replace("/", ""))
            logger.info("Position closed manually.")
        except Exception as e:
            logger.error(f"Close failed: {e}")

    else:
        logger.info("No Action (Waiting).")


if __name__ == "__main__":
    logger.info(f"Bot starting... Trading {TRADING_PAIR} based on {FEATURE_PAIR}")
    logger.info(f"Settings: Amount=${TRADE_AMOUNT}, TP={TAKE_PROFIT_PCT}, SL={STOP_LOSS_PCT}")

    while True:
        # Schlaf bis zur nächsten vollen Minute + 5 Sekunden
        now = datetime.now()
        seconds_to_wait = 60 - now.second + 5

        if seconds_to_wait > 5:
            logger.info(f"Sleeping {seconds_to_wait}s until next candle...")

        time.sleep(seconds_to_wait)
        execute_trade_cycle()