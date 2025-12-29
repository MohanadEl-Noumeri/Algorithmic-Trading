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
        logging.FileHandler("paper_trading_live.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONF_DIR = BASE_DIR / "conf"

try:
    params = yaml.safe_load(open(CONF_DIR / "params.yaml"))
    keys = yaml.safe_load(open(CONF_DIR / "keys.yaml"))
except FileNotFoundError:
    logger.error("Configuration files not found!")
    exit(1)

# API Keys (PAPER TRADING)
API_KEY = keys['KEYS']['APCA-API-KEY-ID']
SECRET_KEY = keys['KEYS']['APCA-API-SECRET-KEY']

# Paths
DATA_PATH = Path(params["DATA_ACQUISITON"]["DATA_PATH"])
MODEL_PATH = Path(params["MODELING"]["MODEL_PATH"])
FEATURE_PATH = params["DATA_PREP"]["FEATURE_PATH"]

# --- STRATEGY SETTINGS ---
TRADING_PAIR = "BTC/USD"
FEATURE_PAIR = "ETH/USD"
TIMEFRAME = TimeFrame.Minute
LOOKBACK_MIN = 350  # Extra buffer for indicators
TRADE_AMOUNT = 1000  # USD per trade

# --- THRESHOLDS (Conservative) ---
CONFIDENCE_THRESHOLD = 0.515  # zwischen 51-52 realistisch
EXIT_THRESHOLD = 0.495

# --- RISK MANAGEMENT (ATR-Based) ---
ATR_WINDOW = 14
TAKE_PROFIT_ATR_MULT = 3.0  # 3x ATR for take profit
STOP_LOSS_ATR_MULT = 1.5  # 1.5x ATR for stop loss
MIN_TP_PCT = 0.005  # Minimum 0.5% TP
MIN_SL_PCT = 0.003  # Minimum 0.3% SL


# --- MODEL CLASS ---
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
logger.info("=" * 60)
logger.info("PAPER TRADING BOT - FIXED VERSION")
logger.info("=" * 60)

# Verify Paper Trading
if not API_KEY.startswith("PK"):
    logger.error("❌ API Key does not start with 'PK' - NOT PAPER TRADING!")
    exit(1)
else:
    logger.info("[OK] Paper Trading Mode Confirmed")

# Initialize Clients
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = CryptoHistoricalDataClient(API_KEY, SECRET_KEY)

# Load Features
with open(FEATURE_PATH, "r") as f:
    FEATURE_LIST = [line.strip() for line in f.readlines()]
logger.info(f"[OK] Loaded {len(FEATURE_LIST)} features")

# Load Scaler
scaler = pickle.load(open(MODEL_PATH / "feature_scaler.pkl", "rb"))
logger.info("[OK] Scaler loaded")

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MODEL_PATH / "best_model.pt", map_location=device)
config = checkpoint['config']

model = MLP(config['in_dim'], config['h1'], config['h2'], config['dropout'])
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
logger.info(f"[OK] Model loaded (device: {device})")

logger.info(f"\nStrategy Config:")
logger.info(f"  Trading: {TRADING_PAIR}")
logger.info(f"  Features from: {FEATURE_PAIR}")
logger.info(f"  Entry Threshold: {CONFIDENCE_THRESHOLD}")
logger.info(f"  Exit Threshold: {EXIT_THRESHOLD}")
logger.info(f"  Trade Size: ${TRADE_AMOUNT}")
logger.info("=" * 60)


# --- FEATURE ENGINEERING ---
def add_log_return(df):
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


def add_emas(df, periods=[10, 30, 60]):
    for w in periods:
        df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()
    return df


def add_ema_differences(df):
    if 'ema_30' in df.columns and 'ema_10' in df.columns:
        df["ema_30_10"] = df["ema_30"] - df["ema_10"]
    return df


def add_ema_slope(df, window=5):
    if 'ema_10' in df.columns:
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
    df["atr"] = tr.rolling(window).mean()
    df["atr_pct"] = df["atr"] / df["close"]
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


# --- DATA PREPARATION ---
def prepare_latest_features():
    """
    CRITICAL FIX: Fetch data UP TO the LAST CLOSED CANDLE
    This prevents look-ahead bias in live trading
    """

    # Get current time and round down to last closed minute
    now = datetime.now()
    end_dt = now.replace(second=0, microsecond=0)

    # IMPORTANT: Go back 1 minute to get the last CLOSED candle
    end_dt = end_dt - timedelta(minutes=1)
    start_dt = end_dt - timedelta(minutes=LOOKBACK_MIN)

    logger.info(f"Fetching data: {start_dt.strftime('%H:%M')} to {end_dt.strftime('%H:%M')}")

    try:
        # 1. Fetch Data
        req = CryptoBarsRequest(
            symbol_or_symbols=[TRADING_PAIR, FEATURE_PAIR],
            timeframe=TIMEFRAME,
            start=start_dt,
            end=end_dt
        )
        bars = data_client.get_crypto_bars(req).df

        if bars.empty:
            logger.warning("[WARN] Empty dataframe from Alpaca")
            return None, None

        # 2. Separate symbols
        df_btc = bars.loc[TRADING_PAIR].reset_index().sort_values('timestamp')
        df_eth = bars.loc[FEATURE_PAIR].reset_index().sort_values('timestamp')

        if len(df_btc) < 100 or len(df_eth) < 100:
            logger.warning(f"[WARN] Insufficient data: BTC={len(df_btc)}, ETH={len(df_eth)}")
            return None, None

        # 3. Process ETH (Full Feature Engineering)
        df_eth = (df_eth
                  .pipe(add_log_return)
                  .pipe(add_emas)
                  .pipe(add_ema_differences)
                  .pipe(add_ema_slope)
                  .pipe(add_volatility)
                  .pipe(add_rsi)
                  .pipe(add_macd)
                  .pipe(add_bollinger_bands)
                  .pipe(add_atr)
                  .pipe(add_roc)
                  .pipe(add_stochastic))

        # Rename with eth_ prefix
        rename_map = {col: f"eth_{col}" for col in df_eth.columns if col != 'timestamp'}
        df_eth = df_eth.rename(columns=rename_map)

        # 4. Process BTC (Minimal - just for cross-asset features)
        df_btc = add_log_return(df_btc)

        # 5. Join
        df_combined = pd.merge(
            df_btc[['timestamp', 'close', 'log_return', 'high', 'low']],
            df_eth,
            on='timestamp',
            how='inner'
        )

        # 6. Cross-Asset Features
        df_combined['btc_eth_ratio'] = df_combined['close'] / df_combined['eth_close']
        df_combined['btc_eth_return_diff'] = df_combined['log_return'] - df_combined['eth_log_return']
        df_combined['btc_eth_corr_60'] = (
            df_combined['log_return']
            .rolling(60)
            .corr(df_combined['eth_log_return'])
        )

        # 7. Calculate ATR for dynamic stops
        df_combined = add_atr(df_combined, window=ATR_WINDOW)

        # 8. Extract last row for prediction
        last_row = df_combined.iloc[-1:].copy()

        if last_row.isnull().values.any():
            logger.warning("[WARN] NaN values in features")
            return None, None

        # 9. Get feature vector
        try:
            X_raw = last_row[FEATURE_LIST].values
        except KeyError as e:
            logger.error(f"❌ Missing features: {e}")
            return None, None

        # 10. Get ATR for risk management
        atr_pct = last_row['atr_pct'].values[0]

        return X_raw, atr_pct

    except Exception as e:
        logger.error(f"❌ Data fetch failed: {e}")
        return None, None


# --- TRADE LOGGING ---
def log_trade(action, price, prob, size=None, pnl=None):
    """Log trades to CSV for analysis"""
    trade_log_path = Path("trade_journal.csv")

    log_entry = pd.DataFrame([{
        'timestamp': datetime.now(),
        'action': action,
        'price': price,
        'probability': prob,
        'size': size,
        'pnl': pnl
    }])

    if trade_log_path.exists():
        log_entry.to_csv(trade_log_path, mode='a', header=False, index=False)
    else:
        log_entry.to_csv(trade_log_path, index=False)


# --- MAIN TRADING CYCLE ---
def execute_trade_cycle():
    cycle_time = datetime.now().strftime('%H:%M:%S')
    logger.info(f"\n{'=' * 60}")
    logger.info(f"CYCLE: {cycle_time}")
    logger.info(f"{'=' * 60}")

    # 1. Get Data & Predict
    try:
        X_raw, atr_pct = prepare_latest_features()

        if X_raw is None:
            logger.info("[SKIP] Skipping cycle (no data)")
            return

        # Scale and predict
        X_scaled = scaler.transform(X_raw)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

        with torch.no_grad():
            prob = torch.sigmoid(model(X_tensor)).item()

        logger.info(f"[PREDICT] Model Prediction: {prob:.4f}")
        logger.info(f"   Entry Threshold: {CONFIDENCE_THRESHOLD}")
        logger.info(f"   Exit Threshold: {EXIT_THRESHOLD}")

    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return

    # 2. Check Current Position
    try:
        try:
            pos = trading_client.get_open_position(TRADING_PAIR.replace("/", ""))
            has_position = True
            position_qty = float(pos.qty)
            position_pnl = float(pos.unrealized_pl)
            logger.info(f"[POSITION] Current Position: {position_qty:.6f} BTC (PnL: ${position_pnl:.2f})")
        except:
            has_position = False
            position_qty = 0
            position_pnl = 0
            logger.info("[POSITION] Current Position: NONE")

        # Get current market price
        latest_trade = data_client.get_crypto_latest_trade(
            CryptoLatestTradeRequest(symbol_or_symbols=TRADING_PAIR)
        )
        current_price = float(latest_trade[TRADING_PAIR].price)
        logger.info(f"[PRICE] Market Price: ${current_price:.2f}")

    except Exception as e:
        logger.error(f"❌ API Error: {e}")
        return

    # 3. DECISION LOGIC

    # --- BUY SIGNAL ---
    if prob > CONFIDENCE_THRESHOLD and not has_position:
        logger.info(f"\n[BUY SIGNAL] Triggered!")
        logger.info(f"   Confidence: {prob:.1%}")

        # Calculate dynamic stops based on ATR
        if atr_pct is not None and atr_pct > 0:
            tp_pct = max(atr_pct * TAKE_PROFIT_ATR_MULT, MIN_TP_PCT)
            sl_pct = max(atr_pct * STOP_LOSS_ATR_MULT, MIN_SL_PCT)
        else:
            # Fallback to fixed percentages
            tp_pct = 0.015  # 1.5%
            sl_pct = 0.0075  # 0.75%

        take_profit_price = round(current_price * (1 + tp_pct), 2)
        stop_loss_price = round(current_price * (1 - sl_pct), 2)

        logger.info(f"   Entry: ~${current_price:.2f}")
        logger.info(f"   TP: ${take_profit_price:.2f} (+{tp_pct * 100:.2f}%)")
        logger.info(f"   SL: ${stop_loss_price:.2f} (-{sl_pct * 100:.2f}%)")
        logger.info(f"   Risk/Reward: {tp_pct / sl_pct:.2f}:1")

        try:
            req = MarketOrderRequest(
                symbol=TRADING_PAIR,
                notional=TRADE_AMOUNT,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,
                take_profit={'limit_price': take_profit_price},
                stop_loss={'stop_price': stop_loss_price}
            )

            order = trading_client.submit_order(req)
            logger.info(f"[OK] Bracket Order Submitted! Order ID: {order.id}")

            log_trade('BUY', current_price, prob, TRADE_AMOUNT)

        except Exception as e:
            logger.error(f"❌ Order failed: {e}")

    # --- MANUAL EXIT (if model turns bearish) ---
    elif prob < EXIT_THRESHOLD and has_position:
        logger.info(f"\n[SELL SIGNAL] Manual Exit!")
        logger.info(f"   Confidence dropped to: {prob:.1%}")
        logger.info(f"   Current PnL: ${position_pnl:.2f}")

        try:
            trading_client.close_position(TRADING_PAIR.replace("/", ""))
            logger.info(f"[OK] Position closed manually")

            log_trade('SELL', current_price, prob, position_qty, position_pnl)

        except Exception as e:
            logger.error(f"❌ Close failed: {e}")

    # --- HOLD ---
    else:
        if has_position:
            logger.info(f"[HOLD] Position held (Prob: {prob:.1%}, PnL: ${position_pnl:.2f})")
        else:
            logger.info(f"[WAIT] Waiting for signal (Prob: {prob:.1%} - need {CONFIDENCE_THRESHOLD:.1%})")


# --- MAIN LOOP ---
if __name__ == "__main__":
    logger.info(f"\n>>> Bot Started!")
    logger.info(f"Trading: {TRADING_PAIR} (using {FEATURE_PAIR} features)")
    logger.info(f"Trade Size: ${TRADE_AMOUNT}")
    logger.info(f"Mode: PAPER TRADING")
    logger.info(f"\n[!] Press Ctrl+C to stop\n")

    try:
        while True:
            # Wait until next full minute + 10 seconds
            # (gives Alpaca time to close the candle)
            now = datetime.now()
            next_minute = (now + timedelta(minutes=1)).replace(second=10, microsecond=0)
            sleep_seconds = (next_minute - now).total_seconds()

            if sleep_seconds > 10:
                logger.info(f"[SLEEP] Sleeping {int(sleep_seconds)}s until next candle...")

            time.sleep(sleep_seconds)

            # Execute trading logic
            execute_trade_cycle()

    except KeyboardInterrupt:
        logger.info("\n\n[STOP] Bot stopped by user")
        logger.info("Trade journal saved to: trade_journal.csv")
    except Exception as e:
        logger.error(f"\n\n[FATAL] Fatal error: {e}")
        raise