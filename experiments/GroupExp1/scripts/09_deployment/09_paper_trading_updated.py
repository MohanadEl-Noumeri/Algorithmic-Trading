"""
REALISTIC IMPROVED BOT v2.0
Focus: RISK MANAGEMENT for 51-52% accuracy model
"""

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

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("paper_trading_v2.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIG ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONF_DIR = BASE_DIR / "conf"

params = yaml.safe_load(open(CONF_DIR / "params.yaml"))
keys = yaml.safe_load(open(CONF_DIR / "keys.yaml"))

API_KEY = keys['KEYS']['APCA-API-KEY-ID']
SECRET_KEY = keys['KEYS']['APCA-API-SECRET-KEY']

DATA_PATH = Path(params["DATA_ACQUISITON"]["DATA_PATH"])
MODEL_PATH = Path(params["MODELING"]["MODEL_PATH"])
FEATURE_PATH = params["DATA_PREP"]["FEATURE_PATH"]

# --- REALISTIC STRATEGY SETTINGS ---
TRADING_PAIR = "BTC/USD"
FEATURE_PAIR = "ETH/USD"
TIMEFRAME = TimeFrame.Minute
LOOKBACK_MIN = 350

TRADE_AMOUNT = 1000  # USD per trade


CONFIDENCE_THRESHOLD = 0.512
# EXIT_THRESHOLD = 0.467

# --- RISK MANAGEMENT ---
STOP_LOSS_PCT = 0.005  #  stop loss
TAKE_PROFIT_PCT = 0.012  #  take profit



# Position tracking (for logging only)
daily_pnl = 0.0
last_reset_date = datetime.now().date()


# --- MODEL ---
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


# --- INIT ---
logger.info("=" * 60)
logger.info("MLP Other Fixes")
logger.info("=" * 60)

if not API_KEY.startswith("PK"):
    logger.error("NOT PAPER TRADING!")
    exit(1)

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = CryptoHistoricalDataClient(API_KEY, SECRET_KEY)

with open(FEATURE_PATH, "r") as f:
    FEATURE_LIST = [line.strip() for line in f.readlines()]

scaler = pickle.load(open(MODEL_PATH / "feature_scaler.pkl", "rb"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MODEL_PATH / "best_model.pt", map_location=device)
config = checkpoint['config']

model = MLP(config['in_dim'], config['h1'], config['h2'], config['dropout'])
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

logger.info(f"\n RISK-FOCUSED Strategy Config:")
logger.info(f"  Model Accuracy: ~51% (realistic)")
logger.info(f"  Entry Threshold: {CONFIDENCE_THRESHOLD}")
logger.info(f"  Fixed Stop Loss: {STOP_LOSS_PCT * 100}%")
logger.info(f"  Fixed Take Profit: {TAKE_PROFIT_PCT * 100}%")
logger.info(f"  Risk/Reward: {TAKE_PROFIT_PCT / STOP_LOSS_PCT:.1f}:1")
logger.info(f"  Trading: 24/7 (no time restrictions)")
logger.info(f"  Auto-Stop: DISABLED (manual control only)")
logger.info("=" * 60)


# --- FEATURE ENGINEERING (same as before) ---
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


# --- DATA PREP ---
def prepare_latest_features():
    now = datetime.now()
    current_second = now.second

    if current_second < 45:
        logger.warning(f"[SKIP] Too early ({current_second}s)")
        return None

    end_dt = now.replace(second=0, microsecond=0)
    end_dt = end_dt - timedelta(minutes=2)  # 2-minute lag
    start_dt = end_dt - timedelta(minutes=LOOKBACK_MIN)

    try:
        req = CryptoBarsRequest(
            symbol_or_symbols=[TRADING_PAIR, FEATURE_PAIR],
            timeframe=TIMEFRAME,
            start=start_dt,
            end=end_dt
        )
        bars = data_client.get_crypto_bars(req).df

        if bars.empty:
            return None

        df_btc = bars.loc[TRADING_PAIR].reset_index().sort_values('timestamp')
        df_eth = bars.loc[FEATURE_PAIR].reset_index().sort_values('timestamp')

        if len(df_btc) < 100 or len(df_eth) < 100:
            return None

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

        rename_map = {col: f"eth_{col}" for col in df_eth.columns if col != 'timestamp'}
        df_eth = df_eth.rename(columns=rename_map)

        df_btc = add_log_return(df_btc)

        df_combined = pd.merge(
            df_btc[['timestamp', 'close', 'log_return', 'high', 'low']],
            df_eth,
            on='timestamp',
            how='inner'
        )

        df_combined['btc_eth_ratio'] = df_combined['close'] / df_combined['eth_close']
        df_combined['btc_eth_return_diff'] = df_combined['log_return'] - df_combined['eth_log_return']
        df_combined['btc_eth_corr_60'] = (
            df_combined['log_return']
            .rolling(60)
            .corr(df_combined['eth_log_return'])
        )

        last_row = df_combined.iloc[-1:].copy()

        if last_row.isnull().values.any():
            return None

        X_raw = last_row[FEATURE_LIST].values
        return X_raw

    except Exception as e:
        logger.error(f" Data fetch failed: {e}")
        return None


# --- TRADE LOGGING ---
def log_trade(action, price, prob, size=None, pnl=None):
    global daily_pnl

    trade_log_path = Path("trade_journal_v2.csv")

    # Update daily PnL for statistics only (doesn't stop trading)
    if action == 'SELL' and pnl is not None:
        daily_pnl += pnl

    log_entry = pd.DataFrame([{
        'timestamp': datetime.now(),
        'action': action,
        'price': price,
        'probability': prob,
        'size': size,
        'pnl': pnl,
        'daily_pnl': daily_pnl
    }])

    if trade_log_path.exists():
        log_entry.to_csv(trade_log_path, mode='a', header=False, index=False)
    else:
        log_entry.to_csv(trade_log_path, index=False)


# --- DAILY STATISTICS RESET ---
def reset_daily_stats_if_needed():
    """Reset daily statistics at midnight (for tracking only, doesn't stop trading)"""
    global daily_pnl, last_reset_date

    current_date = datetime.now().date()
    if current_date != last_reset_date:
        logger.info(f" New Day - Daily PnL was: ${daily_pnl:.2f}")
        daily_pnl = 0.0
        last_reset_date = current_date


# --- MAIN CYCLE ---
def execute_trade_cycle():
    cycle_time = datetime.now().strftime('%H:%M:%S')
    logger.info(f"\n{'=' * 60}")
    logger.info(f"CYCLE: {cycle_time} | Daily PnL: ${daily_pnl:.2f}")
    logger.info(f"{'=' * 60}")

    # Reset daily stats at midnight (for tracking only)
    reset_daily_stats_if_needed()

    # Get prediction (no risk checks - bot runs 24/7)
    try:
        X_raw = prepare_latest_features()

        if X_raw is None:
            logger.info("[SKIP] No data")
            return

        X_scaled = scaler.transform(X_raw)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

        with torch.no_grad():
            prob = torch.sigmoid(model(X_tensor)).item()

        logger.info(f"[PREDICT] Probability: {prob:.4f} (Threshold: {CONFIDENCE_THRESHOLD})")

    except Exception as e:
        logger.error(f" Prediction error: {e}")
        return

    # 2. Check position
    try:
        try:
            pos = trading_client.get_open_position(TRADING_PAIR.replace("/", ""))
            has_position = True
            position_qty = float(pos.qty)
            position_pnl = float(pos.unrealized_pl)
            entry_price = float(pos.avg_entry_price)
            current_return_pct = position_pnl / (entry_price * position_qty) * 100

            logger.info(f"[POSITION] {position_qty:.6f} BTC @ ${entry_price:.2f}")
            logger.info(f"           PnL: ${position_pnl:.2f} ({current_return_pct:+.2f}%)")
        except:
            has_position = False
            position_qty = 0
            position_pnl = 0
            logger.info("[POSITION] FLAT")

        from alpaca.data.requests import CryptoLatestTradeRequest
        latest_trade = data_client.get_crypto_latest_trade(
            CryptoLatestTradeRequest(symbol_or_symbols=TRADING_PAIR)
        )
        current_price = float(latest_trade[TRADING_PAIR].price)
        logger.info(f"[PRICE] ${current_price:.2f}")

    except Exception as e:
        logger.error(f" API Error: {e}")
        return

    # 3. DECISION LOGIC

    # --- BUY SIGNAL ---
    if prob > CONFIDENCE_THRESHOLD and not has_position:
        logger.info(f"\n✅ BUY SIGNAL (Confidence: {prob:.1%})")

        # Calculate FIXED stops
        take_profit_price = round(current_price * (1 + TAKE_PROFIT_PCT), 2)
        stop_loss_price = round(current_price * (1 - STOP_LOSS_PCT), 2)

        expected_risk = TRADE_AMOUNT * STOP_LOSS_PCT
        expected_reward = TRADE_AMOUNT * TAKE_PROFIT_PCT

        logger.info(f"   Entry: ~${current_price:.2f}")
        logger.info(f"   TP: ${take_profit_price:.2f} (+{TAKE_PROFIT_PCT * 100:.1f}%)")
        logger.info(f"   SL: ${stop_loss_price:.2f} (-{STOP_LOSS_PCT * 100:.1f}%)")
        logger.info(f"   Risk: ${expected_risk:.2f} | Reward: ${expected_reward:.2f}")

        try:
            req = MarketOrderRequest(
                symbol=TRADING_PAIR,
                notional=TRADE_AMOUNT,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,
                take_profit={
                    'limit_price': take_profit_price,
                    'stop_price': take_profit_price  # Trigger sofort bei TP
                },
                stop_loss={'stop_price': stop_loss_price}
            )

            order = trading_client.submit_order(req)
            logger.info(f"🎯 ORDER PLACED! ID: {order.id}")

            log_trade('BUY', current_price, prob, TRADE_AMOUNT)

        except Exception as e:
            logger.error(f"❌ Order failed: {e}")

        # --- HOLD or EXIT ---
    elif has_position:
        # Calculate TP/SL prices based on entry
        tp_price = entry_price * (1 + TAKE_PROFIT_PCT)
        sl_price = entry_price * (1 - STOP_LOSS_PCT)

        # FAILSAFE: Manual exit if TP or SL reached
        # (Needed because Alpaca Crypto API may not support bracket orders properly)
        if current_price >= tp_price:
            logger.info(f"\n TAKE PROFIT REACHED!")
            logger.info(f"   Entry: ${entry_price:.2f}")
            logger.info(f"   Target TP: ${tp_price:.2f}")
            logger.info(f"   Current: ${current_price:.2f}")
            logger.info(f"   PnL: ${position_pnl:.2f} (+{TAKE_PROFIT_PCT * 100:.1f}%)")

            try:
                trading_client.close_position(TRADING_PAIR.replace("/", ""))
                logger.info(f"✅ POSITION CLOSED (Take Profit)")
                log_trade('SELL', current_price, prob, position_qty, position_pnl)
            except Exception as e:
                logger.error(f"❌ Close failed: {e}")

        elif current_price <= sl_price:
            logger.info(f"\n STOP LOSS REACHED!")
            logger.info(f"   Entry: ${entry_price:.2f}")
            logger.info(f"   Target SL: ${sl_price:.2f}")
            logger.info(f"   Current: ${current_price:.2f}")
            logger.info(f"   PnL: ${position_pnl:.2f} (-{STOP_LOSS_PCT * 100:.1f}%)")

            try:
                trading_client.close_position(TRADING_PAIR.replace("/", ""))
                logger.info(f"✅ POSITION CLOSED (Stop Loss)")
                log_trade('SELL', current_price, prob, position_qty, position_pnl)
            except Exception as e:
                logger.error(f"❌ Close failed: {e}")
        else:
            # Still holding - show distance to TP/SL
            to_tp = ((tp_price / current_price) - 1) * 100
            to_sl = ((current_price / sl_price) - 1) * 100

            logger.info(f"⏸️  HOLDING (Prob: {prob:.1%}, PnL: ${position_pnl:.2f})")
            logger.info(f"           Need +{to_tp:.2f}% for TP @ ${tp_price:.2f}")
            logger.info(f"           Safe +{to_sl:.2f}% above SL @ ${sl_price:.2f}")
    else:
        logger.info(f"⏳ WAITING (Prob: {prob:.1%} < {CONFIDENCE_THRESHOLD:.1%})")

#    # --- MANUAL EXIT ---
#    elif has_position and prob < EXIT_THRESHOLD:
#        logger.info(f"\n SELL SIGNAL (Confidence dropped to {prob:.1%})")
#        logger.info(f"   Current PnL: ${position_pnl:.2f}")
#
#        try:
#            trading_client.close_position(TRADING_PAIR.replace("/", ""))
#            logger.info(f"✅ POSITION CLOSED")
#
#            log_trade('SELL', current_price, prob, position_qty, position_pnl)
#
#        except Exception as e:
#            logger.error(f"❌ Close failed: {e}")



# --- MAIN LOOP ---
if __name__ == "__main__":
    logger.info(f"\n>>> BOT v2.0 STARTED!")
    logger.info(f"\n Strategy Focus: FIXED RISK MANAGEMENT")
    logger.info(f"   Expected Win Rate: ~51%")
    logger.info(f"   Fixed Risk/Reward: 3:1")
    logger.info(f"   Trading: 24/7 (no auto-stop)")
    logger.info(f"   Control: MANUAL ONLY (Ctrl+C to stop)")
    logger.info(f"\n[!] Press Ctrl+C to stop\n")

    try:
        while True:
            now = datetime.now()
            next_minute = (now + timedelta(minutes=1)).replace(second=50, microsecond=0)
            sleep_seconds = (next_minute - now).total_seconds()

            if sleep_seconds > 10:
                logger.info(f"[SLEEP] {int(sleep_seconds)}s...")

            time.sleep(sleep_seconds)
            execute_trade_cycle()

    except KeyboardInterrupt:
        logger.info("\n\n[STOP] Bot stopped")
        logger.info(f"Final Daily PnL: ${daily_pnl:.2f}")
        logger.info("Journal: trade_journal_v2.csv")
    except Exception as e:
        logger.error(f"\n\n[FATAL] {e}")
        raise