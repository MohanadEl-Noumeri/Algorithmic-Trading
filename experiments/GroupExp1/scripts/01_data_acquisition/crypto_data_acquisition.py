# Python - Data Acquisition für Krypto (1-Minute OHLCV)

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd
from datetime import datetime
import yaml

# --- 1. API Credentials laden ---
keys = yaml.safe_load(open("../../conf/keys.yaml"))
API_KEY = keys['KEYS']['APCA-API-KEY-ID']
SECRET_KEY = keys['KEYS']['APCA-API-SECRET-KEY']

# --- 2. Parameter für Datenabruf ---
params = yaml.safe_load(open("../../conf/params.yaml"))
PATH_BARS = params['DATA_ACQUISITON']['DATA_PATH']
START_DATE = datetime.strptime(params['DATA_ACQUISITON']['START_DATE'], "%Y-%m-%d")
END_DATE = datetime.strptime(params['DATA_ACQUISITON']['END_DATE'], "%Y-%m-%d")

# --- 3. Alpaca Crypto Client initialisieren ---
client = CryptoHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)

# --- 4. Liste der Symbole für Krypto ---
symbols = ["BTC/USD", "ETH/USD"]

# --- 5. Daten abrufen und speichern ---
for symbol in symbols:
    print(f"Fetching 1-Minute bars for {symbol} from {START_DATE.date()} to {END_DATE.date()}")

    request = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=START_DATE,
        end=END_DATE
    )

    bars = client.get_crypto_bars(request)
    df = bars.df
    df = df[df.index.get_level_values("symbol") == symbol]  # Filter nur für dieses Symbol
    df.reset_index(inplace=True)
    if 'symbol' in df.columns:
        df.drop(columns=['symbol'], inplace=True)

    # Speichern als CSV und Parquet
    df.to_csv(f"{PATH_BARS}/{symbol.replace('/', '')}_1m_raw.csv", index=False)
    df.to_parquet(f"{PATH_BARS}/{symbol.replace('/', '')}_1m_raw.parquet", index=False)

    print(f"Saved {symbol} data. Shape: {df.shape}")
