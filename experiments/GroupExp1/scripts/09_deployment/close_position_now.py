"""
Close BTC Position Immediately
"""

import yaml
import requests
from pathlib import Path

# --- CONFIG ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONF_DIR = BASE_DIR / "conf"
keys = yaml.safe_load(open(CONF_DIR / "keys.yaml"))

API_KEY = keys['KEYS']['APCA-API-KEY-ID']
SECRET_KEY = keys['KEYS']['APCA-API-SECRET-KEY']

headers = {
    'APCA-API-KEY-ID': API_KEY,
    'APCA-API-SECRET-KEY': SECRET_KEY
}

BASE_URL = 'https://paper-api.alpaca.markets/v2'

print("=" * 60)
print("CLOSING BTC POSITION")
print("=" * 60)

# Get current position first
print("\nChecking current position...")
response = requests.get(f'{BASE_URL}/positions/BTCUSD', headers=headers)

if response.status_code == 200:
    pos = response.json()
    print(f"\n✅ Position found:")
    print(f"   Qty: {pos['qty']} BTC")
    print(f"   Entry: ${float(pos['avg_entry_price']):.2f}")
    print(f"   Current: ${float(pos['current_price']):.2f}")
    print(f"   Unrealized PnL: ${float(pos['unrealized_pl']):.2f}")
    print(f"   Unrealized %: {float(pos['unrealized_plpc']) * 100:.2f}%")

    # Close position
    print(f"\nClosing position...")

    response = requests.delete(
        f'{BASE_URL}/positions/BTCUSD',
        headers=headers
    )

    if response.status_code == 200:
        result = response.json()
        print(f"\n🎉 POSITION CLOSED!")
        print(f"   Exit Price: ${float(result.get('avg_exit_price', 0)):.2f}")
        print(f"   Realized PnL: ${float(result.get('unrealized_pl', 0)):.2f}")
        print(f"\n✅ Trade completed successfully!")
        print(f"\nNext steps:")
        print(f"  1. Restart bot: pkill -f realistic_improved_bot && python realistic_improved_bot.py")
        print(f"  2. New trades will use the improved TP logic")

    else:
        print(f"\n❌ Error closing position: {response.status_code}")
        print(f"Response: {response.text}")

else:
    print(f"\n❌ No position found or error: {response.status_code}")
    if response.status_code == 404:
        print(f"   Position already closed or doesn't exist")
    else:
        print(f"   Response: {response.text}")

print("\n" + "=" * 60)