"""Seed the stock universe with US and HK tickers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ck_trading.config import settings
from ck_trading.storage.metadata_store import MetadataStore

# Major US value stocks + broad universe
US_TICKERS = [
    # Mega caps
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "BRK-B", "JPM", "JNJ",
    "V", "PG", "UNH", "HD", "MA", "DIS", "CSCO", "VZ", "PFE",
    "KO", "PEP", "MRK", "ABT", "WMT", "TMO", "COST", "NKE",
    # Value-oriented
    "BRK-B", "BAC", "WFC", "C", "GS", "MS",  # Financials
    "XOM", "CVX", "COP",  # Energy
    "CAT", "DE", "MMM", "HON", "GE",  # Industrials
    "IBM", "INTC", "QCOM", "TXN",  # Tech value
    "T", "CMCSA", "TMUS",  # Telecom
    "MO", "PM", "BTI",  # Consumer staples
    "CVS", "CI", "HUM",  # Healthcare
    "F", "GM",  # Auto
    # Benchmarks
    "SPY", "QQQ", "IWM", "VTV",  # ETFs for comparison
]

# Hong Kong blue chips (Hang Seng components)
HK_TICKERS = [
    {"ticker": "0005.HK", "name": "HSBC Holdings", "sector": "Financials"},
    {"ticker": "0700.HK", "name": "Tencent Holdings", "sector": "Technology"},
    {"ticker": "9988.HK", "name": "Alibaba Group", "sector": "Technology"},
    {"ticker": "0941.HK", "name": "China Mobile", "sector": "Telecom"},
    {"ticker": "1299.HK", "name": "AIA Group", "sector": "Financials"},
    {"ticker": "0388.HK", "name": "HKEX", "sector": "Financials"},
    {"ticker": "0027.HK", "name": "Galaxy Entertainment", "sector": "Consumer"},
    {"ticker": "2318.HK", "name": "Ping An Insurance", "sector": "Financials"},
    {"ticker": "0003.HK", "name": "CK Infrastructure", "sector": "Utilities"},
    {"ticker": "0001.HK", "name": "CK Hutchison", "sector": "Conglomerates"},
]


def main():
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    store = MetadataStore()

    # Save US universe
    us_records = [
        {"ticker": t, "market": "US", "name": t, "sector": "", "industry": ""}
        for t in US_TICKERS
    ]
    store.save_universe(us_records)
    print(f"Saved {len(us_records)} US tickers")

    # Save HK universe
    hk_records = [
        {"ticker": t["ticker"], "market": "HK", "name": t["name"],
         "sector": t["sector"], "industry": ""}
        for t in HK_TICKERS
    ]
    store.save_universe(hk_records)
    print(f"Saved {len(hk_records)} HK tickers")

    store.close()
    print("Universe seeded successfully!")


if __name__ == "__main__":
    main()
