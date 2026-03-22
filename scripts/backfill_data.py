"""Backfill historical price and fundamental data."""

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ck_trading.collectors.fundamentals import FundamentalsCollector
from ck_trading.collectors.hk_market import HKMarketCollector
from ck_trading.collectors.us_market import USMarketCollector
from ck_trading.config import settings
from ck_trading.models.market_data import Market
from ck_trading.storage.metadata_store import MetadataStore
from ck_trading.storage.parquet_store import ParquetStore


def main():
    years_back = 10
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * years_back)

    print(f"Backfilling data from {start_date} to {end_date}")

    store = ParquetStore()
    meta = MetadataStore()

    # Get universe
    us_tickers = [t["ticker"] for t in meta.get_universe("US")]
    hk_tickers = [t["ticker"] for t in meta.get_universe("HK")]

    # --- US Prices ---
    if us_tickers:
        print(f"\nCollecting US prices for {len(us_tickers)} tickers...")
        us_collector = USMarketCollector()
        # Batch in groups of 50 to avoid rate limits
        batch_size = 50
        for i in range(0, len(us_tickers), batch_size):
            batch = us_tickers[i : i + batch_size]
            print(f"  Batch {i // batch_size + 1}: {len(batch)} tickers")
            try:
                prices = us_collector.collect_prices(batch, start_date, end_date)
                if not prices.is_empty():
                    store.save_prices(prices, "us")
                    print(f"    Saved {prices.height} price records")
            except Exception as e:
                print(f"    Error: {e}")

    # --- HK Prices ---
    if hk_tickers:
        print(f"\nCollecting HK prices for {len(hk_tickers)} tickers...")
        hk_collector = HKMarketCollector()
        try:
            prices = hk_collector.collect_prices(hk_tickers, start_date, end_date)
            if not prices.is_empty():
                store.save_prices(prices, "hk")
                print(f"  Saved {prices.height} price records")
        except Exception as e:
            print(f"  Error: {e}")

    # --- US Fundamentals ---
    if us_tickers:
        # Exclude ETFs from fundamentals
        etfs = {"SPY", "QQQ", "IWM", "VTV"}
        fund_tickers = [t for t in us_tickers if t not in etfs]
        print(f"\nCollecting US fundamentals for {len(fund_tickers)} tickers...")
        fund_collector = FundamentalsCollector()

        batch_size = 10
        for i in range(0, len(fund_tickers), batch_size):
            batch = fund_tickers[i : i + batch_size]
            print(f"  Batch {i // batch_size + 1}: {', '.join(batch)}")
            try:
                fundamentals = fund_collector.collect(batch, Market.US)
                if not fundamentals.is_empty():
                    store.save_fundamentals(fundamentals, "us")
                    print(f"    Saved {fundamentals.height} fundamental records")
            except Exception as e:
                print(f"    Error: {e}")

    # --- HK Fundamentals ---
    if hk_tickers:
        print(f"\nCollecting HK fundamentals for {len(hk_tickers)} tickers...")
        fund_collector = FundamentalsCollector()
        try:
            fundamentals = fund_collector.collect(hk_tickers, Market.HK)
            if not fundamentals.is_empty():
                store.save_fundamentals(fundamentals, "hk")
                print(f"  Saved {fundamentals.height} fundamental records")
        except Exception as e:
            print(f"  Error: {e}")

    meta.close()
    print("\nBackfill complete!")


if __name__ == "__main__":
    main()
