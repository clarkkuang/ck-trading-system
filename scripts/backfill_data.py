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

    # --- Extended FRED Macro ---
    print("\nCollecting macro data from FRED...")
    try:
        from ck_trading.collectors.macro import MacroCollector

        if settings.fred_api_key:
            macro_collector = MacroCollector()
            macro_data = macro_collector.collect(start_date=start_date, end_date=end_date)
            if not macro_data.is_empty():
                store.save_macro(macro_data)
                print(f"  Saved {macro_data.height} macro records")
        else:
            print("  Skipped: FRED_API_KEY not configured")
    except Exception as e:
        print(f"  Error: {e}")

    # --- Earnings Data ---
    if us_tickers:
        print(f"\nCollecting earnings data for {len(fund_tickers)} tickers...")
        try:
            from ck_trading.collectors.earnings import EarningsCollector

            earnings_collector = EarningsCollector(fmp_api_key=settings.fmp_api_key or None)
            earnings = earnings_collector.collect_earnings_history(fund_tickers)
            if not earnings.is_empty():
                store.save_alternative(earnings, "earnings", "combined")
                print(f"  Saved {earnings.height} earnings records")
        except Exception as e:
            print(f"  Error: {e}")

    # --- FINRA Short Interest ---
    if us_tickers:
        print(f"\nCollecting short interest data...")
        try:
            from ck_trading.collectors.short_interest import ShortInterestCollector

            si_collector = ShortInterestCollector()
            si_data = si_collector.collect(fund_tickers, start_date, end_date)
            if not si_data.is_empty():
                store.save_alternative(si_data, "short_interest", "finra")
                print(f"  Saved {si_data.height} short interest records")
        except Exception as e:
            print(f"  Error: {e}")

    # --- Crypto (BTC/ETH) ---
    print("\nCollecting crypto prices...")
    try:
        from ck_trading.collectors.crypto import CryptoCollector

        crypto_collector = CryptoCollector()
        crypto_data = crypto_collector.collect_prices(
            coins=["BTC", "ETH"], start_date=start_date, end_date=end_date
        )
        if not crypto_data.is_empty():
            store.save_prices(crypto_data, "crypto")
            print(f"  Saved {crypto_data.height} crypto price records")
    except Exception as e:
        print(f"  Error: {e}")

    # --- SEC Insider Trades ---
    if us_tickers:
        print(f"\nCollecting insider trades...")
        try:
            from ck_trading.collectors.insider_trades import InsiderTradesCollector

            insider_collector = InsiderTradesCollector()
            insider_data = insider_collector.collect(fund_tickers[:20], days_back=180)  # Limit to 20 tickers to avoid rate limits
            if not insider_data.is_empty():
                store.save_alternative(insider_data, "insider_trades", "edgar")
                print(f"  Saved {insider_data.height} insider trade records")
        except Exception as e:
            print(f"  Error: {e}")

    meta.close()
    print("\nBackfill complete!")


if __name__ == "__main__":
    main()
