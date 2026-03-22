"""Scheduled jobs for data collection and signal generation."""

from datetime import date

from ck_trading.collectors.fundamentals import FundamentalsCollector
from ck_trading.collectors.hk_market import HKMarketCollector
from ck_trading.collectors.us_market import USMarketCollector
from ck_trading.models.market_data import Market
from ck_trading.signals.generator import SignalGenerator
from ck_trading.signals.manager import SignalManager
from ck_trading.signals.notifier import Notifier
from ck_trading.storage.metadata_store import MetadataStore
from ck_trading.storage.parquet_store import ParquetStore
from ck_trading.strategies.composite_value import CompositeValueStrategy
from ck_trading.strategies.graham_defensive import GrahamDefensiveStrategy
from ck_trading.strategies.magic_formula import MagicFormulaStrategy
from ck_trading.strategies.piotroski_f_score import PiotroskiFScoreStrategy


def daily_price_update():
    """Update daily prices for all tickers in the universe."""
    meta = MetadataStore()
    store = ParquetStore()

    from datetime import timedelta
    end = date.today()
    start = end - timedelta(days=5)  # Overlap to catch any missed days

    # US
    us_tickers = [t["ticker"] for t in meta.get_universe("US")]
    if us_tickers:
        collector = USMarketCollector()
        prices = collector.collect_prices(us_tickers, start, end)
        if not prices.is_empty():
            store.save_prices(prices, "us")

    # HK
    hk_tickers = [t["ticker"] for t in meta.get_universe("HK")]
    if hk_tickers:
        collector = HKMarketCollector()
        prices = collector.collect_prices(hk_tickers, start, end)
        if not prices.is_empty():
            store.save_prices(prices, "hk")

    meta.close()


def weekly_fundamental_update():
    """Update fundamental data weekly."""
    meta = MetadataStore()
    store = ParquetStore()
    collector = FundamentalsCollector()

    us_tickers = [t["ticker"] for t in meta.get_universe("US")]
    etfs = {"SPY", "QQQ", "IWM", "VTV"}
    us_tickers = [t for t in us_tickers if t not in etfs]

    if us_tickers:
        for i in range(0, len(us_tickers), 10):
            batch = us_tickers[i : i + 10]
            try:
                data = collector.collect(batch, Market.US)
                if not data.is_empty():
                    store.save_fundamentals(data, "us")
            except Exception:
                continue

    hk_tickers = [t["ticker"] for t in meta.get_universe("HK")]
    if hk_tickers:
        try:
            data = collector.collect(hk_tickers, Market.HK)
            if not data.is_empty():
                store.save_fundamentals(data, "hk")
        except Exception:
            pass

    meta.close()


def generate_and_notify_signals():
    """Run all strategies and send notifications for new signals."""
    store = ParquetStore()
    meta = MetadataStore()

    prices = store.load_prices("us")
    fundamentals = store.load_fundamentals("us")

    generator = SignalGenerator([
        GrahamDefensiveStrategy(),
        PiotroskiFScoreStrategy(),
        MagicFormulaStrategy(),
        CompositeValueStrategy(),
    ])

    signals = generator.generate(prices, fundamentals)

    if signals:
        manager = SignalManager(meta)
        processed = manager.process_signals(signals)

        if processed:
            notifier = Notifier()
            notifier.notify(processed)

    meta.close()
