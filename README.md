# CK Trading System

Quantitative value investing system for US and HK stocks.

## Features

- **Data Pipeline**: Automated collection of price data, financials, and macro indicators
- **Value Investing Strategies**: Graham Defensive, Piotroski F-Score, Magic Formula, DCF, Composite
- **Backtesting Engine**: Vectorized backtesting optimized for monthly/quarterly rebalancing
- **Signal Generation**: Automated screening with notifications via email/Telegram
- **Dashboard**: Streamlit-based portfolio monitoring and analysis
- **AI Model Share Monitor**: Weekly GitHub Action tracking open-source/Chinese-lab
  competitive threat to Anthropic (dollar-weighted OpenRouter share, npm/PyPI
  adoption, flagship price cuts) → [docs](docs/ai_model_share_monitoring.md)
- **AT&T vs SpaceX Monitor**: Weekly telecom-threat tracker (T price vs valuation
  scenario bands, churn/FCF threshold rules on quarterly inputs, editable
  scenario probabilities) → [docs](docs/att_spacex_monitoring.md)
- **NVDA Framework Monitor**: Three-layer buy/sell signal system (calibrated
  12.7% dip + 200dma technicals, weekly auto forward P/E, quarterly thesis
  rules) with verdict banner → [report](docs/nvda_analysis_report.md) ·
  [docs](docs/nvda_monitoring.md)
- **NFLX Framework Monitor**: Pre-committed buy ladder (T1 $70 / T2 $63) with
  quarterly invalidation rules incl. Nielsen attention share →
  [docs](docs/nflx_monitoring.md)
- **INTC Framework Monitor**: Binary foundry-turnaround option — P/B +
  sum-of-parts valuation (not forward P/E), absolute-price value ladder
  ($72/$55/$40), foundry-milestone thesis rules →
  [report](docs/intel_analysis_report.md) · [docs](docs/intc_monitoring.md)
- **Signal semantics**: how the dip system (page 07, timing layer) composes
  with the framework pages (authorization layer) →
  [docs](docs/signal_semantics.md)

## Tech Stack

- **Python 3.12+** with uv package manager
- **Polars** for high-performance data processing
- **DuckDB + Parquet** for analytical storage (no server required)
- **FastAPI** backend + **Streamlit** dashboard
- **yfinance** for market data

## Quick Start

```bash
# Install dependencies
uv sync

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Seed stock universe and backfill data
uv run python scripts/seed_universe.py
uv run python scripts/backfill_data.py

# Launch dashboard
uv run streamlit run src/ck_trading/dashboard/app.py

# Run API server
uv run fastapi dev src/ck_trading/api/main.py
```

## Project Structure

```
src/ck_trading/
├── collectors/     # Data collection (US/HK prices, fundamentals, macro)
├── cleaning/       # Data normalization and cleaning
├── storage/        # Parquet + DuckDB + SQLite storage layer
├── models/         # Pydantic data models
├── strategies/     # Value investing strategies
├── backtesting/    # Vectorized backtesting engine
├── signals/        # Signal generation and notifications
├── portfolio/      # Position tracking and risk metrics
├── api/            # FastAPI backend
├── dashboard/      # Streamlit UI
└── scheduler/      # APScheduler periodic jobs
```
