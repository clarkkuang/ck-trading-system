"""Financial statement and fundamental data collector."""

from datetime import date

import polars as pl
import yfinance as yf

from ck_trading.models.market_data import Market


class FundamentalsCollector:
    """Collects financial statements and ratios from yfinance."""

    def collect(self, tickers: list[str], market: Market = Market.US) -> pl.DataFrame:
        """Collect financial fundamentals for given tickers."""
        records = []
        for ticker in tickers:
            try:
                data = self._collect_single(ticker, market)
                records.extend(data)
            except Exception:
                continue

        if not records:
            return pl.DataFrame()

        return pl.DataFrame(records)

    def _collect_single(self, ticker: str, market: Market) -> list[dict]:
        """Collect fundamentals for a single ticker."""
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        records = []

        # Get financial statements
        financials = yf_ticker.financials
        balance_sheet = yf_ticker.balance_sheet
        cashflow = yf_ticker.cashflow

        if financials is None or financials.empty:
            return self._collect_from_info(ticker, market, info)

        # Process each period
        for col in financials.columns:
            period_end = col.date() if hasattr(col, "date") else col
            record = {
                "ticker": ticker,
                "market": market.value,
                "period_end": period_end,
                "report_date": period_end,  # Approximation; FMP has actual report dates
            }

            # Income statement
            record["revenue"] = _safe_get(financials, "Total Revenue", col)
            record["gross_profit"] = _safe_get(financials, "Gross Profit", col)
            record["operating_income"] = _safe_get(financials, "Operating Income", col)
            record["net_income"] = _safe_get(financials, "Net Income", col)
            record["ebit"] = _safe_get(financials, "EBIT", col)
            record["ebitda"] = _safe_get(financials, "EBITDA", col)

            # Balance sheet
            if balance_sheet is not None and col in balance_sheet.columns:
                record["total_assets"] = _safe_get(balance_sheet, "Total Assets", col)
                record["total_liabilities"] = _safe_get(
                    balance_sheet, "Total Liabilities Net Minority Interest", col
                )
                record["total_equity"] = _safe_get(
                    balance_sheet, "Stockholders Equity", col
                )
                record["current_assets"] = _safe_get(
                    balance_sheet, "Current Assets", col
                )
                record["current_liabilities"] = _safe_get(
                    balance_sheet, "Current Liabilities", col
                )
                record["cash_and_equivalents"] = _safe_get(
                    balance_sheet, "Cash And Cash Equivalents", col
                )
                record["total_debt"] = _safe_get(balance_sheet, "Total Debt", col)

            # Cash flow
            if cashflow is not None and col in cashflow.columns:
                record["operating_cash_flow"] = _safe_get(
                    cashflow, "Operating Cash Flow", col
                )
                record["capex"] = _safe_get(cashflow, "Capital Expenditure", col)
                record["free_cash_flow"] = _safe_get(cashflow, "Free Cash Flow", col)

            # Ratios from info (current only, applied to most recent period)
            if col == financials.columns[0]:
                record["pe_ratio"] = info.get("trailingPE")
                record["pb_ratio"] = info.get("priceToBook")
                record["ps_ratio"] = info.get("priceToSalesTrailing12Months")
                record["roe"] = info.get("returnOnEquity")
                record["roa"] = info.get("returnOnAssets")
                record["dividend_yield"] = info.get("dividendYield")
                record["enterprise_value"] = info.get("enterpriseValue")
                record["eps"] = info.get("trailingEps")
                record["book_value_per_share"] = info.get("bookValue")
                record["market_cap"] = info.get("marketCap")

            # Derived ratios
            ca = record.get("current_assets")
            cl = record.get("current_liabilities")
            if ca and cl and cl != 0:
                record["current_ratio"] = ca / cl

            td = record.get("total_debt")
            te = record.get("total_equity")
            if td is not None and te and te != 0:
                record["debt_to_equity"] = td / te

            rev = record.get("revenue")
            gp = record.get("gross_profit")
            oi = record.get("operating_income")
            ni = record.get("net_income")
            if rev and rev != 0:
                if gp is not None:
                    record["gross_margin"] = gp / rev
                if oi is not None:
                    record["operating_margin"] = oi / rev
                if ni is not None:
                    record["net_margin"] = ni / rev

            ta = record.get("total_assets")
            if ta and ta != 0 and rev:
                record["asset_turnover"] = rev / ta

            records.append(record)

        return records

    def _collect_from_info(
        self, ticker: str, market: Market, info: dict
    ) -> list[dict]:
        """Fallback: collect what we can from yfinance .info only."""
        today = date.today()
        return [
            {
                "ticker": ticker,
                "market": market.value,
                "period_end": today,
                "report_date": today,
                "pe_ratio": info.get("trailingPE"),
                "pb_ratio": info.get("priceToBook"),
                "ps_ratio": info.get("priceToSalesTrailing12Months"),
                "roe": info.get("returnOnEquity"),
                "roa": info.get("returnOnAssets"),
                "dividend_yield": info.get("dividendYield"),
                "enterprise_value": info.get("enterpriseValue"),
                "eps": info.get("trailingEps"),
                "book_value_per_share": info.get("bookValue"),
                "market_cap": info.get("marketCap"),
                "revenue": info.get("totalRevenue"),
                "net_income": info.get("netIncomeToCommon"),
            }
        ]


def _safe_get(df, row_name: str, col) -> float | None:
    """Safely get a value from a pandas DataFrame."""
    try:
        val = df.loc[row_name, col]
        if val is not None and str(val) != "nan":
            return float(val)
    except (KeyError, ValueError, TypeError):
        pass
    return None
