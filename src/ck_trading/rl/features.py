"""Feature engineering for RL trading environment.

Builds a fixed-size observation vector from price + fundamental data.
Used both during training (Gymnasium env) and inference (RLStrategy.screen()).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl


class FeatureBuilder:
    """Build observation vectors from raw price and fundamental data.

    Per-ticker features (20 total):
      Price-derived (10): 1M/3M/6M/12M returns, 21d/63d volatility,
        RSI-14, SMA ratio, volume ratio, drawdown from 63d high.
      Momentum rank (4): cross-sectional percentile rank of 1M/3M/6M/12M
        returns. Rank in [0, 1] where 1 = strongest momentum.
      Fundamental-derived (6): PE (log), ROE, log(market_cap),
        D/E, gross_margin, operating_margin.

    All features are cross-sectionally z-score standardized.
    Output is flattened to 1D: shape (n_tickers * n_features,).
    """

    N_PRICE_FEATURES = 10
    N_MOMENTUM_RANK_FEATURES = 4
    N_FUND_FEATURES = 6

    def __init__(self, tickers: list[str], lookback_window: int = 252):
        self.tickers = list(tickers)
        self.lookback_window = lookback_window

    @property
    def n_features(self) -> int:
        return self.N_PRICE_FEATURES + self.N_MOMENTUM_RANK_FEATURES + self.N_FUND_FEATURES

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return (len(self.tickers) * self.n_features,)

    def build_observation(
        self,
        prices: pl.DataFrame,
        fundamentals: pl.DataFrame,
        as_of_date: date,
    ) -> np.ndarray:
        """Build flattened observation vector.

        Strictly uses only data with date <= as_of_date (no lookahead).
        Missing tickers are zero-filled.
        """
        n_tickers = len(self.tickers)
        matrix = np.zeros((n_tickers, self.n_features), dtype=np.float32)

        # Filter prices to as_of_date
        if not prices.is_empty():
            hist = prices.filter(pl.col("date") <= as_of_date)
        else:
            hist = pl.DataFrame()

        # Filter fundamentals to as_of_date
        if not fundamentals.is_empty() and "period_end" in fundamentals.columns:
            fund = fundamentals.filter(pl.col("period_end") <= as_of_date)
        else:
            fund = pl.DataFrame()

        for i, ticker in enumerate(self.tickers):
            price_feats = self._price_features(hist, ticker)
            fund_feats = self._fundamental_features(fund, ticker)
            matrix[i, :self.N_PRICE_FEATURES] = price_feats
            # Momentum rank features are filled after all price features are computed
            fund_start = self.N_PRICE_FEATURES + self.N_MOMENTUM_RANK_FEATURES
            matrix[i, fund_start:] = fund_feats

        # Compute cross-sectional momentum rank features (percentile of returns)
        # Uses the raw return features (indices 0-3: 1M/3M/6M/12M returns)
        rank_start = self.N_PRICE_FEATURES
        for feat_idx in range(4):  # 1M, 3M, 6M, 12M returns
            raw_returns = matrix[:, feat_idx]
            ranks = self._percentile_rank(raw_returns)
            matrix[:, rank_start + feat_idx] = ranks

        # Cross-sectional z-score normalization (per feature across tickers)
        matrix = self._z_score(matrix)

        return matrix.flatten().astype(np.float32)

    def _price_features(self, hist: pl.DataFrame, ticker: str) -> np.ndarray:
        """Extract 10 price-based features for one ticker."""
        feats = np.zeros(self.N_PRICE_FEATURES, dtype=np.float32)

        if hist.is_empty():
            return feats

        t_prices = hist.filter(pl.col("ticker") == ticker).sort("date")
        if t_prices.height < 5:
            return feats

        closes = t_prices["close"].to_numpy()
        n = len(closes)
        current = closes[-1]

        # 1-4: returns at 21, 63, 126, 252 days
        for idx, lookback in enumerate([21, 63, 126, 252]):
            if n > lookback and closes[-lookback - 1] > 0:
                feats[idx] = current / closes[-lookback - 1] - 1

        # 5-6: annualized volatility (21d, 63d)
        if n > 2:
            daily_ret = np.diff(closes) / closes[:-1]
            for idx, window in enumerate([21, 63]):
                offset = 4 + idx
                w = min(window, len(daily_ret))
                if w > 1:
                    feats[offset] = np.std(daily_ret[-w:]) * np.sqrt(252)

        # 7: RSI-14
        if n > 15:
            daily_ret = np.diff(closes[-15:])
            gains = np.where(daily_ret > 0, daily_ret, 0)
            losses = np.where(daily_ret < 0, -daily_ret, 0)
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 1e-8
            rs = avg_gain / max(avg_loss, 1e-8)
            feats[6] = 1 - 1 / (1 + rs)  # RSI in [0, 1]

        # 8: close / 50-day SMA
        sma_window = min(50, n)
        if sma_window > 0:
            sma = np.mean(closes[-sma_window:])
            if sma > 0:
                feats[7] = current / sma

        # 9: volume ratio (5d avg / 21d avg)
        if "volume" in hist.columns:
            vols = t_prices["volume"].to_numpy().astype(float)
            if len(vols) >= 21:
                v5 = np.mean(vols[-5:])
                v21 = np.mean(vols[-21:])
                if v21 > 0:
                    feats[8] = v5 / v21

        # 10: drawdown from 63-day high
        dd_window = min(63, n)
        if dd_window > 0:
            peak = np.max(closes[-dd_window:])
            if peak > 0:
                feats[9] = current / peak - 1  # negative during drawdown

        return feats

    def _fundamental_features(self, fund: pl.DataFrame, ticker: str) -> np.ndarray:
        """Extract 6 fundamental features for one ticker."""
        feats = np.zeros(self.N_FUND_FEATURES, dtype=np.float32)

        if fund.is_empty():
            return feats

        t_fund = fund.filter(pl.col("ticker") == ticker).sort("period_end", descending=True)
        if t_fund.is_empty():
            return feats

        latest = t_fund.row(0, named=True)

        # 1: PE ratio (log-transformed, capped)
        pe = latest.get("pe_ratio")
        if pe is not None and pe > 0:
            feats[0] = np.log(min(pe, 500))

        # 2: ROE
        roe = latest.get("roe")
        if roe is not None:
            feats[1] = float(np.clip(roe, -1, 2))

        # 3: log(market_cap)
        mc = latest.get("market_cap")
        if mc is not None and mc > 0:
            feats[2] = np.log(mc)

        # 4: Debt-to-Equity (capped)
        de = latest.get("debt_to_equity")
        if de is not None:
            feats[3] = float(np.clip(de, 0, 10))

        # 5: Gross margin
        gm = latest.get("gross_margin")
        if gm is not None:
            feats[4] = float(np.clip(gm, -1, 1))

        # 6: Operating margin
        om = latest.get("operating_margin")
        if om is not None:
            feats[5] = float(np.clip(om, -1, 1))

        return feats

    @staticmethod
    def _percentile_rank(values: np.ndarray) -> np.ndarray:
        """Convert values to percentile ranks in [0, 1].

        Rank 1.0 = highest value (strongest momentum).
        Ties get the average rank. All-zero or single-element → 0.5.
        """
        n = len(values)
        if n <= 1:
            return np.full(n, 0.5, dtype=np.float32)
        # argsort of argsort gives rank (0-based)
        order = values.argsort().argsort().astype(np.float32)
        # Normalize to [0, 1]
        if n > 1:
            return order / (n - 1)
        return np.full(n, 0.5, dtype=np.float32)

    @staticmethod
    def _z_score(matrix: np.ndarray) -> np.ndarray:
        """Cross-sectional z-score normalization (per feature, across tickers)."""
        mean = np.mean(matrix, axis=0, keepdims=True)
        std = np.std(matrix, axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)  # avoid division by zero
        result = (matrix - mean) / std
        # Replace any NaN/Inf with 0
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result
