"""Application configuration using pydantic-settings."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Data directory
    data_dir: Path = Path("./data")

    # API keys
    fmp_api_key: str = ""
    fred_api_key: str = ""
    openrouter_api_key: str = ""  # optional; enables OpenRouter rankings collection

    # Notification URLs (comma-separated Apprise URLs)
    notification_urls: str = ""

    # Market defaults
    default_us_benchmark: str = "SPY"
    default_hk_benchmark: str = "^HSI"

    # Backtesting defaults
    default_initial_capital: float = 1_000_000
    default_transaction_cost_bps: float = 10.0
    default_max_positions: int = 20

    @property
    def prices_dir(self) -> Path:
        return self.data_dir / "prices"

    @property
    def fundamentals_dir(self) -> Path:
        return self.data_dir / "fundamentals"

    @property
    def macro_dir(self) -> Path:
        return self.data_dir / "macro"

    @property
    def monitoring_dir(self) -> Path:
        """Git-tracked time series for the AI model share monitor."""
        return self.data_dir / "monitoring"

    @property
    def duckdb_path(self) -> Path:
        return self.data_dir / "analytics.duckdb"

    @property
    def metadata_db_path(self) -> Path:
        return self.data_dir / "metadata.db"


settings = Settings()
