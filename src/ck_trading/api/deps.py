"""FastAPI dependencies."""

from functools import lru_cache

from ck_trading.storage.duckdb_store import DuckDBStore
from ck_trading.storage.metadata_store import MetadataStore
from ck_trading.storage.parquet_store import ParquetStore


@lru_cache
def get_parquet_store() -> ParquetStore:
    return ParquetStore()


@lru_cache
def get_duckdb_store() -> DuckDBStore:
    return DuckDBStore()


@lru_cache
def get_metadata_store() -> MetadataStore:
    return MetadataStore()
