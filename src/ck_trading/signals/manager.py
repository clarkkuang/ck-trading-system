"""Signal manager - deduplication, filtering, and logging."""

from ck_trading.models.signals import Signal
from ck_trading.storage.metadata_store import MetadataStore


class SignalManager:
    """Manage signal lifecycle: dedup, filter, log, dispatch."""

    def __init__(self, metadata_store: MetadataStore):
        self.store = metadata_store

    def process_signals(
        self,
        signals: list[Signal],
        min_score: float = 0.0,
        max_signals: int = 20,
    ) -> list[Signal]:
        """Process raw signals: deduplicate, filter, and log."""
        # Filter by minimum score
        filtered = [s for s in signals if s.score >= min_score]

        # Deduplicate: keep highest score per ticker
        seen: dict[str, Signal] = {}
        for signal in filtered:
            key = signal.ticker
            if key not in seen or signal.score > seen[key].score:
                seen[key] = signal

        deduped = sorted(seen.values(), key=lambda s: s.score, reverse=True)

        # Limit count
        final = deduped[:max_signals]

        # Log to database
        for signal in final:
            self.store.save_signal(signal)

        return final

    def get_recent_signals(self, limit: int = 50) -> list[dict]:
        """Get recent signals from the database."""
        return self.store.get_recent_signals(limit)
