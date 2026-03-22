"""Notification dispatcher using Apprise."""

from ck_trading.config import settings
from ck_trading.models.signals import Signal


class Notifier:
    """Send signal notifications via Apprise (email, Telegram, etc.)."""

    def __init__(self, urls: str | None = None):
        self.urls = urls or settings.notification_urls

    def notify(self, signals: list[Signal]) -> bool:
        """Send notification for new signals."""
        if not self.urls or not signals:
            return False

        try:
            import apprise

            apobj = apprise.Apprise()
            for url in self.urls.split(","):
                url = url.strip()
                if url:
                    apobj.add(url)

            title = f"Trading Signals: {len(signals)} new"
            body = self._format_signals(signals)

            return apobj.notify(title=title, body=body)
        except ImportError:
            print("Apprise not installed. Run: uv add apprise")
            return False
        except Exception as e:
            print(f"Notification error: {e}")
            return False

    def _format_signals(self, signals: list[Signal]) -> str:
        lines = []
        for s in signals:
            price_str = f" @ ${s.price_at_signal:.2f}" if s.price_at_signal else ""
            lines.append(
                f"[{s.signal_type}] {s.ticker}{price_str} "
                f"(Score: {s.score:.2f}, Strategy: {s.strategy_name})\n"
                f"  {s.rationale}"
            )
        return "\n\n".join(lines)
