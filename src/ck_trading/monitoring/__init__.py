"""AI model market-share monitoring.

Research telemetry for tracking the open-source / Chinese-lab competitive
threat to Anthropic. NOT a trading strategy — nothing here registers in the
strategy registry or emits BUY/SELL signals.

Data flows:
    collectors (OpenRouter pricing + rankings, npm/PyPI downloads)
    -> MonitoringStore (git-tracked parquet under data/monitoring/)
    -> metrics (dollar-weighted bloc shares, flagship price series)
    -> rules (threshold engine) -> alerts.json (episode state)
    -> dashboard page 08_ai_model_share
"""
