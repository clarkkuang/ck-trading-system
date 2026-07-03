"""User-editable configuration for the AI model share monitor.

This is the ONLY file you should need to edit to tune the monitor:
thresholds, watched packages, flagship families, and the L2/L3 manual
checklist all live here.

Thresholds encode the investment-thesis trigger table:

    | Signal                                   | Threshold          | Action        |
    |------------------------------------------|--------------------|---------------|
    | OpenRouter CN dollar share (programming) | >35% for 8 weeks   | Bear 25%->30% |
    | Claude flagship $/M output               | -30% within 365d   | Cut GM tier   |
    | claude-code npm vs competitor CLIs       | 3-week decline     | advisory      |
"""

from __future__ import annotations

from ck_trading.monitoring.rules import AlertRule

# ---------------------------------------------------------------------------
# Threshold rules
# ---------------------------------------------------------------------------
RULES: tuple[AlertRule, ...] = (
    AlertRule(
        rule_id="or_cn_dollar_share_gt35_8w",
        description=(
            "Chinese-bloc dollar-weighted OpenRouter share (programming) "
            "above 35% for 8 consecutive weeks"
        ),
        metric_key="openrouter.weekly_dollar_share",
        dimensions={"bloc": "chinese", "scope": "programming"},
        fallback_dimensions=(
            {"bloc": "chinese", "scope": "programming:unverified"},
            {"bloc": "chinese", "scope": "all"},
        ),
        comparator="gt_consecutive",
        threshold=0.35,
        consecutive_periods=8,
        action_label="Bear 25%→30%",
        severity="trigger",
    ),
    AlertRule(
        rule_id="claude_opus_price_cut_gt30_365d",
        description="Claude Opus flagship completion price cut >30% within 365d",
        metric_key="openrouter.flagship_completion_price",
        dimensions={"family": "anthropic/claude-opus"},
        comparator="pct_drop_from_trailing_max",
        threshold=0.30,
        window_days=365,
        action_label="Cut GM tier",
        severity="trigger",
    ),
    AlertRule(
        rule_id="claude_sonnet_price_cut_gt30_365d",
        description="Claude Sonnet flagship completion price cut >30% within 365d",
        metric_key="openrouter.flagship_completion_price",
        dimensions={"family": "anthropic/claude-sonnet"},
        comparator="pct_drop_from_trailing_max",
        threshold=0.30,
        window_days=365,
        action_label="Cut GM tier",
        severity="trigger",
    ),
    AlertRule(
        rule_id="claude_code_npm_decline_3w",
        description=(
            "claude-code npm weekly downloads declining 3 consecutive weeks "
            "while a competitor CLI grows"
        ),
        metric_key="pkg.weekly_downloads",
        dimensions={"registry": "npm", "package": "@anthropic-ai/claude-code"},
        guard_dimensions=(
            {"registry": "npm", "package": "@openai/codex"},
            {"registry": "npm", "package": "@google/gemini-cli"},
            {"registry": "npm", "package": "@qwen-code/qwen-code"},
        ),
        comparator="decline_streak_with_competitor_growth",
        threshold=0.0,
        consecutive_periods=3,
        action_label="Advisory: dev adoption slipping vs competitor CLI",
        severity="advisory",
    ),
)

# ---------------------------------------------------------------------------
# Package watchlist: (registry, package, role)
# ---------------------------------------------------------------------------
PKG_WATCHLIST: tuple[tuple[str, str, str], ...] = (
    ("npm", "@anthropic-ai/claude-code", "subject"),
    ("npm", "@openai/codex", "competitor"),
    ("npm", "@google/gemini-cli", "competitor"),
    ("npm", "@qwen-code/qwen-code", "competitor"),
    ("pypi", "anthropic", "subject"),
    ("pypi", "openai", "competitor"),
    ("pypi", "dashscope", "competitor"),
    ("pypi", "google-genai", "competitor"),
)

# Flagship families whose completion price we track for cut detection.
FLAGSHIP_FAMILIES: tuple[str, ...] = (
    "anthropic/claude-opus",
    "anthropic/claude-sonnet",
)

# ---------------------------------------------------------------------------
# L2/L3 manual checklist (rendered by the dashboard, never evaluated)
# ---------------------------------------------------------------------------
DEFAULT_CHECKLIST: tuple[dict, ...] = (
    {
        "id": "menlo_report",
        "label": "Menlo Ventures enterprise LLM report — Anthropic still #1 share?",
        "url": "https://menlovc.com/perspective/",
        "cadence_days": 182,
    },
    {
        "id": "a16z_survey",
        "label": "a16z enterprise AI survey — spend intentions vs Anthropic",
        "url": "https://a16z.com/enterprise/",
        "cadence_days": 365,
    },
    {
        "id": "edgar_s1",
        "label": "EDGAR full-text search: AI lab S-1 filings (NRR / Claude Code revenue share)",
        "url": "https://efts.sec.gov/LATEST/search-index?q=%22Anthropic%22&dateRange=custom&forms=S-1",
        "cadence_days": 92,
    },
    {
        "id": "ramp_ai_index",
        "label": "Ramp AI Index — Anthropic paid-adoption curve still rising?",
        "url": "https://ramp.com/data",
        "cadence_days": 31,
    },
    {
        "id": "swe_bench_gap",
        "label": "SWE-bench Pro — Claude best vs open-source best gap (pp)",
        "url": "https://www.swebench.com/",
        "cadence_days": 31,
    },
    {
        "id": "aa_price_intel",
        "label": "Artificial Analysis price-intelligence frontier — frontier line shift",
        "url": "https://artificialanalysis.ai/models",
        "cadence_days": 31,
    },
)
