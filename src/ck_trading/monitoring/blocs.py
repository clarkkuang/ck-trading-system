"""Bloc classification for OpenRouter model ids.

Maps each model to one of six blocs for the dollar-weighted share metric:

    anthropic       — Anthropic (its own bloc; the thesis subject)
    western_closed  — Western API-only labs (OpenAI, Google, xAI, ...)
    western_open    — Western open-weight labs (Meta, Mistral open tier, ...)
    chinese         — Chinese labs, open or closed (DeepSeek, Qwen, GLM, ...)
    unclassified    — real model whose org isn't mapped yet (fix by updating
                      ORG_BLOC_MAP; surfaced in logs and job_runs results)
    other           — OpenRouter's top-50 truncation aggregate row
                      (measurement error we cannot attribute)

Two-level precedence: exact model-id overrides beat the org-prefix table —
needed for orgs that ship both open and closed tiers (Mistral).
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

BLOC_ANTHROPIC = "anthropic"
BLOC_WESTERN_CLOSED = "western_closed"
BLOC_WESTERN_OPEN = "western_open"
BLOC_CHINESE = "chinese"
BLOC_UNCLASSIFIED = "unclassified"
BLOC_OTHER = "other"

ALL_BLOCS = (
    BLOC_ANTHROPIC,
    BLOC_WESTERN_CLOSED,
    BLOC_WESTERN_OPEN,
    BLOC_CHINESE,
    BLOC_UNCLASSIFIED,
    BLOC_OTHER,
)

# Sentinel model_id for OpenRouter's "everything outside the top 50" row.
OTHER_MODEL_ID = "__other__"

# Precedence 1: exact model-id overrides (checked after stripping ":variant").
# Mistral's commercial/closed tier — org default below is western_open.
MODEL_BLOC_OVERRIDES: dict[str, str] = {
    "mistralai/mistral-large": BLOC_WESTERN_CLOSED,
    "mistralai/mistral-large-2411": BLOC_WESTERN_CLOSED,
    "mistralai/mistral-medium": BLOC_WESTERN_CLOSED,
    "mistralai/mistral-medium-3": BLOC_WESTERN_CLOSED,
    "mistralai/codestral-2501": BLOC_WESTERN_CLOSED,
    "mistralai/mistral-saba": BLOC_WESTERN_CLOSED,
}

# Precedence 2: org prefix (the part before "/").
ORG_BLOC_MAP: dict[str, str] = {
    # -- anthropic
    "anthropic": BLOC_ANTHROPIC,
    # -- chinese labs
    "deepseek": BLOC_CHINESE,
    "qwen": BLOC_CHINESE,
    "alibaba": BLOC_CHINESE,
    "z-ai": BLOC_CHINESE,        # Zhipu / GLM (current slug)
    "zhipuai": BLOC_CHINESE,     # Zhipu (legacy slug)
    "thudm": BLOC_CHINESE,       # GLM academic slug
    "moonshotai": BLOC_CHINESE,  # Kimi
    "minimax": BLOC_CHINESE,
    "01-ai": BLOC_CHINESE,
    "baichuan": BLOC_CHINESE,
    "tencent": BLOC_CHINESE,     # Hunyuan
    "bytedance": BLOC_CHINESE,   # Doubao / Seed / UI-TARS
    "bytedance-research": BLOC_CHINESE,
    "bytedance-seed": BLOC_CHINESE,
    "stepfun": BLOC_CHINESE,
    "inclusionai": BLOC_CHINESE,
    "baidu": BLOC_CHINESE,       # ERNIE
    "xiaomi": BLOC_CHINESE,      # MiMo
    "kwaipilot": BLOC_CHINESE,   # Kuaishou KAT
    # -- western closed (API-only labs)
    "openai": BLOC_WESTERN_CLOSED,
    "google": BLOC_WESTERN_CLOSED,
    "x-ai": BLOC_WESTERN_CLOSED,
    "cohere": BLOC_WESTERN_CLOSED,
    "amazon": BLOC_WESTERN_CLOSED,
    "ai21": BLOC_WESTERN_CLOSED,
    "perplexity": BLOC_WESTERN_CLOSED,
    "inflection": BLOC_WESTERN_CLOSED,
    "rekaai": BLOC_WESTERN_CLOSED,
    "writer": BLOC_WESTERN_CLOSED,     # Palmyra
    "poolside": BLOC_WESTERN_CLOSED,   # Laguna coding models
    "inception": BLOC_WESTERN_CLOSED,  # Mercury diffusion LMs
    # -- western open-weights
    "meta-llama": BLOC_WESTERN_OPEN,
    "mistralai": BLOC_WESTERN_OPEN,   # default; closed tier via overrides
    "microsoft": BLOC_WESTERN_OPEN,   # Phi / WizardLM
    "nousresearch": BLOC_WESTERN_OPEN,
    "nvidia": BLOC_WESTERN_OPEN,
    "allenai": BLOC_WESTERN_OPEN,
    "liquid": BLOC_WESTERN_OPEN,
    "ibm-granite": BLOC_WESTERN_OPEN,
    "arcee-ai": BLOC_WESTERN_OPEN,
    "deepcogito": BLOC_WESTERN_OPEN,
    # open-weight finetuner community (roleplay-heavy; western open weights)
    "cognitivecomputations": BLOC_WESTERN_OPEN,
    "thedrummer": BLOC_WESTERN_OPEN,
    "sao10k": BLOC_WESTERN_OPEN,
    "anthracite-org": BLOC_WESTERN_OPEN,
    "gryphe": BLOC_WESTERN_OPEN,
    "undi95": BLOC_WESTERN_OPEN,
}


def org_of(model_id: str) -> str:
    """'deepseek/deepseek-v4:free' -> 'deepseek'. No slash -> whole id.

    OpenRouter alias entries prefix the org with '~' (e.g.
    '~anthropic/claude-fable-latest'); the tilde is stripped so aliases
    classify with their real org.
    """
    return model_id.split("/", 1)[0].lower().strip().lstrip("~")


def strip_variant(model_id: str) -> str:
    """Drop the ':variant' suffix: 'qwen/qwen3-coder:free' -> 'qwen/qwen3-coder'."""
    return model_id.split(":", 1)[0]


def classify_model_id(model_id: str, *, warned_orgs: set[str] | None = None) -> str:
    """Classify a model id into a bloc.

    Args:
        model_id: OpenRouter model id, possibly with ':variant' suffix, or the
            OTHER_MODEL_ID sentinel.
        warned_orgs: optional set threaded by the caller so each unknown org
            warns only once per run. Mutated in place.
    """
    if model_id == OTHER_MODEL_ID:
        return BLOC_OTHER

    base_id = strip_variant(model_id).lower()
    override = MODEL_BLOC_OVERRIDES.get(base_id)
    if override is not None:
        return override

    org = org_of(model_id)
    bloc = ORG_BLOC_MAP.get(org)
    if bloc is not None:
        return bloc

    if warned_orgs is None or org not in warned_orgs:
        logger.warning(
            "Unclassified OpenRouter org %r (model %r) — add to ORG_BLOC_MAP",
            org,
            model_id,
        )
        if warned_orgs is not None:
            warned_orgs.add(org)
    return BLOC_UNCLASSIFIED


# Trailing tokens that are version/variant noise, not family identity.
_FAMILY_STRIP_RE = re.compile(
    r"""
    (
        [-_.]v?\d+(\.\d+)*      # -4, -4.5, -v2, .5
      | [-_.]\d{6,8}            # -20250514 date stamps
      | [-_.](latest|preview|beta|exp|it|instruct|chat|thinking)
    )+$
    """,
    re.VERBOSE,
)


def family_key(model_id: str) -> str:
    """Version-churn-tolerant family id.

    'anthropic/claude-sonnet-4.5:free' -> 'anthropic/claude-sonnet'
    'anthropic/claude-opus-4-1'        -> 'anthropic/claude-opus'
    """
    base = strip_variant(model_id).lower()
    if "/" in base:
        org, name = base.split("/", 1)
    else:
        org, name = "", base
    prev = None
    while prev != name:
        prev = name
        name = _FAMILY_STRIP_RE.sub("", name)
    return f"{org}/{name}" if org else name


# Generic qualifier tokens dropped from the canonical key — they describe a
# SKU variant (speed/size/modality/mode), not the model family identity.
_GENERIC_TOKENS = frozenset({
    "pro", "flash", "lite", "max", "plus", "mini", "nano", "micro",
    "thinking", "instruct", "chat", "preview", "latest", "code", "coder",
    "fast", "turbo", "vl", "image", "multi", "agent", "it", "base", "exp",
    "beta", "edge", "small", "medium", "large", "xl", "ultra", "high", "low",
    "air", "standard", "reasoner", "distill", "online", "vision", "audio",
})

# A version-ish token: digits with optional dots, optional leading 'v',
# e.g. "4", "4.7", "v3", "3.2". Pure date stamps handled separately.
_VERSION_RE = re.compile(r"^v?\d+(\.\d+)*$")
# Date-stamp fragments appended by the rankings permaslug format.
_DATE_TOKEN_RE = re.compile(r"^(\d{8}|\d{6}|\d{4}|\d{2}-\d{2})$")


def canonical_family_key(model_id: str) -> str:
    """Order-insensitive family key that matches across OpenRouter's two id
    formats (rankings permaslug vs /models id).

        'anthropic/claude-4.7-opus-20260416' -> 'anthropic|claude-opus|4.7'
        'anthropic/claude-opus-4.7'           -> 'anthropic|claude-opus|4.7'
        'anthropic/claude-sonnet-5-20260630'  -> 'anthropic|claude-sonnet|5'

    Word tokens (family identity) are sorted so word order doesn't matter;
    generic SKU qualifiers and date stamps are dropped; the largest version
    number is kept as the version component. Embedded-number model names
    (qwen3, gpt-4o) keep the token verbatim as a word so they still group.
    """
    org = org_of(model_id)
    base = strip_variant(model_id).lower()
    name = base.split("/", 1)[1] if "/" in base else base

    words: list[str] = []
    versions: list[float] = []
    raw_versions: list[str] = []
    i = 0
    parts = name.replace("_", "-").split("-")
    while i < len(parts):
        tok = parts[i]
        # rejoin "MM-DD" style date fragments
        if _DATE_TOKEN_RE.match(tok):
            i += 1
            continue
        if i + 1 < len(parts) and re.match(r"^\d{2}$", tok) and re.match(r"^\d{2}$", parts[i + 1]):
            # e.g. "04-02" trailing date
            i += 2
            continue
        if _VERSION_RE.match(tok):
            num = tok.lstrip("v")
            try:
                versions.append(float(num))
                raw_versions.append(num)
            except ValueError:
                words.append(tok)
        elif tok in _GENERIC_TOKENS:
            pass
        elif tok:
            words.append(tok)
        i += 1

    word_part = "-".join(sorted(words)) if words else name.split("-")[0]
    if versions:
        # keep the raw string of the max version to preserve "4.7" vs "4"
        vmax = raw_versions[versions.index(max(versions))]
    else:
        vmax = ""
    return f"{org}|{word_part}|{vmax}"
