"""Tests for bloc classification."""

import logging

from ck_trading.monitoring.blocs import (
    BLOC_ANTHROPIC,
    BLOC_CHINESE,
    BLOC_OTHER,
    BLOC_UNCLASSIFIED,
    BLOC_WESTERN_CLOSED,
    BLOC_WESTERN_OPEN,
    MODEL_BLOC_OVERRIDES,
    ORG_BLOC_MAP,
    OTHER_MODEL_ID,
    classify_model_id,
    family_key,
    org_of,
    strip_variant,
)


class TestClassify:
    def test_anthropic(self):
        assert classify_model_id("anthropic/claude-sonnet-5") == BLOC_ANTHROPIC

    def test_chinese_orgs(self):
        for mid in [
            "deepseek/deepseek-v4",
            "qwen/qwen3-coder",
            "moonshotai/kimi-k2",
            "z-ai/glm-5",
            "minimax/minimax-m2",
            "bytedance/doubao-2",
            "tencent/hunyuan-large",
            "baidu/ernie-5",
        ]:
            assert classify_model_id(mid) == BLOC_CHINESE, mid

    def test_western_closed(self):
        for mid in ["openai/gpt-6", "google/gemini-3-pro", "x-ai/grok-5"]:
            assert classify_model_id(mid) == BLOC_WESTERN_CLOSED, mid

    def test_western_open(self):
        for mid in ["meta-llama/llama-5", "nousresearch/hermes-4"]:
            assert classify_model_id(mid) == BLOC_WESTERN_OPEN, mid

    def test_variant_suffix_stripped(self):
        assert classify_model_id("qwen/qwen3-coder:free") == BLOC_CHINESE

    def test_override_beats_org(self):
        # mistralai default is western_open; large is closed-tier
        assert classify_model_id("mistralai/mistral-large") == BLOC_WESTERN_CLOSED
        assert classify_model_id("mistralai/mistral-7b-instruct") == BLOC_WESTERN_OPEN

    def test_override_with_variant(self):
        assert (
            classify_model_id("mistralai/mistral-large:nitro")
            == BLOC_WESTERN_CLOSED
        )

    def test_other_sentinel(self):
        assert classify_model_id(OTHER_MODEL_ID) == BLOC_OTHER

    def test_tilde_alias_classified_as_real_org(self):
        assert classify_model_id("~anthropic/claude-fable-latest") == BLOC_ANTHROPIC
        assert classify_model_id("~moonshotai/kimi-latest") == BLOC_CHINESE

    def test_new_chinese_orgs(self):
        assert classify_model_id("xiaomi/mimo-v2.5-pro") == BLOC_CHINESE
        assert classify_model_id("kwaipilot/kat-coder-pro-v2") == BLOC_CHINESE
        assert classify_model_id("bytedance-seed/seed-2.0-lite") == BLOC_CHINESE

    def test_unknown_org_unclassified(self):
        assert classify_model_id("mysterylab/model-x") == BLOC_UNCLASSIFIED

    def test_unknown_org_warns_once(self, caplog):
        warned: set[str] = set()
        with caplog.at_level(logging.WARNING):
            classify_model_id("newlab/m1", warned_orgs=warned)
            classify_model_id("newlab/m2", warned_orgs=warned)
        msgs = [r for r in caplog.records if "newlab" in r.getMessage()]
        assert len(msgs) == 1
        assert "newlab" in warned

    def test_all_seed_orgs_have_valid_blocs(self):
        valid = {
            BLOC_ANTHROPIC, BLOC_CHINESE,
            BLOC_WESTERN_CLOSED, BLOC_WESTERN_OPEN,
        }
        for org, bloc in ORG_BLOC_MAP.items():
            assert bloc in valid, org
        for mid, bloc in MODEL_BLOC_OVERRIDES.items():
            assert bloc in valid, mid


class TestHelpers:
    def test_org_of(self):
        assert org_of("deepseek/deepseek-v4:free") == "deepseek"
        assert org_of("noslash") == "noslash"

    def test_strip_variant(self):
        assert strip_variant("qwen/qwen3:free") == "qwen/qwen3"
        assert strip_variant("qwen/qwen3") == "qwen/qwen3"

    def test_family_key_collapses_versions(self):
        assert family_key("anthropic/claude-sonnet-4") == "anthropic/claude-sonnet"
        assert family_key("anthropic/claude-sonnet-4.5") == "anthropic/claude-sonnet"
        assert family_key("anthropic/claude-sonnet-4.5:free") == "anthropic/claude-sonnet"
        assert family_key("anthropic/claude-opus-4-1") == "anthropic/claude-opus"

    def test_family_key_date_stamp(self):
        assert family_key("anthropic/claude-opus-20250514") == "anthropic/claude-opus"

    def test_family_key_distinct_families_stay_distinct(self):
        assert family_key("anthropic/claude-opus-4") != family_key(
            "anthropic/claude-sonnet-4"
        )
