"""Tests for the per-turn tier override (parse_tier_override + Router.route())."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from kani.config import (
    KaniConfig,
    ProviderConfig,
    ProfileConfig,
    TierModelConfig,
)
from kani.router import Router, parse_tier_override


def _make_config() -> KaniConfig:
    """Create a minimal routing config covering all four tiers."""
    return KaniConfig(
        host="0.0.0.0",
        port=18420,
        providers={
            "default": ProviderConfig(
                name="default",
                base_url="https://api.example.com/v1",
                api_key="test-key",
            )
        },
        default_provider="default",
        profiles={
            "auto": ProfileConfig(
                tiers={
                    "SIMPLE": TierModelConfig(primary="simple-model"),
                    "MEDIUM": TierModelConfig(primary="medium-model"),
                    "COMPLEX": TierModelConfig(primary="complex-model"),
                    "REASONING": TierModelConfig(primary="reasoning-model"),
                }
            )
        },
        default_profile="auto",
    )


class TestParseTierOverride:
    """Unit tests for the parse_tier_override helper."""

    @pytest.mark.parametrize("tier", ["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"])
    def test_valid_tier_string_content(self, tier: str) -> None:
        """A valid token at position 0 of string content sets the override."""
        messages = [{"role": "user", "content": f"/kani:{tier} hello world"}]
        override, stripped = parse_tier_override(messages)
        assert override == tier
        assert stripped[-1]["content"] == "hello world"

    def test_list_content_first_text_part(self) -> None:
        """List content strips the token from the first text part."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "/kani:complex describe this"},
                    {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                ],
            }
        ]
        override, stripped = parse_tier_override(messages)
        assert override == "COMPLEX"
        assert stripped[-1]["content"][0]["text"] == "describe this"
        # original input not mutated
        assert messages[0]["content"][0]["text"] == "/kani:complex describe this"

    def test_list_content_token_in_later_part_ignored(self) -> None:
        """A token in a later part (not the first text part) is ignored."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "plain text first"},
                    {"type": "text", "text": "/kani:reasoning later"},
                ],
            }
        ]
        override, stripped = parse_tier_override(messages)
        assert override is None
        assert stripped is messages

    def test_token_and_leading_whitespace_stripped(self) -> None:
        """The token and its trailing whitespace are stripped from content."""
        messages = [{"role": "user", "content": "/kani:medium   spaced text"}]
        override, stripped = parse_tier_override(messages)
        assert override == "MEDIUM"
        assert stripped[-1]["content"] == "spaced text"

    @pytest.mark.parametrize("token", ["reasoning", "REASONING", "Reasoning"])
    def test_case_insensitive_tier_matching(self, token: str) -> None:
        """Tier matching is case-insensitive."""
        messages = [{"role": "user", "content": f"/kani:{token} question"}]
        override, _ = parse_tier_override(messages)
        assert override == "REASONING"

    def test_invalid_tier_stripped_no_override(self) -> None:
        """An invalid tier yields override=None but the token is stripped."""
        messages = [{"role": "user", "content": "/kani:foo hello"}]
        override, stripped = parse_tier_override(messages)
        assert override is None
        assert stripped[-1]["content"] == "hello"

    def test_invalid_tier_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """An invalid tier logs a warning at log.warning level."""
        messages = [{"role": "user", "content": "/kani:nope hello"}]
        with caplog.at_level("WARNING", logger="kani.router"):
            parse_tier_override(messages)
        assert any("Invalid tier override" in rec.message for rec in caplog.records)

    def test_no_prefix_returns_original(self) -> None:
        """Messages without the token are returned unchanged (same object)."""
        messages: list[dict[str, Any]] = [{"role": "user", "content": "plain"}]
        override, stripped = parse_tier_override(messages)
        assert override is None
        assert stripped is messages

    def test_token_in_assistant_message_ignored(self) -> None:
        """Tokens in assistant messages do not trigger an override."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "/kani:reasoning reply"},
        ]
        override, stripped = parse_tier_override(messages)
        assert override is None
        assert stripped is messages

    def test_token_in_earlier_user_message_ignored(self) -> None:
        """Only the latest user message is scanned."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "/kani:reasoning old turn"},
            {"role": "assistant", "content": "response"},
            {"role": "user", "content": "plain current turn"},
        ]
        override, stripped = parse_tier_override(messages)
        assert override is None
        assert stripped is messages

    def test_empty_content_after_stripping_preserved(self) -> None:
        """Stripping to empty preserves "" (the message is not removed)."""
        messages = [{"role": "user", "content": "/kani:simple"}]
        override, stripped = parse_tier_override(messages)
        assert override == "SIMPLE"
        assert len(stripped) == 1
        assert stripped[-1]["content"] == ""


class TestRouterTierOverride:
    """Integration tests for Router.route() with tier_override."""

    @pytest.mark.parametrize("tier", ["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"])
    def test_override_pins_tier(self, tier: str) -> None:
        """A valid override pins decision.tier and the tier's model."""
        router = Router(_make_config())
        decision = router.route(
            [{"role": "user", "content": "anything"}],
            profile="auto",
            tier_override=tier,
        )
        assert decision.tier == tier
        assert decision.model == f"{tier.lower()}-model"

    def test_override_lowercase_accepted(self) -> None:
        """A lowercase override is normalized to uppercase."""
        router = Router(_make_config())
        decision = router.route(
            [{"role": "user", "content": "anything"}],
            profile="auto",
            tier_override="reasoning",
        )
        assert decision.tier == "REASONING"

    def test_override_skips_scorer(self) -> None:
        """A valid override means _classify is never called."""
        router = Router(_make_config())
        with patch.object(Router, "_classify", new=MagicMock()) as mock_classify:
            decision = router.route(
                [{"role": "user", "content": "anything"}],
                profile="auto",
                tier_override="REASONING",
            )
        mock_classify.assert_not_called()
        assert decision.score == 1.0
        assert decision.confidence == 1.0
        assert decision.signals == ["tier_override"]
        assert decision.agentic_score == 0.0

    def test_invalid_override_falls_through_to_scorer(self) -> None:
        """An invalid override logs a warning and runs normal scoring."""
        router = Router(_make_config())
        with (
            patch.object(Router, "_classify", new=MagicMock()) as mock_classify,
            patch("kani.router.log.warning") as mock_warning,
        ):
            router.route(
                [{"role": "user", "content": "anything"}],
                profile="auto",
                tier_override="not-a-tier",
            )
        mock_classify.assert_called_once()
        assert any(
            "Invalid tier_override" in str(call.args[0])
            for call in mock_warning.call_args_list
        )

    def test_invalid_override_still_routes(self) -> None:
        """An invalid override still produces a routing decision."""
        router = Router(_make_config())
        decision = router.route(
            [{"role": "user", "content": "simple question"}],
            profile="auto",
            tier_override="not-a-tier",
        )
        assert decision.model  # some model selected by normal scoring

    def test_override_respects_capability_escalation(self) -> None:
        """Override with required_capabilities still escalates tiers if needed."""
        config = KaniConfig(
            host="0.0.0.0",
            port=18420,
            providers={
                "default": ProviderConfig(
                    name="default",
                    base_url="https://api.example.com/v1",
                    api_key="test-key",
                )
            },
            default_provider="default",
            profiles={
                "auto": ProfileConfig(
                    tiers={
                        "SIMPLE": TierModelConfig(primary="text-only-model"),
                        "MEDIUM": TierModelConfig(primary="text-only-model"),
                        "COMPLEX": TierModelConfig(primary="vision-model"),
                        "REASONING": TierModelConfig(primary="vision-model"),
                    }
                )
            },
            default_profile="auto",
            model_rules=[],
        )
        from kani.config import ModelCapabilityEntry

        config.model_rules = [
            ModelCapabilityEntry(prefix="vision-model", capabilities=["vision"]),
        ]
        router = Router(config)
        decision = router.route(
            [{"role": "user", "content": "describe this"}],
            profile="auto",
            required_capabilities={"vision"},
            tier_override="SIMPLE",
        )
        # Escalated from overridden SIMPLE to a tier with a vision-capable model
        assert decision.model == "vision-model"
