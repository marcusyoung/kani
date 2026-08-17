"""Tests for the per-turn tier override (parse_tier_override + Router.route())."""

from __future__ import annotations

import logging
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


class TestProxyTierOverride:
    """Proxy integration tests for the tier override (chat completions + debug)."""

    _CONFIG_YAML = """\
default_provider: dummy
default_profile: auto
providers:
  dummy:
    name: dummy
    base_url: "http://localhost:9999/v1"
    api_key: "fake"
profiles:
  auto:
    tiers:
      SIMPLE: {primary: "auto-simple"}
      MEDIUM: {primary: "auto-medium"}
      COMPLEX: {primary: "auto-complex"}
      REASONING: {primary: "auto-reason"}
"""

    _UPSTREAM_RESPONSE = {
        "id": "x",
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }

    @pytest.fixture()
    def proxy_client(self, tmp_path, monkeypatch):
        """Configured proxy app with upstream forwarding mocked.

        Yields (client, captured) where `captured` accumulates dicts holding
        the exact body and decision handed to _proxy_upstream.
        """
        import kani.proxy as proxy_mod
        from fastapi.responses import JSONResponse
        from fastapi.testclient import TestClient

        cfg = tmp_path / "config.yaml"
        cfg.write_text(self._CONFIG_YAML)
        monkeypatch.setenv("KANI_DATA_DIR", str(tmp_path / "data"))
        proxy_mod.configure(str(cfg))

        captured: list[dict[str, Any]] = []

        async def fake_proxy_upstream(
            base_url: str,
            api_key: str,
            body: dict[str, Any],
            decision: Any,
            profile: Any = None,
            **kwargs: Any,
        ):
            captured.append({"body": body, "decision": decision})
            response = JSONResponse(content=self._UPSTREAM_RESPONSE)
            if decision is not None:
                for key, value in proxy_mod._kani_headers(decision).items():
                    response.headers[key] = value
            return response

        monkeypatch.setattr(proxy_mod, "_proxy_upstream", fake_proxy_upstream)

        with TestClient(proxy_mod.app) as client:
            yield client, captured

    @pytest.mark.parametrize("tier", ["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"])
    def test_valid_override_pins_tier_via_header(self, proxy_client, tier: str) -> None:
        """A valid /kani:<tier> token forces the overridden tier (AC #4)."""
        client, _ = proxy_client
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "kani/auto",
                "messages": [{"role": "user", "content": f"/kani:{tier} hi"}],
            },
        )
        assert resp.status_code == 200
        assert resp.headers["X-Kani-Tier"] == tier

    def test_token_stripped_from_upstream_body(self, proxy_client) -> None:
        """The /kani:<tier> token never reaches the upstream model (AC #5)."""
        client, captured = proxy_client
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "kani/auto",
                "messages": [{"role": "user", "content": "/kani:complex hi there"}],
            },
        )
        assert resp.status_code == 200
        upstream_body = captured[0]["body"]
        contents = [m["content"] for m in upstream_body["messages"]]
        assert contents == ["hi there"]
        assert not any("/kani:" in str(c) for c in contents)
        assert upstream_body["model"] == "auto-complex"

    def test_tier_override_log_line(self, proxy_client, caplog) -> None:
        """A TIER_OVERRIDE INFO line is logged with request id + tier (AC #3)."""
        client, _ = proxy_client
        with caplog.at_level(logging.INFO, logger="kani.proxy"):
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "kani/auto",
                    "messages": [{"role": "user", "content": "/kani:reasoning hi"}],
                },
            )
        tier_lines = [r for r in caplog.records if "TIER_OVERRIDE" in r.getMessage()]
        assert tier_lines, "expected a TIER_OVERRIDE log line"
        assert "tier_override=REASONING" in tier_lines[0].getMessage()

    def test_invalid_tier_stripped_and_normal_routing(self, proxy_client) -> None:
        """/kani:foo strips the token but routes normally (AC #6)."""
        client, captured = proxy_client
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "kani/auto",
                "messages": [{"role": "user", "content": "/kani:foo hello"}],
            },
        )
        assert resp.status_code == 200
        contents = [m["content"] for m in captured[0]["body"]["messages"]]
        assert contents == ["hello"]
        # Normal scoring still ran — tier is not pinned to an invalid value
        assert resp.headers["X-Kani-Tier"] in {
            "SIMPLE",
            "MEDIUM",
            "COMPLEX",
            "REASONING",
        }

    def test_token_in_history_not_triggered(self, proxy_client) -> None:
        """A token in history does not trigger override or stripping (AC #7)."""
        client, captured = proxy_client
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "kani/auto",
                "messages": [
                    {"role": "user", "content": "/kani:reasoning earlier turn"},
                    {"role": "assistant", "content": "earlier reply"},
                    {"role": "user", "content": "plain current turn"},
                ],
            },
        )
        assert resp.status_code == 200
        contents = [m["content"] for m in captured[0]["body"]["messages"]]
        assert contents[0] == "/kani:reasoning earlier turn"

    def test_token_in_assistant_message_not_triggered(self, proxy_client) -> None:
        """A token in an assistant message does not trigger anything (AC #7)."""
        client, captured = proxy_client
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "kani/auto",
                "messages": [
                    {"role": "assistant", "content": "/kani:complex quoted"},
                    {"role": "user", "content": "plain question"},
                ],
            },
        )
        assert resp.status_code == 200
        contents = [m["content"] for m in captured[0]["body"]["messages"]]
        assert contents[0] == "/kani:complex quoted"

    def test_route_debug_honours_override(self, proxy_client) -> None:
        """route_debug parses and passes the override to route() (ACs #8/#9)."""
        client, _ = proxy_client
        resp = client.post(
            "/v1/route",
            json={"messages": [{"role": "user", "content": "/kani:complex hi"}]},
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["tier_override"] == "COMPLEX"
        assert payload["tier"] == "COMPLEX"
        assert payload["model"] == "auto-complex"

    def test_route_debug_no_token_reports_null(self, proxy_client) -> None:
        """route_debug without a token reports tier_override: null (AC #9)."""
        client, _ = proxy_client
        resp = client.post(
            "/v1/route",
            json={"messages": [{"role": "user", "content": "plain prompt"}]},
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["tier_override"] is None
