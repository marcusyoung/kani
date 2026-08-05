"""Tests for classification_context.build_classification_input."""

from __future__ import annotations

from kani.classification_context import (
    DEFAULT_CLASSIFICATION_INPUT_MAX_CHARS,
    build_classification_input,
)


def _msg(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


class TestContextText:
    """TASK-037.02: context_text holds prior role-prefixed turns only."""

    def test_context_text_excludes_last_user_message(self) -> None:
        messages = [
            _msg("user", "what is kani?"),
            _msg("assistant", "a routing proxy"),
            _msg("user", "how do I configure it?"),
        ]
        result = build_classification_input(messages)

        assert "how do I configure it?" not in result.context_text
        assert "what is kani?" in result.context_text
        assert "a routing proxy" in result.context_text

    def test_context_text_uses_role_prefixed_lines(self) -> None:
        messages = [
            _msg("user", "first question"),
            _msg("assistant", "first answer"),
            _msg("user", "second question"),
        ]
        result = build_classification_input(messages)

        for line in result.context_text.split("\n"):
            assert line.startswith(("user:", "assistant:"))

    def test_context_text_has_no_conversation_header(self) -> None:
        messages = [
            _msg("user", "hello"),
            _msg("assistant", "hi there"),
            _msg("user", "bye"),
        ]
        result = build_classification_input(messages)

        assert "[conversation]" not in result.context_text

    def test_context_text_empty_when_only_user_message(self) -> None:
        messages = [_msg("user", "standalone question")]
        result = build_classification_input(messages)

        assert result.context_text == ""
        assert result.last_user_message == "standalone question"

    def test_text_still_includes_last_user_message(self) -> None:
        """AC #2: text field remains backward compatible (full conversation)."""
        messages = [
            _msg("user", "prior question"),
            _msg("assistant", "prior answer"),
            _msg("user", "current question"),
        ]
        result = build_classification_input(messages)

        assert "current question" in result.text
        assert "prior question" in result.text
        assert "prior answer" in result.text

    def test_context_text_truncated_to_max_chars(self) -> None:
        long_assistant = "x" * (DEFAULT_CLASSIFICATION_INPUT_MAX_CHARS + 200)
        messages = [
            _msg("assistant", long_assistant),
            _msg("user", "short"),
        ]
        result = build_classification_input(messages, max_chars=100)

        assert len(result.context_text) <= 100
        assert result.context_text.endswith("x")

    def test_context_text_preserves_chronological_order(self) -> None:
        messages = [
            _msg("user", "alpha"),
            _msg("assistant", "beta"),
            _msg("user", "gamma"),
        ]
        result = build_classification_input(messages)

        lines = result.context_text.split("\n")
        assert lines == ["user: alpha", "assistant: beta"]


class TestDualEmbeddingInputs:
    """TASK-037.02 AC #3: dual_embedding_inputs property."""

    def test_returns_tuple_in_order(self) -> None:
        messages = [
            _msg("user", "context here"),
            _msg("assistant", "reply"),
            _msg("user", "the request"),
        ]
        result = build_classification_input(messages)

        request, context = result.dual_embedding_inputs

        assert request == "the request"
        assert context == result.context_text

    def test_property_on_frozen_dataclass(self) -> None:
        """Property must work on a frozen dataclass without raising."""
        messages = [_msg("user", "solo")]
        result = build_classification_input(messages)

        assert result.dual_embedding_inputs == ("solo", "")

    def test_empty_inputs_when_no_user_message(self) -> None:
        messages = [
            _msg("system", "you are helpful"),
            _msg("assistant", "unsolicited greeting"),
        ]
        result = build_classification_input(messages)

        assert result.dual_embedding_inputs == ("", "")
        assert result.context_text == ""


class TestBackwardCompatibility:
    """AC #2 + #4: existing behaviour preserved."""

    def test_text_has_conversation_header(self) -> None:
        messages = [
            _msg("user", "hello"),
            _msg("assistant", "hi"),
            _msg("user", "again"),
        ]
        result = build_classification_input(messages)

        assert result.text.startswith("[conversation]")

    def test_last_user_message_unchanged(self) -> None:
        messages = [
            _msg("user", "first"),
            _msg("user", "second"),
        ]
        result = build_classification_input(messages)

        assert result.last_user_message == "second"

    def test_selected_counts_unchanged(self) -> None:
        messages = [
            _msg("user", "q1"),
            _msg("assistant", "a1"),
            _msg("user", "q2"),
            _msg("assistant", "a2"),
            _msg("user", "q3"),
        ]
        result = build_classification_input(messages)

        assert result.selected_user_turn_count == 3
        assert result.selected_assistant_turn_count == 2

    def test_short_followup_detection_unchanged(self) -> None:
        messages = [
            _msg("user", "explain quantum mechanics"),
            _msg("assistant", "long explanation..."),
            _msg("user", "ok"),
        ]
        result = build_classification_input(messages)

        assert result.last_user_is_short_followup is True
