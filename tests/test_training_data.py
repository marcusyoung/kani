"""Cross-cutting tests for dual-embedding and full-context annotation paths."""

from __future__ import annotations

from kani.training_data import (
    FULL_CONVERSATION_MAX_CHARS,
    _classification_dual_inputs_from_record,
    _full_conversation_from_record,
)


class TestFullConversationFromRecord:
    def test_normalizes_multimodal_text_part_content(self) -> None:
        record = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image_url", "image_url": {"url": "file:///x.png"}},
                        {"type": "text", "text": "Please explain."},
                    ],
                },
            ]
        }

        result = _full_conversation_from_record(record)

        assert "What is this?" in result
        assert "Please explain." in result

    def test_returns_empty_without_messages_key(self) -> None:
        record: dict = {"prompt": "Hello", "signals": {}}

        assert _full_conversation_from_record(record) == ""

    def test_returns_empty_with_empty_messages_list(self) -> None:
        record: dict = {"messages": []}

        assert _full_conversation_from_record(record) == ""

    def test_filters_non_standard_roles(self) -> None:
        record = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "tool", "content": "tool output"},
                {"role": "assistant", "content": "Hi there"},
                {"role": "function", "content": "fn result"},
            ]
        }

        result = _full_conversation_from_record(record)

        assert "system: You are helpful." in result
        assert "user: Hello" in result
        assert "assistant: Hi there" in result
        assert "tool" not in result
        assert "function" not in result

    def test_truncates_at_full_conversation_max_chars(self) -> None:
        long_text = "x" * (FULL_CONVERSATION_MAX_CHARS + 500)
        record = {
            "messages": [
                {"role": "user", "content": long_text},
            ]
        }

        result = _full_conversation_from_record(record)

        assert len(result) == FULL_CONVERSATION_MAX_CHARS
        # The role prefix "user: " is part of the conversation string before truncation
        assert result.startswith("user: ")
        assert result == ("user: " + long_text)[:FULL_CONVERSATION_MAX_CHARS]

    def test_skips_messages_with_empty_content(self) -> None:
        record = {
            "messages": [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "   "},
                {"role": "user", "content": "valid"},
            ]
        }

        result = _full_conversation_from_record(record)

        assert "valid" in result
        assert "assistant:" not in result


class TestClassificationDualInputsBackwardCompat:
    def test_handles_null_context_field_values(self) -> None:
        record = {
            "prompt": "Hello",
            "classification_context": {
                "text": "[conversation]\nuser: Hello",
                "last_user_message": None,
                "context_text": None,
            },
        }

        text, last_user_message, context_text = _classification_dual_inputs_from_record(
            record
        )

        assert text == "[conversation]\nuser: Hello"
        assert last_user_message == ""
        assert context_text == ""

    def test_partial_dual_fields_fall_back_to_empty(self) -> None:
        record = {
            "prompt": "Hello",
            "classification_context": {
                "text": "[conversation]\nuser: Hello",
                "last_user_message": "Hello",
                # context_text missing
            },
        }

        text, last_user_message, context_text = _classification_dual_inputs_from_record(
            record
        )

        assert text == "[conversation]\nuser: Hello"
        assert last_user_message == "Hello"
        assert context_text == ""
