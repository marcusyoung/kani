# Issue: Smarter tool capability detection avoids false-positive model routing

## Problem

When a client sends a request with `tools` or `functions` in the body but no active tool usage (no `tool_calls` in assistant messages, no `role=tool` messages after the last user turn), the proxy currently treats it as requiring a tools-capable model. This causes unnecessary up-tiering to heavier models that support tool use, even when the conversation is just a simple question that happens to carry a tool schema for context.

Common scenario: an OpenCode-style client always includes its full tool schema in every request. The model never actually calls a tool — the conversation is purely conversational — but the router sees `tools` in the body and routes to a tool-capable (often more expensive) tier.

## Our solution

We modified `_detect_required_capabilities` in `proxy.py` to distinguish between **decorative** tool declarations and **active** tool usage:

```python
# Check for tools/functions capability
# Only require tools if the conversation has active tool usage
# (tool_calls or role=tool messages) since the last user message.
if "tools" in body or "functions" in body:
    # Find the last user message
    last_user_idx = -1
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_user_idx = idx
            break
    # Check for tool activity after the last user message
    found_tool_activity = False
    for idx in range(last_user_idx + 1, len(messages)):
        msg = messages[idx]
        if isinstance(msg, dict):
            if msg.get("role") == "tool" or "tool_calls" in msg:
                found_tool_activity = True
                break
    if found_tool_activity:
        required.add("tools")
```

### How it works

1. If `tools` or `functions` fields are present, scan the message history.
2. Find the last user message.
3. Check for **active tool usage** after that user message: any `role=tool` message or any assistant message containing `tool_calls`.
4. Only add `"tools"` to required capabilities if active tool usage is found.
5. Bare `tools`/`functions` declarations without active tool calls are treated as decorative (context-only) and do not up-tier the model.

### Why this matters for routing

Most LLM proxy clients (OpenCode, Cursor, Continue) send their full tool schema on every request. In the majority of turns, the model responds with plain text — no tool calls. Treating every tool-carrying request as tools-required means nearly every request gets routed to a tools-capable model, defeating the purpose of lightweight/fast tier routing for simple conversational turns.

## Compatibility note

The current upstream tests for `_detect_required_capabilities` (in `test_capability_routing.py`) expect `"tools"` to be detected whenever the `tools` or `functions` field is present, regardless of active tool usage. These tests would need to be updated to reflect the smarter detection logic:

- `test_detect_tools_capability_via_tools_field` — currently passes a body with `tools` but no tool activity; should assert `"tools" not in caps` or be updated with tool activity.
- `test_detect_tools_capability_via_functions_field` — same pattern.
- `test_detect_multiple_capabilities` — includes `tools` field without tool activity; should either add tool activity or remove the `"tools" in caps` assertion.

## Suggested tests

To properly cover both cases, the test suite should include:

1. **Tool declaration with active tool usage** — `tools` field present, assistant message with `tool_calls` after the last user message → `"tools" in caps`
2. **Tool declaration with tool response** — `tools` field present, `role=tool` message after last user message → `"tools" in caps`
3. **Tool declaration without active usage** — `tools` field present, no tool activity after last user message → `"tools" not in caps`
4. **Functions field with active usage** — same pattern for the legacy `functions` field
5. **Multi-turn with tools resolved** — earlier tool activity before the last user message, none after → `"tools" not in caps` (tools are no longer active)

## Files involved

- `src/kani/proxy.py` — `_detect_required_capabilities` function (our modification)
- `tests/test_capability_routing.py` — `TestCapabilityDetection` class (needs test updates)