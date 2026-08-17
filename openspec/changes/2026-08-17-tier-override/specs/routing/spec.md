## ADDED Requirements

### Requirement: Per-turn tier override

Kani MUST support a per-turn tier override via a `/kani:<tier>` token at position 0 of the latest user message content. When a valid tier name is supplied, kani MUST skip the scorer and pin the tier for that request. The token MUST be stripped from the message content before forwarding upstream. Only the latest user message MUST be scanned; tokens in assistant, system, or earlier user messages MUST NOT trigger an override.

#### Scenario: Valid override pins the tier and skips the scorer

**Given** the latest user message content starts with `/kani:reasoning`
**When** kani routes the request
**Then** kani MUST set the tier to `REASONING`
**And** kani MUST NOT invoke the distilled feature classifier
**And** the routing decision MUST carry `score=1.0`, `confidence=1.0`, `signals=["tier_override"]`, and `agentic_score=0.0`

#### Scenario: All four valid tiers are honored

**Given** the latest user message content starts with `/kani:simple` (or `medium`, `complex`, `reasoning`)
**When** kani routes the request
**Then** kani MUST set the tier to the corresponding uppercase tier name

#### Scenario: Tier matching is case-insensitive

**Given** the latest user message content starts with `/kani:Reasoning` or `/kani:REASONING`
**When** kani routes the request
**Then** kani MUST set the tier to `REASONING`

#### Scenario: Invalid tier warns and falls through to normal scoring

**Given** the latest user message content starts with `/kani:foo`
**When** kani routes the request
**Then** kani MUST set `tier_override` to `None`
**And** kani MUST still strip the `/kani:foo` token from the message content
**And** kani MUST log a warning
**And** kani MUST run the distilled feature classifier normally

#### Scenario: Token not at position 0 is ignored

**Given** the latest user message content contains `/kani:reasoning` but not at the very start
**When** kani routes the request
**Then** kani MUST NOT trigger an override
**And** kani MUST NOT strip any text from the message

#### Scenario: Token in assistant or history message is ignored

**Given** a `/kani:<tier>` token appears in an assistant message or in an earlier (non-latest) user message
**When** kani routes the request
**Then** kani MUST NOT trigger an override
**And** kani MUST NOT strip any text from any message

#### Scenario: Token stripped before upstream forwarding

**Given** the latest user message content starts with `/kani:reasoning prove P != NP`
**When** kani proxies the request upstream
**Then** the upstream payload message content MUST be `prove P != NP`
**And** the token `/kani:reasoning` MUST NOT appear in the upstream payload

#### Scenario: Token stripped before compaction

**Given** the latest user message content starts with a `/kani:<tier>` token
**And** context compaction is enabled
**When** kani runs compaction on the message history
**Then** the token MUST NOT appear in the compacted content or summary

#### Scenario: Multimodal list content checks first text part only

**Given** the latest user message content is a list of content blocks
**And** the first `{"type": "text", "text": ...}` part starts with `/kani:reasoning`
**When** kani routes the request
**Then** kani MUST trigger the override
**And** kani MUST strip the token from that first text part

#### Scenario: Multimodal list content with token in non-first text part

**Given** the latest user message content is a list of content blocks
**And** the first text part does NOT start with `/kani:`
**And** a later text part starts with `/kani:reasoning`
**When** kani routes the request
**Then** kani MUST NOT trigger an override

#### Scenario: Empty content after stripping is preserved

**Given** the latest user message content is `/kani:reasoning` with no trailing text
**When** kani strips the token
**Then** the resulting message content MUST be an empty string `""`
**And** kani MUST NOT remove the message from the message list

#### Scenario: Override respects capability filtering and input-limit checks

**Given** a valid tier override is set
**And** the request requires a capability (e.g. `vision`, `tools`, `json_mode`)
**When** kani routes the request
**Then** kani MUST still apply capability filtering and input-limit checks
**And** kani MUST fail closed if no candidate in the pinned tier satisfies the requirement

#### Scenario: Override does not bypass tier fallback

**Given** a valid tier override pins the tier to `COMPLEX`
**And** the profile does not define a `COMPLEX` tier
**When** kani routes the request
**Then** kani MUST fall back to an adjacent tier using the standard tier fallback order