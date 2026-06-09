## Requirements

### Requirement: Configurable synchronous context compaction

kani MUST allow operators to enable or disable synchronous request-time context compaction independently from other smart-proxy features.

#### Scenario: Inline compaction runs for oversized routed requests

**Given** smart-proxy synchronous compaction is enabled
**And** a routed `POST /v1/chat/completions` request exceeds the configured compaction threshold
**When** kani processes the request
**Then** kani MUST replace the compactable middle region of `messages` with a generated handoff summary before proxying upstream
**And** kani MUST preserve configured protected head and tail turns
**And** kani MUST return an operator-visible signal that inline compaction was applied

#### Scenario: Inline compaction is skipped safely

**Given** smart-proxy synchronous compaction is disabled or the request cannot be compacted safely
**When** kani processes the routed request
**Then** kani MUST proxy the request without compaction
**And** kani MUST preserve existing routing behavior
**And** kani MUST expose that compaction was skipped or disabled

### Requirement: Session-aware background precompaction

kani MUST support session-aware background precompaction that can be enabled or disabled independently from synchronous compaction.

#### Scenario: Background summary is queued after threshold crossing

**Given** background precompaction is enabled
**And** kani can resolve a stable session identifier for a routed conversation
**And** the session crosses the configured precompaction trigger threshold
**When** kani finishes proxying the request and records usage
**Then** kani MUST persist the current session snapshot and enqueue or refresh a background summary job for that snapshot
**And** kani MUST avoid duplicating an equivalent in-flight job for the same session snapshot

#### Scenario: Ready summary is reused on a later request

**Given** background precompaction is enabled
**And** a ready summary exists for the current session snapshot
**When** a later routed request for the same session arrives
**Then** kani MUST be able to reuse the cached summary instead of recomputing it inline
**And** kani MUST surface that a cached compaction artifact was used

### Requirement: Switchable session and telemetry controls

kani MUST expose switchable controls and operator-visible telemetry for smart-proxy context reduction behavior.

#### Scenario: Explicit session header is preferred

**Given** a client sends the configured session header
**When** kani resolves session identity for smart-proxy compaction
**Then** kani MUST use the explicit header as the session identifier
**And** kani MUST make the explicit resolution mode observable for operators

#### Scenario: No session header means no session identity

**Given** a client does not send the configured session header
**When** kani resolves session identity for smart-proxy compaction
**Then** kani MUST NOT derive a replacement session identifier from model or message content
**And** kani MUST expose that no session identity was resolved
**And** inline compaction MAY still run for an oversized request
**And** cache reuse, persistence, incremental summarization, and background precompaction MUST be unavailable for that request because they require a stable session identifier

#### Scenario: Compaction failures do not break proxying

**Given** smart-proxy compaction is enabled
**And** summary generation, persistence, or cache lookup fails
**When** kani handles the request
**Then** kani MUST continue proxying the request without returning a compaction-specific failure to the client
**And** kani MUST record the failure for operators through logs or metrics


#