## ADDED Requirements

### Requirement: Per-turn tier override honored by all entry points

The `/kani:<tier>` per-turn tier override MUST be honored by the chat completions proxy endpoint, the `/v1/route` debug endpoint, and the `kani route` CLI command. Each entry point MUST call the shared `parse_tier_override` helper and pass the extracted override to `Router.route()`.

#### Scenario: Chat completions proxy honors the override

**Given** a `POST /v1/chat/completions` request with the latest user message starting with `/kani:reasoning`
**When** kani routes and proxies the request
**Then** the routing decision MUST use the `REASONING` tier
**And** the upstream payload MUST NOT contain the `/kani:reasoning` token
**And** the response headers MUST include `X-Kani-Tier: REASONING` and `X-Kani-Signals` containing `tier_override`

#### Scenario: Debug endpoint reflects the override

**Given** a `POST /v1/route` request with the latest user message starting with `/kani:complex`
**When** kani returns the routing decision
**Then** the response JSON MUST include `tier: COMPLEX`
**And** the response MUST include `tier_override: "COMPLEX"` in the payload
**And** kani MUST NOT proxy the request upstream

#### Scenario: CLI route command honors the override

**Given** the `kani route` command is invoked with a prompt starting with `/kani:simple`
**When** kani routes the prompt
**Then** the routing decision output MUST show `tier: SIMPLE`
**And** the scorer MUST NOT be invoked

#### Scenario: Invalid override in proxy falls through gracefully

**Given** a `POST /v1/chat/completions` request with the latest user message starting with `/kani:foo`
**When** kani routes and proxies the request
**Then** kani MUST run the distilled feature classifier normally
**And** the `/kani:foo` token MUST be stripped from the upstream payload
**And** kani MUST NOT return an HTTP error for the invalid tier

#### Scenario: Override token stripped before compaction in proxy

**Given** a `POST /v1/chat/completions` request with the latest user message starting with a `/kani:<tier>` token
**And** context compaction is enabled and triggers on this request
**When** kani compacts the message history
**Then** the token MUST NOT appear in the compacted content sent upstream