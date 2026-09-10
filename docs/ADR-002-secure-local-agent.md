# ADR 002: bounded local Guardrail v2

Status: accepted for the alpha implementation.

## Decision

Use the smallest testable local trust boundary. Expose fixed operations through the official MCP SDK 2.0.0 and a matching CLI. External messages are data. No remote shell, caller filesystem path, deployment, autonomous promotion or federation writes are exposed.

## Threat model

Untrusted JSON can attempt malformed policy or signature input, expensive patterns, oversized bodies, replay, or command injection. Domain validation fails closed. MCP input is bounded to 32 KiB application payload and 64 KiB transport frames. Child processes use a fixed command list, stripped environment, one concurrent slot, 30 second deadline and 64 KiB combined output. Validation requires local operator opt in. Unsigned receipt hashes are diagnostics, not verified execution attestations.

## Domain boundaries

The local policy engine is deterministic validation, not a model safety guarantee. Provider routes require OPENAI_API_KEY and AUTH_TOKEN of at least 32 characters. Provider output remains untrusted data. No live paid provider test or production load qualification is claimed.

## Evidence and acceptance

Run repository tests, runtime SDK stdio tests, benchmark and dependency audit through CI. Benchmarks describe fixture latency on the current host, not claims of SOTA or production throughput. Adversarial tests cover the concrete failures fixed here. External credentials and deployed federation membership are separately configured by an operator.

## Rollback

Revert the change commit and restore a backup before changing database formats. Do not restore revoked identities from an older database snapshot without replaying the revocation log. Do not reenable insecure historical prototypes during rollback.
