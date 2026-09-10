# Repository agent runtime

Install `npm ci --prefix .harness/runtime`. Run `node .harness/runtime/cli.mjs status` or `node .harness/runtime/cli.mjs mcp` for SDK 2 stdio. The CLI also accepts `test`, `benchmark`, and domain JSON on stdin through `evaluate` (guardrail) or `verify` (ANS).

Test and benchmark execution require operator opt in `RUV_ALLOW_VALIDATION=1`. Fixed commands run with a stripped environment, one process at a time, 30 seconds and 64 KiB output. No caller paths, shell commands, publication or automatic promotion. Receipts contain unsigned SHA256 digests; they are process local and are not attestations. MCP exposes project status, validation, benchmark, receipts and a domain tool plus a repository policy resource.

ANS requires operator configured `ANS_TRUSTED_ISSUER_FILE` containing an Ed25519 PUBLIC key. `identity_verify` validates signature, trusted issuer and lifetime only; revocation is explicitly not checked. Use the Registry API for durable revocation and challenge replay prevention. Guardrail requires installed Python dependencies; `RUV_PYTHON` is a trusted operator override.
