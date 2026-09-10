# ADR 0001: Fail closed when required analysis fields are missing

Status: Accepted for this patch

## Context

At baseline `67cc25b7c52acf69bcdea00b094962e06760efe5`, an `exists` condition on an absent field succeeds. `extract_value` substitutes an empty object for a missing key, and the condition accepts anything other than null. A required policy field can therefore be absent while its existence condition passes.

## Decision

Route only `exists` conditions through a dedicated presence resolver. Every object key must actually exist. Null, invalid paths, scalar traversal and absent array elements deny. Dotted paths, array indexes including existing negative indexes, and terminal array wildcards are supported. A terminal wildcard tests whether the array exists, including an empty array; it does not assert that every item has a field. Wildcards followed by additional path components deny because that interpretation is ambiguous.

Existing false, zero, empty strings, empty objects and empty arrays still count as present. Existence is not truthiness or policy approval. Other comparison operators retain their existing implementation and semantics.

## Verification

Run `python -m unittest discover -s tests -v`. Ten tests cover missing and nested fields, null, falsy values, array traversal, malformed paths and representative unchanged comparisons. Tests load the actual pure functions from `main.py` through Python AST, so they require neither legacy web dependencies nor provider credentials. This is a unit regression suite, not HTTP or provider integration validation.

The original missing field case returned true before this patch and false after it. A latency benchmark would not establish a meaningful improvement for this correctness change and is intentionally omitted.

## Limits and rollback

This patch does not address unrestricted regular expressions in other conditions, provider call budgets, dependency qualification or model output schema enforcement. Existing clients relying on absent fields or malformed paths passing `exists` will now receive a failed condition. Rollback is a Git revert, but restores the fail open behavior.
