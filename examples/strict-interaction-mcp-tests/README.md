# Strict Interaction MCP Tests

This directory stores English-only, one-prompt-per-file test cards for stricter HydroModelMCP evaluation.

The cards are intentionally interaction-heavy. They check whether an agent can:

- follow the documented HydroModelMCP surface
- select the right skill for the job
- keep readiness checks explicit
- stop on blocking issues instead of bluffing
- avoid invented tool fields or undocumented recovery paths
- preserve auditability through storage, validation, and cache boundaries

## Review Standard

Use the same severity language as the harness documents:

- `P0`: blocking protocol or orchestration failure
- `P1`: major correctness issue
- `P2`: non-blocking weakness or incomplete reporting

Suggested overall verdict:

- `passed`: no `P0` and no `P1`
- `partial`: no `P0`, but at least one `P1`
- `failed`: any `P0`

## Suite Index

| ID | File | Main Focus | Recommended Skill |
| --- | --- | --- | --- |
| S01 | `S01-fixed-smax-request-safety-gate.md` | Caravan basin capability realism for unsupported parameter locking | `hydromodel-mcp-strict-tests` |
| S02 | `S02-caravan-80-20-calibration-validation.md` | Caravan basin calibration and holdout evaluation | `hydromodel-mcp-strict-tests` |
| S03 | `S03-caravan-sensitivity-storage-audit.md` | Caravan basin sensitivity-first workflow and storage read-back | `hydromodel-mcp-strict-tests` |
| S04 | `S04-explicit-parameter-validation-no-fallback.md` | Caravan basin clean `run_validation` contract with explicit parameters | `hydromodel-mcp-strict-tests` |
| S05 | `S05-calibration-readiness-fail-fast.md` | stop rules when calibration inputs are not ready | `hydromodel-mcp-strict-tests` |
| S06 | `S06-strict-inference-manual-mapping-recovery.md` | strict inference rejection and mapping recovery | `hydromodel-mcp-strict-tests` |
| S07 | `S07-session-cache-boundary-validation.md` | Caravan basin cache cleanup and no hidden parameter reuse | `hydromodel-mcp-strict-tests` |
| S08 | `S08-cross-model-fair-budget-benchmark.md` | Caravan basin same-budget comparison across models | `hydromodel-mcp-strict-tests` |

## Reviewer Checklist

- Check `expected_calls` against the actual tool trace.
- Mark the first mismatch precisely.
- Penalize any skipped readiness check before simulation, calibration, validation, or metrics.
- Penalize any invented request field that is not part of the documented public surface.
- Penalize any success claim that cannot be supported by the actual MCP outputs.
- On successful paths, expect one `clear_session_cache` call near the end unless the card explicitly tests cache behavior.

## Notes

- These cards are intentionally stricter than the baseline dialogue script.
- Some cards are positive workflow tests; some are negative or safety-gate tests.
- All large-sample basin cards in this suite use `source_type=caravan` only. They must pass explicit `dataset_name` plus `gauge_id`/`gage_id` and must not fall back to any retired legacy basin route or local CSV guesses.
- The suite intentionally spans multiple Caravan basins instead of repeating only `01013500`.
- If a required fixture is missing, mark the case as `skipped_missing_fixture` instead of `failed`.
