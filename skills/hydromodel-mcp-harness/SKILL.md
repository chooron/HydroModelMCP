---
name: hydromodel-mcp-harness
description: run a fail-fast, trace-first automated MCP harness for HydroModelMCP. use when the user wants to evaluate MCP completeness, protocol correctness, orchestration reliability, regression stability, or obvious contract vulnerabilities. prioritize finding MCP issues over producing hydrological outputs.
---

# HydroModelMCP Harness Skill

Use this skill to run a high-automation MCP test harness.

Primary goal: detect MCP problems early, stop on blocking failures, and return precise diagnostics.

## Scope

Use this skill when the user asks to:

- validate MCP completeness
- check for obvious protocol/contract vulnerabilities
- run regression-style tool orchestration tests
- verify fail-fast behavior and stopping rules

Do not use this skill when the user only wants one hydrology result.

## Operating principle

- Test-first, not result-first.
- Fail-fast on MCP blocking errors.
- Never hide contract mismatch behind fallback behavior.
- Keep a full execution trace.

## Mandatory rules

- Do not browse the web.
- Treat current workspace as the execution boundary.
- Do not read raw CSV/JSON artifact content into the conversation.
- Use MCP tools/resources/templates/prompts as the primary test surface.
- Before simulation/calibration/validation/metrics on workspace files, call `inspect_hydro_data`.
- If `inspect_hydro_data` reports blocking issues, stop downstream calls immediately.
- If any tool returns error or contract mismatch, stop immediately in `fail_fast=true` mode.
- On successful completion, call `clear_session_cache` once.

## Harness modes

- `smoke`: minimal quick checks for core availability.
- `standard`: default regression mode.
- `full`: extended coverage including strict negative tests and stability loops.

Default: `mode=standard`, `fail_fast=true`.

## Canonical protocol checks (must enforce)

- `run_simulation` must use v2 request shape: `model + inputs` (optional `output/options`).
- `calibrate_model` must use v2 request shape: `model + inputs`.
- `run_validation` should include `inputs.parameters` when possible; omission is only acceptable when same-session parameter fallback exists and is explicitly reported.
- Legacy flat payloads (for example top-level `source_type/path`) should be rejected where no longer supported.

## Suggested case matrix

Run these in order unless user requests a subset:

1. model discovery surface (`find_model`, fallback `list_models`, `get_model_info`, `get_model_parameters`)
2. workspace file listing (`list_workspace_files`)
3. simulation readiness (`inspect_hydro_data`, intended_use=`simulation`)
4. v2 simulation to CSV (`run_simulation`)
5. metrics flow (`compute_metrics`) when observed runoff exists
6. simulation redis output (`run_simulation` with `output.result_source_type=redis`)
7. calibration readiness stop rule (`inspect_hydro_data`, intended_use=`calibration`)
8. v2 low-cost calibration (`configure_objectives`, `calibrate_model`, `diagnose_calibration`)
9. v2 validation (`run_validation`, with explicit `inputs.parameters` or documented same-session fallback)
10. ensemble path (`run_ensemble_parameters`)
11. resources/templates/prompts surfaces (`resources/templates/list`, `resources/read`, `prompts/list`)
12. negative legacy simulation payload (expect reject)
13. negative legacy calibration payload (expect reject)
14. strict inference negative (`options.strict_infer=true` on ambiguous input when fixture exists)
15. cache cleanup (`clear_session_cache`)

## Deviation and severity policy

Classify findings:

- `P0`: blocking protocol/runtime failure (contract break, service unavailable, fail-fast violation)
- `P1`: major correctness issue (wrong sequence, missing required output fields, cleanup omission)
- `P2`: warning-level issue (non-blocking inconsistency or weak diagnostics)

Gate:

- any `P0` => overall `failed`
- else if any `P1` => overall `partial`
- else => overall `passed`

## Required response structure

Always respond in this structure:

# MCP harness report

## Harness config
- mode
- fail_fast
- workspace

## Case results
For each case include:
- case id and name
- expected calls
- actual calls
- status (`passed`/`failed`/`skipped`)
- deviation list
- severity (`none`/`P2`/`P1`/`P0`)

## Blocking failure
- present or `none`
- failing step
- exact MCP interface
- minimal error summary
- downstream steps not attempted

## Overall status
- overall status
- counts: passed/failed/skipped
- counts: p0/p1/p2

## Next action
- one smallest fix-first action only

## JSON report
- append a machine-readable JSON block that mirrors the same results

## Fixtures and defaults

- default model candidate: `exphydro`
- default main fixture: `./data/03604000.csv`
- default output dir: `./result`
- if optional negative fixtures are missing, mark relevant cases as `skipped_missing_fixture`

## References

- `docs/mcp-llm-harness.md`
- `docs/test-dialogue.md`
- `skills/baseline-workflow/references/actual-tool-contracts.md`
- `skills/hydromodel-mcp-harness/references/calibration-stage2-scenarios.md`
