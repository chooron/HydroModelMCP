# HydroModelMCP Strict Tests

Use this skill to execute and review the strict MCP scenario suite under `examples/strict-interaction-mcp-tests/`.

## When to use

- The user wants to test HydroModelMCP behavior against the strict scenario cards.
- The user wants one scenario, multiple named scenarios, or the full strict suite.
- The user wants PASS/PARTIAL/FAIL judgments grounded in actual MCP traces.

## Authoritative inputs

- Read `examples/strict-interaction-mcp-tests/README.md` first.
- Treat each scenario markdown file in that directory as the source of truth for:
  - strict user prompt
  - expected MCP call sequence
  - correctness checks
  - failure signals
  - verdict standard

## Suite files

- `S01-fixed-smax-request-safety-gate.md`
- `S02-caravan-80-20-calibration-validation.md`
- `S03-caravan-sensitivity-storage-audit.md`
- `S04-explicit-parameter-validation-no-fallback.md`
- `S05-calibration-readiness-fail-fast.md`
- `S06-strict-inference-manual-mapping-recovery.md`
- `S07-session-cache-boundary-validation.md`
- `S08-cross-model-fair-budget-benchmark.md`

## Execution rules

1. Read the suite README before running any scenario.
2. Read the selected scenario file in full before taking action.
3. Follow the scenario's strict prompt and expected call order exactly.
4. Use only documented HydroModelMCP tools and resources.
5. Do not read raw dataset files into the conversation unless the scenario explicitly allows it.
6. For Caravan basin cases, use only `source_type=caravan` with explicit `dataset_name` and `gauge_id`/`gage_id`.
7. If a scenario fixture is missing and the scenario defines `skipped_missing_fixture`, mark it as skipped instead of failed.
8. Stop immediately on blocking failures when the scenario requires fail-fast behavior.
9. On successful paths, expect `clear_session_cache` near the end unless the scenario says otherwise.

## Review method

For every scenario, produce:

- scenario id
- executed or skipped status
- actual MCP call sequence
- first mismatch against expected sequence, if any
- findings labeled `P0`, `P1`, or `P2`
- final verdict: `passed`, `partial`, `failed`, or `skipped_missing_fixture`
- short justification tied to observed MCP outputs

## Full-suite behavior

- If the user asks for the full suite, run all scenario cards in filename order.
- Continue after individual scenario failures unless the user explicitly asks to stop on first failure.
- End with a suite summary table containing scenario id, verdict, and first blocking issue.

## Output contract

- Findings first, ordered by severity.
- Then per-scenario verdicts.
- Then a short overall suite summary.
- Never claim success without matching tool evidence.
