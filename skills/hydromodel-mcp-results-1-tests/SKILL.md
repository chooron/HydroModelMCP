# HydroModelMCP Results 1 Tests

Use this skill to design, execute, or review the Results 1 scenario suite under `examples/paper-results-1-scenarios/`.

## When to Use

- The user wants to turn a Results 1 scenario card into an executable test script.
- The user wants to run one Results 1 scenario against the live HydroModelMCP server.
- The user wants to run the full Results 1 suite in order.
- The user wants PASS/PARTIAL/FAIL judgments grounded in actual HydroModelMCP traces and artifacts.

## Modes

- `design`: expand a selected scenario card into an executable test case, ordered pseudocode, fixture checklist, and reporting skeleton.
- `execute`: run the selected scenario against the live server and collect auditable outputs.
- `review`: inspect a completed execution trace, logs, or outputs against the scenario card and issue findings plus a verdict.

If the user does not specify a mode, choose `execute` when a live server is available and the request asks for testing, otherwise choose `design`.

## Authoritative Inputs

- Read `examples/paper-results-1-scenarios/README.md` first.
- Treat each markdown file in `examples/paper-results-1-scenarios/` as the source of truth for that scenario.
- Use `examples/paper-results-r1-basic-scenarios.md` only as the consolidated reference pack.
- Use the project `README.md` as the authority for documented HydroModelMCP tool surfaces and contract details.

## Suite Files

- `R1-S1-model-catalog-coverage-and-metadata-completeness.md`
- `R1-S2-minimal-executability-benchmark.md`
- `R1-S3-single-basin-standardized-cross-model-simulation-benchmark.md`
- `R1-S4-simple-single-objective-calibration-validation-closed-loop.md`

## Execution Rules

1. Read the suite README before doing anything else.
2. Read the selected scenario file in full before taking action.
3. Keep Results 1 bounded to the scenario card being handled.
4. Use only documented HydroModelMCP tools, fields, and request shapes.
5. Use the unified v2 contract for simulation, validation, and calibration.
6. Run readiness checks through `inspect_hydro_data` before simulation or calibration branches.
7. For official basin routes, use only `source_type=caravan` with explicit `dataset_name` and `gauge_id` or `gage_id`.
8. Never use `load_camels_data` or `source_type=camels`.
9. Do not introduce skill-comparison, transport-comparison, backend-comparison, self-repair, or multi-objective branches.
10. If a required declared fixture is missing, mark the scenario `skipped_missing_fixture` instead of improvising a replacement.
11. If a branch is unsupported for a model, log `skipped` or `excluded` with tool-grounded reasons.
12. Preserve raw tool outputs, warnings, inference reports, metrics, and artifact references needed for later paper tables and figures.

## Design Mode Rules

When the task is to design a test script from a scenario card, output exactly these sections:

1. Executable test case
2. Ordered pseudocode
3. Fixture checklist
4. Reporting table skeleton

Additional design requirements:

- Preserve the scenario objective and exclusions exactly.
- Keep the required tool workflow in numbered order.
- Make pass, partial, fail, skip, and exclusion branches explicit.
- Keep the script usable by a weaker LLM client that should not improvise.

## Execute Mode Rules

When the task is to run a scenario:

1. Build a scenario manifest first.
2. Record declared fixtures, fixed metric sets, split rules, budgets, and model subsets before the first mutable workflow step.
3. Follow the required tool workflow exactly.
4. Stop on blocking failures when the scenario demands it.
5. Produce the required logs, tables, figures, and artifact list for the scenario.
6. Never claim success without matching tool evidence.

## Review Method

For every reviewed scenario, produce:

- scenario id
- scenario name
- mode used
- executed or skipped status
- fixture status
- actual HydroModelMCP call sequence
- first mismatch against the required workflow, if any
- findings labeled `P0`, `P1`, or `P2`
- generated outputs and missing outputs
- final verdict: `passed`, `partial`, `failed`, or `skipped_missing_fixture`
- short justification tied to observed tool outputs

## Full-Suite Behavior

- If the user asks for the full Results 1 suite, run the cards in this order: `R1-S1`, `R1-S2`, `R1-S3`, `R1-S4`.
- Continue after an individual scenario failure unless the user explicitly requests fail-fast execution.
- End with a suite summary table containing scenario id, verdict, first blocking issue, and produced paper artifacts.

## Output Contract

- Findings first, ordered by severity.
- Then per-scenario verdicts.
- Then a short suite or task summary.
- Never claim paper-ready evidence without logs, metrics, or artifacts to support it.
