# Paper Results 1 Scenarios

This directory stores the standalone Results 1 scenario cards for HydroModelMCP.

These files are the working source for designing and executing the four paper-facing Results 1 tests:

- unified discovery coverage
- minimal executability
- single-basin standardized comparison
- simple single-objective calibration-validation closure

The consolidated pack remains available at `examples/paper-results-r1-basic-scenarios.md`.
This directory splits that pack into one scenario per file so another LLM or harness can expand, execute, and review each case independently.

## Global Rules

1. Use only documented HydroModelMCP surfaces from the current README.
2. Keep Results 1 limited to the four scenarios in this directory.
3. Use the unified v2 request contract for simulation, validation, and calibration paths.
4. Use `inspect_hydro_data` before simulation or calibration branches.
5. Official basin routes must use `source_type=caravan` with explicit `dataset_name` and `gauge_id` or `gage_id`.
6. Do not use `load_camels_data` or `source_type=camels`.
7. Do not turn these scenarios into skill-comparison, transport-comparison, backend-comparison, self-repair, or multi-objective experiments.
8. Preserve warnings, inference reports, metrics, and artifact references as evidence.
9. If a fixture is missing, mark the case as `skipped_missing_fixture` instead of inventing a replacement.
10. If a model branch is unsupported, log it as `skipped` or `excluded` with tool-grounded reasons.

## Review Standard

Use the following severity labels when reviewing scenario execution:

- `P0`: blocking protocol, workflow, or evidence failure
- `P1`: major correctness or fairness issue
- `P2`: non-blocking weakness, incomplete packaging, or weak reporting

Suggested overall verdict:

- `passed`: no `P0` and no `P1`
- `partial`: no `P0`, but at least one `P1`
- `failed`: any `P0`
- `skipped_missing_fixture`: required declared fixture is unavailable

## Scenario Index

| ID | File | Main Focus | Recommended Skill |
| --- | --- | --- | --- |
| `R1-S1` | `R1-S1-model-catalog-coverage-and-metadata-completeness.md` | Unified discovery layer and metadata completeness | `hydromodel-mcp-results-1-tests` |
| `R1-S2` | `R1-S2-minimal-executability-benchmark.md` | Readiness-gated minimal execution | `hydromodel-mcp-results-1-tests` |
| `R1-S3` | `R1-S3-single-basin-standardized-cross-model-simulation-benchmark.md` | One-basin shared-metric comparison | `hydromodel-mcp-results-1-tests` |
| `R1-S4` | `R1-S4-simple-single-objective-calibration-validation-closed-loop.md` | Bounded calibration-validation workflow | `hydromodel-mcp-results-1-tests` |

## Execution Order

Run the scenarios in this order when building the paper Results 1 section:

1. `R1-S1`
2. `R1-S2`
3. `R1-S3`
4. `R1-S4`

## Expected Reviewer Output

For each executed scenario, report:

- scenario id
- scenario name
- fixture status
- actual tool sequence
- first mismatch against the required workflow, if any
- findings labeled `P0`, `P1`, or `P2`
- generated tables, figures, logs, and artifacts
- final verdict
- short justification tied to observed tool outputs

## Low-Tier Executor Reminder

- Read this README first.
- Read the selected scenario file in full before acting.
- Follow the numbered tool workflow exactly.
- Do not hallucinate unsupported branches.
- Keep the shortest valid tool chain that satisfies the scenario card.
