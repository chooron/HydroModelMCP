---
name: hydromodel-mcp-sensitivity
description: run a parameter sensitivity or screening workflow through HydroModelMCP. use when Codex should identify influential parameters, rank sensitivity, screen insensitive parameters before calibration, or explain identifiability risk. do not use this skill for full calibration, validation, or low-cost parameter optimization.
---

# HydroModelMCP Sensitivity Workflow

Use this skill for standalone sensitivity analysis and parameter screening.

## Scope

This skill is for:
- parameter sensitivity screening
- parameter ranking and screening before calibration
- identifiability-oriented pre-analysis
- deciding which parameters may be fixed before calibration

Do not use this skill when:
- the user wants a full calibration workflow
- the user wants validation metrics
- the user wants a quick low-cost parameter optimization run

Hand off instead:
- use `hydromodel-mcp-calibration` for full calibration and validation workflows
- use `hydromodel-mcp-optimization` for cheap single-objective parameter search

## Canonical workflow

1. `find_model`
2. `get_model_info`
3. `get_model_parameters`
4. `inspect_hydro_data`
5. `load_hydro_csv`
6. `sensitivity_analysis` or `run_sensitivity`
7. `clear_session_cache`

Optional supporting steps:
- `analyze_distribution_from_handle` when low-flow or log-sensitive behavior matters
- `configure_objectives` when the user wants sensitivity measured against a specific objective

## Default sensitivity settings

- prefer `morris` for low-cost screening
- default `n_samples` around `100` for quick screening
- use `sobol` only when the user explicitly asks for a more rigorous decomposition
- prefer `NSE` or `KGE` as the objective unless the user requests a specific low-flow or balance-oriented metric

## Stopping rules

- stop if the selected model has missing required inputs
- stop if observed runoff is unavailable and the requested sensitivity objective depends on calibration against observations
- stop if the user asks for full calibration or validation and hand off to the appropriate skill

## Response structure

# Sensitivity summary

## Model
- selected model
- why it matched
- required inputs

## Data readiness
- source path or handle
- model-required input coverage
- observed runoff availability
- blocking issues or `none`

## Sensitivity setup
- method
- objective
- sample size
- whether the run used a low-cost default budget

## Results
- sensitive parameters
- insensitive parameters
- ranking or scores summary

## Interpretation
- which parameters should likely remain calibratable
- which parameters may be fixed or deprioritized
- identifiability caveats

## Outputs observed
- handles, result ids, or artifacts if any

## Next action
- one small next step only

## Test samples

Read `references/test-samples.md` for concrete prompts and expected MCP tool sequences.
