---
name: hydromodel-mcp-optimization
description: run a low-cost hydrological parameter optimization workflow through HydroModelMCP. use when a local data csv contains meteorological inputs and observed runoff, the user wants parameter optimization rather than one-off simulation, the workflow should stop immediately if model-required inputs or observed runoff are missing, and the default optimization budget should stay small unless the user explicitly asks for a broader search.
---

# HydroModelMCP Optimization Workflow

Use this skill for practical, low-cost parameter optimization in the current workspace.

## Fixed operating rules

- Do not browse the web.
- Do not search external databases.
- Treat the current workspace as the execution boundary.
- Default input directory: `./data`
- Default output directory: `./result`
- Do not read raw CSV or JSON contents into the conversation.
- Use `inspect_hydro_data` before loading calibration data.
- Stop immediately if the selected model has missing required inputs.
- Stop immediately if observed runoff is unavailable for calibration.
- Keep the default optimization cost low unless the user explicitly asks for a wider search.

## Canonical tool path

1. `find_model`
2. `get_model_info`
3. `get_model_parameters`
4. `inspect_hydro_data`
5. `load_hydro_csv`
6. `analyze_distribution_from_handle`
7. `configure_objectives`
8. `calibrate_model`
9. `diagnose_calibration`
10. `clear_session_cache`

## Default optimization settings

- algorithm: `BBO`
- objective: `KGE`
- maxiters: `80`
- n_trials: `1`
- log_transform: false unless low-flow behavior clearly justifies it

Use a larger budget only when the user explicitly asks for more exhaustive optimization.

## Data and readiness rules

- Resolve local CSV paths from `./data`.
- Prefer explicit user paths when provided.
- Use `inspect_hydro_data` with:
  - `model`
  - `intended_use="calibration"`
- Treat generic `P/T/Ep` detection as informational.
- Treat model-required inputs and observed runoff as blocking checks.

## Calibration rules

- Use `calibrate_model` through the unified v2 request contract (`model + inputs`).
- Include both forcing and observation source descriptors in `inputs` for calibration.
- Prefer single-objective calibration unless the user explicitly asks for Pareto tradeoffs.
- Prefer `KGE` for general fit.
- Consider `LogKGE` only when low-flow performance is the real target.

## Response structure

# Optimization summary

## Model
- selected model
- why it matched
- required inputs

## Data readiness
- source path
- model-required input coverage
- observed runoff availability
- blocking issues or `none`

## Optimization setup
- algorithm
- objective
- maxiters
- n_trials
- whether the run used a low-cost default budget

## Results
- best objective
- best parameters
- runtime summary

## Diagnostics
- convergence status
- important warnings
- recommended next adjustment

## Outputs observed
- handles, result ids, or artifacts if any

## Next action
- one small next step only

## Test samples

Read `references/test-samples.md` when you need concrete prompt examples and expected MCP tool sequences for this skill.
