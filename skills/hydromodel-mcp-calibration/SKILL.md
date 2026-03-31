---
name: hydromodel-mcp-calibration
description: run a full hydrological calibration workflow through HydroModelMCP. use when Codex should execute or design a complete calibration process with data readiness checks, optional sensitivity screening, objective selection, calibration runs, diagnostics, and optionally validation. do not use this skill for quick low-cost parameter search only or for standalone sensitivity screening.
---

# HydroModelMCP Calibration Workflow

Use this skill for a full calibration workflow, not just a quick optimization run.

## Scope

This skill is for:
- complete calibration workflows
- calibration design and method choices
- identifiability and convergence interpretation
- objective-function and algorithm selection
- validation-oriented calibration review

Do not use this skill when:
- the user only wants a quick low-cost parameter search
- the user only wants parameter sensitivity screening

Hand off instead:
- use `hydromodel-mcp-optimization` for low-cost single-objective parameter search
- use `hydromodel-mcp-sensitivity` for standalone parameter screening and ranking

## Start

- Confirm whether the user wants workflow design, full execution, diagnostics, or validation review.
- Read [references/project-tool-map.md](references/project-tool-map.md) before naming tools or resources.
- Read [references/paper-ten-strategies.md](references/paper-ten-strategies.md) when the request is about calibration strategy, experiment design, or best practice.
- Read [references/service-gaps.md](references/service-gaps.md) before promising functionality that depends on partially exposed server features.

## Canonical workflow

Use this order unless the user explicitly requests a narrower path.

1. Discover the model surface.

- `list_models`
- `find_model`
- `get_model_info`
- `get_model_parameters`
- `get_model_variables`

2. Confirm data readiness.

- `inspect_hydro_data`
- stop if the selected model has missing required inputs
- stop if observed runoff is unavailable for calibration or validation

3. Load data once and reuse handles.

- `load_hydro_csv`
- prefer `data_handle` reuse after loading

4. Prepare the calibration design.

- `analyze_distribution_from_handle`
- `run_sensitivity` or `sensitivity_analysis` when screening is needed
- `configure_objectives`
- `split_data` when validation is required
- `init_calibration_setup` when the client needs a packaged recommendation

5. Execute calibration.

- `calibrate_model` for single-objective calibration
- `calibrate_multiobjective` only when the user explicitly needs a Pareto tradeoff

6. Check the result.

- `diagnose_calibration`
- `compute_diagnostics_full`
- `run_validation` when validation data were prepared (prefer explicit `inputs.parameters`; same-session fallback is acceptable only when clearly reported)
- `compute_metrics` when validation or comparison metrics are needed

7. Finish cleanly.

- `clear_session_cache`

## Calibration defaults

Use stronger defaults than the low-cost optimization skill.

- prefer `n_trials >= 3` when the user asks about robustness, identifiability, or convergence
- use a larger `maxiters` than the optimization skill unless the user explicitly asks for a cheap exploratory run
- prefer `KGE` for general fit
- recommend `LogKGE` or `LogNSE` only when low-flow behavior is the real target
- use multi-objective calibration only when the user truly needs tradeoffs, not merely multiple reported metrics

## Decision rules

- screen insensitive parameters before broad calibration when identifiability is uncertain
- revise parameter ranges after diagnostics instead of treating defaults as final
- separate design choices from execution choices in the response
- distinguish exploratory optimization from calibration-grade evidence

## Current limits

- treat criteria integration carefully; only the currently validated objective set should be promised
- treat parameter-constraint workflows as a current gap
- treat multi-objective diagnostics as partial because there is still no dedicated MCP tool for `diagnose_multiobjective`
- treat validation, storage, and HTTP transport behavior as partially tested surfaces

## Response structure

# Calibration summary

## Calibration target
- selected model
- why it matched
- required inputs

## Data readiness
- source path or handle
- model-required input coverage
- observed runoff availability
- blocking issues or `none`

## Workflow design
- sensitivity screening used or skipped
- objective choice
- algorithm choice
- iteration and trial budget
- validation split used or skipped

## Calibration result
- best objective
- best parameters
- runtime summary

## Diagnostics
- convergence status
- identifiability warnings
- range or algorithm adjustment suggestions

## Validation
- validation used or skipped
- metrics when available

## Outputs observed
- handles, result ids, or artifacts if any

## Next action
- one small next step only

## Test samples

Read `references/test-samples.md` for concrete prompts and expected MCP tool sequences.
