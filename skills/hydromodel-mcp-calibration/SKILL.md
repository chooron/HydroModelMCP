---
name: hydromodel-mcp-calibration
description: Operate the HydroModelMCP Julia MCP server for hydrological model discovery, data loading, sensitivity analysis, calibration, validation, and diagnostic workflows. Use when Codex needs to design or execute a hydrological calibration workflow, interpret identifiability or convergence issues, choose objective functions or algorithms, map Juliane Mai's ten-strategy calibration lifecycle onto this project, or orchestrate HydroModelMCP tools for model selection, parameter screening, multi-objective calibration, or calibration review.
---

# HydroModelMCP Calibration Workflow

## Start

- Confirm whether the task is workflow design, tool orchestration, result interpretation, or service-gap review.
- Read [references/project-tool-map.md](references/project-tool-map.md) before naming tools or resources.
- Read [references/paper-ten-strategies.md](references/paper-ten-strategies.md) when the request is about calibration method, experiment design, diagnostics, or "best practice."

## Use the repository in a paper-aligned order

1. Discover the model surface first.

- Call `list_models`, `get_model_info`, `get_model_parameters`, and `get_model_variables` before recommending a model or parameter range.
- Use `hydro://models/catalog` when the client benefits from browseable resource URIs.

2. Load data once and reuse handles when possible.

- Prefer `load_hydro_csv` for local forcing-plus-observation CSV workflows.
- Use `load_camels_data` only when the user already has a CAMELS NPZ artifact.
- Prefer handle-based calls such as `sensitivity_analysis` and `calibrate_model` after loading data, so the client does not keep resending large arrays.

3. Walk the prepare -> execute -> check loop explicitly.

- Prepare with `run_sensitivity` or `sensitivity_analysis`, `analyze_distribution_from_handle`, `split_data`, `configure_objectives`, and `init_calibration_setup`.
- Execute with `generate_samples`, `calibrate_model`, `calibrate_multiobjective`, `run_validation`, and `run_ensemble_parameters`.
- Check with `compute_metrics`, `diagnose_calibration`, and `compute_diagnostics_full`.
- Prefer `n_trials >= 3` whenever the user wants convergence or identifiability judgments.

4. Make calibration decisions explicit instead of implicit.

- Screen insensitive parameters before broad calibration.
- Recommend log-domain metrics only when the observation magnitude ratio or the low-flow objective justifies them.
- Use multi-objective calibration only when the user truly needs a Pareto tradeoff, not merely multiple summary metrics.
- Revise parameter ranges after diagnostics instead of treating defaults as final.

## Respect current project limits

- Treat result-template metadata as aspirational unless a matching read path exists. Read [references/service-gaps.md](references/service-gaps.md) before promising stored-result retrieval.
- Call out parameter-constraint workflows as a current gap. The core code contains delta and pie-share helpers, but the MCP tool layer does not yet expose constraint-aware calibration inputs.
- Call out reproducibility limits. `random_seed` and `sampling_method` appear in the calibration schema, but they are not fully wired through the optimization path.
- Call out HTTP configuration carefully. `start_http.jl` reads `JULIA_HTTP_*`, while `run_http_server` defaults to `MCP_*`.
- Treat validation, simulation, storage, and HTTP behavior as partially tested surfaces.

## Produce useful outputs

- Return a concrete tool sequence, not generic hydrology advice.
- Separate "what the server can do now" from "what should be added next."
- Read [references/service-gaps.md](references/service-gaps.md) and [../../docs/mcp-service-todo.md](../../docs/mcp-service-todo.md) before planning roadmap work.
