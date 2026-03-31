# HydroModelMCP Tool Map

## MCP tools exposed by the repository

### Data loading and preprocessing

- `inspect_hydro_data`
- `load_camels_data`
- `analyze_distribution_from_handle`
- `load_hydro_csv`

### Model discovery

- `list_models`
- `find_model`
- `get_model_info`
- `get_model_variables`
- `get_model_parameters`

### Simulation and validation

- `run_simulation`
- `run_ensemble_parameters`
- `run_validation`

### Calibration workflow

- `compute_metrics`
- `split_data`
- `run_sensitivity`
- `sensitivity_analysis`
- `generate_samples`
- `calibrate_model`
- `calibrate_multiobjective`
- `diagnose_calibration`
- `configure_objectives`
- `init_calibration_setup`
- `compute_diagnostics_full`

## Resources exposed by the repository

- `hydro://models/catalog`
- `hydro://guides/model-discovery`
- `hydro://guides/algorithms`
- `hydro://guides/objectives`
- `hydro://guides/metrics`
- `hydro://guides/data-handles`
- `hydro://guides/runoff-workspace`
- `hydro://guides/result-artifacts`
- `hydro://meta/resource-templates`
- `hydro://calibration/results`
- `hydro://sensitivity/results`
- `hydro://ensemble/results`

### Exact resource families that may exist

- `hydro://calibration/results/<result_id>` for IDs already present in storage
- `hydro://sensitivity/results/<result_id>` for IDs already present in storage
- `hydro://ensemble/results/<result_id>` for IDs already present in storage

## Prompt exposed by the repository

- `hydrology_expert_review`
- `runoff_workspace_workflow`
- `calibration_workflow_plan`
- `hydrology_result_review`

## Recommended skill split

- `hydromodel-mcp-optimization`: quick low-cost parameter search with small budgets
- `hydromodel-mcp-sensitivity`: standalone parameter screening and ranking
- `hydromodel-mcp-calibration`: full calibration, diagnostics, and optional validation

## Fast workflow recipes

### Basic calibration setup

1. `list_models`
2. `get_model_info`
3. `get_model_parameters`
4. `inspect_hydro_data`
5. `load_hydro_csv`
6. `analyze_distribution_from_handle`
7. `run_sensitivity` or `sensitivity_analysis`
8. `configure_objectives`
9. `split_data`
10. `calibrate_model`
11. `diagnose_calibration`

### Low-cost optimization setup

1. `find_model`
2. `get_model_info`
3. `get_model_parameters`
4. `inspect_hydro_data`
5. `load_hydro_csv`
6. `configure_objectives`
7. `calibrate_model`
8. `diagnose_calibration`

### Multi-objective review

1. `inspect_hydro_data`
2. `load_hydro_csv`
3. `run_sensitivity`
4. `configure_objectives`
5. `calibrate_multiobjective`
6. `compute_diagnostics_full`

### Validation-focused workflow

1. `inspect_hydro_data`
2. `load_hydro_csv`
3. `split_data`
4. `calibrate_model`
5. `run_validation`
6. `compute_metrics`

## Notes that matter in practice

- `load_hydro_csv` is the most practical entry point for local CSV-driven workflows in this repository.
- `sensitivity_analysis` supports the `data_handle + model_name` path for handle-centric screening workflows.
- `calibrate_model` now uses the unified v2 request contract (`model + inputs`, with optional `options`).
- `inspect_hydro_data` should be used before calibration to verify both model-required inputs and observed runoff availability.
- Model templates are now discoverable through `resources/templates/list` and can be read dynamically through `resources/read`, but the unified discovery tools are still the preferred path for orchestration and fuzzy model resolution.

## Runtime and algorithm canonical options (CN/EN aliases)

- Solver canonical values: `DISCRETE`, `ODE`, `MUTABLE`, `IMMUTABLE`
  - Common aliases: `discrete` / `离散`, `continuous` / `连续`, `mutable` / `可变`, `immutable` / `不可变`
- Interpolation canonical values: `LINEAR`, `CONSTANT`, `DIRECT`
  - Common aliases: `linear` / `线性`, `constant` / `常量`, `direct` / `直接`
- Single-objective algorithm canonical values: `BBO`, `DE`, `PSO`, `CMAES`, `ECA`
  - Common aliases: `biogeography-based optimization` / `生物地理优化`, `differential evolution` / `差分进化`, `particle swarm optimization` / `粒子群优化`
- Multi-objective algorithm canonical values: `NSGA2`, `NSGA3`
  - Common aliases: `NSGA-II` / `多目标遗传算法2`, `NSGA-III` / `多目标遗传算法3`
