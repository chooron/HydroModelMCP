# HydroModelMCP Tool Map

## MCP tools exposed by the repository

### Data loading and preprocessing

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

## Fast workflow recipes

### Basic calibration setup

1. `list_models`
2. `get_model_info`
3. `get_model_parameters`
4. `load_hydro_csv`
5. `analyze_distribution_from_handle`
6. `run_sensitivity` or `sensitivity_analysis`
7. `configure_objectives`
8. `split_data`
9. `calibrate_model`
10. `diagnose_calibration`

### Multi-objective review

1. `load_hydro_csv`
2. `run_sensitivity`
3. `configure_objectives`
4. `calibrate_multiobjective`
5. `compute_diagnostics_full`

### Validation-focused workflow

1. `split_data`
2. `calibrate_model`
3. `run_validation`
4. `compute_metrics`

## Notes that matter in practice

- `load_hydro_csv` is the most practical entry point for local CSV-driven workflows in this repository.
- `sensitivity_analysis` and `calibrate_model` both support `data_handle`, which avoids repeated payload transfer.
- Model templates are discoverability metadata only; use the unified discovery tools for on-demand model details. Stored-result resources may exist for IDs already present at startup, but newly created results still require a server rebuild or restart before they appear as resources.
