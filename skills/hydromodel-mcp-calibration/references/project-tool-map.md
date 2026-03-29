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
- `hydro://guides/algorithms`
- `hydro://guides/objectives`
- `hydro://meta/resource-templates`
- `hydro://calibration/results`
- Exact per-model resources:
  - `hydro://models/<model_name>/info`
  - `hydro://models/<model_name>/parameters`
  - `hydro://models/<model_name>/variables`

## Prompt exposed by the repository

- `hydrology_expert_review`

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
- The repo advertises more result-template URIs than it can currently read back. Treat stored-result workflows cautiously.
