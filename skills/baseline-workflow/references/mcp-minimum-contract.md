# Minimum MCP contract for this simplified skill

This version of the skill intentionally avoids database lookup and broad data discovery. It only needs enough MCP support to execute one workspace run cleanly.

## Required tools

- `find_model`
- `list_models` as a fallback
- `get_model_info`
- `get_model_parameters`
- `run_simulation`
- `compute_metrics`
- `load_hydro_csv`
- `list_workspace_files`

## Required behavior

### 1. random parameter support in `run_simulation`

When parameters are omitted, MCP should generate a random valid parameter set and return:
- parameter values used
- parameter seed
- run id
- warnings

### 2. deterministic export to `./result`

`run_simulation` should support `output_dir`, default it to `./result`, and write:
- simulation CSV under `./result/simulation/`
- machine-readable metadata JSON next to the CSV
- a short markdown summary under `./result/`

### 3. metric computation by result path or data handle

`compute_metrics` should accept either:
- a simulation output path plus an observed discharge path
- or data handles for either side

It should return at least:
- NSE
- KGE
- RMSE
- sample size used
- warnings
- metrics artifact path under `./result/metrics/`

### 4. workspace inspection

`list_workspace_files` should allow lightweight inspection of `./data` and `./result` without reading file contents.

## Notes

- `run_simulation` may infer common forcing column aliases such as `prcp(mm/day)` -> `P`
- `run_simulation` still uses file paths, not data handles
- `load_hydro_csv` remains the preferred entry point for handle-based calibration workflows
