# Workspace test workflow

This reference describes the baseline MCP test flow for one interactive workspace.

## Intended sequence

1. Ask for the target model only when the user has not named one and the requested test requires a model.
2. Resolve the model with MCP tools.
3. Resolve local input paths from `./data` without opening file contents.
4. Inspect the input source with `inspect_hydro_data` before downstream execution.
5. Execute the requested MCP path, usually one simulation-path baseline test.
6. Record every MCP interaction in an execution trace.
7. Stop immediately on the first blocking failure.
8. Return the trace, final status, and the exact stopping reason.

## Data resolution order

Resolve forcing data in this order:
1. explicit user path
2. `./data/forcing.csv`
3. `./data/meteo.csv`
4. `./data/meteorology.csv`
5. `./data/input.csv`
6. `list_workspace_files(directory="./data")` if confirmation is needed

Resolve observation data in this order:
1. explicit user path
2. `./data/obs.csv`
3. `./data/observed.csv`
4. `./data/discharge.csv`
5. `./data/streamflow.csv`

If several candidates exist and none is clearly preferred, do not guess. Stop and report the ambiguity.

## Testing default

When the user does not supply parameters:
- allow MCP to randomize parameters
- label the run as exploratory
- avoid strong forecast-confidence claims
- record returned `params_used` and `params_seed` when available

## Input readiness rule

Use `inspect_hydro_data` before simulation, calibration, validation, or metric computation when the input source is a workspace file.

Stop immediately when:
- the selected model has missing required inputs after alias and mapping resolution
- observed runoff is missing for a calibration or validation workflow

## Canonical interface rule

- For simulation, use `run_simulation` with unified v2 request fields (`model + inputs`, optional `output/options`).
- For single-objective calibration, call `calibrate_model` with unified v2 fields (`model + inputs`) and include both forcing and observation sources.
- For validation, call `run_validation` with v2 fields and include `inputs.parameters` when possible.
- Do not use legacy flat simulation fields.

### Parameter sourcing for simulation and validation

- Prefer explicit `inputs.parameters` from user dialogue when available (full or partial object).
- Supported parameter source descriptors include `json`, `csv`, `data_handle`, and `calibration_result`.
- If validation omits `inputs.parameters`, same-session fallback from the latest calibration/simulation may be used.
- For simulation only, unresolved partial parameters are auto-filled with random valid values and reported in warnings.
- Validation periods support either `start_index/end_index` or `start/end`; when forcing has no date column and `start/end` is used for both periods, MCP falls back to a synthetic timeline and emits a warning.

### Runtime and algorithm alias matching

- Runtime solver/interpolation values may appear in Chinese or English; MCP normalizes to canonical enums.
- Examples:
  - solver: `ODE`, `DISCRETE`, `连续`, `离散`
  - interpolation: `LINEAR`, `CONSTANT`, `DIRECT`, `线性`, `常量`, `直接`
- Calibration algorithm aliases are also normalized, e.g. `粒子群优化` -> `PSO`, `差分进化` -> `DE`.

## Error rule

Stop immediately when any MCP call fails or any required return field is missing.

Do not continue to downstream steps only to produce a nicer final answer.

## Output expectations

Successful tests may report:
- simulation CSV path under `./result/simulation/`
- simulation metadata JSON path under `./result/simulation/`
- run summary markdown path under `./result/`
- metrics JSON path under `./result/metrics/`

Failed tests should report:
- the last successful step
- the failing step
- the exact MCP interface that failed
- which later steps were intentionally not attempted
