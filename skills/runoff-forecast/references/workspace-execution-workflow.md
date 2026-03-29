# Workspace execution workflow

This skill is optimized for one interactive workspace.

## Intended sequence

1. Ask for the target model only when the user has not named one.
2. Resolve the model with MCP.
3. Use the current project's `./data` folder as the default input area.
4. Run a single simulation with random exploratory parameters when no parameters are supplied.
5. Evaluate the run if observed discharge is available.
6. Write outputs into `./result`.
7. Return a short summary and one next action.

## Data resolution order

Resolve forcing data in this order:
1. explicit user path
2. `./data/forcing.csv`
3. `./data/meteo.csv`
4. `./data/meteorology.csv`
5. `./data/input.csv`
6. `list_workspace_files(directory="./data")` if you need to confirm candidates

Resolve observation data in this order:
1. explicit user path
2. `./data/obs.csv`
3. `./data/observed.csv`
4. `./data/discharge.csv`
5. `./data/streamflow.csv`

If several candidates exist and none is clearly preferred, do not guess. Return a minimal handoff request for the data-discovery skill.

## Exploratory default

When the user does not supply parameters:
- allow MCP to randomize parameters
- label the run as exploratory
- avoid strong forecast-confidence claims
- include returned `params_used` and `params_seed`

## Evaluation rule

Use `compute_metrics` only when observed discharge exists.
Otherwise provide:
- runtime status
- warnings
- compact hydrological behavior summary

## Output layout

Current MCP behavior:
- simulation CSV: `./result/simulation/<base>_result_<timestamp>.csv`
- simulation metadata JSON: `./result/simulation/<base>_result_<timestamp>.metadata.json`
- run summary markdown: `./result/run_summary_<timestamp>.md`
- metrics JSON: `./result/metrics/<base>_metrics_<timestamp>.json`
