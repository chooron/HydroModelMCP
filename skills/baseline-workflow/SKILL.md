---
name: runoff-forecast
description: interactively execute hydrological runoff simulations inside a project workspace with an mcp service. use when chatgpt should ask for or accept a target model, default to the current project's data folder for inputs, let mcp generate random parameters when none are supplied, run one simulation, optionally compute metrics when observed discharge is available, export outputs and a short summary into the project's result folder, and hand off only missing-data issues to a separate data-discovery skill instead of searching databases or the web.
---

# Runoff Forecast

Use this skill as a workspace executor, not as a data-discovery skill.

## Fixed operating rules

- Do not browse the web.
- Do not search databases or external systems from this skill.
- Assume the current project workspace is the execution boundary.
- Default input directory: `./data`
- Default output directory: `./result`
- Default intermediate cache: transient session cache, preferably Redis-backed when available
- Keep the interaction minimal: ask for the model only if the user did not name one.
- Do not read raw CSV contents into the conversation when MCP can consume file paths directly.
- If parameters are not supplied, allow MCP to generate a random parameter set and label the run as exploratory.
- Do not use model resource URIs such as hydro://models/<model_name>/info, hydro://models/<model_name>/parameters, or hydro://models/<model_name>/variables. Resolve model metadata through tools only.

Read these references when needed:
- `references/workspace-execution-workflow.md`
- `references/mcp-minimum-contract.md`
- `references/actual-tool-contracts.md`

## resources, templates, and prompts in this mcp service

### read-only resources available

- `hydro://models/catalog` for compact model browsing and preferred discovery-tool hints
- `hydro://guides/model-discovery` for model-browsing and tool-vs-resource guidance
- `hydro://guides/algorithms` for calibration algorithm selection guidance
- `hydro://guides/objectives` for objective-function selection guidance
- `hydro://guides/metrics` for metric behavior and direction hints
- `hydro://guides/data-handles` for deciding between file paths and reusable handles
- `hydro://guides/runoff-workspace` for workspace defaults and execution order
- `hydro://guides/result-artifacts` for `./result` layout and stored-result notes
- `hydro://meta/resource-templates` for template metadata and server behavior notes
- `hydro://calibration/results` for stored calibration result identifiers
- `hydro://sensitivity/results` for stored sensitivity result identifiers
- `hydro://ensemble/results` for stored ensemble result identifiers

Resource usage rules for this skill:
- treat resources as optional context, not as the main execution path
- keep model-detail access tool-first: `find_model`, `get_model_info`, `get_model_parameters`, `get_model_variables`
- do not replace model resolution with `resources/list` or `resources/read`

### resource template policy

Declared templates in the current service:
- `hydro://calibration/results/{result_id}`
- `hydro://sensitivity/results/{result_id}`

Execution policy:
- treat template URIs as capability metadata unless a concrete `resources/read` call succeeds
- do not promise historical-result retrieval through template URIs by default
- if a user asks for a historical result, first check IDs from `hydro://calibration/results`, then report the current retrieval limitation when no reliable read path exists

### prompt usage (optional)

Available MCP prompt:
- `hydrology_expert_review` with required `task` and optional `context`

When to use it:
- use only when the user asks for expert critique, method justification, or risk review
- do not block a requested simulation run on prompt invocation

Recommended English prompt payload:

```json
{
  "name": "hydrology_expert_review",
  "arguments": {
    "task": "Review whether this exploratory runoff simulation is decision-ready.",
    "context": "Model: exphydro; forcing: ./data/forcing.csv; observed discharge: unavailable; warnings: alias mapping was applied."
  }
}
```

Expected response focus:
1. key assumptions and uncertainties
2. recommended workflow or tool sequence
3. critical parameter and data checks
4. risks that may invalidate conclusions

## Interaction pattern

1. If the user has not named a model, ask only: which hydrological model should be run?
2. Once a model is known, proceed autonomously with workspace defaults.
3. Ask another question only if MCP cannot resolve a blocking issue from the workspace and default conventions.

## Default conventions

- Project root: current workspace root
- Forcing source: prefer an explicit path; otherwise resolve from `./data`
- Preferred forcing filenames, in order: `forcing.csv`, `meteo.csv`, `meteorology.csv`, `input.csv`
- Optional observed discharge filenames, in order: `obs.csv`, `observed.csv`, `discharge.csv`, `streamflow.csv`
- Parameter source: random exploratory parameters from MCP when the user gives none
- Execution mode: single-run simulation
- Output location: `./result`

If multiple plausible files exist and no safe choice is possible, stop and return a minimal handoff note for the data-discovery skill.

## Core workflow

### 1) normalize the request

Extract these fields when present:
- model name
- explicit forcing file path
- explicit observed discharge path
- optional period, warmup, solver, interpolation, or parameter values

### 2) resolve the model

Use tools in this order:
1. `find_model`
2. `list_models` only if the match is empty or ambiguous
3. `get_model_info`
4. `get_model_parameters`

Do not substitute model-resolution steps with `resources/list` or `resources/read`.

### 3) resolve workspace data

Use this order:
1. explicit user path if given
2. default file conventions in `./data`
3. `list_workspace_files` if you need to confirm candidates in `./data` or `./result`

If no forcing file can be resolved, stop with a short handoff message that states:
- required forcing file is missing from `./data`
- the selected model
- the required input contract from `get_model_info`
- that a data-discovery skill should provide the correct local file path

### 4) run the simulation

Use `run_simulation` as the default execution path.

Execution rules:
- pass `model`
- pass the resolved forcing file path with `source_type="csv"`
- omit `params` when the user did not provide them so MCP can randomize
- allow MCP to infer common forcing column aliases such as `prcp(mm/day)` -> `P`
- preserve `run_id`, `seed`, `solver`, `interpolation`, `warnings`, and `params_used` when available

### 5) evaluate the run

If observed discharge is available in the workspace, call `compute_metrics` after the simulation.

If observed discharge is not available:
- do not fabricate accuracy metrics
- provide only a simulation-quality assessment from runtime diagnostics and summary statistics
- state clearly that no observation-based evaluation was possible

### 6) outputs

Current MCP behavior:
- simulation outputs are written under `./result/simulation/`
- a machine-readable metadata file is written next to the simulation CSV
- a short summary markdown file is written under `./result/`
- metrics artifacts are written under `./result/metrics/` when `compute_metrics` is used
- transient data handles are stored in session cache rather than temporary files

### 6b) end-of-conversation cleanup

Before ending the conversation, call `clear_session_cache`.

Cleanup rules:
- do this only after all requested outputs have been written or reported
- this clears transient Redis or in-memory session handles only
- do not claim that `./result` artifacts were deleted

### 7) respond in this fixed structure

# Runoff run summary

## Model
- selected model
- why it matched
- whether the run is exploratory or constrained

## Workspace inputs
- forcing path
- observed discharge path or `not available`
- directories used: `./data` and `./result`

## Execution
- tool used
- parameter source
- period, warmup, solver, interpolation if available
- run id or output handle if available

## Results
- compact runoff summary
- key warnings or anomalies

## Evaluation
- metric results when observation exists
- otherwise state that only runtime and behavior evaluation was possible

## Outputs written
- files or folders written to `./result`

## Next action
Give one small next step only.

## Failure modes

Use one of these explicit failure labels when needed:
- model not found
- forcing file not found in workspace
- multiple forcing files need resolution
- simulation failed
- evaluation skipped because observation is unavailable

For each failure, give the smallest actionable unblock message.

## Data handling strategy

This skill passes file paths to MCP tools without loading CSV contents into the LLM context.

### Option 1: Direct file paths

When the user provides or you resolve a file path:
1. Pass the absolute path directly to `run_simulation`
2. Example: `{"model": "exphydro", "source_type": "csv", "path": "./data/forcing.csv"}`
3. MCP loads the data internally and writes artifacts into `./result`

### Option 2: Data handle pattern

When working with data across multiple operations:
1. Call `load_hydro_csv`
2. Receive a `data_handle` identifier
3. Pass the handle to `calibrate_model`, `sensitivity_analysis`, or `compute_metrics`
4. `run_simulation` still uses direct file paths rather than data handles
5. Before closing the conversation, call `clear_session_cache` to remove transient session handles
