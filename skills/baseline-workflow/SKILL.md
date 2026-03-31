---
name: baseline-workflow
description: test HydroModelMCP behavior inside the current workspace. use when chatgpt should validate an mcp workflow, trace each tool/resource/template step, avoid opening raw csv/json data files in the conversation, stop immediately on any error or contract mismatch, and report the exact failing step instead of forcing a final successful result.
---

# Baseline Workflow Test

Use this skill to test the MCP service itself.

Do not optimize for a polished final hydrology result. Optimize for accurate workflow validation, transparent trace logging, and immediate failure surfacing.

## Fixed operating rules

- Do not browse the web.
- Do not search databases or external systems.
- Treat the current project workspace as the execution boundary.
- Default input directory: `./data`
- Default output directory: `./result`
- Do not read raw contents from workspace data or artifact files such as `.csv`, `.json`, `.metadata.json`, `.txt`, or similar files into the conversation.
- It is acceptable to pass file paths such as CSV or JSON paths directly into MCP tools when the tool contract supports that.
- Use `list_workspace_files` to confirm file existence or candidate paths instead of opening files.
- Prefer MCP tools and MCP resources over ad hoc file inspection.
- Every MCP interaction must be recorded in the final execution trace.
- If any MCP call returns an error, contract mismatch, ambiguous blocking condition, or missing required field, stop immediately.
- After a blocking failure, do not continue with downstream calls just to obtain a final result.
- Do not fabricate success, metrics, or artifact summaries after a blocking failure.
- Call `clear_session_cache` only at the normal end of a successful or intentionally completed test path.

Read these references when needed:
- `references/workspace-execution-workflow.md`
- `references/mcp-minimum-contract.md`
- `references/actual-tool-contracts.md`

## Available MCP surfaces

### read-only resources

- `hydro://models/catalog`
- `hydro://models/knowledge-index`
- `hydro://models/knowledge-coverage`
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

### resource templates

- `hydro://models/{model_name}/info`
- `hydro://models/{model_name}/parameters`
- `hydro://models/{model_name}/variables`
- `hydro://models/{model_name}/knowledge`
- `hydro://calibration/results/{result_id}`
- `hydro://sensitivity/results/{result_id}`
- `hydro://ensemble/results/{result_id}`

### prompts

- `hydrology_expert_review`
- `runoff_workspace_workflow`
- `calibration_workflow_plan`
- `hydrology_result_review`

### key tools for input readiness

- `inspect_hydro_data` to check generic hydrometeorological coverage, model-required inputs, and observed runoff availability before downstream execution

### usage policy

- For tool-path validation, use tools directly.
- For template-path validation, explicitly use `resources/templates/list` and `resources/read` with a concrete template URI.
- For normal model resolution, keep the workflow tool-first: `find_model`, `get_model_info`, `get_model_parameters`, `get_model_variables`.
- Use `hydro://models/{model_name}/knowledge` only as supplementary knowledge, not as the runtime contract source.

## Interaction pattern

1. If the user already named a model, workflow, or tool path to test, start from that target.
2. If a model is required but missing, ask only: which hydrological model should be tested?
3. Ask another question only when a required local file path cannot be resolved safely from workspace defaults.

## Default conventions

- Project root: current workspace root
- Default baseline mode: one end-to-end simulation-path test
- Forcing source resolution order:
  1. explicit user path
  2. `./data/forcing.csv`
  3. `./data/meteo.csv`
  4. `./data/meteorology.csv`
  5. `./data/input.csv`
  6. `list_workspace_files(directory="./data")` if confirmation is needed
- Optional observed discharge resolution order:
  1. explicit user path
  2. `./data/obs.csv`
  3. `./data/observed.csv`
  4. `./data/discharge.csv`
  5. `./data/streamflow.csv`
- Parameter source: random exploratory parameters when the user gives none
- Output location: `./result`

If multiple plausible files exist and there is no safe default, stop immediately and report the ambiguity as a failure.

## Core workflow

### 1) normalize the request

Extract when present:
- target model
- target workflow or tool path
- explicit forcing file path
- explicit observed discharge path
- optional period, warmup, solver, interpolation, seed, or parameter values

### 2) choose the test mode

Use one of these modes:
- `simulation-baseline`: default when the user asks to run a model in the workspace
- `calibration-baseline`: when the user asks to test parameter optimization or calibration readiness
- `template-check`: when the user asks to validate resource templates or resource reads
- `prompt-check`: when the user asks to validate MCP prompts
- `result-retrieval-check`: when the user asks to inspect stored results by result id

### 3) resolve the model when needed

Use tools in this order:
1. `find_model`
2. `list_models` only if the match is empty or ambiguous
3. `get_model_info`
4. `get_model_parameters`

Stop immediately if model resolution fails or becomes ambiguous.

### 4) resolve workspace inputs when needed

Use this order:
1. explicit user path
2. default file conventions in `./data`
3. `list_workspace_files`

Do not open file contents.

### 4b) inspect input readiness before downstream execution

Before `run_simulation`, `calibrate_model`, `run_validation`, or `compute_metrics` when the source is a workspace file:
1. call `inspect_hydro_data`
2. pass the source as a structured descriptor such as `{"source": {"source_type": "csv", "path": "./data/03604000.csv"}}`
3. set `intended_use` to the planned workflow such as `simulation`, `calibration`, or `validation`
4. pass `model` when model-specific input coverage should be checked

Interpretation rules:
- if the selected model has missing required inputs, stop immediately
- if `intended_use` is `calibration` or `validation` and observed runoff is unavailable, stop immediately and tell the user to provide observed discharge data
- do not proceed to downstream MCP calls after a failed readiness check

### 5) execute the requested MCP path

#### simulation-baseline

- call `inspect_hydro_data` first unless a trusted recent inspection result already exists in the same conversation
- call `run_simulation`
- pass `model`
- pass the resolved forcing path through the canonical structured `forcing` object
- omit `params` when the user gave none so MCP can randomize
- if the user provides partial parameters, allow MCP to complete missing ones and record completion warnings
- normalize solver/interpolation choices to canonical enums even when user phrasing is Chinese or English aliases
- preserve `run_id`, `seed`, `solver`, `interpolation`, `warnings`, `params_used`, and output paths when returned
- call `compute_metrics` only when an observed discharge path is available and the test explicitly needs evaluation

#### calibration-baseline

- call `inspect_hydro_data` first with `intended_use="calibration"`
- if the check passes, call `load_hydro_csv`
- call `calibrate_model` with the unified v2 contract (`model + inputs`, with forcing and observation sources)
- keep the default budget small unless the user explicitly asks for a broader search
- call `diagnose_calibration` after calibration when the run succeeds

#### template-check

- call `resources/templates/list` first
- read one or more concrete template URIs with `resources/read`
- record whether the template was discoverable and whether the concrete read succeeded

#### prompt-check

- call `prompts/list` or `prompts/get` only when the client path supports prompt validation
- record the prompt name, arguments, and whether prompt expansion succeeded

#### result-retrieval-check

- use the index resources first:
  - `hydro://calibration/results`
  - `hydro://sensitivity/results`
  - `hydro://ensemble/results`
- only then test a concrete result template URI

### 6) stopping rule

Stop immediately when any of these occurs:
- MCP tool error
- resource read error
- template read error
- prompt error
- model not found
- missing required workspace input
- data inspection failed
- ambiguous input selection
- missing required output field in a response
- inconsistent or clearly invalid response structure

Do not continue to later steps after a blocking failure.

### 7) successful end

Only on success or intentional completion:
- report the full trace
- report returned artifact paths if any
- call `clear_session_cache` when transient session handles were created and cleanup is appropriate

## Response structure

Always respond in this structure.

# MCP workflow test summary

## Test target
- requested workflow or mode
- selected model or `not required`
- workspace scope used

## Execution trace
For every MCP interaction, record:
- step number
- interface used: tool, resource, template, or prompt
- exact name or URI
- purpose of the step
- key inputs summarized as names, ids, and file paths only
- outcome: `success`, `warning`, or `error`
- important returned identifiers, warnings, or artifact paths

## Status
- overall status: `passed`, `failed`, or `partial`
- final stopping reason

## Outputs observed
- returned artifact paths, resource URIs, result ids, or `none`

## Failure
- if failed: failure label
- failing step
- exact MCP interface that failed
- minimal error summary
- explicitly state which downstream steps were not attempted because execution stopped

## Next action
- give the smallest useful next check only

## Failure labels

Use one of these labels when needed:
- model not found
- forcing file not found in workspace
- multiple forcing files need resolution
- mcp tool error
- resource read error
- template read error
- prompt error
- data inspection failed
- contract mismatch
- evaluation skipped because observation is unavailable

## Data handling strategy

- Pass file paths to MCP tools without loading file contents into the LLM context.
- It is acceptable to pass CSV or JSON file paths directly to MCP when supported by the tool contract.
- Do not inspect generated result CSV or JSON artifacts directly in this skill.
- If the user explicitly asks to inspect artifact contents, stop and state that this skill is for MCP workflow testing rather than artifact-content reading.
