# HydroModelMCP

HydroModelMCP is a Julia MCP server built on top of `HydroModels.jl`, `HydroModelLibrary.jl`, and `ModelContextProtocol.jl`.
It exposes hydrological model discovery, simulation, calibration, validation, and ensemble workflows to MCP clients over `stdio` or Streamable HTTP.

## Project Summary Report (Current Capabilities)

This section is a practical status report for what HydroModelMCP can do today.

### 1) End-to-end hydrology workflows exposed as MCP tools

- Model discovery: `list_models`, `find_model`, `get_model_info`, `get_model_parameters`, `get_model_variables`.
- Data readiness and source inspection: `inspect_hydro_data` with model-aware input coverage checks.
- Data loading and reuse: `load_hydro_csv`, `load_caravan_data`, and in-session `data_handle` reuse.
- Simulation: unified v2 `run_simulation` with role-based `inputs` (`forcing`, `runtime`, `parameters`, `observation`).
- Calibration: `calibrate_model` (single-objective), `calibrate_multiobjective` (Pareto workflow when needed).
- Validation and metrics: `run_validation`, `compute_metrics`, and `split_data`.
- Ensemble runs: `run_ensemble_parameters`.
- Storage and retrieval: `list_stored_results`, `get_stored_result` for calibration/sensitivity/ensemble outputs.

### 2) Strategy-aligned calibration support (KNOWLEDEGE 1-10)

- Sensitivity-first screening: `run_sensitivity` / `sensitivity_analysis` before full calibration.
- Constraint-aware calibration inputs: pie-share and inequality (`delta_method`) contracts are supported.
- Magnitude-aware objective handling: distribution inspection and optional log-transform logic.
- Data split support: auto split and period split for train/test calibration review.
- Objective mapping by goal: `configure_objectives` aligns metrics to use-case intent.
- Strategy-8 algorithm adaptation: `AUTO` can choose budget-aware strategy aliases (`DDS`, `SCE`) and stable backends.
- Diagnostic loop tools: `diagnose_calibration`, `compute_diagnostics_full`, and `diagnose_multiobjective`.
- One-click automation: `auto_calibration_workflow` for strategy-oriented closed-loop execution.

### 3) MCP-first usability surfaces for LLM clients

- Resources: model catalogs, algorithm/objective/metrics guides, data-handle guides, stored-result indexes.
- Templates: model/resource templates discoverable through `resources/templates/list`.
- Prompts: expert/workflow prompts for compact planning, routing, and review.
- Low-token routing support: quickstart/workflow resources designed for weaker model clients.

### 4) Runtime and integration model

- Supports `stdio` and Streamable HTTP server modes.
- Supports file and Redis storage backends, plus session cache cleanup via `clear_session_cache`.
- Uses a unified v2 payload contract across simulation/calibration/validation/ensemble paths.
- Reads runtime configuration from process environment variables first, then falls back to project-root `.env` via `DotEnv` only when a value is missing.

### 5) Current implementation notes

- `DDS` and `SCE` are strategy aliases in the calibration interface.
- Runtime execution uses stable library backends (`DDS -> DE`, `SCE -> PSO`) for service robustness.
- Constraints are supported with feasibility checks and optimization-time penalties; post-run constraint audit is still recommended for strict acceptance gates.
- Large-sample official basin loading is Caravan-first. Basin requests should use `source_type=caravan` with explicit `dataset_name` and `gauge_id`/`gage_id`.
- The old CAMELS route is retired from the public MCP surface; do not use `load_camels_data` or `source_type=camels` in new clients.

## What This Server Exposes

The server currently provides:

- Discovery tools such as `list_models`, `find_model`, `get_model_info`, `get_model_variables`, and `get_model_parameters`.
- Data tools such as `inspect_hydro_data`, `load_hydro_csv`, `load_caravan_data`, `analyze_distribution_from_handle`, and `list_workspace_files`.
- Simulation and validation tools such as `run_simulation`, `list_workspace_files`, `clear_session_cache`, `run_ensemble_parameters`, and `run_validation`.
- Calibration workflow tools such as `compute_metrics`, `split_data`, `run_sensitivity`, `sensitivity_analysis`, `generate_samples`, `calibrate_model`, `calibrate_multiobjective`, `diagnose_calibration`, `diagnose_multiobjective`, `configure_objectives`, `init_calibration_setup`, `compute_diagnostics_full`, and `auto_calibration_workflow`.
- Surface inspection tool `list_mcp_surfaces` for lightweight routing/debugging.
- Storage retrieval tools `list_stored_results` and `get_stored_result` for listing and fetching persisted calibration/sensitivity/ensemble outputs.
- Resources for model catalog browsing, workflow guides, template catalog metadata, and stored-result indexes.
- Model details remain available through unified discovery tools on demand.
- Six MCP prompts: `hydrology_expert_review`, `hydro_minimal_tool_router`, `runoff_workspace_workflow`, `calibration_workflow_plan`, `hydrology_result_review`, and `hydro_minimal_workflow_router`.

## Upstream MCP Design Notes

This project was reviewed against the upstream `ModelContextProtocol.jl` README and example servers:

- Upstream README: [JuliaSMLM/ModelContextProtocol.jl](https://github.com/JuliaSMLM/ModelContextProtocol.jl)
- Upstream docs: [stable docs](https://juliasmlm.github.io/ModelContextProtocol.jl/stable/)
- Upstream example used for comparison: [`examples/time_server.jl`](https://github.com/JuliaSMLM/ModelContextProtocol.jl/blob/main/examples/time_server.jl)

One implementation detail matters here: HydroModelMCP patches the installed `ModelContextProtocol.jl` runtime so `resources/templates/list` is exposed and supported URI templates can be read dynamically through `resources/read`. Runtime model access still stays tool-first, while template-addressable resources provide compact discoverability.

## Installation

```bash
git clone https://github.com/chooron/HydroModelMCP
cd HydroModelMCP
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

## Storage Configuration

Configuration precedence for this project is:

1. Process environment variables
2. Project-root `.env`
3. Built-in defaults

This lets the MCP server resolve dataset and runtime settings internally without asking the model to inspect `.env` manually.

## Dataset Configuration

Official large-sample basin loading is Caravan-only.

Recommended Caravan settings:

| Variable | Meaning |
| --- | --- |
| `CARAVAN_DATASET_ROOT` | Caravan dataset root containing `attributes/` and `timeseries/` |
| `CARAVAN_DATASET_PATH` | Alias of `CARAVAN_DATASET_ROOT` |
| `CARAVAN_TIMESERIES_ROOT` | Optional `timeseries/` root override |
| `CARAVAN_NETCDF_ROOT` | Optional `timeseries/netcdf/` root override |
| `CARAVAN_NETCDF_PATH` | Alias of `CARAVAN_NETCDF_ROOT` |

Typical request shape for basin data:

```json
{
  "source_type": "caravan",
  "dataset_name": "camels",
  "gauge_id": "01013500"
}
```

Behavior rules:

- `dataset_name` or `source_dataset` is required for Caravan basin lookup.
- If the requested basin is missing, the server returns a direct not-found error.
- Caravan basin requests do not fall back to local CSV discovery.

Environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `STORAGE_BACKEND` | `file` | `file` or `redis` |
| `HYDRO_STORAGE_PATH` | `~/.hydro_mcp/storage` | File-backend storage directory |
| `HYDRO_STORAGE_TTL` | `604800` | TTL in seconds, `0` means no expiration |
| `HYDRO_SESSION_CACHE_BACKEND` | `auto` | `auto`, `redis`, or `memory` for transient data-handle storage |
| `HYDRO_SESSION_CACHE_TTL` | `86400` | TTL in seconds for Redis-backed transient session cache |
| `REDIS_HOST` | `127.0.0.1` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |

Examples:

```bash
# Default file backend
julia --project=. start.jl

# Custom file backend
set STORAGE_BACKEND=file
set HYDRO_STORAGE_PATH=D:\hydro-storage
set HYDRO_STORAGE_TTL=604800
julia --project=. start.jl

# Redis backend
set STORAGE_BACKEND=redis
set REDIS_HOST=127.0.0.1
set REDIS_PORT=6379
set HYDRO_STORAGE_TTL=604800
julia --project=. start.jl
```

Caravan example:

```bash
set CARAVAN_DATASET_ROOT=G:\Dataset\Caravan
julia --project=. start.jl
```

## Running The Server

### stdio transport

```bash
julia --project=. start.jl
```

This is the mode you want for Claude Desktop and Inspector CLI testing.

If a runtime setting is not present in the process environment, `start.jl` and `start_http.jl` will still allow the server to resolve it from the project-root `.env` file.

### Streamable HTTP transport

```bash
julia --project=. start_http.jl
```

By default the HTTP server listens on `http://127.0.0.1:3000/`.

Optional environment variables for `start_http.jl`:

| Variable | Default |
| --- | --- |
| `JULIA_HTTP_HOST` | `127.0.0.1` |
| `JULIA_HTTP_PORT` | `3000` |
| `JULIA_HTTP_ALLOWED_ORIGINS` | `*` |

Runtime precedence note:

- `run_http_server` now prefers `JULIA_HTTP_HOST` / `JULIA_HTTP_PORT` and falls back to `MCP_HOST` / `MCP_PORT` for backward compatibility.

## MCP Inspector Testing

The current Inspector CLI syntax should use `--cli`:

```bash
npx @modelcontextprotocol/inspector --cli julia --project=. start.jl --method tools/list
```

Common checks:

```bash
npx @modelcontextprotocol/inspector --cli julia --project=. start.jl --method tools/list
npx @modelcontextprotocol/inspector --cli julia --project=. start.jl --method resources/list
npx @modelcontextprotocol/inspector --cli julia --project=. start.jl --method prompts/list
npx @modelcontextprotocol/inspector --cli julia --project=. start.jl --method tools/call --tool-name list_models
```

If `julia` resolves to a Windows Store shim, use the absolute Julia binary path instead.

## Resource Layout

Always-available top-level resources:

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
- `hydro://guides/llm-hints`
- `hydro://guides/llm-quickstart`
- `hydro://meta/resource-templates`
- `hydro://calibration/results`
- `hydro://sensitivity/results`
- `hydro://ensemble/results`

Stored result resources are registered at startup for any IDs already present in storage:

- `hydro://calibration/results/<result_id>`
- `hydro://sensitivity/results/<result_id>`
- `hydro://ensemble/results/<result_id>`

Template-addressable model resources:

- `hydro://models/<model_name>/info`
- `hydro://models/<model_name>/parameters`
- `hydro://models/<model_name>/variables`
- `hydro://models/<model_name>/knowledge`
- `hydro://hints/<feature>`
- `hydro://workflows/<intent>`

Discovery tools remain the most reliable workflow path:

- `find_model`
- `get_model_info`
- `get_model_parameters`
- `get_model_variables`

The model catalog includes model names, template metadata, knowledge resource URIs, and preferred tools so clients can browse compactly and then switch into tool-oriented workflows.

Resource templates are discoverable through the protocol itself:

- `resources/templates/list`

## Project Layout

```text
HydroModelMCP/
|-- src/
|   |-- core/          # hydrology and optimization logic
|   |-- utils/         # data/runtime/storage utilities
|   |-- resources/     # MCP resources
|   |-- tools/         # MCP tool definitions
|   |-- prompts/       # MCP prompts
|   `-- HydroModelMCP.jl
|-- test/
|-- data/
|-- start.jl
|-- start_http.jl
`-- README.md
```

## Test Suite

Julia tests:

```bash
julia --project=. test/runtests.jl
```

The test suite includes:

- Core discovery, metrics, sampling, sensitivity, calibration, and ensemble checks.
- Contract checks for simulation/calibration/validation surfaces, including Caravan loading, output artifacts, workspace listing, and session-cache cleanup.
- Server assembly checks for tool registration, prompt registration, and compact resource registration.

## Strict MCP Scenarios

The repository includes strict scenario cards under `examples/strict-interaction-mcp-tests/`.

- These cards exercise documented MCP behavior, fail-fast rules, and auditability.
- They are designed to be run through the custom repository skill `hydromodel-mcp-strict-tests`.
- Large-sample basin cards in that suite use Caravan only.

## Example MCP Usage

List models:

```json
{
  "tool": "list_models"
}
```

Read the catalog resource:

```text
hydro://models/catalog
```

Read a supplementary knowledge resource:

```text
hydro://models/gr4j/knowledge
```

Read dynamic model info through a template URI:

```text
hydro://models/gr4j/info
```

Run a simulation:

```json
{
  "tool": "run_simulation",
  "params": {
    "model": "exphydro",
    "inputs": {
      "forcing": {
        "source_type": "csv",
        "path": "./data/03604000.csv"
      },
      "runtime": {
        "source_type": "json",
        "data": {
          "solver": "DISCRETE",
          "interpolation": "LINEAR",
          "seed": 1234
        }
      }
    },
    "output": {
      "result_source_type": "csv",
      "output_dir": "./result"
    },
    "options": {
      "auto_infer": true,
      "strict_infer": false
    }
  }
}
```

Load a Caravan basin into a reusable handle:

```json
{
  "tool": "load_caravan_data",
  "params": {
    "dataset_name": "camels",
    "gauge_id": "01013500"
  }
}
```

Run validation with explicit observation source:

```json
{
  "tool": "run_validation",
  "params": {
    "model": "exphydro",
    "inputs": {
      "forcing": {
        "source_type": "csv",
        "path": "./data/03604000.csv"
      },
      "observation": {
        "source_type": "csv",
        "path": "./data/03604000.csv"
      }
    },
    "calibration_period": {
      "start": "1989-01-01",
      "end": "1995-12-31"
    },
    "validation_period": {
      "start": "1996-01-01",
      "end": "1998-12-31"
    },
    "metrics": ["NSE", "KGE", "RMSE"]
  }
}
```

## Unified v2 Workflow Protocol

HydroModelMCP now uses a unified request contract for simulation, validation, ensemble, and v2 calibration tools:

- Required top-level fields: `model`, `inputs`
- Optional top-level fields: `output`, `options`, plus workflow-specific fields such as `metrics`, `objectives`, `parameter_sets`, `calibration_period`, and `validation_period`
- `inputs` is role-based: `forcing`, `observation`, `parameters`, `runtime`
- Each input role uses a source descriptor with `source_type` in: `csv`, `json`, `redis`, `caravan`, `npz`, `data_handle`

Example source descriptors:

```json
{
  "source_type": "csv",
  "path": "./data/03604000.csv"
}
```

```json
{
  "source_type": "redis",
  "host": "127.0.0.1",
  "port": 6379,
  "key": "hydro:forcing:03604000"
}
```

```json
{
  "source_type": "caravan",
  "dataset_name": "camels",
  "gauge_id": "01013500"
}
```

### Intelligent Input Inference

- Forcing variables are auto-mapped to model inputs (for example `P`, `T`, `Ep`) using aliases and numeric-series heuristics.
- Observed runoff is auto-detected from aliases such as `obs`, `qmean`, `target`, `flow`, `streamflow`, and `discharge`.
- Every workflow response returns machine-readable `inference_report` and `warnings` to help clients inspect mapping confidence.
- Use `options.input_mapping` to pin explicit mappings and `options.strict_infer=true` to fail on low-confidence inference.

List workspace files:

```json
{
  "tool": "list_workspace_files",
  "params": {
    "directory": "./data",
    "extensions": ["csv"]
  }
}
```

Clear transient session cache after a workflow:

```json
{
  "tool": "clear_session_cache",
  "params": {}
}
```

Use the prompt:

```json
{
  "name": "hydrology_expert_review",
  "arguments": {
    "task": "Review whether exphydro is a reasonable model choice for a snow-dominated catchment",
    "context": "Observed flow record is short and low-flow behavior matters"
  }
}
```

## Current Constraints

- Per-model template URIs are metadata only because the installed `ModelContextProtocol.jl` handler resolves exact registered URIs rather than wildcard templates, and this server intentionally avoids per-model resource registration to keep startup and `resources/list` compact.
- Stored result IDs are listed through `hydro://calibration/results`, `hydro://sensitivity/results`, and `hydro://ensemble/results`. If you need fresh per-result resource registration after new runs, restart the server or rebuild the server resource registry.
- Final result artifacts are written to `./result` by default unless `output.result_source_type="redis"` is used for cache-oriented output.

## Acknowledgements

- [HydroModels.jl](https://github.com/chooron/HydroModels.jl)
- [HydroModelLibrary.jl](https://github.com/chooron/HydroModelLibrary.jl)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [JuliaSMLM/ModelContextProtocol.jl](https://github.com/JuliaSMLM/ModelContextProtocol.jl)
