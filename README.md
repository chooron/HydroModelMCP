# HydroModelMCP

HydroModelMCP is a Julia MCP server built on top of `HydroModels.jl`, `HydroModelLibrary.jl`, and `ModelContextProtocol.jl`.
It exposes hydrological model discovery, simulation, calibration, validation, and ensemble workflows to MCP clients over `stdio` or Streamable HTTP.

## What This Server Exposes

The server currently provides:

- Discovery tools such as `list_models`, `find_model`, `get_model_info`, `get_model_variables`, and `get_model_parameters`.
- Simulation and validation tools such as `run_simulation`, `list_workspace_files`, `clear_session_cache`, `run_ensemble_parameters`, and `run_validation`.
- Calibration workflow tools such as `compute_metrics`, `split_data`, `run_sensitivity`, `generate_samples`, `calibrate_model`, `calibrate_multiobjective`, `diagnose_calibration`, `configure_objectives`, `init_calibration_setup`, and `compute_diagnostics_full`.
- Static resources for model catalog browsing, algorithm guidance, objective guidance, and calibration-result listing.
- Exact per-model resources for `info`, `parameters`, and `variables`.
- One MCP prompt, `hydrology_expert_review`, for structured hydrology reasoning.

## Upstream MCP Design Notes

This project was reviewed against the upstream `ModelContextProtocol.jl` README and example servers:

- Upstream README: [JuliaSMLM/ModelContextProtocol.jl](https://github.com/JuliaSMLM/ModelContextProtocol.jl)
- Upstream docs: [stable docs](https://juliasmlm.github.io/ModelContextProtocol.jl/stable/)
- Upstream example used for comparison: [`examples/time_server.jl`](https://github.com/JuliaSMLM/ModelContextProtocol.jl/blob/main/examples/time_server.jl)

One implementation detail matters here: in the installed `ModelContextProtocol.jl v0.4.0`, `resources/read` resolves exact resource URIs in the handler. Because of that, HydroModelMCP registers exact model resource URIs and exposes template metadata separately at `hydro://meta/resource-templates` instead of relying on wildcard resource URIs.

## Installation

```bash
git clone https://github.com/chooron/HydroModelMCP
cd HydroModelMCP
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

## Storage Configuration

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

## Running The Server

### stdio transport

```bash
julia --project=. start.jl
```

This is the mode you want for Claude Desktop and Inspector CLI testing.

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

Always-available resources:

- `hydro://models/catalog`
- `hydro://guides/algorithms`
- `hydro://guides/objectives`
- `hydro://meta/resource-templates`
- `hydro://calibration/results`

Per-model exact resources are generated at startup:

- `hydro://models/<model_name>/info`
- `hydro://models/<model_name>/parameters`
- `hydro://models/<model_name>/variables`

The model catalog includes all exact URIs, including `variables_uri`, so clients can browse first and read later.

## Project Layout

```text
HydroModelMCP/
|-- src/
|   |-- core/          # hydrology and optimization logic
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
- Contract checks for the runoff-forecast skill surface, including simulation output artifacts, workspace listing, and session-cache cleanup.
- Server assembly checks for tool registration, prompt registration, and exact resource registration.

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

Run a simulation:

```json
{
  "tool": "run_simulation",
  "params": {
    "model": "exphydro",
    "source_type": "csv",
    "path": "./data/03604000.csv",
    "output_dir": "./result",
    "solver": "DISCRETE",
    "interpolation": "LINEAR",
    "seed": 1234
  }
}
```

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

- Exact per-model resources are supported. Template metadata is published separately for clients because wildcard resource URIs are not resolved by the installed `ModelContextProtocol.jl` handler.
- Stored result IDs are listed through `hydro://calibration/results`. If you need fresh per-result resource registration after new runs, restart the server or add an explicit result-fetch tool.
- Final result artifacts are written to `./result`, while transient `data_handle` payloads prefer session cache and can be cleared with `clear_session_cache`.

## Acknowledgements

- [HydroModels.jl](https://github.com/chooron/HydroModels.jl)
- [HydroModelLibrary.jl](https://github.com/chooron/HydroModelLibrary.jl)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [JuliaSMLM/ModelContextProtocol.jl](https://github.com/JuliaSMLM/ModelContextProtocol.jl)
