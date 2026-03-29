# Actual Tool Contracts

This document records the current input and output contracts for MCP tools used by the `runoff-forecast` skill.

## run_simulation

### Input schema

```json
{
  "type": "object",
  "properties": {
    "model": {"type": "string"},
    "source_type": {"type": "string", "enum": ["csv", "json"]},
    "path": {"type": "string"},
    "params": {"type": "object"},
    "period": {"type": "array", "items": {"type": "string"}},
    "warmup": {"type": "integer"},
    "solver": {"type": "string"},
    "interpolation": {"type": "string"},
    "input_mapping": {"type": "object"},
    "output_dir": {"type": "string"},
    "seed": {"type": "integer"}
  },
  "required": ["model", "source_type"]
}
```

### Output format

```json
{
  "status": "success",
  "run_id": "uuid",
  "model": "exphydro",
  "output_path": "./result/simulation/forcing_result_20260329120000.csv",
  "metadata_path": "./result/simulation/forcing_result_20260329120000.metadata.json",
  "summary_path": "./result/run_summary_20260329120000.md",
  "params_used": {"f": 0.01, "Smax": 250.0},
  "params_source": "random",
  "params_seed": 1234,
  "run_info": {
    "model": "exphydro",
    "solver": "DISCRETE",
    "interpolation": "LINEAR",
    "runtime_seconds": 0.45
  },
  "warnings": []
}
```

### Key behaviors

- Defaults `output_dir` to `./result`
- Writes simulation artifacts into `./result/simulation/`
- Returns randomized parameter values and the seed when parameters are omitted
- Infers common forcing aliases such as `prcp(mm/day)` -> `P`
- Still records `period` and `warmup` as metadata only

---

## compute_metrics

### Input schema

```json
{
  "type": "object",
  "properties": {
    "simulated": {"type": "object"},
    "observed": {"type": "object"},
    "metrics": {"type": "array", "items": {"type": "string"}},
    "output_dir": {"type": "string"}
  },
  "required": ["simulated", "observed"]
}
```

Each source object may use either:

```json
{"source_type": "csv", "path": "./result/simulation/foo.csv"}
```

or:

```json
{"data_handle": "hydro_foo"}
```

### Output format

```json
{
  "status": "success",
  "metrics": {
    "NSE": 0.82,
    "KGE": 0.78,
    "RMSE": 12.34,
    "PBIAS": -5.2
  },
  "sample_size": 365,
  "simulated_column": "Result",
  "observed_column": "flow(mm)",
  "output_path": "./result/metrics/foo_metrics_20260329120000.json",
  "warnings": []
}
```

### Key behaviors

- Accepts either file paths or data handles
- Infers `Result` for simulated CSV and common discharge aliases for observed CSV
- Writes a metrics artifact into `./result/metrics/`
- Truncates to the shorter series if lengths differ and returns a warning

---

## load_hydro_csv

### Input schema

```json
{
  "type": "object",
  "properties": {
    "path": {"type": "string"},
    "data_type": {"type": "string", "enum": ["forcing", "observed", "simulated"]},
    "validation": {"type": "boolean"},
    "obs_column": {"type": "string"},
    "prcp_column": {"type": "string"},
    "temp_column": {"type": "string"},
    "pet_column": {"type": "string"},
    "simulated_column": {"type": "string"}
  },
  "required": ["path", "data_type"]
}
```

### Output format

```json
{
  "status": "success",
  "data_handle": "hydro_03604000",
  "metadata": {
    "rows": 365,
    "columns": ["date", "prcp(mm/day)", "tmean(C)", "pet(mm)", "flow(mm)"],
    "start_date": "2020-01-01",
    "end_date": "2020-12-31"
  },
  "warnings": []
}
```

### Key behaviors

- Returns a handle, not the data itself
- Preserves richer combined forcing + observation payloads when those columns exist
- Remains suitable for calibration and sensitivity workflows

---

## find_model

### Input schema

```json
{
  "type": "object",
  "properties": {
    "query": {"type": "string"}
  },
  "required": ["query"]
}
```

### Output format

```json
{
  "status": "success",
  "query": "exp",
  "matches": [
    {
      "name": "exphydro",
      "full_name": "exphydro",
      "match_score": 0.75
    }
  ]
}
```

---

## get_model_info

### Input schema

```json
{
  "type": "object",
  "properties": {
    "model": {"type": "string"}
  },
  "required": ["model"]
}
```

### Output format

```json
{
  "status": "success",
  "model": "exphydro",
  "full_name": "exphydro",
  "description": "Model description",
  "inputs": ["P", "T", "Ep"],
  "outputs": ["Qt"],
  "parameter_count": 6
}
```

---

## get_model_parameters

### Input schema

```json
{
  "type": "object",
  "properties": {
    "model": {"type": "string"}
  },
  "required": ["model"]
}
```

### Output format

```json
{
  "status": "success",
  "model": "exphydro",
  "parameters": [
    {
      "name": "f",
      "description": "Parameter description",
      "min": 0.0,
      "max": 1.0,
      "default": null,
      "unit": "-"
    }
  ]
}
```

---

## list_workspace_files

### Input schema

```json
{
  "type": "object",
  "properties": {
    "directory": {"type": "string"},
    "extensions": {"type": "array", "items": {"type": "string"}},
    "include_size": {"type": "boolean"},
    "include_modified": {"type": "boolean"}
  },
  "required": ["directory"]
}
```

### Output format

```json
{
  "status": "success",
  "directory": "./data",
  "count": 2,
  "files": [
    {
      "name": "forcing.csv",
      "path": "./data/forcing.csv",
      "size_bytes": 12345,
      "modified": "2026-03-29T12:00:00"
    }
  ]
}
```

### Key behaviors

- Lists files only, not directory contents recursively
- Auto-creates the requested directory when it is missing and returns `created_directory: true`
- Rejects paths outside the current workspace root

---

## clear_session_cache

### Input schema

```json
{
  "type": "object",
  "properties": {
    "include_handles": {"type": "boolean"}
  }
}
```

### Output format

```json
{
  "status": "success",
  "backend": "redis",
  "prefix": "hydro:session:...:",
  "cleared_count": 3,
  "cleared_handles": ["hydro_03604000", "obs_03604000"]
}
```

### Key behaviors

- Clears transient session cache entries only
- Does not delete files written to `./result`
- Should be called at the end of a `runoff-forecast` conversation
