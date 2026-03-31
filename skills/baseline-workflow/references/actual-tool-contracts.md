# Actual Tool Contracts

This document records the current canonical tool contracts used by the `baseline-workflow` skill.

## inspect_hydro_data

### Input shape

```json
{
  "source": {
    "source_type": "csv",
    "path": "./data/03604000.csv"
  },
  "model": "exphydro",
  "intended_use": "calibration"
}
```

### Output highlights

- `forcing_elements`: generic `P/T/Ep` coverage report
- `observed_runoff`: detected observed discharge column if present
- `model_check`: model-specific required-input coverage when `model` is provided
- `blocking_issues`: reasons to stop the workflow immediately
- `readiness`: booleans derived from model-required inputs and intended use

### Important behavior

- Generic `P/T/Ep` detection is informational.
- Workflow blocking is driven by model-required inputs when `model` is provided.
- `calibration`, `validation`, and `metrics` readiness also require observed runoff.

---

## run_simulation

### Canonical input shape

```json
{
  "model": "exphydro",
  "inputs": {
    "forcing": {
      "source_type": "csv",
      "path": "./data/03604000.csv"
    }
  },
  "output": {
    "result_source_type": "csv",
    "output_dir": "./result"
  }
}
```

### Redis output shape

```json
{
  "model": "exphydro",
  "inputs": {
    "forcing": {
      "source_type": "csv",
      "path": "./data/03604000.csv"
    }
  },
  "output": {
    "result_source_type": "redis",
    "result_host": "127.0.0.1",
    "result_port": 6379
  }
}
```

### Important behavior

- `inputs.forcing` is required.
- Legacy flat fields such as top-level `source_type` and `path` are no longer part of the public contract.
- `output.result_source_type` is the result-sink selector.
- `inputs.parameters` accepts these practical forms:
  - inline object: `{"f":0.03,"Smax":500,...}`
  - source descriptor: `json/csv/data_handle/calibration_result`
  - partial object is allowed in simulation and missing parameters are filled with random valid values
- runtime option values support Chinese/English aliases and are normalized to canonical values (`ODE` / `DISCRETE`, `LINEAR` / `CONSTANT` / `DIRECT`).

---

## load_hydro_csv

### Canonical input shape

```json
{
  "path": "./data/03604000.csv",
  "data_type": "forcing"
}
```

### Output highlights

- `data_handle`
- `metadata.rows`
- `metadata.columns`
- `warnings`

### Important behavior

- Stores any resolvable canonical forcing inputs among `P`, `T`, and `Ep`.
- Stores observed runoff when a matching observed column exists.
- Returns a handle, not the raw data.

---

## calibrate_model

### Canonical input shape

```json
{
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
  "objective": "KGE",
  "algorithm": "BBO",
  "maxiters": 80,
  "n_trials": 1
}
```

### Important behavior

- `model` and `inputs` are required.
- `inputs` must include both forcing and observed runoff sources for calibration.
- The tool validates model-required inputs before entering optimization.
- The baseline workflow should keep `maxiters` small unless the user explicitly asks for a broader search.

---

## run_validation

### Canonical input shape

```json
{
  "model": "exphydro",
  "inputs": {
    "forcing": {
      "source_type": "csv",
      "path": "./data/03604000.csv"
    },
    "observation": {
      "source_type": "csv",
      "path": "./data/03604000.csv"
    },
    "parameters": {
      "source_type": "json",
      "data": {
        "f": 0.03,
        "Smax": 500,
        "Qmax": 20,
        "Df": 2.0,
        "Tmax": 1.0,
        "Tmin": -1.5
      }
    }
  },
  "calibration_period": {
    "start_index": 1,
    "end_index": 700
  },
  "validation_period": {
    "start_index": 701,
    "end_index": 1200
  }
}
```

### Parameter source options

- inline object in `inputs.parameters` (no `source_type` required)
- source descriptor with `source_type=json/csv/data_handle/calibration_result`
- direct calibration result object using keys such as `best_params`, `best_parameters`, `calibrated_params`, or `params_used`
- same-session fallback: if `inputs.parameters` is omitted, MCP can reuse the latest calibration/simulation parameters when available

### Important behavior

- `model`, `inputs.forcing`, `inputs.observation`, `calibration_period`, `validation_period` are required.
- `inputs.parameters` is strongly recommended; omission only works when same-session fallback is available.
- if parameter values are partial, MCP first attempts same-session completion; unresolved required parameters cause fail-fast errors.
- if both periods use `start/end` dates but forcing has no date column, MCP can synthesize a timeline from requested boundaries and returns a warning; use `start_index/end_index` for strict reproducibility.

---

## compute_metrics

### Canonical input shape

```json
{
  "simulated": {
    "source_type": "csv",
    "path": "./result/simulation/foo.csv"
  },
  "observed": {
    "source_type": "csv",
    "path": "./data/03604000.csv"
  },
  "output_dir": "./result"
}
```

### Important behavior

- Accepts either file paths or data handles.
- Infers common simulated and observed column names when they are not provided.
- Writes a metrics artifact into `./result/metrics/`.
