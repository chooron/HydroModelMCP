# Stage2 Calibration Harness Scenarios

This document defines new harness scenarios for stage-2 calibration behavior in HydroModelMCP.

## Goal

- verify `calibrate_model` can calibrate on train split and report test-set metrics
- verify both split modes are supported:
  - auto split (`method/ratio/warmup`)
  - period split (`calibration_period` + `validation_period`)
- keep initial runs lightweight

## Lightweight defaults (first-pass)

- `maxiters`: `6` to `12`
- `n_trials`: `1`
- `algorithm`: `BBO`
- `metrics`: `["NSE", "KGE", "RMSE"]`
- fixture: `./data/03604000.csv`

These defaults are for quick regression checks, not for final hydrological quality.

## Scenario S2-01: auto split train/test

### Input template

```json
{
  "tool": "calibrate_model",
  "params": {
    "model": "exphydro",
    "inputs": {
      "forcing": {"source_type": "csv", "path": "./data/03604000.csv"},
      "observation": {"source_type": "csv", "path": "./data/03604000.csv", "column": "flow(mm)"}
    },
    "algorithm": "BBO",
    "objective": "KGE",
    "maxiters": 8,
    "n_trials": 1,
    "method": "split_sample",
    "ratio": 0.75,
    "warmup": 30,
    "metrics": ["NSE", "KGE"]
  }
}
```

### Assertions

- response contains `stage2_evaluation`
- `stage2_evaluation.split_mode == "auto"`
- `train_length > 0` and `test_length > 0`
- `stage2_evaluation.train_metrics` is present
- `stage2_evaluation.test_metrics` is present when `test_available=true`

## Scenario S2-02: period split train/test

### Input template (index-based)

```json
{
  "tool": "calibrate_model",
  "params": {
    "model": "exphydro",
    "inputs": {
      "forcing": {"source_type": "csv", "path": "./data/03604000.csv"},
      "observation": {"source_type": "csv", "path": "./data/03604000.csv", "column": "flow(mm)"}
    },
    "algorithm": "BBO",
    "objective": "KGE",
    "maxiters": 6,
    "n_trials": 1,
    "calibration_period": {"start_index": 1, "end_index": 900},
    "validation_period": {"start_index": 901, "end_index": 1200},
    "metrics": ["NSE"]
  }
}
```

### Assertions

- `stage2_evaluation.split_mode == "period"`
- `train_indices == [1, 900]`
- `test_indices == [901, 1200]`
- `train_length == 900` and `test_length == 300`

## Scenario S2-03: period split with date bounds and no forcing dates

Use `start/end` dates while forcing metadata has no date column.

### Assertions

- call succeeds
- `stage2_evaluation.used_synthetic_dates == true`
- warnings mention synthetic timeline fallback

## Alias policy note

For synonym replacement, list-based definitions are easier to read, but canonical mapping is lookup-heavy.

- harness docs may describe aliases as lists
- runtime normalization can keep dictionary mapping for O(1) lookup and lower overhead

Current implementation keeps dictionary-based canonical mapping in code for stability and speed.
