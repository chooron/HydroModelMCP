# Calibration Test Samples

## Sample 1: full single-objective calibration with diagnostics

### User prompt

```text
请使用 HydroModelMCP 对 ./data/03604000.csv 的 exphydro 做一次正式率定，不要只做低成本优化。先检查输入，再完成率定诊断，并告诉我是否还需要扩大预算。
```

### Expected MCP sequence

1. `find_model`
2. `get_model_info`
3. `get_model_parameters`
4. `inspect_hydro_data`
5. `load_hydro_csv`
6. `analyze_distribution_from_handle`
7. `configure_objectives`
8. `calibrate_model`
9. `diagnose_calibration`
10. `compute_diagnostics_full` when multi-trial review is needed
11. `clear_session_cache`

### Expected behavior

- use a stronger budget than the optimization skill by default
- use unified v2 calibration requests (`model + inputs`)
- report diagnostics, not just best parameters

## Sample 2: calibration with sensitivity screening first

### User prompt

```text
请先做参数敏感性分析，再根据筛选结果对 ./data/03604000.csv 的 exphydro 做率定。
```

### Expected MCP sequence

1. `find_model`
2. `get_model_info`
3. `inspect_hydro_data`
4. `load_hydro_csv`
5. `sensitivity_analysis` or `run_sensitivity`
6. `configure_objectives`
7. `calibrate_model`
8. `diagnose_calibration`

## Sample 3: calibration plus validation split

### User prompt

```text
请把 ./data/03604000.csv 划分为率定期和验证期，再完成 exphydro 率定，并给出验证指标。
```

### Expected MCP sequence

1. `find_model`
2. `inspect_hydro_data`
3. `load_hydro_csv`
4. `split_data`
5. `configure_objectives`
6. `calibrate_model`
7. `run_validation`
8. `compute_metrics`
9. `diagnose_calibration`

## Sample 4: stop because data are not calibration-ready

### User prompt

```text
请检查 ./data/partial_forcing.csv 是否可以用于正式率定。如果输入不满足模型要求或没有实测径流，请立刻停止。
```

### Expected MCP sequence

1. `find_model`
2. `get_model_info`
3. `inspect_hydro_data`

### Expected stop behavior

- do not call `load_hydro_csv`
- do not call `calibrate_model`
- clearly report the blocking issue
