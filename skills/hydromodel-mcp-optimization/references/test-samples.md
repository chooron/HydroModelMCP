# Optimization Test Samples

## Sample 1: low-cost optimization succeeds

### User prompt

```text
请使用 HydroModelMCP，对 ./data/03604000.csv 的 exphydro 做一次低成本参数优化。不要读原始 csv 内容，先检查输入是否满足率定要求，预算保持较小。
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
10. `clear_session_cache`

### Expected defaults

- `algorithm=BBO`
- `objective=KGE`
- `maxiters=80`
- `n_trials=1`

## Sample 2: stop because observed runoff is missing

### User prompt

```text
请对 ./data/forcing_only.csv 的 exphydro 做参数优化。如果缺少实测径流，不要继续率定，直接告诉我缺什么。
```

### Expected MCP sequence

1. `find_model`
2. `get_model_info`
3. `inspect_hydro_data`

### Expected stop behavior

- Stop after `inspect_hydro_data`
- Do not call `load_hydro_csv`
- Do not call `calibrate_model`

## Sample 3: stop because the model-required inputs are missing

### User prompt

```text
请检查 ./data/partial_forcing.csv 是否能用于某个指定模型的参数优化；如果模型输入不满足，就立刻停止。
```

### Expected MCP sequence

1. `find_model`
2. `get_model_info`
3. `inspect_hydro_data`

### Expected stop behavior

- Stop after `inspect_hydro_data`
- Report the missing model-required inputs

## Sample 4: user explicitly asks for a broader search

### User prompt

```text
请对 ./data/03604000.csv 的 exphydro 做参数优化，这次不要走最低成本，至少运行 300 次迭代，并给出诊断结果。
```

### Expected MCP sequence

1. `find_model`
2. `get_model_info`
3. `inspect_hydro_data`
4. `load_hydro_csv`
5. `configure_objectives`
6. `calibrate_model`
7. `diagnose_calibration`

### Expected setup override

- `maxiters` should follow the user's larger budget request instead of the low-cost default
