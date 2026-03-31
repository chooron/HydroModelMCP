# Sensitivity Test Samples

## Sample 1: low-cost sensitivity screening

### User prompt

```text
请使用 HydroModelMCP，对 ./data/03604000.csv 的 exphydro 先做一次低成本参数敏感性分析，告诉我哪些参数最值得率定。
```

### Expected MCP sequence

1. `find_model`
2. `get_model_info`
3. `get_model_parameters`
4. `inspect_hydro_data`
5. `load_hydro_csv`
6. `sensitivity_analysis`
7. `clear_session_cache`

### Expected defaults

- `method=morris`
- `n_samples` around `100`

## Sample 2: sensitivity with explicit objective

### User prompt

```text
请对 ./data/03604000.csv 的 exphydro 做敏感性分析，并以 KGE 作为目标来筛选重要参数。
```

### Expected MCP sequence

1. `find_model`
2. `inspect_hydro_data`
3. `load_hydro_csv`
4. `sensitivity_analysis`

### Expected behavior

- pass `objective=KGE`
- report sensitive vs insensitive parameters

## Sample 3: stop when calibration-ready data are missing

### User prompt

```text
请检查 ./data/partial_forcing.csv 是否足以对指定模型做参数敏感性分析；如果模型输入不满足，就不要继续。
```

### Expected MCP sequence

1. `find_model`
2. `get_model_info`
3. `inspect_hydro_data`

### Expected stop behavior

- do not call `load_hydro_csv`
- do not call `sensitivity_analysis`
- report the missing model-required inputs
