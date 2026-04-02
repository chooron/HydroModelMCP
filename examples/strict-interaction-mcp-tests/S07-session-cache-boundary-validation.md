# S07. Session Cache Boundary Validation

## Goal

Test whether `clear_session_cache` really removes hidden parameter fallback and whether validation succeeds only when parameters are supplied explicitly afterward.

## Recommended Skill

- `hydromodel-mcp-strict-tests`

## Strict User Prompt

```text
Use the hydromodel-mcp-strict-tests skill.

Calibrate exphydro on Caravan basin 01605500, using the explicit sub-dataset that contains this basin, with an 80/20 split. Then clear the session cache. After that, intentionally try one validation call without passing parameters and confirm that hidden fallback is no longer available. Finally, run validation again with the explicit optimized parameters and report the metrics. I want to verify that cache cleanup is real.
```

## Expected MCP Call Sequence

1. `find_model`
2. `get_model_info`
3. `get_model_parameters`
4. `inspect_hydro_data` with `source_type=caravan`, explicit `dataset_name`, basin `01605500`, and `intended_use=calibration`
5. `load_caravan_data`
6. `configure_objectives`
7. `split_data` with an explicit 80/20 policy
8. `calibrate_model`
9. `clear_session_cache`
10. `run_validation` without `inputs.parameters`, expecting fail-fast or an explicit no-fallback error
11. `run_validation` again with explicit `inputs.parameters`
12. `compute_metrics`

## Required Correctness Checks

- The first post-cleanup validation call must not succeed by silently reusing old parameters.
- The data request must use `source_type=caravan` with explicit `dataset_name`.
- The second validation call must supply the optimized parameter set explicitly.
- The report must explain that cache cleanup changed the session behavior.

## Automatic Failure Signals

- The first post-cleanup validation call succeeds through hidden fallback.
- Any retired legacy basin-loading route appears anywhere in the workflow.
- The agent skips the negative check entirely.
- The second validation call still omits `inputs.parameters`.

## Review Verdict

- Pass: cleanup removes fallback, the negative check fails cleanly, and the explicit-parameter retry succeeds.
- Partial: the main idea is correct but one of the validation calls is not demonstrated clearly enough.
- Fail: hidden reuse survives cleanup or the contract boundary is not tested.
