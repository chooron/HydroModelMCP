# S04. Explicit-Parameter Validation Without Fallback

## Goal

Test whether `run_validation` is called with explicit optimized parameters instead of relying on hidden same-session fallback.

## Recommended Skill

- `hydromodel-mcp-strict-tests`

## Strict User Prompt

```text
Use the hydromodel-mcp-strict-tests skill.

Calibrate exphydro on Caravan basin 01170100, using the explicit sub-dataset that contains this basin, with an 80/20 split, then run validation with the optimized parameter set passed explicitly into the validation request. Do not rely on same-session fallback. I want the validation call to be contract-clean and reproducible.
```

## Expected MCP Call Sequence

1. `find_model`
2. `get_model_info`
3. `get_model_parameters`
4. `inspect_hydro_data` with `source_type=caravan`, explicit `dataset_name`, basin `01170100`, and `intended_use=calibration`
5. `load_caravan_data`
6. `configure_objectives`
7. `split_data` with an explicit 80/20 policy
8. `calibrate_model`
9. `run_validation` with explicit `inputs.parameters`
10. `compute_metrics`
11. `clear_session_cache`

## Required Correctness Checks

- The validation request must include explicit optimized parameters.
- The data request must use `source_type=caravan` with explicit `dataset_name`.
- The agent must not describe same-session fallback as the primary mechanism.
- The calibration and validation periods must be explicit and reproducible.
- The final answer must report validation metrics tied to the explicit parameter set.

## Automatic Failure Signals

- `run_validation` omits `inputs.parameters` and silently uses session fallback.
- Any retired legacy basin-loading route appears anywhere in the workflow.
- The agent provides only a prose promise of reproducibility without a contract-clean validation call.
- The split is implicit or undocumented.

## Review Verdict

- Pass: validation is executed with explicit parameters and the contract is clearly reproducible.
- Partial: the workflow succeeds but the validation contract is not demonstrated strongly enough.
- Fail: the agent relies on hidden fallback or cannot prove the parameter source.
