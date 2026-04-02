# S02. Caravan Basin 80/20 Calibration and Validation

## Goal

Test a full positive-path Caravan calibration workflow with explicit 80/20 splitting, holdout evaluation, diagnostics, and runoff-process reporting.

## Recommended Skill

- `hydromodel-mcp-strict-tests`

## Strict User Prompt

```text
Use the hydromodel-mcp-strict-tests skill.

Read basin 01491000 from the Caravan dataset, using the explicit sub-dataset that contains this basin, and calibrate exphydro on it. Use the first 80% of the data for calibration and the last 20% for evaluation. Return the best parameter set, the holdout evaluation metrics, and the simulated runoff process. Keep the workflow fully inside HydroModelMCP and do not read raw data into the conversation.
```

## Expected MCP Call Sequence

1. `find_model`
2. `get_model_info`
3. `get_model_parameters`
4. `inspect_hydro_data` with `source_type=caravan`, explicit `dataset_name`, basin `01491000`, and `intended_use=calibration`
5. `load_caravan_data`
6. `analyze_distribution_from_handle`
7. `configure_objectives`
8. `split_data` with an explicit 80/20 policy
9. `calibrate_model` with the unified `model + inputs` request shape
10. `run_validation`
11. `compute_metrics`
12. `diagnose_calibration`
13. `compute_diagnostics_full`
14. `clear_session_cache`

## Required Correctness Checks

- `inspect_hydro_data` must happen before calibration.
- The data request must use `source_type=caravan` with explicit `dataset_name`.
- The split must be explicit; the agent must not calibrate and evaluate on the same full series.
- `run_validation` must target the holdout segment, not the calibration segment.
- The final answer must include best parameters and evaluation metrics from the holdout period.
- The runoff process must be represented by a result artifact, handle, path, or structured summary that is grounded in MCP outputs.

## Automatic Failure Signals

- No `split_data` call before validation.
- The agent reports evaluation metrics without proving a holdout split.
- Any retired legacy basin-loading route appears anywhere in the workflow.
- `calibrate_model` uses legacy flat payload fields instead of the v2 request shape.
- The agent pastes raw Caravan records into the conversation.
- No cleanup on the successful path.

## Review Verdict

- Pass: the full workflow executes in order, uses an explicit 80/20 split, and returns holdout metrics plus runoff output evidence.
- Partial: the workflow mostly works but has a major weakness such as incomplete diagnostics or weak runoff-process reporting.
- Fail: the split/evaluation logic is incorrect or a blocking contract/orchestration error occurs.
