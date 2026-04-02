# S03. Caravan Basin Sensitivity and Storage Audit

## Goal

Test whether the agent can run a sensitivity-first calibration workflow, persist the calibration result, and audit the stored record immediately afterward.

## Recommended Skill

- `hydromodel-mcp-strict-tests`

## Strict User Prompt

```text
Use the hydromodel-mcp-strict-tests skill.

For Caravan basin 01547700, using the explicit sub-dataset that contains this basin, and the exphydro model, first run parameter sensitivity screening, then do an 80/20 calibration-validation workflow, persist the calibration result, list stored calibration results, and read back the newest stored record. I want one final PASS or FAIL verdict telling me whether the stored record is consistent with the live calibration response.
```

## Expected MCP Call Sequence

1. `find_model`
2. `get_model_info`
3. `get_model_parameters`
4. `inspect_hydro_data` with `source_type=caravan`, explicit `dataset_name`, basin `01547700`, and `intended_use=calibration`
5. `load_caravan_data`
6. `analyze_distribution_from_handle`
7. `run_sensitivity` or `sensitivity_analysis`
8. `configure_objectives`
9. `split_data` with an explicit 80/20 policy
10. `calibrate_model` with storage enabled
11. `run_validation`
12. `compute_metrics`
13. `diagnose_calibration`
14. `list_stored_results` with `category=calibration`
15. `get_stored_result` for the returned calibration `result_id`
16. `clear_session_cache`

## Required Correctness Checks

- Sensitivity screening must happen before calibration.
- The data request must use `source_type=caravan` with explicit `dataset_name`.
- The stored calibration result must expose a `result_id` or equivalent storage handle.
- The read-back result must match the live calibration response on result identity and key summary fields.
- The final verdict must state whether the storage record is consistent or inconsistent.
- The agent must not inspect raw storage files outside MCP.

## Automatic Failure Signals

- Sensitivity screening is skipped.
- Any retired legacy basin-loading route appears anywhere in the workflow.
- Storage retrieval is skipped after persistence.
- The agent reads back a different result than the one just created.
- The final verdict says storage is consistent without checking the stored record.

## Review Verdict

- Pass: the workflow performs sensitivity screening, calibration, storage retrieval, and a grounded consistency verdict.
- Partial: the core path works but the storage audit is incomplete or weakly justified.
- Fail: persistence or retrieval is mishandled, or the verdict is unsupported.
