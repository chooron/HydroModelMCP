# S05. Calibration Readiness Fail-Fast

## Goal

Test whether the agent stops immediately when a calibration request is not ready because observed runoff or required forcing inputs are missing.

## Recommended Skill

- `hydromodel-mcp-strict-tests`

## Strict User Prompt

```text
Use the hydromodel-mcp-strict-tests skill.

Check whether ./data/partial_forcing.csv can be used to calibrate exphydro. If observed runoff is missing or the model-required forcing inputs are incomplete, stop immediately and explain the blocking issue. Do not call downstream calibration or validation tools after a blocking readiness failure.
```

## Preconditions

- If `./data/partial_forcing.csv` is missing, mark this case as `skipped_missing_fixture`.

## Expected MCP Call Sequence

1. `find_model`
2. `get_model_info`
3. `get_model_parameters`
4. `inspect_hydro_data` with `source_type=csv`, path `./data/partial_forcing.csv`, and `intended_use=calibration`
5. Stop on blocking readiness failure

## Required Correctness Checks

- The agent must use `inspect_hydro_data` before any calibration attempt.
- A blocking readiness issue must terminate the workflow immediately.
- The final answer must identify the exact missing requirement.

## Automatic Failure Signals

- `load_hydro_csv` is called after a blocking readiness report.
- `calibrate_model`, `run_validation`, or `compute_metrics` is called anyway.
- The agent gives a calibration result despite missing readiness.

## Review Verdict

- Pass: the workflow stops immediately and reports the blocking issue precisely.
- Partial: the stop behavior is correct but the explanation is vague.
- Fail: downstream execution continues after a blocking readiness failure.
