# S06. Strict Inference and Manual Mapping Recovery

## Goal

Test whether ambiguous inputs are rejected safely under strict inference and then recovered through documented manual mapping.

## Recommended Skill

- `hydromodel-mcp-strict-tests`

## Strict User Prompt

```text
Use the hydromodel-mcp-strict-tests skill.

I have an ambiguous CSV at ./data/ambiguous_columns.csv. With exphydro, first inspect it and try strict inference mode so that uncertain input mapping is rejected instead of guessed. Then retry the same task with an explicit input mapping and report whether the workflow recovers. Do not read the raw CSV text into the conversation.
```

## Preconditions

- If `./data/ambiguous_columns.csv` is missing, mark this case as `skipped_missing_fixture`.

## Expected MCP Call Sequence

1. `find_model`
2. `get_model_info`
3. `inspect_hydro_data` for `./data/ambiguous_columns.csv`
4. `run_simulation` with `options.strict_infer=true` and an expected safe rejection
5. `run_simulation` again with `options.strict_infer=true` plus documented `options.input_mapping`
6. `clear_session_cache`

## Required Correctness Checks

- Strict mode must reject uncertain mapping instead of guessing silently.
- Recovery must use documented `options.input_mapping`, not an invented mapping API.
- The final report must compare rejection and recovery behavior.

## Automatic Failure Signals

- Strict mode guesses columns and runs anyway without high-confidence mapping.
- The recovery path uses undocumented fields or invented tools.
- The agent reads and pastes raw CSV content into the conversation.

## Review Verdict

- Pass: strict rejection happens first and explicit mapping recovers the workflow through the documented path.
- Partial: the recovery works but the strict rejection or reporting is not clean enough.
- Fail: the agent guesses silently or invents an unsupported recovery interface.
