# P2. Minimal Executability Benchmark

## Scenario Metadata

- Scenario ID: `P2`
- Paper role: core results
- Status: ready now
- Independence: independent once a shared fixture is selected

## Objective

Test whether each discoverable model can complete at least one minimal simulation when compatible inputs are available.

## Why It Matters for the Paper

This scenario distinguishes between models that are only listed and models that are genuinely runnable under the unified execution protocol.

## Preconditions

- at least one shared forcing or forcing-plus-observation fixture is available
- the workflow can inspect readiness before execution
- output artifacts can be recorded per model

## Data Requirements

- one shared baseline dataset
- optional fallback fixtures for model families with stricter input needs

## Model Scope

- all models returned by `list_models`
- only models that pass readiness should be executed

## Example User Requests

- "Check which models can run with the current dataset."
- "Run one minimal simulation for each compatible model."
- "Skip models that fail readiness and explain why."
- "Summarize successful outputs by model."
- "Report runnable, skipped, and failed counts."

## Core Tool Workflow

1. Resolve the model catalog.
2. For each model, call `inspect_hydro_data` with the chosen fixture and intended simulation use.
3. Execute `run_simulation` only for models that pass readiness.
4. Record skipped models and missing-input reasons.
5. Aggregate outputs and failures.

## Expected Outputs

- runnable/skipped/failed matrix
- per-model execution notes
- returned run summaries or artifact paths for successful runs

## Pass/Fail Criteria

- Pass: the model completes one valid simulation.
- Skip: readiness correctly reports incompatible or missing inputs.
- Fail: readiness passes but simulation crashes or returns invalid output.

## Tables and Figures

- Table: minimal executability matrix
- Figure: runnable vs skipped vs failed ratio

## Pseudocode Scaffold

```text
models = list_models()
for model in models:
    inspect = inspect_hydro_data(source=fixture, model=model, intended_use="simulation")
    if inspect blocks execution:
        mark skipped
        continue
    result = run_simulation(model=model, inputs=...)
    record success or failure
render executability summary
```
