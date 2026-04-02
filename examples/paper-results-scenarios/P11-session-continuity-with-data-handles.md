# P11. Session Continuity with Data Handles

## Scenario Metadata

- Scenario ID: `P11`
- Paper role: extended results
- Status: cleaned up
- Independence: independent once one reusable CSV fixture is fixed

## Objective

Test whether a dataset can be loaded once, reused across multiple downstream steps, and cleared at the end of the workflow.

## Why It Matters for the Paper

This scenario demonstrates session continuity for long-running workflows without forcing repeated file ingestion.

## Preconditions

- one CSV source with forcing is available
- observed runoff is available if validation or metrics are required
- the workflow uses one session context throughout
- when `data_handle` is used as a `run_simulation` input source, `output.result_source_type` must be set explicitly

## Data Requirements

- one CSV source with forcing and optionally observed runoff

## Model Scope

- one representative model or a minimal workflow subset

## Example User Requests

- "Load the CSV once and keep reusing the data handle."
- "Run a simulation from the same session state."
- "Continue with validation without reloading the file."
- "Compute metrics from the same workflow context."
- "Clear the session cache when done."

## Core Tool Workflow

1. Call `load_hydro_csv` and capture the resulting handle.
2. If `run_simulation` uses `source_type="data_handle"`, set `output.result_source_type` explicitly, typically `csv`.
3. Execute `run_simulation` from the loaded state or aligned source.
4. Execute `run_validation` in the same session when appropriate.
5. Execute `compute_metrics` when observed runoff is available.
6. Call `clear_session_cache` at the intentional end of the workflow.

## Expected Outputs

- handle reuse trace
- session continuity log
- cleanup confirmation

## Pass/Fail Criteria

- Pass: the same session state supports multiple downstream calls without redundant reloads.
- Fail: handle reuse breaks, yields inconsistent results, or cleanup is incomplete.

## Tables and Figures

- Figure: session workflow trace with data-handle reuse

## Pseudocode Scaffold

```text
handle = load_hydro_csv(path=..., data_type=..., handle_name=...)
sim = run_simulation(
    model=...,
    inputs={"forcing": {"source_type": "data_handle", "data_handle": handle}},
    output={"result_source_type": "csv"}
)
validation = run_validation(model=..., inputs=...)
metrics = compute_metrics(...)
clear_session_cache()
record reuse and cleanup behavior
```
