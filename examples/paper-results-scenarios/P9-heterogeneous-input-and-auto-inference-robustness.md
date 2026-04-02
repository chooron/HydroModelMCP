# P9. Heterogeneous Input and Auto-Inference Robustness

## Scenario Metadata

- Scenario ID: `P9`
- Paper role: core results
- Status: ready now
- Independence: independent once the source-type fixture set is fixed

## Objective

Evaluate readiness detection and input inference across CSV, Caravan, Redis, and data-handle-driven workflows.

## Why It Matters for the Paper

This scenario shows that HydroModelMCP can support realistic data heterogeneity instead of relying on one idealized file format.

## Preconditions

- at least one canonical CSV fixture is available
- at least one non-standard CSV fixture is available
- Caravan access is configured if Caravan is included
- Redis is available if Redis-backed inputs are included

## Data Requirements

- one canonical CSV fixture
- one aliased or non-standard CSV fixture
- one Caravan basin fixture or configured Caravan dataset root
- optional Redis-backed input source

## Model Scope

- one representative model or a small compatibility subset

## Example User Requests

- "Inspect whether this non-standard CSV can be recognized automatically."
- "Run a model directly from Caravan data."
- "Load data once and continue from a handle."
- "Compare warnings across source types."
- "Summarize inference success rates."

## Core Tool Workflow

1. Call `inspect_hydro_data` for each source type.
2. Use `load_hydro_csv` where canonical handle loading is needed.
3. Use `load_caravan_data` for Caravan-backed basin cases.
4. Execute `run_simulation` on compatible inputs.
5. Aggregate warnings, inference behavior, and compatibility notes.

## Expected Outputs

- inference success rate by source type
- warning summary
- compatibility matrix by source and model

## Pass/Fail Criteria

- Pass: readiness and execution behave consistently across source types when inputs are sufficient.
- Fail: inference hides blocking issues or incompatible inputs are treated as valid.

## Tables and Figures

- Figure: inference success rate by input source
- Table: source compatibility summary

## Pseudocode Scaffold

```text
for source_case in source_cases:
    inspect = inspect_hydro_data(source=source_case, model=..., intended_use="simulation")
    if source_case requires loading:
        handle = load_hydro_csv(...) or load_caravan_data(...)
    if inspect passes:
        sim = run_simulation(model=..., inputs=...)
    record warnings and outcomes
render heterogeneity summary
```
