# P10. Strict Readiness Gating and Manual Mapping Recovery

## Scenario Metadata

- Scenario ID: `P10`
- Paper role: extended results
- Status: cleaned up
- Independence: independent once one ambiguous fixture is fixed

## Objective

Verify that ambiguous inputs can be rejected safely under strict inference and recovered through documented manual mapping.

## Why It Matters for the Paper

This scenario supports the claim that the platform is safe under uncertainty and recoverable with minimal, explicit user intervention.

## Preconditions

- at least one ambiguous CSV or forcing fixture exists
- the ambiguity can trigger low-confidence inference
- a valid explicit mapping is available for recovery

## Data Requirements

- one ambiguous input fixture
- one explicit mapping object for canonical fields

## Model Scope

- one representative model or a small model subset affected by the ambiguous inputs

## Example User Requests

- "Inspect this file and do not guess if the mapping is uncertain."
- "Run in strict inference mode."
- "Tell me which variables are ambiguous."
- "Retry with an explicit input mapping."
- "Compare rejection and recovery behavior."

## Core Tool Workflow

1. Call `inspect_hydro_data` as the pre-check.
2. Attempt execution with `options.strict_infer=true`.
3. Confirm that ambiguous inference is rejected safely.
4. Retry the same workflow with explicit `options.input_mapping`.
5. Measure rejection and recovery rates.

## Expected Outputs

- strict-mode rejection rate
- recovery success rate after explicit mapping
- ambiguity category summary

## Pass/Fail Criteria

- Pass: strict mode blocks uncertain inference and explicit mapping recovers the workflow.
- Fail: strict mode guesses silently or explicit mapping still cannot normalize the data path.

## Tables and Figures

- Table: strict gating vs manual mapping recovery

## Pseudocode Scaffold

```text
inspect = inspect_hydro_data(source=ambiguous_fixture, model=..., intended_use="simulation")
strict_result = run_simulation(model=..., inputs=..., options={"strict_infer": true})
mapped_result = run_simulation(
    model=...,
    inputs=...,
    options={"strict_infer": true, "input_mapping": explicit_mapping}
)
compare rejection and recovery outcomes
```
