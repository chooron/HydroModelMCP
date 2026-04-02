# P6. Calibration Bootstrap and Fair Budget Comparison

## Scenario Metadata

- Scenario ID: `P6`
- Paper role: extended results
- Status: ready now
- Independence: independent once one budget class and comparison set are fixed

## Objective

Use the built-in setup recommendation workflow to initialize controlled calibrations and compare outcomes under a fair budget.

## Why It Matters for the Paper

This scenario shows that HydroModelMCP can standardize calibration setup decisions in addition to executing the calibration itself.

## Preconditions

- one shared goal is selected
- one shared budget class is selected
- a comparison model set is fixed

## Data Requirements

- one basin or basin set with forcing and observed runoff
- one shared budget category such as `low`, `medium`, or `high`

## Model Scope

- a controlled comparison set such as `gr4j`, `hbv`, `xinanjiang`, and `exphydro`

## Example User Requests

- "Recommend a calibration setup for each target model."
- "Use the same budget class across all models."
- "Compare pre-calibration and post-calibration performance."
- "Summarize the recommended algorithms and objectives."
- "Report convergence quality under the matched budget."

## Core Tool Workflow

1. Call `init_calibration_setup` for each target model.
2. Keep the budget class fixed.
3. Run `calibrate_model` with the recommended setup.
4. Compare gains and diagnostics across models.

## Expected Outputs

- recommended setup summary
- pre/post calibration improvement table
- convergence comparison across models

## Pass/Fail Criteria

- Pass: each model receives a valid setup and completes a comparable calibration run.
- Fail: initialization or controlled comparison cannot be completed consistently.

## Tables and Figures

- Table: recommended calibration setups by model
- Figure: gain under a controlled budget

## Pseudocode Scaffold

```text
for model in model_set:
    setup = init_calibration_setup(model=model, inputs=..., budget=...)
    result = calibrate_model(model=model, inputs=..., ...recommended settings...)
    record setup and gain
render fair-budget comparison
```
