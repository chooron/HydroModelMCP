# P7. Sensitivity to Multiobjective Workflow

## Scenario Metadata

- Scenario ID: `P7`
- Paper role: extended results
- Status: ready now
- Independence: independent once one model and objective pair are fixed

## Objective

Start with sensitivity screening, then run multiobjective calibration on the retained parameter set.

## Why It Matters for the Paper

This scenario demonstrates that advanced scientific workflows can be executed through a unified protocol rather than ad hoc scripts.

## Preconditions

- one representative model is selected
- forcing and observed runoff are available
- at least two objectives are fixed

## Data Requirements

- one model-ready basin fixture
- one objective pair such as `KGE` and `LogKGE`

## Model Scope

- one representative model or a small advanced-workflow subset

## Example User Requests

- "Run sensitivity analysis first."
- "Keep the sensitive parameters for the next stage."
- "Run multiobjective calibration on the retained parameter set."
- "Summarize Pareto trade-offs."
- "Retrieve the stored result for later analysis."

## Core Tool Workflow

1. Call `run_sensitivity` or `sensitivity_analysis`.
2. Select retained parameters according to the sensitivity result.
3. Execute `calibrate_multiobjective`.
4. Retrieve or summarize the resulting Pareto output.

## Expected Outputs

- sensitive-parameter ranking
- Pareto summary
- trade-off interpretation notes

## Pass/Fail Criteria

- Pass: sensitivity outputs are generated and the multiobjective workflow returns a non-empty Pareto result.
- Fail: either stage fails or the linkage between stages is not reproducible.

## Tables and Figures

- Figure: parameter sensitivity ranking
- Figure: Pareto frontier

## Pseudocode Scaffold

```text
sense = run_sensitivity(model=..., inputs=...)
retained = choose sensitive parameters
pareto = calibrate_multiobjective(model=..., inputs=..., objectives=[...])
stored = get_stored_result(...)
render advanced-workflow summary
```
