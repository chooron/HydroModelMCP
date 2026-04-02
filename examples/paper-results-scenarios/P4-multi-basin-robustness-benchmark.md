# P4. Multi-Basin Robustness Benchmark

## Scenario Metadata

- Scenario ID: `P4`
- Paper role: extended results
- Status: data-dependent
- Independence: independent once a basin set is fixed

## Objective

Repeat the single-basin benchmark on multiple basin types to test whether conclusions are stable across hydrological regimes.

## Why It Matters for the Paper

This scenario prevents overclaiming from one-basin results and shows whether cross-model findings generalize beyond a single case study.

## Preconditions

- at least three representative basins are selected
- the same evaluation protocol is reused across all basins
- the basin set is justified in the paper

## Data Requirements

- at least three basin fixtures
- a consistent metric set and evaluation window policy

## Model Scope

- all models that pass readiness within each basin-specific run

## Example User Requests

- "Repeat the benchmark on three basin types."
- "Measure rank volatility across basins."
- "Identify robust models and basin-dependent models."
- "Keep the metric protocol fixed across basins."
- "Summarize how rankings shift by basin class."

## Core Tool Workflow

1. Reuse the P3 workflow for each target basin.
2. Compute per-basin rankings.
3. Aggregate rank volatility and score variance.
4. Classify models by stability across basins.

## Expected Outputs

- cross-basin metrics table
- rank variance summary
- robustness categorization

## Pass/Fail Criteria

- Pass: the workflow completes on the intended basin set or logs justified skips.
- Fail: the comparison protocol cannot be reproduced consistently across basins.

## Tables and Figures

- Table: cross-basin benchmark summary
- Figure: rank volatility across basin classes

## Pseudocode Scaffold

```text
for basin in basins:
    run P3 workflow
aggregate rankings across basins
compute stability statistics
render robustness summary
```
