# P3. Single-Basin Cross-Model Benchmark

## Scenario Metadata

- Scenario ID: `P3`
- Paper role: core results
- Status: ready now
- Independence: independent once one basin fixture is fixed

## Objective

Run all compatible models on one shared basin and compare them under one metric set.

## Why It Matters for the Paper

This scenario provides the main benchmark table for the Results section and demonstrates standardized cross-model comparison under a controlled setup.

## Preconditions

- one basin fixture is selected
- forcing and observation are aligned over one evaluation window
- the metric set is fixed before execution

## Data Requirements

- one basin with forcing and observed runoff
- one shared time window
- one shared metric set such as `NSE`, `KGE`, and `RMSE`

## Model Scope

- all models that pass readiness for the chosen basin

## Example User Requests

- "Run all compatible models on the same basin."
- "Rank models by NSE and KGE."
- "Compute a shared metric table."
- "Show representative hydrographs for top and lower performers."
- "Explain which models were excluded and why."

## Core Tool Workflow

1. Call `inspect_hydro_data` for each candidate model.
2. Execute `run_simulation` for each eligible model.
3. Compute scores with `compute_metrics` or `run_validation`.
4. Rank the results under the shared metric set.
5. Prepare representative outputs for the paper.

## Expected Outputs

- metric leaderboard
- ranking table
- representative hydrograph set
- exclusion notes for ineligible models

## Pass/Fail Criteria

- Pass: simulation and metric computation succeed for an eligible model.
- Excluded: the input contract is not satisfied for that model.
- Fail: execution or scoring fails after readiness passed.

## Tables and Figures

- Table: single-basin model comparison
- Figure: score ranking distribution
- Figure: representative hydrographs

## Pseudocode Scaffold

```text
for model in candidate_models:
    inspect = inspect_hydro_data(..., model=model, intended_use="simulation")
    if blocked:
        record exclusion
        continue
    sim = run_simulation(model=model, inputs=...)
    metrics = compute_metrics(...) or run_validation(...)
    record scores
rank models and render figures
```
