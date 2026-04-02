# P5. Calibration-Validation Closed Loop

## Scenario Metadata

- Scenario ID: `P5`
- Paper role: core results
- Status: ready now
- Independence: independent once one basin/model set and one budget are fixed

## Objective

Demonstrate a full workflow from data split to calibration, validation, diagnostics, and stored-result retrieval.

## Why It Matters for the Paper

This scenario supports the central claim that HydroModelMCP is a workflow platform, not only a simulation runner.

## Preconditions

- forcing and observed runoff are available
- a calibration budget is fixed
- the objective and metric policy are fixed
- result storage is enabled

## Data Requirements

- 1 to 3 representative basins
- forcing plus observed runoff
- one split policy or explicit periods

## Model Scope

- 3 to 5 representative models or a smaller pilot subset

## Example User Requests

- "Split the dataset into calibration and validation periods."
- "Configure objectives for a general-fit calibration."
- "Calibrate GR4J under the shared budget."
- "Validate the calibrated result on the holdout period."
- "Return diagnostics and stored result IDs."

## Core Tool Workflow

1. Call `split_data` if the split must be materialized or described explicitly.
2. Call `configure_objectives`.
3. Execute `calibrate_model`.
4. Execute `run_validation`.
5. Run `diagnose_calibration` or `compute_diagnostics_full`.
6. Query `list_stored_results` and `get_stored_result` for auditability.

## Expected Outputs

- calibration vs validation metric table
- overfitting gap summary
- best-parameter summary
- stored result identifiers
- calibration diagnostics

## Pass/Fail Criteria

- Pass: the full loop executes and stored results are queryable.
- Fail: a critical step breaks or the final result cannot be retrieved or audited.

## Tables and Figures

- Table: calibration-validation performance
- Figure: train vs validation gap

## Pseudocode Scaffold

```text
objective_cfg = configure_objectives(goal=...)
calibration = calibrate_model(model=..., inputs=..., ...)
validation = run_validation(model=..., inputs=..., ...)
diagnostics = diagnose_calibration(...) or compute_diagnostics_full(...)
stored = list_stored_results(category="calibration")
render closed-loop summary
```
