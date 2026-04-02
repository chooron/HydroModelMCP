# R1-S4 Simple Single-Objective Calibration-Validation Closed Loop

## Scenario Metadata

- Scenario ID: `R1-S4`
- Paper role: minimal workflow closure
- Recommended skill: `hydromodel-mcp-results-1-tests`
- Source pack: `examples/paper-results-r1-basic-scenarios.md`
- Status: ready for test-script design and execution

## Objective

Demonstrate one bounded research workflow from data split through single-objective calibration, holdout validation, and diagnostic review.

## Why It Matters for Results 1

This is the closing scenario for Results 1. It converts the earlier discovery and execution evidence into a minimal scientific loop that can support a development and technical paper claim.

## What It Proves

It proves that HydroModelMCP can support a simple single-objective calibration-validation workflow with one shared split policy, one shared budget rule, one shared objective family, and post-run diagnostics.

## Preconditions

- One declared basin fixture with forcing and observed runoff is available.
- One model subset of one to three representative models is declared before execution.
- One shared calibration budget rule is declared before execution.
- One shared train/validation split rule is declared before execution.
- One shared objective family is declared before execution.
- `split_data`, `configure_objectives`, `calibrate_model`, `run_validation`, and at least one diagnostic tool are available.

## Model Scope

One to three representative models selected from the runtime catalog after readiness confirmation for the declared basin fixture. Do not expand this scenario to the entire catalog.

## Data Scope

- Exactly one basin.
- One chronological train/validation split.
- Single-objective calibration only.
- No multi-objective calibration.
- No self-repair loop.

## Example User Requests

- Using one declared basin, split the record chronologically, calibrate a small model subset under the same budget, and validate each result on the holdout period.
- Configure one single-objective runoff-fit target and run the same closed loop for up to three representative models.
- Return calibration metrics, validation metrics, best parameters, and overfitting gaps for each selected model.
- Diagnose each calibration result and summarize which model generalizes best on the validation period.
- Keep the workflow strictly inside the documented HydroModelMCP calibration and validation tools without any agentic recovery loop.

## Required Tool Workflow

1. Call `list_models` and select one to three representative models for this scenario. Record the fixed subset before calibration begins.
2. For each selected model, call `inspect_hydro_data` on the declared basin fixture for the intended calibration branch.
3. If readiness is blocked for a selected model, log the model as `skipped` and do not call downstream calibration tools for that model.
4. Call `split_data` once using the shared chronological split rule and record the exact train and validation boundaries.
5. Call `configure_objectives` once using the shared single-objective family and preserve the returned objective configuration.
6. For each ready selected model, call `calibrate_model` under the same declared budget rule and the unified v2 request shape.
7. For each successful calibration, call `run_validation` on the holdout period using the calibrated result and the same declared metric set.
8. For each successful calibration, call `diagnose_calibration` and, when available, `compute_diagnostics_full`.
9. Aggregate calibration metrics, validation metrics, best-parameter summaries, and overfitting gap summaries across the selected model subset.

## Input Fixture Specification

- Basin fixture: one declared official basin request using `source_type=caravan`, explicit `dataset_name`, and explicit `gauge_id` or `gage_id`.
- Default pack fixture: reuse the same official basin fixture as `R1-S3` unless the experiment manifest explicitly overrides it.
- Split rule: chronological 80/20 split, implemented once through `split_data`, with the first 80 percent for calibration and the last 20 percent for validation.
- Objective family: one single-objective discharge-fit target configured through `configure_objectives`; the pack default is an NSE-oriented objective family.
- Budget rule: one fixed numerical calibration budget applied identically to every selected model; the pack default is 100 objective evaluations per model, and the exact live-tool field mapping shall be written into the log.
- Metric reporting set: `NSE`, `KGE`, and `RMSE` for both calibration and validation summaries when the live surface supports them.

## Expected Outputs

- Calibration versus validation metrics for every selected and successfully calibrated model.
- Best-parameter summary for every selected and successfully calibrated model.
- Overfitting gap summary computed from calibration and validation metrics under the same metric family.
- Diagnostic notes from `diagnose_calibration` and, when available, `compute_diagnostics_full`.
- Artifact bundle containing calibration outputs, validation outputs, and any returned result identifiers.

## Pass / Partial / Fail Criteria

Pass: the scenario respects one fixed model subset, one fixed split rule, one fixed single-objective family, and one fixed budget rule; every successful model returns best parameters, calibration metrics, validation metrics, and diagnostics; every skipped model is explicitly logged.

Partial: the closed loop is mostly completed, but one or more selected models end in logged skips or diagnostics are incomplete while the train/validation separation and budget fairness remain intact.

Fail: calibration and validation are run on the same unsplit series, the budget or objective changes across models, multi-objective logic is introduced, or the workflow relies on self-repair or undocumented orchestration.

## Required Logs

- One scenario manifest log containing the selected model subset, split rule, objective family, metric set, and budget rule.
- One per-model readiness log.
- One per-model calibration log with best parameters, warnings, inference report, and artifacts.
- One per-model validation log with holdout metrics.
- One per-model diagnostics log with the chosen diagnostic tools and summarized findings.
- One final summary log containing overfitting gaps and the final model ranking by validation performance.

## Tables and Figures to Generate

- Table R1-7: Calibration versus validation metrics for the selected model subset.
- Table R1-8: Best-parameter summary and declared calibration budget per model.
- Table R1-9: Overfitting gap summary with calibration-minus-validation performance differences.
- Figure R1-6: Train versus validation performance comparison for the selected models.
- Figure R1-7: Validation hydrographs for the best calibrated model and one comparison model.
- Appendix diagnostic table: Condensed findings from `diagnose_calibration` and `compute_diagnostics_full`.

## Notes for Weaker LLM Executors

- Keep the subset small. Do not expand this scenario to all models.
- Do not use `auto_calibration_workflow` unless the harness explicitly requires it and logs the exact bounded configuration. The default path is the explicit manual tool chain above.
- Do not change the split, objective, or budget once the scenario manifest is declared.
- If a selected model is unsupported for the declared basin or calibration branch, log `skipped` and continue with the remaining selected models.
- Do not invent self-correction loops, alternative optimizers, or hidden preprocessing steps.
