# R1-S3 Single-Basin Standardized Cross-Model Simulation Benchmark

## Scenario Metadata

- Scenario ID: `R1-S3`
- Paper role: standardized comparison
- Recommended skill: `hydromodel-mcp-results-1-tests`
- Source pack: `examples/paper-results-r1-basic-scenarios.md`
- Status: ready for test-script design and execution

## Objective

Compare compatible models on one fixed basin under one shared observation source, one fixed metric set, and one fixed evaluation window.

## Why It Matters for Results 1

Results 1 needs a standardized comparison, not only proof of isolated execution. This scenario produces the first cross-model evidence table grounded in one controlled basin setup.

## What It Proves

It proves that HydroModelMCP can support a normalized single-basin benchmark where compatible models are screened, executed, scored, ranked, and excluded transparently when they do not fit the declared setup.

## Preconditions

- One basin fixture is declared before execution and remains fixed for the full scenario.
- One evaluation window is declared before the first model run and remains fixed for the full scenario.
- One metric set is declared before the first model run and remains fixed for the full scenario.
- `inspect_hydro_data`, `run_simulation`, and either `compute_metrics` or `run_validation` are available.

## Model Scope

All models from `list_models` that pass readiness for the declared single-basin fixture.

## Data Scope

- Exactly one basin.
- Exactly one forcing and observation condition set.
- Exactly one evaluation window.
- No calibration.
- No multi-basin aggregation.

## Example User Requests

- Run all compatible models on one declared basin and rank them under one shared metric set.
- Use the same evaluation window for every compatible model and log any excluded models with reasons.
- Produce a single-basin leaderboard with NSE, KGE, and RMSE for all included models.
- Generate representative hydrographs for the best, median, and weakest included models.
- Show which models were excluded before execution because the basin fixture did not satisfy their readiness requirements.

## Required Tool Workflow

1. Call `list_models` to obtain the candidate model set.
2. For each model, call `inspect_hydro_data` on the declared basin fixture for the intended simulation branch.
3. If readiness is blocked, log the model in the exclusion log and do not call downstream simulation or scoring tools for that model.
4. For each ready model, call `run_simulation` on the same basin fixture using the same evaluation window rule.
5. For each successful simulation, compute the fixed metric set using either `compute_metrics` or `run_validation`.
6. Record the metric results, warnings, inference reports, and artifact evidence for every included model.
7. Rank included models under one declared primary ranking metric and preserve the full metric table.
8. Generate the exclusion log and representative hydrograph set from the included-model results.

## Input Fixture Specification

- Basin fixture: one declared official basin request using `source_type=caravan`, explicit `dataset_name`, and explicit `gauge_id` or `gage_id`.
- Default pack fixture: `source_type=caravan`, `dataset_name=camels`, `gauge_id=01013500` or `gage_id=01013500`.
- Evaluation window: one explicit start date and one explicit end date declared in the scenario manifest before model iteration. The same window shall be used for every included model.
- Metric set: `NSE`, `KGE`, and `RMSE`.
- Ranking rule: rank by `NSE`, then use `KGE` as the first tie-breaker.

## Expected Outputs

- Metric leaderboard for all included models.
- Representative hydrograph set using one common observed runoff reference.
- Exclusion log for every model that did not enter simulation.
- One benchmark artifact bundle containing the leaderboard table input, hydrograph plotting input, and exclusion notes.

## Pass / Partial / Fail Criteria

Pass: all candidate models are screened, every included model is evaluated under the same basin, metric set, and evaluation window, the leaderboard is complete, and every excluded model has a logged reason.

Partial: the fixed setup is respected, but some included models have incomplete artifact packaging or the hydrograph set is incomplete while the metric and exclusion evidence remains usable.

Fail: more than one basin is used, the metric set or evaluation window changes across models, excluded models are not logged, or downstream scoring is reported without prior readiness evidence.

## Required Logs

- One scenario manifest log containing the basin fixture, evaluation window, metric set, and ranking metric.
- One per-model readiness log.
- One per-model execution and scoring log with `metrics`, `warnings`, `inference_report`, and `artifacts`.
- One exclusion log entry for every blocked model with `failure_type=readiness_blocked` or `failure_type=unsupported_model_branch`.

## Tables and Figures to Generate

- Table R1-5: Single-basin leaderboard with NSE, KGE, RMSE, and inclusion status.
- Table R1-6: Exclusion log with model name and exclusion reason.
- Figure R1-4: Ranked performance plot for the included models.
- Figure R1-5: Representative hydrographs for the best, median, and weakest included models under the common basin setup.

## Notes for Weaker LLM Executors

- Do not turn this into a multi-basin experiment.
- Do not change the metric set after the first scored model.
- Use the minimal valid chain: readiness, simulation, scoring, logging.
- If a model is unsupported for the declared basin setup, log it as excluded and continue.
- Do not invent a second observation source or a custom ranking rule.
