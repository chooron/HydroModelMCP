# R1-S2 Minimal Executability Benchmark

## Scenario Metadata

- Scenario ID: `R1-S2`
- Paper role: execution baseline
- Recommended skill: `hydromodel-mcp-results-1-tests`
- Source pack: `examples/paper-results-r1-basic-scenarios.md`
- Status: ready for test-script design and execution

## Objective

Verify that discoverable models are executable under at least one minimal, documented input branch after an explicit readiness check.

## Why It Matters for Results 1

Results 1 must show that the server does more than advertise names. This scenario converts catalog entries into evidence of real execution or evidence-based skips.

## What It Proves

It proves that minimal execution is available through the documented simulation surface and that readiness gating can separate runnable models from unsupported or insufficient-input branches.

## Preconditions

- The server is reachable.
- `inspect_hydro_data` and `run_simulation` are available.
- Two fixed input fixtures are declared before execution: one local CSV fixture and one Caravan basin fixture.
- No skill-driven planning is used.

## Model Scope

All models returned by `list_models`. Each model must end in exactly one terminal status: `runnable`, `skipped`, or `failed`.

## Data Scope

- Local CSV fixture branch for minimal simulation.
- Official Caravan basin branch for minimal simulation.
- No calibration. No validation benchmark. No multi-basin execution.

## Example User Requests

- Check which listed models can run a minimal simulation on the repository CSV fixture and which ones need a Caravan basin instead.
- Perform a readiness check for every discovered model, then run one minimal simulation only for the models that pass.
- Use the local CSV fixture first, fall back to the declared Caravan fixture only when the CSV branch is blocked, and explain every skip.
- Produce a runnable, skipped, and failed table for the full model catalog.
- For each successful model, return the minimum execution evidence and any warnings or inference details.

## Required Tool Workflow

1. Call `list_models` to obtain the runtime model catalog.
2. For each model, optionally call `get_model_info` or `get_model_variables` only if needed to interpret readiness or expected inputs.
3. For each model, call `inspect_hydro_data` on the local CSV fixture with the intended simulation branch.
4. If the local CSV branch passes readiness, call `run_simulation` once on the local CSV fixture and stop evaluation for that model.
5. If the local CSV branch is blocked, call `inspect_hydro_data` on the Caravan fixture using `source_type=caravan` with explicit `dataset_name` and `gauge_id` or `gage_id`.
6. If the Caravan branch passes readiness, call `run_simulation` once on the Caravan fixture and stop evaluation for that model.
7. If both branches are blocked, log the model as `skipped` with branch-specific reasons.
8. If readiness passes but `run_simulation` fails, log the model as `failed` and preserve tool-grounded error evidence.
9. Aggregate all model outcomes into one runnable/skipped/failed table and one per-model execution note set.

## Input Fixture Specification

- Local CSV fixture: `source_type=csv`, `path=./data/03604000.csv`.
- Official Caravan fixture: `source_type=caravan`, `dataset_name=camels`, plus `gauge_id=01013500` or `gage_id=01013500` according to the live schema.
- Execution rule: the local CSV branch is evaluated first for every model; the Caravan branch is only evaluated when the CSV branch is blocked.
- Output rule: every successful minimal simulation must preserve its artifact path, result summary, or equivalent structured execution evidence.

## Expected Outputs

- Runnable / skipped / failed table for the full model catalog.
- Per-model minimal execution note stating which branch ran, which branch was blocked, or why the model failed.
- Skip-reason list grouped by readiness cause.
- Artifact index for successful minimal simulations.

## Pass / Partial / Fail Criteria

Pass: every discoverable model is processed through the declared branch order, every executed model passed readiness before `run_simulation`, and the final table records one terminal status with evidence for each model.

Partial: the benchmark covers the full catalog, but some executed models lack artifact details or some skip reasons are weakly specified while the branch outcomes remain auditable.

Fail: readiness is skipped, undocumented branches are used, official basin requests do not use `source_type=caravan` with explicit `dataset_name` and `gauge_id` or `gage_id`, or one or more models end without a logged terminal status.

## Required Logs

- One per-model log entry for the CSV readiness branch.
- One per-model log entry for the Caravan readiness branch when the CSV branch is blocked.
- One per-model terminal log entry containing `success`, `failure_type`, `warnings`, `inference_report`, and `artifacts` where available.
- One batch summary log with runnable, skipped, and failed counts plus grouped skip reasons.

## Tables and Figures to Generate

- Table R1-3: Full runnable / skipped / failed matrix with executed branch and reason codes.
- Table R1-4: Per-model minimal execution notes and artifact evidence.
- Figure R1-2: Outcome distribution across the catalog.
- Figure R1-3: Skip-reason frequency by readiness category.

## Notes for Weaker LLM Executors

- Do not run `run_simulation` before `inspect_hydro_data`.
- Do not invent a third fixture, fallback route, or manual data rewrite path.
- `dataset_name=camels` inside a Caravan request is valid; `source_type=camels` is not valid.
- Use only one successful minimal simulation per model.
- If both fixtures are blocked, log `skipped`; do not hallucinate a runnable path.
