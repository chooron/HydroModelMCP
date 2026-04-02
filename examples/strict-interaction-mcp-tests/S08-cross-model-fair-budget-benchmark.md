# S08. Cross-Model Fair-Budget Benchmark

## Goal

Test whether the agent can compare two models on the same basin while preserving the same split policy, objective, budget, and reporting standard.

## Recommended Skill

- `hydromodel-mcp-strict-tests`

## Strict User Prompt

```text
Use the hydromodel-mcp-strict-tests skill.

On Caravan basin 01552500, using the explicit sub-dataset that contains this basin, compare exphydro and gr4j under the same 80/20 split, the same calibration objective, the same calibration budget, and the same reporting template. Rank the two models by validation KGE and tell me which one wins. Do not silently give one model a larger search budget or a different objective.
```

## Expected MCP Call Sequence

1. `find_model` for `exphydro`
2. `get_model_info` for `exphydro`
3. `get_model_parameters` for `exphydro`
4. `find_model` for `gr4j`
5. `get_model_info` for `gr4j`
6. `get_model_parameters` for `gr4j`
7. `inspect_hydro_data` for `source_type=caravan`, explicit `dataset_name`, basin `01552500` with a model-specific readiness check
8. `load_caravan_data`
9. `configure_objectives`
10. `split_data` with one shared 80/20 policy
11. `calibrate_model` for `exphydro`
12. `run_validation` for `exphydro`
13. `compute_metrics` for `exphydro`
14. `diagnose_calibration` for `exphydro`
15. `calibrate_model` for `gr4j`
16. `run_validation` for `gr4j`
17. `compute_metrics` for `gr4j`
18. `diagnose_calibration` for `gr4j`
19. `clear_session_cache`

## Required Correctness Checks

- Both models must use the same split policy.
- The data request must use `source_type=caravan` with explicit `dataset_name`.
- Both models must use the same objective and the same optimization budget.
- Validation metrics must be reported in a comparable format.
- The final ranking must be grounded in holdout metrics, not calibration metrics.

## Automatic Failure Signals

- The agent silently changes `maxiters`, `n_trials`, or objective across models.
- Any retired legacy basin-loading route appears anywhere in the workflow.
- One model is evaluated on a different split.
- The final ranking uses calibration performance instead of validation performance.

## Review Verdict

- Pass: the benchmark is fair, traceable, and ranked by comparable holdout metrics.
- Partial: both models run but fairness is not fully demonstrated.
- Fail: the comparison is not budget-fair, split-fair, or metric-fair.
