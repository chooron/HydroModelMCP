# Current Service Gaps

## Result storage and retrieval are incomplete

- `hydro://calibration/results/{result_id}` and `hydro://sensitivity/results/{result_id}` are declared as templates.
- Only `hydro://calibration/results` is actually registered as a resource.
- `Storage.load_result` exists in core storage code, but no MCP tool exposes it.
- `run_ensemble_parameters` is the only tool that currently writes to storage.

## Multi-objective diagnostics are only half exposed

- Core calibration code includes `diagnose_multiobjective`.
- The MCP tool layer exposes `calibrate_multiobjective`, but not a matching diagnosis tool.

## Constraint-aware calibration is not exposed end to end

- Core sampling code includes `pie_share_sampling` and `delta_method_sampling`.
- MCP calibration and sampling tools do not accept constraint schemas that would route into those helpers.

## Reproducibility and telemetry are weaker than the schemas imply

- `calibrate_model` accepts `random_seed` and `sampling_method`, but they are not fully wired through the optimization path.
- The tool sets `converged` from a fixed `best_objective >= 0.7` rule, which is not metric-aware.
- The diagnostics path often relies on placeholder or synthesized history instead of real optimization traces.

## HTTP configuration is inconsistent

- `start_http.jl` reads `JULIA_HTTP_HOST`, `JULIA_HTTP_PORT`, and `JULIA_HTTP_ALLOWED_ORIGINS`.
- `run_http_server` in `src/HydroModelMCP.jl` defaults to `MCP_HOST` and `MCP_PORT`.

## Test coverage is partial

- `test_simulation.jl` is present but commented out in `test/runtests.jl`.
- There is no dedicated validation, storage, or HTTP transport test file in the current suite.
