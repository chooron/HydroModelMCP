# Current Service Gaps

## Result storage and retrieval are only partially dynamic

- `hydro://models/{model_name}/...` templates are metadata only by design; model details are intentionally fetched through unified tools on demand.
- Stored-result templates are also declared as metadata.
- Exact stored-result resources for IDs already present at startup can be registered.
- Newly created stored results do not automatically appear as readable resources until the server rebuilds its resource registry.
- `Storage.load_result` is used by resource providers, but there is still no dedicated MCP tool that fetches an arbitrary stored result ID on demand.
- `run_ensemble_parameters` is still the only tool that currently writes to storage.

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
