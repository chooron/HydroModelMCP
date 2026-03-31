# Current Service Gaps

## Result storage and retrieval are only partially dynamic

- Model and stored-result templates can now be listed and read dynamically through the protocol.
- Stored-result index resources still remain the safest discovery entry point before reading a concrete result template URI.
- There is still no dedicated MCP tool that fetches an arbitrary stored result ID independently of the resource layer.
- `run_ensemble_parameters` is still the only tool that currently writes to storage.

## Multi-objective diagnostics are only half exposed

- Core calibration code includes `diagnose_multiobjective`.
- The MCP tool layer exposes `calibrate_multiobjective`, but not a matching diagnosis tool.

## Constraint-aware calibration is not exposed end to end

- Core sampling code includes `pie_share_sampling` and `delta_method_sampling`.
- MCP calibration and sampling tools do not accept constraint schemas that would route into those helpers.

## Criteria coverage is narrower than HydroModelLibrary suggests

- `HydroModelLibrary` contains many criteria source files under `src/criteria/`.
- HydroModelMCP should only promise the currently validated objective set until each criterion is tested end to end inside the optimization path.

## Reproducibility and telemetry are still limited

- The tool sets `converged` from a fixed `best_objective >= 0.7` rule, which is not metric-aware.
- The diagnostics path often relies on placeholder or synthesized history instead of real optimization traces.

## HTTP configuration is inconsistent

- `start_http.jl` reads `JULIA_HTTP_HOST`, `JULIA_HTTP_PORT`, and `JULIA_HTTP_ALLOWED_ORIGINS`.
- `run_http_server` in `src/HydroModelMCP.jl` defaults to `MCP_HOST` and `MCP_PORT`.

## Test coverage is partial

- `test_simulation.jl` is present but commented out in `test/runtests.jl`.
- There is no dedicated validation, storage, or HTTP transport test file in the current suite.
