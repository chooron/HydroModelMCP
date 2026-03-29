# Paper-to-Project Mapping

Source paper: "Ten strategies towards successful calibration of environmental models" by Juliane Mai.

## Strategy mapping

| Strategy | What the paper asks for | Current project mapping | Status |
| --- | --- | --- | --- |
| 1 | Use sensitivity analysis to identify parameters to calibrate | `run_sensitivity`, `sensitivity_analysis`, `init_calibration_setup` | Supported |
| 2 | Handle parameter constraints efficiently | Core helpers `pie_share_sampling` and `delta_method_sampling` exist, but they are not exposed through MCP calibration inputs | Gap |
| 3 | Transform data that span orders of magnitude when needed | `analyze_distribution_from_handle`, `compute_metrics`, `configure_objectives`, `init_calibration_setup` | Partial |
| 4 | Choose data for calibration and validation deliberately | `split_data`, `run_validation` | Supported |
| 5 | Choose an appropriate parameter sampling strategy | `generate_samples` plus algorithm-specific sampling inside calibration | Partial |
| 6 | Adjust parameter ranges to cover optima without wasting search budget | `get_model_parameters`, `diagnose_calibration`, `compute_diagnostics_full` | Partial |
| 7 | Choose objective functions that match model purpose | `configure_objectives`, `compute_metrics`, `calibrate_model`, `calibrate_multiobjective` | Supported |
| 8 | Choose the right calibration algorithm | `hydro://guides/algorithms`, `init_calibration_setup`, `calibrate_model` | Partial |
| 9 | Decide whether to calibrate multiple objectives | `calibrate_multiobjective` exists, but there is no exposed MCP tool for `diagnose_multiobjective` | Partial |
| 10 | Diagnose calibration performance after the run | `diagnose_calibration`, `compute_diagnostics_full` | Supported |

## Interpretation rules for this repository

- Treat Strategies 1, 4, 7, and 10 as the strongest parts of the current server.
- Treat Strategies 2, 6, 8, and 9 as places where the codebase has useful pieces but not a complete end-to-end MCP contract.
- Treat Strategy 3 as a decision layer. The server can recommend or compute around log-sensitive metrics, but it does not fully automate every transform choice.
- Treat Strategy 5 as partly supported because the MCP layer exposes generic sampling, while the actual optimization path hides most algorithm internals from clients.

## Practical guidance

- Recommend `run_sensitivity` before calibration whenever the user is uncertain about identifiability.
- Recommend `split_data` and `run_validation` whenever the user asks about robustness or deployment realism.
- Recommend `diagnose_calibration` and `compute_diagnostics_full` whenever the user asks whether a run is "good enough."
- Do not over-promise constrained sampling, stored-result browsing, or multi-objective diagnostics without calling out the current gap first.
