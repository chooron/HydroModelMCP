# HydroModelMCP Paper Results Scenario Design

## 1. Purpose

This file is now the index for the paper-oriented scenario catalog.
Detailed scenario cards are stored as independent files under `examples/paper-results-scenarios/` so each scenario can be expanded, reviewed, versioned, and prompted separately.

This structure is intended for two downstream uses:

- selective prompting of another LLM with one scenario at a time
- clean separation between core-paper scenarios and second-wave or appendix scenarios

## 2. Global Rules

All scenario files should follow these rules:

1. Use only currently exposed HydroModelMCP capabilities.
2. Keep readiness checks explicit before simulation, calibration, validation, or metrics.
3. Prefer the currently exposed MCP discovery surface in the running server.
4. Treat `inspect_hydro_data` outputs as inspection evidence, not as a guaranteed native readiness field.
5. Keep each scenario independently executable at the design level.
6. Keep each scenario independently promptable for another LLM.

## 3. Standard Output Package

When another LLM expands one scenario file into an executable plan, it should return:

1. Scenario objective
2. Why the scenario matters for the paper
3. Preconditions
4. Data requirements
5. Model scope
6. Example user requests
7. Tool workflow
8. Fixture or placeholder specification
9. Expected outputs
10. Pass/fail criteria
11. Logging fields
12. Tables and figures to generate
13. Pseudocode or test-case scaffold

## 4. Unified Logging Schema

These are recommended test-log fields, not guaranteed native MCP response fields:

- `scenario_id`
- `scenario_name`
- `paper_section`
- `client_mode`
- `storage_backend`
- `model_name`
- `basin_id`
- `input_source_type`
- `tool_chain`
- `start_time`
- `end_time`
- `success`
- `failure_type`
- `warnings`
- `inspect_result`
- `inference_report`
- `result_id`
- `metrics`
- `artifacts`
- `notes`

## 5. Directory Layout

- `examples/paper-results-scenario-design.md`: this index
- `examples/paper-results-scenarios/README.md`: quick navigation guide
- `examples/paper-results-scenarios/P1-*.md` to `P15-*.md`: one scenario per file

## 6. Scenario Index

| ID | Scenario Name | File | Status | Phase |
| --- | --- | --- | --- | --- |
| P1 | Model Catalog Coverage | `examples/paper-results-scenarios/P1-model-catalog-coverage.md` | Ready now | Core |
| P2 | Minimal Executability Benchmark | `examples/paper-results-scenarios/P2-minimal-executability-benchmark.md` | Ready now | Core |
| P3 | Single-Basin Cross-Model Benchmark | `examples/paper-results-scenarios/P3-single-basin-cross-model-benchmark.md` | Ready now | Core |
| P4 | Multi-Basin Robustness Benchmark | `examples/paper-results-scenarios/P4-multi-basin-robustness-benchmark.md` | Data-dependent | Second wave |
| P5 | Calibration-Validation Closed Loop | `examples/paper-results-scenarios/P5-calibration-validation-closed-loop.md` | Ready now | Core |
| P6 | Calibration Bootstrap and Fair Budget Comparison | `examples/paper-results-scenarios/P6-calibration-bootstrap-and-fair-budget-comparison.md` | Ready now | Second wave |
| P7 | Sensitivity to Multiobjective Workflow | `examples/paper-results-scenarios/P7-sensitivity-to-multiobjective-workflow.md` | Ready now | Second wave |
| P8 | Ensemble Sampling and Result Persistence | `examples/paper-results-scenarios/P8-ensemble-sampling-and-result-persistence.md` | Ready now | Second wave |
| P9 | Heterogeneous Input and Auto-Inference Robustness | `examples/paper-results-scenarios/P9-heterogeneous-input-and-auto-inference-robustness.md` | Ready now | Core |
| P10 | Strict Readiness Gating and Manual Mapping Recovery | `examples/paper-results-scenarios/P10-strict-readiness-gating-and-manual-mapping-recovery.md` | Cleaned up | Second wave |
| P11 | Session Continuity with Data Handles | `examples/paper-results-scenarios/P11-session-continuity-with-data-handles.md` | Cleaned up | Second wave |
| P12 | Replay and Reproducibility Audit | `examples/paper-results-scenarios/P12-replay-and-reproducibility-audit.md` | Ready now | Core |
| P14 | Simple Natural-Language Task Execution | `examples/paper-results-scenarios/P14-simple-natural-language-task-execution.md` | Agent-eval dependent | Second wave |
| P15 | Complex Planning and Safe Recovery | `examples/paper-results-scenarios/P15-complex-planning-and-safe-recovery.md` | Agent-eval dependent | Second wave |

## 7. Recommended Development Order

### First expansion wave

Only expand these into executable test-development tasks first:

1. P1
2. P2
3. P3
4. P5
5. P9

### Additional ready scenarios

These align well with the current capability surface, but can wait until after the first wave:

1. P12

### Second-wave scenarios

These are worth keeping in the catalog, but are more data-, harness-, or scoring-dependent:

1. P4
2. P6
3. P7
4. P8
5. P10
6. P11
7. P14
8. P15

## 8. Discovery Wording Standard

When a scenario needs prompt/resource/template discovery, write it as:

- `list_mcp_surfaces` for discovery of tools, prompts, resources, and templates

Do not assume direct protocol discovery methods are available in the current server path.

## 9. Mapping and Safety Wording Standard

For ambiguous inputs:

- use `inspect_hydro_data` as the pre-check
- use `options.strict_infer=true` to force safe rejection when needed
- use `options.input_mapping` as the documented manual recovery path

Do not assume a recovery workflow that depends on undocumented custom mapping interfaces.

## 10. Master Prompt Template

Use this when asking another LLM to expand one scenario file into an executable paper-oriented plan:

```text
Generate an executable paper-oriented evaluation plan for HydroModelMCP using the scenario file [examples/paper-results-scenarios/PX-...md].

Output the following sections in English:
1. Scenario objective
2. Why this scenario matters for the paper
3. Preconditions
4. Data requirements
5. Model scope
6. At least 5 example user requests
7. Tool workflow
8. Fixture or placeholder specification
9. Expected outputs
10. Pass/fail criteria
11. Logging fields
12. Tables and figures to generate
13. Executable pseudocode or a test-case scaffold

Requirements:
- use only currently exposed HydroModelMCP capabilities
- do not invent tools, prompts, resources, or recovery interfaces
- keep the output structured and implementation-ready
- explicitly note any assumptions or skipped branches
```
