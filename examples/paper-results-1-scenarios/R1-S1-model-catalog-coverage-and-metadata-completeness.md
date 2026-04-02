# R1-S1 Model Catalog Coverage and Metadata Completeness

## Scenario Metadata

- Scenario ID: `R1-S1`
- Paper role: discovery baseline
- Recommended skill: `hydromodel-mcp-results-1-tests`
- Source pack: `examples/paper-results-r1-basic-scenarios.md`
- Status: ready for test-script design and execution

## Objective

Verify the unified discovery layer by enumerating all currently listed models and retrieving structured metadata through documented model query tools.

## Why It Matters for Results 1

Results 1 begins with the claim that HydroModelMCP is a unified interface rather than a collection of disconnected model-specific entry points. This scenario establishes that baseline without relying on simulation or calibration.

## What It Proves

It proves that the server can expose one runtime model catalog and can resolve per-model information, parameter metadata, and variable metadata through documented discovery tools.

## Preconditions

- The server is reachable.
- The live tool surface includes `list_models`, `get_model_info`, `get_model_parameters`, and `get_model_variables`.
- No hydrological data fixture is required.

## Model Scope

All models returned by `list_models` at runtime.

## Data Scope

No basin data. No simulation input. No calibration input.

## Example User Requests

- List every model currently exposed by HydroModelMCP and show whether each one has complete metadata.
- For each discoverable model, return its model information, parameter definitions, and variable definitions.
- Build a coverage matrix for the model catalog and highlight any metadata gaps.
- Report which models expose all required discovery surfaces and which ones fail on specific metadata calls.
- Check the advertised MCP surfaces quickly, then verify real model metadata through the actual model tools.

## Required Tool Workflow

1. Optionally call `list_mcp_surfaces` once as a quick surface-presence check and record the result.
2. Call `list_models` and record the returned catalog exactly as exposed by the server.
3. For each model from `list_models`, call `get_model_info`.
4. For each model from `list_models`, call `get_model_parameters`.
5. For each model from `list_models`, call `get_model_variables`.
6. Build one per-model record containing success state, returned metadata presence, and failure notes for each required discovery call.
7. Generate the model coverage matrix and the metadata completeness summary from those records.

## Input Fixture Specification

- Fixture type: runtime catalog snapshot only.
- Required runtime evidence: one raw `list_models` response and one raw response for each per-model metadata tool.
- No hydrological dataset is allowed in this scenario.

## Expected Outputs

- Model coverage matrix with one row per model and one column each for `get_model_info`, `get_model_parameters`, and `get_model_variables`.
- Metadata completeness summary with counts of complete, partial, and failed models.
- Per-model failure notes that preserve exact tool-level failure branches.
- One catalog artifact that can be cited in the Results 1 discovery subsection.

## Pass / Partial / Fail Criteria

Pass: `list_models` succeeds, every listed model is queried with all three metadata tools, the coverage matrix is complete, and every tool outcome is logged.

Partial: `list_models` succeeds, but one or more models return incomplete or failed metadata retrieval while the failures are fully logged and summarized.

Fail: `list_models` fails, one or more listed models are not queried with the required metadata tools, or the coverage and failure outputs are missing.

## Required Logs

- One batch log entry with `model_name=catalog_batch` for the catalog retrieval.
- One per-model log entry with the ordered discovery `tool_chain`.
- `failure_type` for every failed metadata branch.
- `notes` field stating whether `list_mcp_surfaces` was used only as a quick check.

## Tables and Figures to Generate

- Table R1-1: Model catalog coverage matrix with discovery surface completeness by model.
- Table R1-2: Metadata completeness summary with counts and percentages.
- Figure R1-1: Discovery surface success rate by tool.
- Appendix log table: Per-model failure notes for any missing or errored metadata branch.

## Notes for Weaker LLM Executors

- Do not replace `list_models` with `list_mcp_surfaces`; the latter is only a quick presence check.
- Do not attempt simulation or calibration in this scenario.
- If one metadata tool fails for a model, keep querying the remaining required metadata tools unless the server itself becomes unavailable.
- Use the shortest valid tool chain: `list_models` plus the three per-model metadata calls.
