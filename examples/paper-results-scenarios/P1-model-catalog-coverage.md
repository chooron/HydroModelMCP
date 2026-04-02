# P1. Model Catalog Coverage

## Scenario Metadata

- Scenario ID: `P1`
- Paper role: core results
- Status: ready now
- Independence: fully independent, no basin data required

## Objective

Verify that HydroModelMCP exposes a complete and structured discovery layer for supported hydrological models.

## Why It Matters for the Paper

This scenario supports an early Results claim that the platform offers a unified entry point for heterogeneous models instead of a fragmented collection of model-specific interfaces.

## Preconditions

- the MCP server is available
- model discovery tools are reachable
- the test loop can iterate across the current model catalog

## Data Requirements

- no hydrometeorological dataset is required

## Model Scope

- all models returned by `list_models`

## Example User Requests

- "List all models currently supported by the server."
- "Return summary information for every model."
- "Return parameter metadata for each model."
- "Return variable metadata for each model."
- "Report which models have incomplete metadata surfaces."

## Core Tool Workflow

1. Call `list_models`.
2. Iterate through the returned model names.
3. Call `get_model_info` for each model.
4. Call `get_model_parameters` for each model.
5. Call `get_model_variables` for each model.
6. Aggregate completeness and failure statistics.

## Expected Outputs

- model catalog table
- metadata completeness matrix
- success and failure counts by surface
- missing-field notes where relevant

## Pass/Fail Criteria

- Pass: a model returns structured info, parameter metadata, and variable metadata.
- Partial: a model is discoverable but one metadata surface is empty or weakly structured.
- Fail: discovery or metadata retrieval breaks for a target model.

## Tables and Figures

- Table: model catalog and metadata coverage
- Figure: coverage success rate by discovery surface

## Pseudocode Scaffold

```text
models = list_models()
for model in models:
    info = get_model_info(model)
    params = get_model_parameters(model)
    vars = get_model_variables(model)
    record completeness and errors
render coverage table and summary chart
```
