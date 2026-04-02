# P14. Simple Natural-Language Task Execution

## Scenario Metadata

- Scenario ID: `P14`
- Paper role: extended results
- Status: agent-eval dependent
- Independence: independent once one prompt set and scoring rubric are fixed

## Objective

Evaluate whether short natural-language requests map onto correct low-complexity tool sequences.

## Why It Matters for the Paper

This scenario provides evidence for the user-facing assistant layer while keeping task complexity low enough for transparent scoring.

## Preconditions

- a simple task set is defined
- a scoring rubric for tool-call accuracy is defined
- protocol discovery methods are available when prompt or resource discovery is part of the workflow

## Data Requirements

- a low-ambiguity task set
- minimal model-ready data fixtures

## Model Scope

- one representative model or a small candidate set

## Example User Requests

- "Find a suitable model for this basin and try one run."
- "Check the data first, then simulate."
- "List the available prompts before you continue."
- "Show the available model resources."
- "Summarize the result briefly."

## Core Tool Workflow

1. Use `list_mcp_surfaces` for discovery of tools, prompts, resources, and templates.
2. Use `find_model` for model resolution when the model is not fixed.
3. Use `inspect_hydro_data` before execution.
4. Use `run_simulation` for the final task step.
5. Score completion rate, call correctness, and intervention count.

## Expected Outputs

- task completion rate
- tool-call accuracy
- average number of corrective interventions

## Pass/Fail Criteria

- Pass: the agent completes the task with the expected tool family and correct stopping behavior.
- Fail: the agent skips readiness checks, uses the wrong tool family, or invents results.

## Tables and Figures

- Table: simple task execution performance

## Pseudocode Scaffold

```text
for task in simple_task_set:
    run agent on task
    capture actual tool chain
    compare against expected low-complexity workflow
aggregate completion and correction statistics
```
