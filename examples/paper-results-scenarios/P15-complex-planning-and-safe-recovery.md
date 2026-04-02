# P15. Complex Planning and Safe Recovery

## Scenario Metadata

- Scenario ID: `P15`
- Paper role: extended results
- Status: agent-eval dependent
- Independence: independent once one multi-step task suite and one gold workflow set are fixed

## Objective

Evaluate whether a more capable agent can decompose multi-step tasks, stop on blocking conditions, and resume after missing information is supplied.

## Why It Matters for the Paper

This scenario supports stronger claims about safe orchestration rather than single-step command execution.

## Preconditions

- a multi-step benchmark suite is defined
- a gold workflow exists for each task
- safe-stop and recovery scoring rules are defined

## Data Requirements

- a small suite of multi-step tasks
- model-ready fixtures for each task branch

## Model Scope

- one representative comparison set or several task-specific models

## Example User Requests

- "Compare two models under the same calibration budget and recommend one."
- "Screen runnable models first, then benchmark them."
- "Retrieve the previous result and continue the diagnosis."
- "Stop if the data is not sufficient and tell me what is missing."
- "Resume after I provide the missing mapping or data."

## Core Tool Workflow

1. Use `list_mcp_surfaces` for discovery of tools, prompts, resources, and templates.
2. Resolve models and resources.
3. Use readiness checks before downstream execution.
4. Execute discovery, simulation, calibration, validation, and retrieval steps as required.
5. Stop immediately on blocking conditions.
6. Retry only after the missing user input is provided.

## Expected Outputs

- task completion rate
- workflow-plan accuracy
- safe-stop rate under blocking conditions
- recovery success rate after user correction

## Pass/Fail Criteria

- Pass: the agent follows the gold workflow closely, stops on blockers, and recovers correctly when possible.
- Fail: the agent continues after blockers, hallucinates missing steps, or cannot resume safely.

## Tables and Figures

- Figure: planned vs actual tool-chain diagram
- Table: recovery and safe-stop statistics

## Pseudocode Scaffold

```text
for task in complex_task_suite:
    run agent on task
    compare actual plan and calls against the gold workflow
    score safe-stop behavior and recovery behavior
aggregate planning and recovery statistics
```
