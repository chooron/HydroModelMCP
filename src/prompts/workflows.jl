module Workflows

using ..ModelContextProtocol: MCPPrompt, PromptArgument, PromptMessage, TextContent

export calibration_workflow_prompt, runoff_workspace_prompt, result_review_prompt, minimal_tool_routing_prompt

const runoff_workspace_prompt = MCPPrompt(
    name = "runoff_workspace_workflow",
    title = "Runoff Workspace Workflow",
    description = "Guide an assistant through a single workspace runoff run using HydroModelMCP tools.",
    arguments = [
        PromptArgument(name = "model", description = "Resolved hydrological model name.", required = true),
        PromptArgument(name = "forcing_path", description = "Optional forcing CSV path.", required = false),
        PromptArgument(name = "observed_path", description = "Optional observed discharge path.", required = false),
    ],
    messages = [
        PromptMessage(
            content = TextContent(
                text = """
Operate HydroModelMCP for one runoff simulation in the current workspace.

Resolved model:
{model}{?forcing_path?

Forcing path:
{forcing_path}}{?observed_path?

Observed discharge path:
{observed_path}}

Follow this sequence:
1. Call `get_model_info` and `get_model_parameters` for the resolved model.
2. If the request refers to Caravan or to a Caravan sub-dataset such as `camels`, do not inspect `./data` for CSV files first. Use a `caravan` source with explicit `dataset_name`/`source_dataset` plus `gauge_id`/`gage_id`, and omit `dataset_root`/`netcdf_root` only when `CARAVAN_DATASET_ROOT` or `CARAVAN_NETCDF_ROOT` is configured.
3. If the requested Caravan basin is not found, report that directly and do not fall back to workspace CSV discovery.
4. Otherwise, if no forcing path was provided, inspect `./data` with `list_workspace_files` and choose the safest forcing file.
5. Call `run_simulation` with unified v2 request fields: `model`, `inputs`, optional `output`, and optional `options`.
6. If observed discharge exists, call `compute_metrics` with explicit `simulated` + `observed` sources (or rely on same-session auto-inference only as fallback).
7. End by calling `clear_session_cache` after outputs are written.

Respond with:
1. The selected inputs and why they were used.
2. The executed tool sequence.
3. Key runtime warnings or hydrological anomalies.
4. The files written under `./result`.
""",
            ),
        ),
    ],
)

const calibration_workflow_prompt = MCPPrompt(
    name = "calibration_workflow_plan",
    title = "Calibration Workflow Plan",
    description = "Guide an assistant through a calibration workflow using HydroModelMCP tools.",
    arguments = [
        PromptArgument(name = "model", description = "Target hydrological model.", required = true),
        PromptArgument(name = "goal", description = "Calibration goal such as general_fit or low_flows.", required = false),
        PromptArgument(name = "data_context", description = "Optional data or basin context.", required = false),
    ],
    messages = [
        PromptMessage(
            content = TextContent(
                text = """
Design a HydroModelMCP calibration workflow.

Model:
{model}{?goal?

Goal:
{goal}}{?data_context?

Data context:
{data_context}}

Use this tool-oriented order unless the evidence suggests otherwise:
1. `get_model_info`, `get_model_parameters`, and `get_model_variables`.
2. Prepare `inputs` for the unified v2 protocol (forcing plus observation; optional runtime and parameters).
   - Parameter sources may be inline objects, json/csv/data_handle/calibration_result descriptors, or same-session calibrated parameters.
   - If the user names Caravan or a Caravan sub-dataset such as `camels`, require `source_type=caravan` with explicit `dataset_name`/`source_dataset` and `gauge_id`/`gage_id`; do not search local CSV files first.
   - If the requested Caravan basin is missing, stop with a not-found error instead of falling back to any CSV guess.
3. `sensitivity_analysis` or `run_sensitivity` to screen insensitive parameters.
4. `configure_objectives` and `init_calibration_setup` to shape the experiment.
5. `calibrate_model` or `calibrate_multiobjective`.
6. `diagnose_calibration` or `compute_diagnostics_full` to assess convergence and identifiability.
7. Prefer passing `calibration_result=<calibrate_model result>` explicitly to `diagnose_calibration`; use same-session auto-inference only as fallback.

Respond with:
1. Recommended tool sequence.
2. Metric and algorithm choices with justification.
3. Parameters that should likely be fixed, screened, or widened.
4. Risks to reproducibility or validity.
""",
            ),
        ),
    ],
)

const result_review_prompt = MCPPrompt(
    name = "hydrology_result_review",
    title = "Hydrology Result Review",
    description = "Review a simulation, calibration, or validation result for quality and next actions.",
    arguments = [
        PromptArgument(name = "workflow", description = "Workflow type such as simulation, calibration, or validation.", required = true),
        PromptArgument(name = "result_summary", description = "Compact summary of the produced result.", required = true),
        PromptArgument(name = "context", description = "Optional context such as warnings, basin traits, or data limitations.", required = false),
    ],
    messages = [
        PromptMessage(
            content = TextContent(
                text = """
Review the following HydroModelMCP output.

Workflow:
{workflow}

Result summary:
{result_summary}{?context?

Additional context:
{context}}

Respond with:
1. What looks credible.
2. What warnings or failure modes matter most.
3. Which follow-up tool call would reduce uncertainty the fastest.
4. Whether the result is exploratory, decision-supporting, or not yet trustworthy.
""",
            ),
        ),
    ],
)

const minimal_tool_routing_prompt = MCPPrompt(
    name = "hydro_minimal_workflow_router",
    title = "Hydro Minimal Workflow Router",
    description = "Token-efficient workflow prompt for lightweight models.",
    arguments = [
        PromptArgument(name = "intent", description = "simulation, calibration, validation, metrics, or discovery", required = true),
        PromptArgument(name = "user_goal", description = "Short user goal sentence.", required = true),
    ],
    messages = [
        PromptMessage(
            content = TextContent(
                text = """
You are routing HydroModelMCP calls with minimal tokens.

Intent:
{intent}

User goal:
{user_goal}

Do this:
1. Read hydro://workflows/{intent}.
2. Return next tools (1-3) and minimal required fields.
3. Avoid long explanations.
4. For calibration intent, prefer auto_calibration_workflow first.

Output keys only:
- next_tools
- required_fields
- one_risk
""",
            ),
        ),
    ],
)

end # module Workflows
