module Experts

using ..ModelContextProtocol: MCPPrompt, PromptArgument, PromptMessage, TextContent

export hydro_expert_prompt
export hydro_minimal_router_prompt

const hydro_expert_prompt = MCPPrompt(
    name = "hydrology_expert_review",
    title = "Hydrology Expert Review",
    description = "Structured guidance for hydrological model selection, simulation, calibration, and validation.",
    arguments = [
        PromptArgument(
            name = "task",
            description = "The hydrology task to perform or review.",
            required = true,
        ),
        PromptArgument(
            name = "context",
            description = "Optional basin, data, or model context.",
            required = false,
        ),
    ],
    messages = [
        PromptMessage(
            content = TextContent(
                text = """
You are a hydrology expert supporting model selection, simulation, calibration, and validation.
Focus on physical plausibility, data quality, parameter identifiability, and reproducible reasoning.

Task:
{task}{?context?

Context:
{context}}

Respond with:
1. Key assumptions and uncertainties.
2. A recommended workflow or tool sequence.
3. Important parameter, data, or diagnostic checks.
4. Risks that could invalidate the result.
""",
            ),
        ),
    ],
)

const hydro_minimal_router_prompt = MCPPrompt(
    name = "hydro_minimal_tool_router",
    title = "Hydro Minimal Tool Router",
    description = "Compact routing prompt for lightweight LLMs to pick the next HydroModelMCP tool call.",
    arguments = [
        PromptArgument(
            name = "goal",
            description = "User goal such as simulation, calibration, validation, or metrics.",
            required = true,
        ),
        PromptArgument(
            name = "model_hint",
            description = "Optional model hint from user input.",
            required = false,
        ),
        PromptArgument(
            name = "data_hint",
            description = "Optional data/source hint.",
            required = false,
        ),
    ],
    messages = [
        PromptMessage(
            content = TextContent(
                text = """
Route HydroModelMCP calls with minimal tokens.

Goal:
{goal}{?model_hint?

Model hint:
{model_hint}}{?data_hint?

Data hint:
{data_hint}}

Rules:
1. Return only the next 1-3 tool calls.
2. Prefer compact order: find_model -> inspect_hydro_data -> target workflow tool.
3. Use unified v2 payload shape: model + inputs.
4. For calibration intents, prefer auto_calibration_workflow before manual step chaining.
5. If uncertain, call list_mcp_surfaces or read hydro://guides/llm-quickstart.

Output format:
- next_tools: [tool names]
- reason: one sentence
- blocking_info: [] or list
""",
            ),
        ),
    ],
)

end # module Experts
