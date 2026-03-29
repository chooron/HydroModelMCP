module Experts

using ..ModelContextProtocol: MCPPrompt, PromptArgument, PromptMessage, TextContent

export hydro_expert_prompt

const hydro_expert_prompt = MCPPrompt(
    name = "hydrology_expert_review",
    description = "Structured guidance for hydrological model selection, simulation, calibration, and validation.",
    arguments = [
        PromptArgument(
            name = "task",
            description = "The hydrology task to perform or review.",
            required = true
        ),
        PromptArgument(
            name = "context",
            description = "Optional basin, data, or model context.",
            required = false
        )
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
"""
            )
        )
    ]
)

end # module Experts
