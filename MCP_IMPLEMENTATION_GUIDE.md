# MCP Implementation Guide for HydroModelMCP

## ✅ Current Implementation Status

Your implementation is **correct and follows ModelContextProtocol.jl v0.4.0 patterns**. The tools are properly structured and will work with MCP clients.

## 🔧 Recommended Improvements

### 1. Standardize Error Content Format

**Current Issue**: Error returns use raw Dict instead of Content types.

**Fix**: Update error handlers to use `TextContent`:

```julia
# ❌ Current (works but inconsistent):
catch e
    return CallToolResult(
        content = [Dict("type" => "text", "text" => "Error: $e")],
        is_error = true
    )
end

# ✅ Recommended (consistent):
catch e
    err_msg = sprint(showerror, e, catch_backtrace())
    return CallToolResult(
        content = [TextContent(text = "Error: $err_msg")],
        is_error = true
    )
end
```

### 2. Add Resources for Data Access

Resources allow clients to query data without invoking tools. Consider adding:

```julia
# Example: Model catalog as a resource
model_catalog_resource = MCPResource(
    uri = "hydro://models/catalog",
    name = "Model Catalog",
    description = "Complete catalog of available hydrological models",
    mime_type = "application/json",
    data_provider = () -> begin
        models = Discovery.list_models()
        return TextResourceContents(
            uri = URI("hydro://models/catalog"),
            mime_type = "application/json",
            text = JSON3.write(models)
        )
    end
)

# Example: Calibration results storage
calibration_results_resource = MCPResource(
    uri = "hydro://calibration/results/{id}",
    name = "Calibration Results",
    description = "Access stored calibration results by ID",
    mime_type = "application/json",
    data_provider = (id) -> begin
        # Load from storage
        result = load_calibration_result(id)
        return TextResourceContents(
            uri = URI("hydro://calibration/results/$id"),
            mime_type = "application/json",
            text = JSON3.write(result)
        )
    end
)
```

### 3. Add Resource Templates for Dynamic URIs

For parameterized resources:

```julia
# In HydroModelMCP.jl
ALL_RESOURCE_TEMPLATES = [
    ResourceTemplate(
        name = "model_info",
        uri_template = "hydro://models/{model_name}/info",
        mime_type = "application/json",
        description = "Get detailed information about a specific model"
    ),
    ResourceTemplate(
        name = "calibration_result",
        uri_template = "hydro://calibration/results/{result_id}",
        mime_type = "application/json",
        description = "Access calibration results by ID"
    )
]

# Update server creation:
server = mcp_server(
    name = "HydroModel-Agent-Interface",
    version = "0.1.0",
    tools = ALL_TOOLS,
    resources = ALL_RESOURCES,
    resource_templates = ALL_RESOURCE_TEMPLATES
)
```

### 4. Add Prompts for Common Workflows

Prompts guide LLMs through complex workflows:

```julia
# Example: Calibration workflow prompt
calibration_workflow_prompt = MCPPrompt(
    name = "calibration_workflow",
    description = "Complete workflow for model calibration",
    arguments = [
        PromptArgument(
            name = "model_name",
            description = "Name of the hydrological model to calibrate",
            required = true
        ),
        PromptArgument(
            name = "goal",
            description = "Calibration goal (e.g., 'general_fit', 'peak_flows')",
            required = true
        )
    ],
    messages = [
        PromptMessage(
            content = TextContent(text = """
            You are calibrating the {{model_name}} model with goal: {{goal}}.

            Follow these steps:
            1. Call init_calibration_setup to analyze sensitivity and recommend configuration
            2. Review the recommended configuration
            3. Call calibrate_model with the recommended settings
            4. Call diagnose_calibration to check results quality
            5. If diagnosis suggests improvements, iterate with adjusted parameters

            Always explain your decisions and the reasoning behind parameter choices.
            """),
            role = user
        )
    ]
)
```

### 5. Improve Tool Descriptions with Examples

Add usage examples to tool descriptions:

```julia
calibrate_tool = MCPTool(
    name = "calibrate_model",
    description = """
    Execute hydrological model parameter calibration (single-objective optimization).

    Supports multiple optimization algorithms (BBO/DE/PSO/etc), fixed parameters,
    and custom parameter ranges. Returns optimal parameters, objective function value,
    and convergence information.

    Example usage:
    {
        "model_name": "hbv",
        "forcing": {"source_type": "csv", "path": "data/forcing.csv"},
        "observation": {"source_type": "csv", "path": "data/obs.csv"},
        "obs_column": "discharge",
        "objective": "KGE",
        "algorithm": "BBO",
        "maxiters": 1000
    }
    """,
    # ... rest of definition
)
```

### 6. Add Input Validation

Add validation before processing:

```julia
handler = function(params)
    # Validate required fields
    if !haskey(params, "model_name") || isempty(params["model_name"])
        return CallToolResult(
            content = [TextContent(text = "Error: model_name is required and cannot be empty")],
            is_error = true
        )
    end

    # Validate enum values
    if haskey(params, "algorithm")
        valid_algorithms = ["BBO", "DE", "PSO", "CMAES", "ECA"]
        if !(params["algorithm"] in valid_algorithms)
            return CallToolResult(
                content = [TextContent(text = "Error: algorithm must be one of: $(join(valid_algorithms, ", "))")],
                is_error = true
            )
        end
    end

    try
        # ... actual processing
    catch e
        # ... error handling
    end
end
```

### 7. Add Progress Reporting for Long Operations

For calibration and sensitivity analysis:

```julia
handler = function(params)
    try
        # Create progress token
        progress_token = string(uuid4())

        # Report initial progress
        # (Note: This requires server-side progress notification support)

        # Run calibration with progress updates
        result = Calibration.calibrate_model(
            params["model_name"],
            forcing_nt,
            obs_vec;
            progress_callback = (iter, obj_val) -> begin
                # Report progress
                # send_progress(progress_token, iter, maxiters, "Iteration $iter: $obj_val")
            end,
            # ... other params
        )

        return TextContent(text = JSON3.write(result))
    catch e
        # ... error handling
    end
end
```

## 📚 Resources vs Tools - When to Use What

### Use **Tools** when:
- The operation modifies state (calibration, simulation)
- The operation requires computation (sensitivity analysis)
- The operation has side effects (saving results)
- Parameters vary significantly between calls

### Use **Resources** when:
- Data is relatively static (model catalog, parameter bounds)
- Clients need to query/browse data (list of models)
- Data should be cacheable
- You want to support subscriptions for updates

## 🎯 Priority Recommendations

1. **High Priority**: Fix error content format (quick win, improves consistency)
2. **Medium Priority**: Add resources for model catalog and results storage
3. **Low Priority**: Add prompts for common workflows (nice to have)

## 📖 Official Documentation

Since the official website is not accessible, refer to the package source code:
- Types: `D:\Julia\packages\packages\ModelContextProtocol\moOOc\src\types.jl`
- Examples: Check the package's test files or examples directory

## ✨ Your Implementation is Solid

Your current implementation follows best practices and will work correctly with MCP clients. The suggestions above are enhancements, not fixes for broken code.
