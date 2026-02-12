using ModelContextProtocol
using JSON3
using URIs

# Template definitions
model_info_template = ResourceTemplate(
    name = "model_info",
    uri_template = "hydro://models/{model_name}/info",
    mime_type = "application/json",
    description = "Detailed information about a specific hydrological model"
)

model_params_template = ResourceTemplate(
    name = "model_parameters",
    uri_template = "hydro://models/{model_name}/parameters",
    mime_type = "application/json",
    description = "Parameter bounds and descriptions for a specific model"
)

model_vars_template = ResourceTemplate(
    name = "model_variables",
    uri_template = "hydro://models/{model_name}/variables",
    mime_type = "application/json",
    description = "Input/output/state variables for a specific model"
)

calibration_result_template = ResourceTemplate(
    name = "calibration_result",
    uri_template = "hydro://calibration/results/{result_id}",
    mime_type = "application/json",
    description = "Retrieve a specific calibration result by ID"
)

sensitivity_result_template = ResourceTemplate(
    name = "sensitivity_result",
    uri_template = "hydro://sensitivity/results/{result_id}",
    mime_type = "application/json",
    description = "Retrieve a specific sensitivity analysis result by ID"
)

ALL_RESOURCE_TEMPLATES = [
    model_info_template,
    model_params_template,
    model_vars_template,
    calibration_result_template,
    sensitivity_result_template
]

# URI Parsing Functions
function parse_uri_template(uri_string::String, template::String)
    """Parse URI against template and extract parameters.

    Example:
        parse_uri_template("hydro://models/hbv/info", "hydro://models/{model_name}/info")
        => Dict("model_name" => "hbv")
    """
    # Split both into parts
    uri_parts = split(uri_string, "/")
    template_parts = split(template, "/")

    if length(uri_parts) != length(template_parts)
        return nothing
    end

    params = Dict{String,String}()
    for (uri_part, template_part) in zip(uri_parts, template_parts)
        if startswith(template_part, "{") && endswith(template_part, "}")
            # Extract parameter name
            param_name = template_part[2:end-1]
            params[param_name] = uri_part
        elseif uri_part != template_part
            # Mismatch in static parts
            return nothing
        end
    end

    return params
end

function create_dynamic_resources(storage_backend)
    """Create dynamic resources that handle parameterized URIs."""

    resources = MCPResource[]

    # Resource handler for hydro://models/{model_name}/info
    push!(resources, MCPResource(
        uri = "hydro://models/*/info",
        name = "Model Info (Dynamic)",
        description = "Dynamic model information resource",
        mime_type = "application/json",
        data_provider = (uri_string) -> begin
            params = parse_uri_template(uri_string, "hydro://models/{model_name}/info")
            if isnothing(params)
                error("Invalid URI format")
            end

            model_name = params["model_name"]
            info = Discovery.get_model_info(model_name)

            return TextResourceContents(
                uri = URI(uri_string),
                mime_type = "application/json",
                text = JSON3.write(info)
            )
        end
    ))

    # Resource handler for hydro://models/{model_name}/parameters
    push!(resources, MCPResource(
        uri = "hydro://models/*/parameters",
        name = "Model Parameters (Dynamic)",
        description = "Dynamic model parameters resource",
        mime_type = "application/json",
        data_provider = (uri_string) -> begin
            params = parse_uri_template(uri_string, "hydro://models/{model_name}/parameters")
            if isnothing(params)
                error("Invalid URI format")
            end

            model_name = params["model_name"]
            param_details = Discovery.get_parameters_detail(model_name)

            return TextResourceContents(
                uri = URI(uri_string),
                mime_type = "application/json",
                text = JSON3.write(Dict("model" => model_name, "parameters" => param_details))
            )
        end
    ))

    # Resource handler for hydro://models/{model_name}/variables
    push!(resources, MCPResource(
        uri = "hydro://models/*/variables",
        name = "Model Variables (Dynamic)",
        description = "Dynamic model variables resource",
        mime_type = "application/json",
        data_provider = (uri_string) -> begin
            params = parse_uri_template(uri_string, "hydro://models/{model_name}/variables")
            if isnothing(params)
                error("Invalid URI format")
            end

            model_name = params["model_name"]
            var_details = Discovery.get_variables_detail(model_name)

            return TextResourceContents(
                uri = URI(uri_string),
                mime_type = "application/json",
                text = JSON3.write(Dict("model" => model_name, "variables" => var_details))
            )
        end
    ))

    # Resource handler for hydro://calibration/results/{result_id}
    push!(resources, MCPResource(
        uri = "hydro://calibration/results/*",
        name = "Calibration Result (Dynamic)",
        description = "Dynamic calibration result resource",
        mime_type = "application/json",
        data_provider = (uri_string) -> begin
            params = parse_uri_template(uri_string, "hydro://calibration/results/{result_id}")
            if isnothing(params)
                error("Invalid URI format")
            end

            result_id = params["result_id"]
            result = Storage.load_result(storage_backend, "calibration", result_id)

            return TextResourceContents(
                uri = URI(uri_string),
                mime_type = "application/json",
                text = JSON3.write(result)
            )
        end
    ))

    # Resource handler for hydro://sensitivity/results/{result_id}
    push!(resources, MCPResource(
        uri = "hydro://sensitivity/results/*",
        name = "Sensitivity Result (Dynamic)",
        description = "Dynamic sensitivity result resource",
        mime_type = "application/json",
        data_provider = (uri_string) -> begin
            params = parse_uri_template(uri_string, "hydro://sensitivity/results/{result_id}")
            if isnothing(params)
                error("Invalid URI format")
            end

            result_id = params["result_id"]
            result = Storage.load_result(storage_backend, "sensitivity", result_id)

            return TextResourceContents(
                uri = URI(uri_string),
                mime_type = "application/json",
                text = JSON3.write(result)
            )
        end
    ))

    return resources
end

export ALL_RESOURCE_TEMPLATES, create_dynamic_resources, parse_uri_template
