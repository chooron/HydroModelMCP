"""
Data loading tools for HydroModelMCP.

Provides MCP tools for loading data into Julia and returning handles.
"""

using ModelContextProtocol
using JSON3
using NPZ
using Dates
using Statistics

"""
    load_camels_data_tool

MCP tool for loading CAMELS dataset into Julia memory.

Returns a data handle and metadata instead of raw arrays.
"""
const load_camels_data_tool = MCPTool(
    name = "load_camels_data",
    description = """
    Load CAMELS dataset for a specific gage into Julia memory.

    Returns a data handle (not raw data) and metadata for Python orchestration.
    The actual time series data stays in Julia to avoid large data transmission.

    Use this tool at the start of calibration workflows to load observation data.
    """,
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "dataset_path" => Dict{String,Any}(
                "type" => "string",
                "description" => "Path to CAMELS NPZ dataset file"
            ),
            "gage_id" => Dict{String,Any}(
                "type" => "integer",
                "description" => "USGS gage ID (e.g., 5362000)"
            ),
            "variable" => Dict{String,Any}(
                "type" => "string",
                "description" => "Variable to load (default: streamflow)",
                "default" => "streamflow"
            )
        ),
        "required" => ["dataset_path", "gage_id"]
    ),
    handler = function(params)
        # Validate required parameters
        validation_error = validate_required_params(params, ["dataset_path", "gage_id"])
        if !isnothing(validation_error)
            return create_error_response(validation_error)
        end

        try
            dataset_path = params["dataset_path"]
            gage_id = params["gage_id"]
            variable = get(params, "variable", "streamflow")

            # Load NPZ file
            data = npzread(dataset_path)

            # Find catchment index
            gage_ids = data["gage_ids"]
            catchment_idx = findfirst(==(gage_id), gage_ids)

            if isnothing(catchment_idx)
                return create_error_response("Gage $gage_id not found in dataset")
            end

            # Extract streamflow data (target[:, :, 1] is streamflow in ft³/s)
            target_data = data["target"][catchment_idx, :, 1]

            # Get catchment area for unit conversion (attribute index 13 in Julia, 1-indexed)
            area_km2 = data["attributes"][catchment_idx, 13]

            # Convert from ft³/s to mm/day
            # Formula: flow_mm_day = flow_ft3_s * (10^3) * 0.0283168 * 3600 * 24 / (area * 10^6)
            target_data_mm = target_data .* (1e3) .* 0.0283168 .* 3600 .* 24 ./ (area_km2 .* 1e6)

            # Filter out NaN values
            valid_mask = .!isnan.(target_data_mm)
            valid_values = target_data_mm[valid_mask]

            # Generate timestamps (CAMELS data: 1980-10-01 to 2014-09-30)
            start_date = Date(1980, 10, 1)
            n_timesteps = length(target_data)
            dates = [start_date + Day(i-1) for i in 1:n_timesteps]
            valid_dates = dates[valid_mask]

            # Generate data handle
            handle = "camels_$(gage_id)_$(variable)"

            # Store data in Julia memory
            store_data(handle, valid_values)

            # Also store dates if needed for future operations
            store_data("$(handle)_dates", valid_dates)

            # Compute metadata
            positive_values = valid_values[valid_values .> 0]
            magnitude_ratio = if isempty(positive_values)
                1.0
            else
                maximum(positive_values) / minimum(positive_values)
            end

            metadata = Dict{String,Any}(
                "name" => "$(variable)_$(gage_id)",
                "n_observations" => length(valid_values),
                "start_date" => string(valid_dates[1]),
                "end_date" => string(valid_dates[end]),
                "units" => "mm/day",
                "mean" => mean(valid_values),
                "min" => minimum(valid_values),
                "max" => maximum(valid_values),
                "std" => std(valid_values),
                "magnitude_ratio" => magnitude_ratio,
                "needs_log_transform" => magnitude_ratio > 100,
                "has_missing_data" => any(.!valid_mask)
            )

            result = Dict{String,Any}(
                "data_handle" => handle,
                "metadata" => metadata
            )

            return TextContent(text = JSON3.write(result))
        catch e
            return create_error_response(e)
        end
    end
)


"""
    analyze_distribution_from_handle_tool

MCP tool for analyzing data distribution using a data handle.
"""
const analyze_distribution_from_handle_tool = MCPTool(
    name = "analyze_distribution_from_handle",
    description = """
    Analyze time series data distribution using a data handle.

    This tool accesses data stored in Julia memory via handle reference,
    avoiding the need to transmit large arrays from Python.
    """,
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "data_handle" => Dict{String,Any}(
                "type" => "string",
                "description" => "Julia data handle ID (e.g., 'camels_5362000_streamflow')"
            )
        ),
        "required" => ["data_handle"]
    ),
    handler = function(params)
        # Validate required parameters
        validation_error = validate_required_params(params, ["data_handle"])
        if !isnothing(validation_error)
            return create_error_response(validation_error)
        end

        try
            data_handle = params["data_handle"]

            # Retrieve data from Julia memory
            values = get_data(data_handle)

            # Compute statistics
            positive_values = values[values .> 0]
            magnitude_ratio = if isempty(positive_values)
                1.0
            else
                maximum(positive_values) / minimum(positive_values)
            end

            # Try to get dates for record length calculation
            dates_handle = "$(data_handle)_dates"
            record_length_years = if has_data(dates_handle)
                dates = get_data(dates_handle)
                (dates[end] - dates[1]).value / 365.25
            else
                length(values) / 365.0
            end

            result = Dict{String,Any}(
                "magnitude_ratio" => magnitude_ratio,
                "needs_log_transform" => magnitude_ratio > 100,
                "record_length_years" => record_length_years,
                "has_missing_data" => false,  # Already filtered
                "min_value" => minimum(values),
                "max_value" => maximum(values),
                "mean_value" => mean(values),
                "num_observations" => length(values)
            )

            return TextContent(text = JSON3.write(result))
        catch e
            return create_error_response(e)
        end
    end
)

export load_camels_data_tool, analyze_distribution_from_handle_tool
