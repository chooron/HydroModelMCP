"""
Data loading tools for HydroModelMCP.
"""

using CSV
using DataFrames
using Dates
using JSON3
using ModelContextProtocol
using NPZ
using Statistics

const load_camels_data_tool = MCPTool(
    name = "load_camels_data",
    description = """
    Load CAMELS data for a gage into the in-memory data handle store.
    Returns a lightweight handle and summary metadata.
    """,
    input_schema = Dict(
        "type" => "object",
        "properties" => Dict(
            "dataset_path" => Dict("type" => "string"),
            "gage_id" => Dict("type" => "integer"),
            "variable" => Dict("type" => "string", "default" => "streamflow"),
        ),
        "required" => ["dataset_path", "gage_id"],
    ),
    handler = function(params)
        validation_error = validate_required_params(params, ["dataset_path", "gage_id"])
        !isnothing(validation_error) && return create_error_response(validation_error)

        try
            dataset_path = params["dataset_path"]
            gage_id = params["gage_id"]
            variable = get(params, "variable", "streamflow")

            data = npzread(dataset_path)
            gage_ids = data["gage_ids"]
            catchment_idx = findfirst(==(gage_id), gage_ids)
            isnothing(catchment_idx) && return create_error_response("Gage $gage_id not found in dataset")

            target_data = data["target"][catchment_idx, :, 1]
            area_km2 = data["attributes"][catchment_idx, 13]
            target_data_mm = target_data .* (1e3) .* 0.0283168 .* 3600 .* 24 ./ (area_km2 .* 1e6)

            valid_mask = .!isnan.(target_data_mm)
            valid_values = target_data_mm[valid_mask]
            start_date = Date(1980, 10, 1)
            dates = [start_date + Day(i - 1) for i in eachindex(target_data)]
            valid_dates = dates[valid_mask]

            handle = "camels_$(gage_id)_$(variable)"
            store_data(handle, valid_values)
            store_data("$(handle)_dates", valid_dates)

            metadata = Dict(
                "name" => "$(variable)_$(gage_id)",
                "rows" => length(valid_values),
                "start_date" => string(first(valid_dates)),
                "end_date" => string(last(valid_dates)),
                "units" => "mm/day",
                "mean" => mean(valid_values),
                "min" => minimum(valid_values),
                "max" => maximum(valid_values),
            )

            return create_json_response(Dict(
                "status" => "success",
                "data_handle" => handle,
                "metadata" => metadata,
                "warnings" => Any[],
            ))
        catch e
            return create_error_response(e)
        end
    end,
)

const analyze_distribution_from_handle_tool = MCPTool(
    name = "analyze_distribution_from_handle",
    description = "Analyze a time series already stored in the in-memory data handle store.",
    input_schema = Dict(
        "type" => "object",
        "properties" => Dict(
            "data_handle" => Dict("type" => "string"),
        ),
        "required" => ["data_handle"],
    ),
    handler = function(params)
        validation_error = validate_required_params(params, ["data_handle"])
        !isnothing(validation_error) && return create_error_response(validation_error)

        try
            values = get_data(params["data_handle"])
            positive_values = values[values .> 0]
            magnitude_ratio = isempty(positive_values) ? 1.0 : maximum(positive_values) / minimum(positive_values)

            return create_json_response(Dict(
                "status" => "success",
                "magnitude_ratio" => magnitude_ratio,
                "needs_log_transform" => magnitude_ratio > 100,
                "num_observations" => length(values),
                "min_value" => minimum(values),
                "max_value" => maximum(values),
                "mean_value" => mean(values),
            ))
        catch e
            return create_error_response(e)
        end
    end,
)

function _maybe_date_bounds(df::DataFrame)
    if :date in propertynames(df)
        return string(first(df.date)), string(last(df.date))
    end

    for name in names(df)
        value = df[1, name]
        if value isa Date || value isa DateTime
            return string(first(df[!, name])), string(last(df[!, name]))
        end
    end

    return nothing, nothing
end

function _forcing_from_dataframe(df::DataFrame, prcp_col::String, temp_col::String, pet_col::String)
    required = [prcp_col, temp_col, pet_col]
    for col in required
        Symbol(col) in propertynames(df) || throw(ArgumentError("Missing required forcing column '$col'"))
    end

    return (
        P = Float64.(df[!, Symbol(prcp_col)]),
        T = Float64.(df[!, Symbol(temp_col)]),
        Ep = Float64.(df[!, Symbol(pet_col)]),
    )
end

const load_hydro_csv_tool = MCPTool(
    name = "load_hydro_csv",
    description = """
    Load a hydro CSV file into the in-memory data handle store.
    The canonical public contract is path + data_type + validation.
    Advanced column override fields remain available for calibration workflows.
    """,
    input_schema = Dict(
        "type" => "object",
        "properties" => Dict(
            "path" => Dict("type" => "string", "description" => "Path to the CSV file."),
            "data_type" => Dict(
                "type" => "string",
                "enum" => ["forcing", "observed", "simulated"],
                "description" => "Primary data type expected from the CSV.",
            ),
            "validation" => Dict(
                "type" => "boolean",
                "description" => "Whether to validate required columns for the requested data_type.",
                "default" => true,
            ),
            "obs_column" => Dict("type" => "string", "default" => "flow(mm)"),
            "prcp_column" => Dict("type" => "string", "default" => "prcp(mm/day)"),
            "temp_column" => Dict("type" => "string", "default" => "tmean(C)"),
            "pet_column" => Dict("type" => "string", "default" => "pet(mm)"),
            "simulated_column" => Dict("type" => "string", "default" => "Result"),
            "handle_name" => Dict("type" => "string"),
            "n_rows" => Dict("type" => "integer"),
        ),
        "required" => ["path", "data_type"],
    ),
    handler = function(params)
        validation_error = validate_required_params(params, ["path", "data_type"])
        !isnothing(validation_error) && return create_error_response(validation_error)

        try
            path = resolve_workspace_path(string(params["path"]); must_exist = true)
            data_type = string(params["data_type"])
            validate_columns = Bool(get(params, "validation", true))
            obs_col = string(get(params, "obs_column", "flow(mm)"))
            prcp_col = string(get(params, "prcp_column", "prcp(mm/day)"))
            temp_col = string(get(params, "temp_column", "tmean(C)"))
            pet_col = string(get(params, "pet_column", "pet(mm)"))
            simulated_col = string(get(params, "simulated_column", "Result"))

            df = CSV.read(path, DataFrame)
            if haskey(params, "n_rows")
                df = df[1:min(Int(params["n_rows"]), nrow(df)), :]
            end

            handle = if haskey(params, "handle_name") && !isempty(strip(string(params["handle_name"])))
                string(params["handle_name"])
            else
                "hydro_" * splitext(basename(path))[1]
            end

            stored = Dict{String,Any}()

            has_forcing_columns = all(Symbol(col) in propertynames(df) for col in (prcp_col, temp_col, pet_col))
            has_obs_column = Symbol(obs_col) in propertynames(df)
            has_sim_column = Symbol(simulated_col) in propertynames(df)

            if data_type == "forcing"
                validate_columns && !has_forcing_columns &&
                    throw(ArgumentError("Forcing CSV is missing one or more required forcing columns"))
                if has_forcing_columns
                    stored["forcing_nt"] = _forcing_from_dataframe(df, prcp_col, temp_col, pet_col)
                end
            elseif data_type == "observed"
                validate_columns && !has_obs_column &&
                    throw(ArgumentError("Observed CSV is missing column '$obs_col'"))
                has_obs_column && (stored["obs"] = Float64.(df[!, Symbol(obs_col)]))
            elseif data_type == "simulated"
                validate_columns && !has_sim_column &&
                    throw(ArgumentError("Simulated CSV is missing column '$simulated_col'"))
                has_sim_column && (stored["simulated"] = Float64.(df[!, Symbol(simulated_col)]))
            end

            # Preserve the richer combined-hydro behavior for calibration workflows.
            if has_forcing_columns
                stored["forcing_nt"] = get(stored, "forcing_nt", _forcing_from_dataframe(df, prcp_col, temp_col, pet_col))
            end
            if has_obs_column
                stored["obs"] = get(stored, "obs", Float64.(df[!, Symbol(obs_col)]))
                stored["obs_column"] = obs_col
            end
            if has_sim_column
                stored["simulated"] = get(stored, "simulated", Float64.(df[!, Symbol(simulated_col)]))
                stored["simulated_column"] = simulated_col
            end
            stored["source_path"] = path
            stored["data_type"] = data_type

            store_data(handle, stored)

            start_date, end_date = _maybe_date_bounds(df)
            metadata = Dict(
                "rows" => nrow(df),
                "columns" => String.(names(df)),
                "start_date" => start_date,
                "end_date" => end_date,
            )

            warnings = String[]
            !has_forcing_columns && push!(warnings, "Forcing columns were not fully available in the CSV")
            !has_obs_column && push!(warnings, "Observed discharge column was not available in the CSV")

            return create_json_response(Dict(
                "status" => "success",
                "data_handle" => handle,
                "metadata" => metadata,
                "warnings" => warnings,
            ))
        catch e
            return create_error_response(e)
        end
    end,
)

export load_camels_data_tool, analyze_distribution_from_handle_tool, load_hydro_csv_tool
