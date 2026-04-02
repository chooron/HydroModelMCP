"""
Data loading tools for HydroModelMCP.
"""

using CSV
using DataFrames
using Dates
using JSON3
using ModelContextProtocol
using Statistics

const TOOL_CARAVAN_GAUGE_ID_SCHEMA = Dict(
    "oneOf" => [
        Dict("type" => "integer"),
        Dict("type" => "string"),
    ],
    "description" => "Caravan basin/gauge id. Strings like 01013500 are accepted and should be passed through unchanged, but dataset_name/source_dataset must also be provided.",
)

const TOOL_CARAVAN_GAUGE_ID_SCHEMA = Dict(
    "oneOf" => [
        Dict("type" => "integer"),
        Dict("type" => "string"),
    ],
    "description" => "Caravan gauge/basin id. Prefer the full global id like camels_01013500; local ids such as 01013500 also work when dataset_name/source_dataset is provided.",
)

const TOOL_CARAVAN_DATASET_NAME_SCHEMA = Dict(
    "type" => "string",
    "enum" => ["camels", "camelsaus", "camelsbr", "camelscl", "camelsgb", "hysets", "lamah"],
    "description" => "Caravan source dataset name.",
)

const load_camels_data_tool = MCPTool(
    name = "load_camels_data",
    description = "Deprecated legacy entry. CAMELS loading has been retired; use load_caravan_data with dataset_name=camels and gauge_id/gage_id instead.",
    input_schema = Dict(
        "type" => "object",
        "properties" => Dict(
            "dataset_name" => TOOL_CARAVAN_DATASET_NAME_SCHEMA,
            "gauge_id" => TOOL_CARAVAN_GAUGE_ID_SCHEMA,
            "gage_id" => TOOL_CARAVAN_GAUGE_ID_SCHEMA,
        ),
        "required" => [],
    ),
    handler = function(params)
        return create_error_response(ArgumentError(
            "load_camels_data has been retired. Use load_caravan_data with dataset_name=camels and gauge_id/gage_id instead.",
        ))
    end,
)

const load_caravan_data_tool = MCPTool(
    name = "load_caravan_data",
    description = "Load Caravan netCDF basin data into the in-memory data handle store. Supports global gauge ids like camels_01013500, or local ids plus dataset_name/source_dataset.",
    input_schema = Dict(
        "type" => "object",
        "properties" => Dict(
            "path" => Dict("type" => "string", "description" => "Optional direct path to a Caravan basin netCDF file."),
            "dataset_root" => Dict("type" => "string", "description" => "Optional Caravan dataset root containing attributes/ and timeseries/."),
            "dataset_path" => Dict("type" => "string", "description" => "Optional alias of dataset_root for Caravan datasets."),
            "timeseries_root" => Dict("type" => "string", "description" => "Optional Caravan timeseries root."),
            "netcdf_root" => Dict("type" => "string", "description" => "Optional Caravan netCDF root; omit it to use CARAVAN_DATASET_ROOT/CARAVAN_NETCDF_ROOT when available."),
            "dataset_name" => TOOL_CARAVAN_DATASET_NAME_SCHEMA,
            "source_dataset" => TOOL_CARAVAN_DATASET_NAME_SCHEMA,
            "gauge_id" => TOOL_CARAVAN_GAUGE_ID_SCHEMA,
            "gage_id" => TOOL_CARAVAN_GAUGE_ID_SCHEMA,
            "basin_id" => TOOL_CARAVAN_GAUGE_ID_SCHEMA,
            "variable" => Dict("type" => "string", "default" => "streamflow"),
        ),
        "required" => [],
    ),
    handler = function(params)
        try
            variable = get(params, "variable", "streamflow")
            source = Dict{String,Any}("source_type" => "caravan")
            for key in ("path", "dataset_root", "dataset_path", "timeseries_root", "netcdf_root", "dataset_name", "source_dataset", "gauge_id", "gage_id", "basin_id")
                haskey(params, key) || continue
                value = params[key]
                isempty(strip(string(value))) && continue
                source[key] = value
            end

            haskey(source, "path") || haskey(source, "gauge_id") || haskey(source, "gage_id") || haskey(source, "basin_id") ||
                throw(ArgumentError("Missing required parameter: gauge_id/gage_id/basin_id or path"))
            haskey(source, "path") || haskey(source, "dataset_name") || haskey(source, "source_dataset") ||
                throw(ArgumentError("Missing required parameter: dataset_name/source_dataset"))

            loaded, caravan_meta = DataLoader.load_data(Val(:caravan), source)
            obs_values = Float64.(loaded[Symbol("flow(mm)")])
            forcing_nt = (
                P = Float64.(loaded.P),
                T = Float64.(loaded.T),
                Ep = Float64.(loaded.Ep),
                Tidx = Float64.(loaded.Tidx),
            )

            canonical_gauge_id = get(caravan_meta, :gauge_id, get(caravan_meta, :gage_id, "unknown"))
            handle = "caravan_$(canonical_gauge_id)_$(variable)"

            stored = Dict{String,Any}(
                "forcing_nt" => forcing_nt,
                "obs" => obs_values,
                "obs_column" => "flow(mm)",
                "obs_units" => get(caravan_meta, :flow_units, "mm/day"),
                "source_path" => get(caravan_meta, :path, nothing),
                "data_type" => "caravan",
                "gauge_id" => canonical_gauge_id,
                "gage_id" => get(caravan_meta, :gage_id, nothing),
                "dataset_name" => get(caravan_meta, :dataset_name, nothing),
            )
            if haskey(caravan_meta, :dates)
                stored["dates"] = caravan_meta[:dates]
            end
            store_data(handle, stored)

            start_date = haskey(caravan_meta, :dates) ? string(first(caravan_meta[:dates])) : nothing
            end_date = haskey(caravan_meta, :dates) ? string(last(caravan_meta[:dates])) : nothing

            metadata = Dict(
                "name" => "$(variable)_$(canonical_gauge_id)",
                "rows" => length(obs_values),
                "start_date" => start_date,
                "end_date" => end_date,
                "units" => get(caravan_meta, :flow_units, "mm/day"),
                "mean" => mean(obs_values),
                "min" => minimum(obs_values),
                "max" => maximum(obs_values),
                "dataset_name" => get(caravan_meta, :dataset_name, nothing),
                "gauge_id" => canonical_gauge_id,
                "gage_id" => get(caravan_meta, :gage_id, nothing),
                "netcdf_path" => get(caravan_meta, :path, nothing),
                "dataset_root" => get(caravan_meta, :dataset_root, nothing),
                "area_km2" => get(caravan_meta, :area_km2, nothing),
                "dropped_nan_rows" => get(caravan_meta, :dropped_nan_rows, 0),
            )

            warnings = Any[]
            dropped_rows = get(caravan_meta, :dropped_nan_rows, 0)
            dropped_rows > 0 && push!(warnings, "Dropped $(dropped_rows) rows containing NaN values")
            units = lowercase(string(get(caravan_meta, :flow_units, "mm/day")))
            if !occursin("mm", units)
                push!(warnings, "Caravan streamflow units are reported as '$(get(caravan_meta, :flow_units, "unknown"))'; verify area normalization before using as flow(mm)")
            end

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

const analyze_distribution_from_handle_tool = MCPTool(
    name = "analyze_distribution_from_handle",
    description = "Analyze an observed or simulated time series already stored in the in-memory data handle store.",
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
            handle = string(params["data_handle"])
            stored = get_data(handle)
            values, series_name = _extract_distribution_series(stored)
            positive_values = values[values .> 0]
            magnitude_ratio = isempty(positive_values) ? 1.0 : maximum(positive_values) / minimum(positive_values)

            return create_json_response(Dict(
                "status" => "success",
                "data_handle" => handle,
                "series_name" => series_name,
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

function _extract_distribution_series(stored)
    if stored isa AbstractVector
        return Float64.(stored), "values"
    elseif stored isa AbstractDict
        if haskey(stored, "obs")
            return Float64.(stored["obs"]), get(stored, "obs_column", "obs")
        elseif haskey(stored, "simulated")
            return Float64.(stored["simulated"]), get(stored, "simulated_column", "simulated")
        end

        throw(ArgumentError(
            "data_handle does not contain an observed or simulated time series. Load a CSV with observed runoff or provide a handle that stores a numeric vector.",
        ))
    end

    throw(ArgumentError("Unsupported data_handle payload type: $(typeof(stored))"))
end

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

function _resolve_forcing_column(df::DataFrame, canonical::String, explicit_column::String)
    column_names = String.(names(df))
    lookup = _column_lookup(column_names)

    explicit_match = get(lookup, lowercase(explicit_column), nothing)
    !isnothing(explicit_match) && return explicit_match

    aliases = get(Simulation.DEFAULT_INPUT_ALIASES, canonical, [canonical])
    return _first_matching_column(lookup, aliases)
end

function _forcing_from_dataframe(df::DataFrame, prcp_col::String, temp_col::String, pet_col::String)
    resolved_columns = Dict(
        "P" => _resolve_forcing_column(df, "P", prcp_col),
        "T" => _resolve_forcing_column(df, "T", temp_col),
        "Ep" => _resolve_forcing_column(df, "Ep", pet_col),
    )

    pairs = Pair{Symbol,Vector{Float64}}[]
    for canonical in ("P", "T", "Ep")
        column_name = resolved_columns[canonical]
        isnothing(column_name) && continue
        push!(pairs, Symbol(canonical) => Float64.(df[!, Symbol(column_name)]))
    end

    isempty(pairs) && throw(ArgumentError("Could not resolve any canonical forcing columns from the CSV"))

    names_tuple = Tuple(first(pair) for pair in pairs)
    values_tuple = Tuple(last(pair) for pair in pairs)
    return NamedTuple{names_tuple}(values_tuple)
end

const OBSERVED_RUNOFF_ALIASES = [
    "flow(mm)",
    "flow",
    "discharge",
    "streamflow",
    "runoff",
    "qobs",
    "observed",
    "q",
]

const DATE_COLUMN_ALIASES = ["date", "datetime", "timestamp", "time"]

function _column_lookup(column_names::Vector{String})
    return Dict(lowercase(name) => name for name in column_names)
end

function _first_matching_column(column_lookup::Dict{String,String}, aliases::Vector{String})
    for alias in aliases
        resolved = get(column_lookup, lowercase(alias), nothing)
        !isnothing(resolved) && return resolved
    end
    return nothing
end

function _normalize_optional_input_mapping(value)
    value === nothing && return nothing
    return Simulation._normalize_input_mapping(value)
end

function _resolve_input_for_inspection(
    target_name::String,
    column_names::Vector{String},
    input_mapping::Union{Nothing,Dict{String,String}},
)
    available = Set(column_names)

    if !isnothing(input_mapping)
        if haskey(input_mapping, target_name)
            mapped_source = input_mapping[target_name]
            return mapped_source in available ? mapped_source : nothing, false, mapped_source
        end

        for (source_name, mapped_target) in input_mapping
            if mapped_target == target_name
                return source_name in available ? source_name : nothing, false, source_name
            end
        end
    end

    source_name, inferred = Simulation._resolve_input_source(Symbol(target_name), column_names, nothing)
    return source_name, inferred, nothing
end

function _inspect_required_inputs(
    required_inputs::Vector{String},
    column_names::Vector{String},
    input_mapping::Union{Nothing,Dict{String,String}},
)
    coverage = Dict{String,Any}()
    missing = String[]

    for target_name in required_inputs
        source_name, inferred, requested_mapping = _resolve_input_for_inspection(target_name, column_names, input_mapping)
        entry = Dict{String,Any}(
            "present" => !isnothing(source_name),
            "column" => source_name,
            "matched_by_alias" => inferred,
            "aliases" => get(Simulation.DEFAULT_INPUT_ALIASES, target_name, [target_name]),
        )
        !isnothing(requested_mapping) && (entry["requested_mapping"] = requested_mapping)

        coverage[target_name] = entry
        isnothing(source_name) && push!(missing, target_name)
    end

    return coverage, missing
end

function _infer_series_length(payload)
    if payload isa NamedTuple
        payload_values = collect(Base.values(payload))
        for value in payload_values
            value isa AbstractVector && return length(value)
        end
        return 0
    elseif payload isa AbstractDict
        for value in Base.values(payload)
            value isa AbstractVector && return length(value)
        end
        return 0
    end

    return 0
end

function _inspect_data_handle_columns(source::AbstractDict)
    handle = if haskey(source, "handle")
        String(strip(string(source["handle"])))
    elseif haskey(source, "data_handle")
        String(strip(string(source["data_handle"])))
    else
        throw(ArgumentError("data_handle source must include 'handle' or 'data_handle'"))
    end

    isempty(handle) && throw(ArgumentError("data_handle source must include a non-empty handle"))
    has_data(handle) || throw(ArgumentError("Data handle '$handle' not found"))

    payload = get_data(handle)
    column_names = String[]
    row_count = 0
    summary = Dict{String,Any}(
        "source_type" => "data_handle",
        "data_handle" => handle,
    )

    if payload isa NamedTuple
        column_names = String.(collect(keys(payload)))
        row_count = _infer_series_length(payload)
    elseif payload isa AbstractDict
        if haskey(payload, "forcing_nt") && payload["forcing_nt"] isa NamedTuple
            append!(column_names, String.(collect(keys(payload["forcing_nt"]))))
            row_count = max(row_count, _infer_series_length(payload["forcing_nt"]))
        end

        if haskey(payload, "obs")
            obs_name = string(get(payload, "obs_column", "obs"))
            obs_name in column_names || push!(column_names, obs_name)
            row_count = max(row_count, length(payload["obs"]))
        end

        if haskey(payload, "simulated")
            simulated_name = string(get(payload, "simulated_column", "simulated"))
            simulated_name in column_names || push!(column_names, simulated_name)
            row_count = max(row_count, length(payload["simulated"]))
        end

        if isempty(column_names)
            for (k, v) in pairs(payload)
                v isa AbstractVector || continue
                push!(column_names, string(k))
                row_count = max(row_count, length(v))
            end
        end

        haskey(payload, "source_path") && (summary["path"] = payload["source_path"])
        haskey(payload, "gage_id") && (summary["gage_id"] = payload["gage_id"])
        haskey(payload, "data_type") && (summary["data_type"] = payload["data_type"])
    else
        throw(ArgumentError("Unsupported inspected data_handle payload type: $(typeof(payload))"))
    end

    summary["row_count"] = row_count
    summary["column_names"] = unique(column_names)
    return summary
end

function _inspect_source_columns(source::AbstractDict)
    source_type = lowercase(string(source["source_type"]))

    if source_type == "csv"
        haskey(source, "path") || throw(ArgumentError("CSV source must include 'path'"))
        path = resolve_workspace_path(string(source["path"]); must_exist = true)
        df = CSV.read(path, DataFrame)
        return Dict{String,Any}(
            "source_type" => source_type,
            "path" => path,
            "row_count" => nrow(df),
            "column_names" => String.(names(df)),
        )
    end

    if source_type == "data_handle"
        return _inspect_data_handle_columns(source)
    end

    normalized_source = Dict{String,Any}(string(k) => v for (k, v) in pairs(source))
    if haskey(normalized_source, "path")
        normalized_source["path"] = resolve_workspace_path(string(normalized_source["path"]); must_exist = true)
    end

    loaded_source, metadata = DataLoader.load_data(Val(Symbol(source_type)), normalized_source)
    column_names = if loaded_source isa NamedTuple
        String.(collect(keys(loaded_source)))
    elseif loaded_source isa AbstractDict
        [string(k) for k in keys(loaded_source)]
    else
        throw(ArgumentError("Unsupported inspected payload type: $(typeof(loaded_source))"))
    end

    path = get(metadata, :path, nothing)
    if isnothing(path) && metadata isa AbstractDict
        path = get(metadata, "path", nothing)
    end

    summary = Dict{String,Any}(
        "source_type" => source_type,
        "row_count" => _infer_series_length(loaded_source),
        "column_names" => column_names,
    )
    !isnothing(path) && (summary["path"] = path)
    haskey(source, "key") && (summary["key"] = string(source["key"]))
    haskey(source, "gage_id") && (summary["gage_id"] = source["gage_id"])
    return summary
end

function _observed_runoff_check(column_names::Vector{String})
    lookup = _column_lookup(column_names)
    detected = _first_matching_column(lookup, OBSERVED_RUNOFF_ALIASES)

    return Dict{String,Any}(
        "present" => !isnothing(detected),
        "column" => detected,
        "aliases" => OBSERVED_RUNOFF_ALIASES,
    )
end

function _date_column_check(column_names::Vector{String})
    lookup = _column_lookup(column_names)
    detected = _first_matching_column(lookup, DATE_COLUMN_ALIASES)

    return Dict{String,Any}(
        "present" => !isnothing(detected),
        "column" => detected,
    )
end

function _build_inspection_payload(
    source_summary::Dict{String,Any};
    model_name::Union{Nothing,String} = nothing,
    intended_use::String = "simulation",
    input_mapping = nothing,
)
    column_names = String.(source_summary["column_names"])
    normalized_mapping = _normalize_optional_input_mapping(input_mapping)

    forcing_coverage, missing_forcing = _inspect_required_inputs(["P", "T", "Ep"], column_names, normalized_mapping)
    observed_check = _observed_runoff_check(column_names)
    date_check = _date_column_check(column_names)

    warnings = String[]
    recommendations = String[]
    blocking_issues = String[]

    if !isempty(missing_forcing)
        push!(warnings, "Missing forcing elements: $(join(missing_forcing, ", "))")
        push!(recommendations, "If the selected model requires these elements, provide the missing columns or an explicit input_mapping.")
    end

    requires_observed_runoff = intended_use in ("calibration", "validation", "metrics")
    if requires_observed_runoff && !observed_check["present"]
        push!(warnings, "Observed runoff was not detected, but $(intended_use) requires observed discharge.")
        push!(recommendations, "Provide a runoff column such as flow(mm), discharge, or streamflow before $(intended_use).")
        push!(blocking_issues, "Observed runoff is required for $(intended_use) but was not detected.")
    end

    payload = Dict{String,Any}(
        "status" => "success",
        "source" => source_summary,
        "intended_use" => intended_use,
        "date_column" => date_check,
        "forcing_elements" => Dict(
            "required_inputs" => ["P", "T", "Ep"],
            "coverage" => forcing_coverage,
            "sufficient" => isempty(missing_forcing),
            "missing_inputs" => missing_forcing,
            "note" => "This is a general hydrometeorological check. Blocking readiness is determined by the selected model requirements when a model is provided.",
        ),
        "observed_runoff" => observed_check,
        "blocking_issues" => blocking_issues,
        "readiness" => Dict(
            "ready_for_simulation" => model_name === nothing ? nothing : true,
            "ready_for_calibration" => model_name === nothing ? nothing : observed_check["present"],
            "ready_for_requested_use" => isempty(blocking_issues),
            "requires_observed_runoff" => requires_observed_runoff,
            "model_checked" => !(model_name === nothing),
        ),
        "warnings" => warnings,
        "recommendations" => recommendations,
    )

    if !isnothing(model_name)
        resolved_model = Discovery.find_model(model_name)
        isnothing(resolved_model) && throw(ArgumentError("Model '$(model_name)' was not found."))

        model_info = Discovery.get_model_info(resolved_model)
        required_inputs = String.(model_info["inputs"])
        model_coverage, missing_model_inputs = _inspect_required_inputs(required_inputs, column_names, normalized_mapping)

        payload["model_check"] = Dict(
            "requested_model" => model_name,
            "resolved_model" => resolved_model,
            "required_inputs" => required_inputs,
            "coverage" => model_coverage,
            "sufficient" => isempty(missing_model_inputs),
            "missing_inputs" => missing_model_inputs,
        )

        if !isempty(missing_model_inputs)
            push!(warnings, "Model '$(resolved_model)' is missing required inputs: $(join(missing_model_inputs, ", "))")
            push!(recommendations, "Provide the required model inputs or an explicit input_mapping before running $(resolved_model).")
            push!(blocking_issues, "Model '$(resolved_model)' is missing required inputs: $(join(missing_model_inputs, ", "))")
        end

        payload["readiness"]["ready_for_simulation"] = isempty(missing_model_inputs)
        payload["readiness"]["ready_for_calibration"] = isempty(missing_model_inputs) && observed_check["present"]
        payload["readiness"]["ready_for_requested_use"] = isempty(blocking_issues)
    end

    return payload
end

const inspect_hydro_data_tool = MCPTool(
    name = "inspect_hydro_data",
    description = "Inspect a csv/json/redis/caravan/data_handle source and report detected hydrometeorological elements, model-required input coverage, and observed runoff availability before simulation or calibration.",
    input_schema = Dict(
        "type" => "object",
        "properties" => Dict(
            "source" => Dict(
                "type" => "object",
                "description" => "Source descriptor to inspect. Use the same source_type/path-or-key pattern as other HydroModelMCP tools.",
                "properties" => Dict(
                    "source_type" => Dict("type" => "string", "enum" => ["csv", "json", "redis", "caravan", "data_handle"]),
                    "path" => Dict("type" => "string"),
                    "data" => Dict("type" => "object"),
                    "key" => Dict("type" => "string"),
                    "host" => Dict("type" => "string"),
                    "port" => Dict("type" => "integer"),
                    "dataset_path" => Dict("type" => "string", "description" => "Optional alias of Caravan dataset_root."),
                    "dataset_root" => Dict("type" => "string", "description" => "Optional Caravan dataset root containing attributes/ and timeseries/."),
                    "timeseries_root" => Dict("type" => "string", "description" => "Optional Caravan timeseries root."),
                    "netcdf_root" => Dict("type" => "string", "description" => "Optional Caravan netCDF root; omit it to use CARAVAN_DATASET_ROOT/CARAVAN_NETCDF_ROOT when available."),
                    "dataset_name" => TOOL_CARAVAN_DATASET_NAME_SCHEMA,
                    "source_dataset" => TOOL_CARAVAN_DATASET_NAME_SCHEMA,
                    "gage_id" => TOOL_CARAVAN_GAUGE_ID_SCHEMA,
                    "gauge_id" => TOOL_CARAVAN_GAUGE_ID_SCHEMA,
                    "basin_id" => TOOL_CARAVAN_GAUGE_ID_SCHEMA,
                    "handle" => Dict("type" => "string"),
                    "data_handle" => Dict("type" => "string"),
                ),
                "required" => ["source_type"],
            ),
            "model" => Dict(
                "type" => "string",
                "description" => "Optional model name. When provided, the inspection also checks whether model-required inputs can be resolved.",
            ),
            "intended_use" => Dict(
                "type" => "string",
                "enum" => ["general", "simulation", "calibration", "validation", "metrics"],
                "default" => "simulation",
                "description" => "Workflow intent used to determine whether observed runoff is required.",
            ),
            "input_mapping" => Dict(
                "type" => "object",
                "description" => "Optional model-input to source-column mapping used during inspection.",
            ),
        ),
        "required" => ["source"],
    ),
    handler = function(params)
        validation_error = validate_required_params(params, ["source"])
        !isnothing(validation_error) && return create_error_response(validation_error)

        try
            source = normalize_string_dict(params["source"])
            source_validation_error = validate_required_params(source, ["source_type"])
            !isnothing(source_validation_error) && throw(ArgumentError(source_validation_error))

            intended_use = lowercase(string(get(params, "intended_use", "simulation")))
            intended_use in ("general", "simulation", "calibration", "validation", "metrics") ||
                throw(ArgumentError("intended_use must be one of: general, simulation, calibration, validation, metrics"))

            source_summary = _inspect_source_columns(source)
            payload = _build_inspection_payload(
                source_summary;
                model_name = haskey(params, "model") ? string(params["model"]) : nothing,
                intended_use = intended_use,
                input_mapping = get(params, "input_mapping", nothing),
            )

            return create_json_response(payload)
        catch e
            return create_error_response(e)
        end
    end,
)

const load_hydro_csv_tool = MCPTool(
    name = "load_hydro_csv",
    description = """
    Load a hydro CSV file into the in-memory data handle store.
    Load a hydro CSV file into the in-memory data handle store using one canonical path-based contract.
    The tool stores any resolvable canonical forcing inputs (P/T/Ep) plus observed or simulated series when present.
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
            data_type = lowercase(string(params["data_type"]))
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

            forcing_nt = try
                _forcing_from_dataframe(df, prcp_col, temp_col, pet_col)
            catch
                nothing
            end
            has_forcing_columns = !isnothing(forcing_nt)
            has_obs_column = Symbol(obs_col) in propertynames(df)
            has_sim_column = Symbol(simulated_col) in propertynames(df)

            if data_type == "forcing"
                validate_columns && !has_forcing_columns &&
                    throw(ArgumentError("Forcing CSV does not expose any canonical forcing inputs after alias resolution"))
                has_forcing_columns && (stored["forcing_nt"] = forcing_nt)
            elseif data_type == "observed"
                validate_columns && !has_obs_column &&
                    throw(ArgumentError("Observed CSV is missing column '$obs_col'"))
                has_obs_column && (stored["obs"] = Float64.(df[!, Symbol(obs_col)]))
            elseif data_type == "simulated"
                validate_columns && !has_sim_column &&
                    throw(ArgumentError("Simulated CSV is missing column '$simulated_col'"))
                has_sim_column && (stored["simulated"] = Float64.(df[!, Symbol(simulated_col)]))
            else
                throw(ArgumentError("data_type must be one of: forcing, observed, simulated"))
            end

            # Preserve the richer combined-hydro behavior for calibration workflows.
            if has_forcing_columns
                stored["forcing_nt"] = get(stored, "forcing_nt", forcing_nt)
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

export load_caravan_data_tool, analyze_distribution_from_handle_tool, inspect_hydro_data_tool, load_hydro_csv_tool
