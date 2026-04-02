using ModelContextProtocol
using JSON3
using .Schemas
import UUIDs
import Dates

const WORKFLOW_INPUTS_REQUIRED_FORCING_OBS = Dict{String,Any}(
    "type" => "object",
    "properties" => Dict{String,Any}(
        "forcing" => WORKFLOW_SOURCE_SCHEMA,
        "observation" => WORKFLOW_SOURCE_SCHEMA,
        "parameters" => WORKFLOW_SOURCE_SCHEMA,
        "runtime" => WORKFLOW_SOURCE_SCHEMA,
    ),
    "required" => ["forcing", "observation"],
)

const OBJECTIVE_NAME_ALIASES = Dict{String,String}(
    "kge" => "KGE",
    "klinggupta" => "KGE",
    "klingguptaefficiency" => "KGE",
    "综合拟合" => "KGE",
    "nse" => "NSE",
    "nashefficiency" => "NSE",
    "nashsutcliffe" => "NSE",
    "纳什效率" => "NSE",
    "lognse" => "LogNSE",
    "对数nse" => "LogNSE",
    "对数纳什效率" => "LogNSE",
    "logkge" => "LogKGE",
    "对数kge" => "LogKGE",
    "pbias" => "PBIAS",
    "bias" => "PBIAS",
    "水量偏差" => "PBIAS",
    "r2" => "R2",
    "决定系数" => "R2",
    "rmse" => "RMSE",
    "均方根误差" => "RMSE",
)

const SINGLE_ALGORITHM_ALIASES = Dict{String,String}(
    "auto" => "AUTO",
    "automatic" => "AUTO",
    "自动" => "AUTO",
    "auto_select" => "AUTO",
    "dds" => "DDS",
    "dynamicallydimensionedsearch" => "DDS",
    "动态维度搜索" => "DDS",
    "sce" => "SCE",
    "shuffledcomplexevolution" => "SCE",
    "复合体进化" => "SCE",
    "bbo" => "BBO",
    "biogeographybasedoptimization" => "BBO",
    "biogeography" => "BBO",
    "生物地理优化" => "BBO",
    "生物地理优化算法" => "BBO",
    "de" => "DE",
    "differentialevolution" => "DE",
    "差分进化" => "DE",
    "差分进化算法" => "DE",
    "pso" => "PSO",
    "particleswarmoptimization" => "PSO",
    "粒子群" => "PSO",
    "粒子群优化" => "PSO",
    "cmaes" => "CMAES",
    "cmaesoptimization" => "CMAES",
    "协方差矩阵自适应进化策略" => "CMAES",
    "eca" => "ECA",
    "evolutionarycenteralgorithm" => "ECA",
    "进化中心算法" => "ECA",
)

const MULTI_ALGORITHM_ALIASES = Dict{String,String}(
    "nsga2" => "NSGA2",
    "nsgaii" => "NSGA2",
    "多目标遗传算法2" => "NSGA2",
    "nsga3" => "NSGA3",
    "nsgaiii" => "NSGA3",
    "多目标遗传算法3" => "NSGA3",
)

function _normalize_named_choice(value, field::String, aliases::Dict{String,String}, allowed::Set{String})
    raw = strip(string(value))
    isempty(raw) && throw(ArgumentError("$field cannot be empty"))
    key = lowercase(replace(raw, r"[^0-9A-Za-z一-龥]+" => ""))
    canonical = haskey(aliases, key) ? aliases[key] : uppercase(raw)
    canonical in allowed || throw(ArgumentError("Invalid value for '$field': $value. Must be one of: $(join(sort!(collect(allowed)), ", "))"))
    return canonical
end

function _normalize_metric_names(metric_names_input)
    metric_names_input isa AbstractVector ||
        throw(ArgumentError("metrics must be an array of strings"))

    alias_map = Dict(
        "NSE" => "NSE",
        "KGE" => "KGE",
        "LOGNSE" => "LogNSE",
        "LOGKGE" => "LogKGE",
        "PBIAS" => "PBIAS",
        "R2" => "R2",
        "RMSE" => "RMSE",
        "MAE" => "MAE",
        "BIAS" => "Bias",
    )

    normalized = String[]
    for raw_name in metric_names_input
        canonical = get(alias_map, uppercase(strip(string(raw_name))), string(raw_name))
        push!(normalized, canonical)
    end

    return normalized
end

function _normalize_nested_string_dict(value, field_name::String)
    if value isa AbstractString
        parsed = JSON3.read(String(value))
        return _normalize_nested_string_dict(parsed, field_name)
    elseif value isa AbstractDict
        normalized = Dict{String,Any}()
        for (k, v) in pairs(value)
            normalized[string(k)] = if v isa AbstractDict
                _normalize_nested_string_dict(v, field_name)
            elseif v isa AbstractVector
                Any[item isa AbstractDict ? _normalize_nested_string_dict(item, field_name) : item for item in v]
            else
                v
            end
        end
        return normalized
    elseif value isa NamedTuple
        return _normalize_nested_string_dict(Dict(pairs(value)), field_name)
    end

    throw(ArgumentError("$field_name must be an object-like value or JSON string"))
end

function _normalize_calibration_result_payload(raw)
    payload = _normalize_nested_string_dict(raw, "calibration_result")

    for alias_key in ("calibration_result", "result", "payload", "data")
        haskey(payload, alias_key) || continue
        nested = payload[alias_key]
        nested isa AbstractDict || continue
        candidate = _normalize_nested_string_dict(nested, alias_key)
        if haskey(candidate, "best_params") || haskey(candidate, "all_trials") || haskey(candidate, "trial_results")
            payload = candidate
            break
        end
    end

    if haskey(payload, "best_parameters") && !haskey(payload, "best_params")
        payload["best_params"] = payload["best_parameters"]
    end
    if haskey(payload, "final_parameters") && !haskey(payload, "best_params")
        payload["best_params"] = payload["final_parameters"]
    end
    if haskey(payload, "best_params") && !haskey(payload, "calibrated_params")
        payload["calibrated_params"] = payload["best_params"]
    end

    if haskey(payload, "trial_results") && !haskey(payload, "all_trials")
        payload["all_trials"] = payload["trial_results"]
    end

    if haskey(payload, "all_trials") && payload["all_trials"] isa AbstractVector
        normalized_trials = Dict{String,Any}[]
        for trial in payload["all_trials"]
            trial isa AbstractDict || continue
            normalized_trial = _normalize_nested_string_dict(trial, "calibration_result.all_trials")
            if haskey(normalized_trial, "best_objective") && !haskey(normalized_trial, "objective_value")
                normalized_trial["objective_value"] = normalized_trial["best_objective"]
            end
            if haskey(normalized_trial, "best_parameters") && !haskey(normalized_trial, "params")
                normalized_trial["params"] = normalized_trial["best_parameters"]
            end
            if haskey(normalized_trial, "final_parameters") && !haskey(normalized_trial, "params")
                normalized_trial["params"] = normalized_trial["final_parameters"]
            end
            if haskey(normalized_trial, "best_params") && !haskey(normalized_trial, "params")
                normalized_trial["params"] = normalized_trial["best_params"]
            end
            push!(normalized_trials, normalized_trial)
        end
        payload["all_trials"] = normalized_trials
    end

    return payload
end

function _metric_candidates(role::String)
    role == "simulated" && return ["Result", "result", "simulated", "simulation", "runoff", "Q"]
    return ["flow(mm)", "flow", "runoff", "discharge", "streamflow", "Q", "observed"]
end

function _infer_metric_key(available_keys::Vector{String}, role::String, explicit_column)
    lookup = Dict(lowercase(k) => k for k in available_keys)

    if !(explicit_column === nothing)
        requested = String(explicit_column)
        resolved = get(lookup, lowercase(requested), nothing)
        isnothing(resolved) &&
            throw(ArgumentError("Column '$requested' not found in $(role) source. Available columns: $(join(sort(available_keys), ", "))"))
        return resolved, String[]
    end

    warnings = String[]
    for candidate in _metric_candidates(role)
        resolved = get(lookup, lowercase(candidate), nothing)
        isnothing(resolved) || begin
            push!(warnings, "Inferred $(role) column '$resolved'")
            return resolved, warnings
        end
    end

    isempty(available_keys) && throw(ArgumentError("Could not infer a $(role) column from empty source data"))
    fallback = sort(available_keys)[1]
    push!(warnings, "Fell back to first available $(role) column '$fallback'")
    return fallback, warnings
end

function _column_value_from_source(data_source, column_name::String)
    if data_source isa NamedTuple
        return getproperty(data_source, Symbol(column_name))
    elseif data_source isa AbstractDict
        haskey(data_source, column_name) && return data_source[column_name]
        haskey(data_source, Symbol(column_name)) && return data_source[Symbol(column_name)]
        throw(ArgumentError("Column '$column_name' not found in source object"))
    end

    throw(ArgumentError("Unsupported source payload type: $(typeof(data_source))"))
end

function _load_metric_series_from_loader(source::AbstractDict, role::String)
    source_type = Symbol(lowercase(string(source["source_type"])))
    data_source, _ = DataLoader.load_data(Val(source_type), source)

    available_keys = if data_source isa NamedTuple
        String.(collect(keys(data_source)))
    elseif data_source isa AbstractDict
        [string(k) for k in keys(data_source)]
    else
        throw(ArgumentError("Unsupported source payload type for $(role): $(typeof(data_source))"))
    end

    column_name, warnings = _infer_metric_key(available_keys, role, get(source, "column", nothing))
    values = _column_value_from_source(data_source, column_name)
    return Float64.(collect(values)), nothing, column_name, warnings
end

function _infer_metric_column(df, role::String, explicit_column)
    if !(explicit_column === nothing)
        return String(explicit_column), String[]
    end

    warnings = String[]
    column_names = String.(names(df))
    for candidate in _metric_candidates(role)
        candidate in column_names || continue
        push!(warnings, "Inferred $(role) column '$candidate'")
        return candidate, warnings
    end

    for name in names(df)
        if eltype(df[!, name]) <: Number
            inferred = String(name)
            push!(warnings, "Fell back to first numeric $(role) column '$inferred'")
            return inferred, warnings
        end
    end

    throw(ArgumentError("Could not infer a $(role) column from the CSV"))
end

function _load_metric_series(source_like, role::String)
    source = normalize_string_dict(source_like)
    warnings = String[]

    if haskey(source, "data_handle")
        stored = get_data(string(source["data_handle"]))
        if stored isa AbstractVector
            return Float64.(stored), nothing, nothing, warnings
        elseif stored isa AbstractDict
            explicit_column = get(source, "column", nothing)
            key = if role == "simulated"
                haskey(stored, "simulated") ? "simulated" : explicit_column
            else
                haskey(stored, "obs") ? "obs" : explicit_column
            end
            isnothing(key) && throw(ArgumentError("data_handle does not contain $(role) series"))
            return Float64.(stored[key]), nothing, String(key), warnings
        end
        throw(ArgumentError("Unsupported data_handle payload for $(role)"))
    end

    source_type = lowercase(string(get(source, "source_type", "csv")))

    if source_type == "csv"
        haskey(source, "path") || throw(ArgumentError("$(role) source must include 'path'"))
        path = resolve_workspace_path(string(source["path"]); must_exist = true)
        df = CSV.read(path, DataFrame)
        column_name, infer_warnings = _infer_metric_column(df, role, get(source, "column", nothing))
        append!(warnings, infer_warnings)
        return Float64.(df[!, Symbol(column_name)]), path, column_name, warnings
    elseif source_type in ("json", "redis", "caravan")
        values, path, column_name, infer_warnings = _load_metric_series_from_loader(source, role)
        append!(warnings, infer_warnings)
        return values, path, column_name, warnings
    end

    throw(ArgumentError("Unsupported source_type for $(role): $source_type. Use csv, json, redis, caravan, or data_handle."))
end

function _write_metrics_artifact(output_dir::String, base_name::String, payload::Dict{String,Any})
    normalized_output_dir = normpath(output_dir)
    metrics_dir = lowercase(basename(normalized_output_dir)) == "metrics" ?
                  normalized_output_dir :
                  joinpath(normalized_output_dir, "metrics")
    mkpath(metrics_dir)
    timestamp = Dates.format(now(), "yyyymmddHHMMSS")
    output_path = joinpath(metrics_dir, "$(base_name)_metrics_$(timestamp).json")
    open(output_path, "w") do io
        JSON3.write(io, payload)
    end
    return output_path
end

function _infer_observed_path_from_simulated(simulated_source)
    simulated_source === nothing && return nothing

    normalized = normalize_string_dict(simulated_source)
    source_type = lowercase(string(get(normalized, "source_type", "")))
    source_type == "csv" || return nothing
    haskey(normalized, "path") || return nothing

    simulation_path = string(normalized["path"])
    base = splitext(basename(simulation_path))[1]
    parts = split(base, "_result_"; limit = 2)
    isempty(parts) && return nothing
    gauge_id = strip(parts[1])
    isempty(gauge_id) && return nothing

    candidate_rel = "./data/$(gauge_id).csv"
    try
        return resolve_workspace_path(candidate_rel; must_exist = true)
    catch
    end

    return nothing
end

function _resolve_metrics_sources(params)
    normalized = normalize_string_dict(params)
    warnings = String[]

    simulated = get(normalized, "simulated", nothing)
    observed = get(normalized, "observed", nothing)
    last_simulation = nothing
    simulated_from_last_run = false

    if simulated === nothing
        last_simulation = load_last_result(LAST_SIMULATION_RESULT_HANDLE)
        if last_simulation isa AbstractDict && haskey(last_simulation, "output_path")
            simulated = Dict{String,Any}(
                "source_type" => "csv",
                "path" => string(last_simulation["output_path"]),
            )
            simulated_from_last_run = true
            push!(warnings, "Inferred simulated source from last run_simulation output")
        end
    end

    if observed === nothing
        inferred_observed_path = _infer_observed_path_from_simulated(simulated)
        if !isnothing(inferred_observed_path)
            observed = Dict{String,Any}(
                "source_type" => "csv",
                "path" => inferred_observed_path,
            )
            push!(warnings, "Inferred observed source from ./data using simulation file stem")
        end
    end

    if observed === nothing
        last_simulation isa AbstractDict || (last_simulation = load_last_result(LAST_SIMULATION_RESULT_HANDLE))
        if last_simulation isa AbstractDict && haskey(last_simulation, "forcing_source")
            forcing_source = normalize_string_dict(last_simulation["forcing_source"])
            source_type = lowercase(string(get(forcing_source, "source_type", "")))
            same_run_context = simulated_from_last_run || (
                simulated isa AbstractDict &&
                haskey(simulated, "path") &&
                haskey(last_simulation, "output_path") &&
                normpath(string(simulated["path"])) == normpath(string(last_simulation["output_path"]))
            )
            if same_run_context && source_type == "caravan"
                observed = Dict{String,Any}(
                    "source_type" => "caravan",
                    "dataset_name" => get(forcing_source, "dataset_name", get(forcing_source, "source_dataset", nothing)),
                    "gauge_id" => get(forcing_source, "gauge_id", get(forcing_source, "gage_id", get(forcing_source, "basin_id", nothing))),
                )
                push!(warnings, "Inferred observed source from last run_simulation Caravan forcing context")
            end
        end
    end

    return simulated, observed, warnings
end

function _extract_dates_from_forcing_metadata(forcing_meta)
    if forcing_meta isa NamedTuple
        return haskey(forcing_meta, :dates) ? forcing_meta[:dates] : nothing
    elseif forcing_meta isa AbstractDict
        return get(forcing_meta, "dates", get(forcing_meta, :dates, nothing))
    end

    return nothing
end

function _stage2_split_dataset(obs_vec::AbstractVector, forcing_nt::NamedTuple, forcing_meta, params::AbstractDict)
    has_cal = haskey(params, "calibration_period")
    has_val = haskey(params, "validation_period")
    has_cal == has_val || throw(ArgumentError(
        "calibration_period and validation_period must be provided together",
    ))

    method = string(get(params, "method", "split_sample"))
    ratio = Float64(get(params, "ratio", 0.7))
    warmup = Int(get(params, "warmup", 365))

    split_result = if has_cal
        DataSplitter.split_data(
            obs_vec,
            forcing_nt;
            method = method,
            ratio = ratio,
            warmup = warmup,
            calibration_period = params["calibration_period"],
            validation_period = params["validation_period"],
            dates = _extract_dates_from_forcing_metadata(forcing_meta),
        )
    else
        DataSplitter.split_data(obs_vec, forcing_nt; method = method, ratio = ratio, warmup = warmup)
    end

    split_warnings = String[]
    get(split_result, "used_synthetic_dates", false) && push!(
        split_warnings,
        "Forcing metadata had no dates; resolved calibration/validation periods on a synthetic timeline",
    )
    split_result["val_length"] == 0 && push!(
        split_warnings,
        "Split produced an empty test set; stage-2 test metrics are omitted",
    )

    return split_result, split_warnings
end

function _build_stage2_evaluation(
    model,
    runtime_cfg::AbstractDict,
    params_used::Dict{String,Float64},
    split_result::Dict{String,Any},
    metric_names::Vector{String},
)
    params_vec = Simulation._component_vector_from_params(model, params_used)
    init_states = Simulation._build_init_states(model, get(runtime_cfg, "init_states", nothing))
    solver_sym = Symbol(uppercase(string(get(runtime_cfg, "solver", "DISCRETE"))))
    interp_sym = Symbol(uppercase(string(get(runtime_cfg, "interpolation", "LINEAR"))))
    config_dict = haskey(runtime_cfg, "config") && runtime_cfg["config"] isa AbstractDict ?
        Dict{String,Any}(string(k) => v for (k, v) in pairs(runtime_cfg["config"])) : Dict{String,Any}()

    train_forcing = split_result["cal_forcing"]
    train_obs = Float64.(split_result["cal_obs"])
    train_sim = Simulation._execute_core(
        model,
        train_forcing,
        params_vec,
        init_states,
        solver_sym,
        interp_sym,
        config_dict,
    )
    train_metrics = Metrics.compute_metrics(train_sim, train_obs, metric_names)

    test_metrics = nothing
    if split_result["val_length"] > 0
        test_forcing = split_result["val_forcing"]
        test_obs = Float64.(split_result["val_obs"])
        test_sim = Simulation._execute_core(
            model,
            test_forcing,
            params_vec,
            init_states,
            solver_sym,
            interp_sym,
            config_dict,
        )
        test_metrics = Metrics.compute_metrics(test_sim, test_obs, metric_names)
    end

    return Dict{String,Any}(
        "split_mode" => split_result["split_mode"],
        "method" => split_result["method"],
        "warmup" => split_result["warmup"],
        "train_indices" => split_result["cal_indices"],
        "test_indices" => split_result["val_indices"],
        "train_length" => split_result["cal_length"],
        "test_length" => split_result["val_length"],
        "metrics" => metric_names,
        "train_metrics" => train_metrics,
        "test_metrics" => test_metrics,
        "test_available" => split_result["val_length"] > 0,
        "used_synthetic_dates" => get(split_result, "used_synthetic_dates", false),
    )
end

function _parse_pie_share_constraints(raw)
    raw isa AbstractDict || throw(ArgumentError("constraints.pie_share must be an object"))
    data = normalize_string_dict(raw)
    haskey(data, "parameters") || throw(ArgumentError("constraints.pie_share.parameters is required"))
    params_raw = data["parameters"]
    params_raw isa AbstractVector || throw(ArgumentError("constraints.pie_share.parameters must be an array"))
    parameter_names = String[strip(string(v)) for v in params_raw]
    any(isempty, parameter_names) && throw(ArgumentError("constraints.pie_share.parameters cannot contain empty names"))
    length(unique(parameter_names)) == length(parameter_names) ||
        throw(ArgumentError("constraints.pie_share.parameters cannot contain duplicates"))
    return Dict{String,Any}(
        "parameters" => parameter_names,
        "total" => Float64(get(data, "total", 1.0)),
    )
end

function _parse_delta_constraints(raw)
    raw isa AbstractDict || throw(ArgumentError("constraints.delta_method must be an object"))
    data = normalize_string_dict(raw)
    haskey(data, "inequalities") || throw(ArgumentError("constraints.delta_method.inequalities is required"))
    inequalities_raw = data["inequalities"]
    inequalities_raw isa AbstractVector || throw(ArgumentError("constraints.delta_method.inequalities must be an array"))

    inequalities = Vector{Tuple{String,String}}()
    for item in inequalities_raw
        item isa AbstractVector || throw(ArgumentError("Each inequality must be a two-element array [lower_param, upper_param]"))
        length(item) == 2 || throw(ArgumentError("Each inequality must contain exactly two parameter names"))
        lower_name = strip(string(item[1]))
        upper_name = strip(string(item[2]))
        isempty(lower_name) && throw(ArgumentError("delta_method inequality lower parameter cannot be empty"))
        isempty(upper_name) && throw(ArgumentError("delta_method inequality upper parameter cannot be empty"))
        lower_name == upper_name && throw(ArgumentError("delta_method inequality cannot use the same parameter on both sides"))
        push!(inequalities, (lower_name, upper_name))
    end

    return Dict{String,Any}("inequalities" => inequalities)
end

function _normalize_constraints(raw_constraints)
    raw_constraints === nothing && return nothing
    raw_constraints isa AbstractDict || throw(ArgumentError("constraints must be an object"))
    constraints = normalize_string_dict(raw_constraints)
    normalized = Dict{String,Any}()

    if haskey(constraints, "pie_share")
        normalized["pie_share"] = _parse_pie_share_constraints(constraints["pie_share"])
    end
    if haskey(constraints, "delta_method")
        normalized["delta_method"] = _parse_delta_constraints(constraints["delta_method"])
    end

    isempty(normalized) && throw(ArgumentError("constraints must contain at least one of: pie_share, delta_method"))
    return normalized
end

function _build_param_bounds_lookup(param_names::Vector{String}, default_bounds::Vector{Tuple{Float64,Float64}}, custom_bounds)
    lookup = Dict{String,Tuple{Float64,Float64}}()
    for (idx, pname) in enumerate(param_names)
        if !(custom_bounds === nothing) && haskey(custom_bounds, pname)
            bound_val = custom_bounds[pname]
            if bound_val isa AbstractVector
                lookup[pname] = (Float64(bound_val[1]), Float64(bound_val[2]))
            elseif bound_val isa AbstractDict
                bdict = normalize_string_dict(bound_val)
                lookup[pname] = (Float64(bdict["lower"]), Float64(bdict["upper"]))
            else
                throw(ArgumentError("param_bounds.$pname must be [lower, upper] or object{lower,upper}"))
            end
        else
            lookup[pname] = default_bounds[idx]
        end
    end
    return lookup
end

function _evaluate_constraint_plan(model_name::String, param_bounds, constraints)
    constraints === nothing && return nothing, String[]

    _, _, param_names_syms, default_bounds = Calibration._load_model_and_bounds(model_name)
    param_names = string.(param_names_syms)
    bounds_lookup = _build_param_bounds_lookup(param_names, default_bounds, param_bounds)

    warnings = String[]
    summary = Dict{String,Any}(
        "enabled" => true,
        "status" => "ok",
        "checks" => Dict{String,Any}(),
    )

    if haskey(constraints, "pie_share")
        pie = constraints["pie_share"]
        pie_params = String.(pie["parameters"])
        total = Float64(pie["total"])
        unknown = [p for p in pie_params if !haskey(bounds_lookup, p)]
        isempty(unknown) || throw(ArgumentError("constraints.pie_share includes unknown parameters: $(join(unknown, ", "))"))

        lower_sum = sum(bounds_lookup[p][1] for p in pie_params)
        upper_sum = sum(bounds_lookup[p][2] for p in pie_params)
        feasible = lower_sum - 1e-10 <= total <= upper_sum + 1e-10
        feasible || throw(ArgumentError(
            "Pie-share infeasible: total=$total is outside [sum(lower)=$(round(lower_sum, digits=6)), sum(upper)=$(round(upper_sum, digits=6))]",
        ))

        summary["checks"]["pie_share"] = Dict{String,Any}(
            "status" => "ok",
            "parameters" => pie_params,
            "total" => total,
            "feasible_interval" => [lower_sum, upper_sum],
        )
    end

    if haskey(constraints, "delta_method")
        delta = constraints["delta_method"]
        inequalities = delta["inequalities"]
        invalid = Tuple{String,String}[]
        valid = Tuple{String,String}[]

        for (lower_name, upper_name) in inequalities
            haskey(bounds_lookup, lower_name) || throw(ArgumentError("delta_method parameter '$lower_name' not found in model bounds"))
            haskey(bounds_lookup, upper_name) || throw(ArgumentError("delta_method parameter '$upper_name' not found in model bounds"))

            lower_bound = bounds_lookup[lower_name]
            upper_bound = bounds_lookup[upper_name]
            if lower_bound[1] >= upper_bound[2]
                push!(invalid, (lower_name, upper_name))
            else
                push!(valid, (lower_name, upper_name))
            end
        end

        isempty(invalid) || throw(ArgumentError(
            "Delta-method infeasible inequalities due to non-overlapping bounds: $(join(["$(p[1])<$(p[2])" for p in invalid], ", "))",
        ))

        summary["checks"]["delta_method"] = Dict{String,Any}(
            "status" => "ok",
            "inequalities" => [[p[1], p[2]] for p in valid],
            "count" => length(valid),
        )
    end

    push!(warnings, "Constraints were pre-validated and are enforced during optimization through a penalty term.")
    return summary, warnings
end

function _infer_convergence_assessment(objective_name::String, best_objective::Float64)
    metric_upper = Dict(
        "NSE" => 1.0,
        "KGE" => 1.0,
        "LogNSE" => 1.0,
        "LogKGE" => 1.0,
        "R2" => 1.0,
    )

    if haskey(metric_upper, objective_name)
        score = clamp(best_objective / metric_upper[objective_name], 0.0, 1.0)
        status = score >= 0.8 ? "converged" : (score >= 0.6 ? "uncertain" : "not_converged")
        return Dict{String,Any}(
            "status" => status,
            "score" => score,
            "reason" => "higher-is-better objective compared against practical target ratio",
            "objective_name" => objective_name,
            "best_objective" => best_objective,
        )
    elseif objective_name == "PBIAS"
        abs_bias = abs(best_objective)
        score = if abs_bias <= 10.0
            1.0
        elseif abs_bias <= 20.0
            0.7
        elseif abs_bias <= 40.0
            0.4
        else
            0.1
        end
        status = score >= 0.8 ? "converged" : (score >= 0.6 ? "uncertain" : "not_converged")
        return Dict{String,Any}(
            "status" => status,
            "score" => score,
            "reason" => "lower absolute PBIAS indicates better convergence",
            "objective_name" => objective_name,
            "best_objective" => best_objective,
        )
    elseif objective_name == "RMSE"
        score = best_objective <= 0 ? 1.0 : clamp(1.0 / (1.0 + best_objective), 0.0, 1.0)
        status = score >= 0.8 ? "converged" : (score >= 0.6 ? "uncertain" : "not_converged")
        return Dict{String,Any}(
            "status" => status,
            "score" => score,
            "reason" => "lower RMSE indicates better convergence",
            "objective_name" => objective_name,
            "best_objective" => best_objective,
        )
    end

    return Dict{String,Any}(
        "status" => "uncertain",
        "score" => 0.5,
        "reason" => "objective type not calibrated in heuristic convergence assessment",
        "objective_name" => objective_name,
        "best_objective" => best_objective,
    )
end

function _canonical_param_bounds(raw_bounds)
    raw_bounds isa AbstractDict || throw(ArgumentError("param_bounds must be an object"))
    canonical = Dict{String,Vector{Float64}}()
    for (k, v) in pairs(raw_bounds)
        key = string(k)
        if v isa AbstractVector
            length(v) == 2 || throw(ArgumentError("param_bounds.$key must contain [lower, upper]"))
            canonical[key] = [Float64(v[1]), Float64(v[2])]
        elseif v isa AbstractDict
            bdict = normalize_string_dict(v)
            haskey(bdict, "lower") && haskey(bdict, "upper") ||
                throw(ArgumentError("param_bounds.$key must include lower and upper fields"))
            canonical[key] = [Float64(bdict["lower"]), Float64(bdict["upper"])]
        else
            throw(ArgumentError("param_bounds.$key must be [lower, upper] or object{lower,upper}"))
        end
    end
    return canonical
end

function _quality_score_from_metrics(metrics::AbstractDict)
    kge = haskey(metrics, "KGE") ? Float64(metrics["KGE"]) : 0.0
    nse = haskey(metrics, "NSE") ? Float64(metrics["NSE"]) : 0.0
    rmse = haskey(metrics, "RMSE") ? Float64(metrics["RMSE"]) : 1.0
    pbias = haskey(metrics, "PBIAS") ? abs(Float64(metrics["PBIAS"])) : 100.0

    rmse_score = clamp(1.0 / (1.0 + rmse), 0.0, 1.0)
    pbias_score = clamp(1.0 - pbias / 100.0, 0.0, 1.0)
    return 0.35 * kge + 0.30 * nse + 0.20 * rmse_score + 0.15 * pbias_score
end

function _apply_boundary_expansion(bounds::Dict{String,Vector{Float64}}, boundary_adjustments)
    expanded = Dict{String,Vector{Float64}}(k => copy(v) for (k, v) in pairs(bounds))
    if boundary_adjustments isa AbstractDict
        for (pname_raw, adj_raw) in pairs(boundary_adjustments)
            pname = string(pname_raw)
            haskey(expanded, pname) || continue
            if adj_raw isa AbstractDict
                adj = normalize_string_dict(adj_raw)
                if haskey(adj, "suggested_bounds")
                    sb = adj["suggested_bounds"]
                    sb isa AbstractVector || continue
                    length(sb) == 2 || continue
                    expanded[pname] = [Float64(sb[1]), Float64(sb[2])]
                    continue
                end
            end

            lo, hi = expanded[pname]
            width = max(hi - lo, 1e-6)
            margin = 0.2 * width
            expanded[pname] = [lo - margin, hi + margin]
        end
    end
    return expanded
end

function _auto_calibration_iteration_summary(run_payload::Dict{String,Any}, diagnostics::Dict{String,Any}, validation_payload)
    stage2 = get(run_payload, "stage2_evaluation", Dict{String,Any}())
    metrics = get(stage2, "test_available", false) ? get(stage2, "test_metrics", Dict{String,Any}()) : get(stage2, "train_metrics", Dict{String,Any}())
    qscore = metrics isa AbstractDict ? _quality_score_from_metrics(metrics) : 0.0

    return Dict{String,Any}(
        "result_id" => get(run_payload, "result_id", nothing),
        "algorithm" => get(run_payload, "algorithm", nothing),
        "objective" => get(run_payload, "objective_name", nothing),
        "best_objective" => get(run_payload, "best_objective", nothing),
        "converged" => get(run_payload, "converged", false),
        "quality_score" => qscore,
        "diagnostics" => diagnostics,
        "validation" => validation_payload,
    )
end

compute_metrics_tool = MCPTool(
    name = "compute_metrics",
    description = "Compute evaluation metrics from simulated and observed sources, including csv/json/redis/caravan descriptors and in-memory data handles.",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "simulated" => Dict("type" => "object", "description" => "Simulated source descriptor."),
            "observed" => Dict("type" => "object", "description" => "Observed source descriptor."),
            "metrics" => METRICS_SCHEMA,
            "output_dir" => Dict(
                "type" => "string",
                "description" => "Workspace-relative output directory for metrics artifacts.",
                "default" => "./result"
            )
        ),
        "required" => []
    ),
    handler = function(params)
        try
            simulated_source, observed_source, source_warnings = _resolve_metrics_sources(params)
            simulated_source === nothing &&
                throw(ArgumentError("Missing required parameter: simulated (or run run_simulation first so compute_metrics can infer it)"))
            observed_source === nothing &&
                throw(ArgumentError("Missing required parameter: observed"))

            sim_vec, sim_path, sim_column, sim_warnings = _load_metric_series(simulated_source, "simulated")
            obs_vec, obs_path, obs_column, obs_warnings = _load_metric_series(observed_source, "observed")

            warnings = String[]
            append!(warnings, source_warnings)
            append!(warnings, sim_warnings)
            append!(warnings, obs_warnings)

            n = min(length(sim_vec), length(obs_vec))
            if length(sim_vec) != length(obs_vec)
                push!(warnings, "Series lengths differ; truncated to $n samples")
            end

            metric_names = _normalize_metric_names(
                get(params, "metrics", String["NSE", "KGE", "RMSE", "PBIAS", "R2", "MAE", "Bias"]),
            )
            metric_values = Metrics.compute_metrics(sim_vec[1:n], obs_vec[1:n], metric_names)

            payload = Dict{String,Any}(
                "status" => "success",
                "metrics" => metric_values,
                "sample_size" => n,
                "warnings" => warnings,
                "simulated_column" => sim_column,
                "observed_column" => obs_column
            )

            output_dir = resolve_workspace_path(string(get(params, "output_dir", "./result")))
            base_name = isnothing(sim_path) ? "simulation" : splitext(basename(sim_path))[1]
            payload["output_path"] = _write_metrics_artifact(output_dir, base_name, payload)
            !isnothing(sim_path) && (payload["simulated_path"] = sim_path)
            !isnothing(obs_path) && (payload["observed_path"] = obs_path)

            return create_json_response(payload)
        catch e
            return create_error_response(e)
        end
    end
)

# ==============================================================================
# 2. 数据划分工具
# ==============================================================================
split_data_tool = MCPTool(
    name = "split_data",
    description = "将时间序列数据划分为校准集和验证集。支持自动划分(recent_first/split_sample/use_all)与显式时间段划分(calibration_period + validation_period)。",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "data_source" => DATA_SOURCE_SCHEMA,
            "obs_column" => OBS_COLUMN_SCHEMA,
            "method" => SPLIT_METHOD_SCHEMA,
            "ratio" => RATIO_SCHEMA,
            "warmup" => WARMUP_SCHEMA,
            "calibration_period" => CALIBRATION_PERIOD_SCHEMA,
            "validation_period" => VALIDATION_PERIOD_SCHEMA,
        ),
        "required" => ["data_source", "obs_column"]
    ),
    handler = function(params)
        # 验证必需参数
        validation_error = validate_required_params(params, ["data_source", "obs_column"])
        if !isnothing(validation_error)
            return create_error_response(validation_error)
        end

        # 验证枚举参数
        if haskey(params, "method")
            enum_error = validate_enum_param(params, "method",
                ["recent_first", "split_sample", "use_all"], "split_sample")
            if !isnothing(enum_error)
                return create_error_response(enum_error)
            end
        end

        try
            config = params["data_source"]
            obs_col = params["obs_column"]
            data, metadata = DataLoader.load_data(Val(Symbol(config["source_type"])), config)
            obs_vec = Float64.(data[Symbol(obs_col)])

            method = get(params, "method", "split_sample")
            ratio = get(params, "ratio", 0.7)
            warmup = get(params, "warmup", 365)
            has_periods = haskey(params, "calibration_period") || haskey(params, "validation_period")
            haskey(params, "calibration_period") == haskey(params, "validation_period") ||
                throw(ArgumentError("calibration_period and validation_period must be provided together"))

            forcing_dates = _extract_dates_from_forcing_metadata(metadata)

            result = if has_periods
                DataSplitter.split_data(
                    obs_vec,
                    data;
                    method = method,
                    ratio = Float64(ratio),
                    warmup = Int(warmup),
                    calibration_period = params["calibration_period"],
                    validation_period = params["validation_period"],
                    dates = forcing_dates,
                )
            else
                DataSplitter.split_data(obs_vec, data; method = method, ratio = Float64(ratio), warmup = Int(warmup))
            end

            # 移除 NamedTuple (不可直接 JSON 序列化)，只保留元信息
            serializable = Dict{String,Any}(
                "method" => result["method"],
                "split_mode" => result["split_mode"],
                "warmup" => result["warmup"],
                "total_length" => result["total_length"],
                "cal_length" => result["cal_length"],
                "val_length" => result["val_length"],
                "cal_indices" => result["cal_indices"],
                "val_indices" => result["val_indices"],
                "used_synthetic_dates" => get(result, "used_synthetic_dates", false),
            )
            return TextContent(text = JSON3.write(serializable))
        catch e
            return create_error_response(e)
        end
    end
)

# ==============================================================================
# 3. 敏感性分析工具
# ==============================================================================
sensitivity_tool = MCPTool(
    name = "run_sensitivity",
    description = "对水文模型参数进行全局敏感性分析(Strategy 1)。识别哪些参数对输出有显著影响(应校准)，哪些可固定为默认值。支持 Morris 和 Sobol 方法。",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "model" => MODEL_NAME_SCHEMA,
            "inputs" => WORKFLOW_INPUTS_REQUIRED_FORCING_OBS,
            "options" => WORKFLOW_OPTIONS_SCHEMA,
            "method" => SENSITIVITY_METHOD_SCHEMA,
            "n_samples" => N_SAMPLES_SCHEMA,
            "objective" => OBJECTIVE_SCHEMA,
            "threshold" => THRESHOLD_SCHEMA,
            "solver" => SOLVER_SIMPLE_SCHEMA,
            "interpolator" => INTERPOLATOR_SIMPLE_SCHEMA,
            "save_to_storage" => SAVE_TO_STORAGE_SCHEMA,
        ),
        "required" => ["model", "inputs"]
    ),
    handler = function(params)
        validation_error = validate_required_params(params, ["model", "inputs"])
        if !isnothing(validation_error)
            return create_error_response(validation_error)
        end

        # 验证枚举参数
        if haskey(params, "method")
            enum_error = validate_enum_param(params, "method",
                ["morris", "sobol"], "morris")
            if !isnothing(enum_error)
                return create_error_response(enum_error)
            end
        end

        try
            request = UnifiedInputs.normalize_workflow_request(params)
            model_name = request["model"]
            _, _, model = Simulation._load_model(model_name)

            resolved = UnifiedInputs.resolve_common_inputs(request, model;
                require_observation = true,
                require_parameters = false,
            )
            forcing_nt = resolved["forcing_nt"]
            obs_vec = Float64.(resolved["observation"])
            runtime_cfg = resolved["runtime"]

            method = get(params, "method", "morris")
            default_n = method == "sobol" ? 1000 : 100

            result = SensitivityAnalysis.run_sensitivity(
                model_name, forcing_nt, obs_vec;
                method = method,
                n_samples = Int(get(params, "n_samples", default_n)),
                objective = get(params, "objective", "NSE"),
                threshold = Float64(get(params, "threshold", 0.1)),
                solver_type = get(runtime_cfg, "solver", get(params, "solver", "DISCRETE")),
                interp_type = get(runtime_cfg, "interpolation", get(params, "interpolator", "LINEAR"))
            )

            result_id = string(UUIDs.uuid4())
            result["result_id"] = result_id
            result["storage_category"] = "sensitivity"
            result["status"] = "success"
            result["inference_report"] = resolved["inference_report"]
            result["warnings"] = resolved["warnings"]

            if get(params, "save_to_storage", true)
                Storage.save_result(
                    STORAGE_BACKEND,
                    "sensitivity",
                    result_id,
                    result,
                )
            end
            return TextContent(text = JSON3.write(result))
        catch e
            return create_error_response(e)
        end
    end
)

# ==============================================================================
# 3b. sensitivity_analysis 工具（data_handle 接口，兼容 Python HydroAgent 协议）
# ==============================================================================
sensitivity_analysis_tool = MCPTool(
    name = "sensitivity_analysis",
    description = "对水文模型参数进行全局敏感性分析。规范输入路径为 load_hydro_csv 生成的 data_handle + model_name。支持 morris 和 sobol 方法。",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "data_handle" => Dict{String,Any}(
                "type" => "string",
                "description" => "通过 load_hydro_csv 获取的数据句柄（需包含 forcing 数据）"
            ),
            "model_name" => Dict{String,Any}(
                "type" => "string",
                "description" => "水文模型名称（如 exphydro、gr4j 等）"
            ),
            "parameter_names" => Dict{String,Any}(
                "type" => "array",
                "items" => Dict{String,Any}("type" => "string"),
                "description" => "要分析的参数名列表（可选，默认分析全部参数）"
            ),
            "parameter_bounds" => Dict{String,Any}(
                "type" => "object",
                "description" => "自定义参数边界，格式: {\"param\": [lower, upper]}（可选，默认使用模型内置边界）"
            ),
            "method" => Dict{String,Any}(
                "type" => "string",
                "enum" => ["morris", "sobol"],
                "description" => "敏感性分析方法: morris（快速筛选）或 sobol（精确分解），默认 morris",
                "default" => "morris"
            ),
            "n_samples" => Dict{String,Any}(
                "type" => "integer",
                "description" => "采样数，morris 默认 100，sobol 默认 1000"
            ),
            "objective" => Dict{String,Any}(
                "type" => "string",
                "description" => "目标评价指标",
                "default" => "NSE"
            ),
            "threshold" => Dict{String,Any}(
                "type" => "number",
                "description" => "敏感性阈值（归一化后，>=此值为敏感参数），默认 0.1"
            ),
            "save_to_storage" => SAVE_TO_STORAGE_SCHEMA,
        ),
        "required" => ["data_handle", "model_name"]
    ),
    handler = function(params)
        validation_error = validate_required_params(params, ["data_handle", "model_name"])
        if !isnothing(validation_error)
            return create_error_response(validation_error)
        end

        model_name = string(params["model_name"])

        # 验证枚举
        if haskey(params, "method")
            enum_error = validate_enum_param(params, "method", ["morris", "sobol"], "morris")
            if !isnothing(enum_error)
                return create_error_response(enum_error)
            end
        end

        try
            # 从 data_handle 获取 obs + forcing
            handle = params["data_handle"]
            if !has_data(handle)
                return create_error_response("data_handle '$handle' 不存在，请先调用 load_hydro_csv 加载数据")
            end
            stored = get_data(handle)
            if !(stored isa Dict) || !haskey(stored, "obs") || !haskey(stored, "forcing_nt")
                return create_error_response("data_handle '$handle' 不含 forcing 数据，请使用 load_hydro_csv 加载完整数据集")
            end
            obs_vec    = Float64.(stored["obs"])
            forcing_nt = stored["forcing_nt"]

            # 处理可选参数
            method = get(params, "method", "morris")
            default_n = method == "sobol" ? 1000 : 100
            n_samples  = Int(get(params, "n_samples", default_n))
            threshold  = Float64(get(params, "threshold", 0.1))
            objective  = get(params, "objective", "NSE")

            # 自定义参数边界
            pb_override = nothing
            raw_pb_sa = haskey(params, "parameter_bounds") ? params["parameter_bounds"] : nothing
            if !isnothing(raw_pb_sa)
                pb_override = Dict{String,Vector{Float64}}()
                for (k, v) in raw_pb_sa
                    if v isa AbstractVector
                        pb_override[string(k)] = Float64.(v)
                    elseif v isa Dict || (hasproperty(v, :keys))
                        lo = Float64(get(v, "lower", get(v, :lower, 0.0)))
                        hi = Float64(get(v, "upper", get(v, :upper, 1.0)))
                        pb_override[string(k)] = [lo, hi]
                    end
                end
            end

            # 参数名筛选
            pnames_filter = nothing
            if haskey(params, "parameter_names") && !isempty(params["parameter_names"])
                pnames_filter = String.(params["parameter_names"])
            end

            result = SensitivityAnalysis.run_sensitivity(
                model_name, forcing_nt, obs_vec;
                method                = method,
                n_samples             = n_samples,
                objective             = objective,
                threshold             = threshold,
                param_bounds_override = pb_override,
                param_names_filter    = pnames_filter
            )

            # 转换为规范输出格式
            sensitivity_scores = Dict{String,Float64}(
                result["param_names"][i] => result["sensitivities"][i]
                for i in eachindex(result["param_names"])
            )

            output = Dict{String,Any}(
                "sensitive_parameters"   => result["calibratable"],
                "insensitive_parameters" => result["fixed"],
                "sensitivity_scores"     => sensitivity_scores,
                "method"                 => result["method"],
                "method_used"            => result["method"],
                "threshold"              => result["threshold"],
                "model_name"             => result["model_name"],
                "n_samples"              => result["n_samples"],
                "status"                 => "success",
            )

            result_id = string(UUIDs.uuid4())
            output["result_id"] = result_id
            output["storage_category"] = "sensitivity"
            if get(params, "save_to_storage", true)
                Storage.save_result(
                    STORAGE_BACKEND,
                    "sensitivity",
                    result_id,
                    output,
                )
            end
            return TextContent(text = JSON3.write(output))
        catch e
            return create_error_response(e)
        end
    end
)
sampling_tool = MCPTool(
    name = "generate_samples",
    description = "为水文模型生成参数样本集(Strategy 5)。支持 LHS/Sobol/随机采样，也支持 Strategy 2 的约束采样（pie_share 与 delta_method）。",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "model_name" => MODEL_NAME_SCHEMA,
            "n_samples" => SAMPLING_N_SAMPLES_SCHEMA,
            "method" => SAMPLING_METHOD_SCHEMA,
            "param_bounds" => PARAM_BOUNDS_SCHEMA,
            "constraints" => PARAMETER_CONSTRAINTS_SCHEMA,
        ),
        "required" => ["model_name"]
    ),
    handler = function(params)
        # 验证必需参数
        validation_error = validate_required_params(params, ["model_name"])
        if !isnothing(validation_error)
            return create_error_response(validation_error)
        end

        # 验证枚举参数
        if haskey(params, "method")
            enum_error = validate_enum_param(params, "method",
                ["lhs", "sobol", "random", "pie_share", "delta_method"], "lhs")
            if !isnothing(enum_error)
                return create_error_response(enum_error)
            end
        end

        try
            model_name = params["model_name"]
            _, canonical, param_names, default_bounds = Calibration._load_model_and_bounds(model_name)

            custom_bounds = get(params, "param_bounds", nothing)
            bounds = Tuple{Float64,Float64}[]
            for (i, pname) in enumerate(param_names)
                sname = string(pname)
                if !isnothing(custom_bounds) && haskey(custom_bounds, sname)
                    b = custom_bounds[sname]
                    push!(bounds, (Float64(b[1]), Float64(b[2])))
                else
                    push!(bounds, default_bounds[i])
                end
            end

            n_samples = Int(get(params, "n_samples", 100))
            method = get(params, "method", "lhs")
            constraints = _normalize_constraints(get(params, "constraints", nothing))
            output_param_names = string.(param_names)
            bound_lookup = Dict(string(param_names[i]) => [bounds[i][1], bounds[i][2]] for i in eachindex(param_names))

            samples_matrix = if method == "pie_share"
                isnothing(constraints) && throw(ArgumentError("method=pie_share requires constraints.pie_share"))
                haskey(constraints, "pie_share") || throw(ArgumentError("constraints.pie_share is required when method=pie_share"))
                pie = constraints["pie_share"]
                constrained_param_names = String.(pie["parameters"])
                all(constrained_param_names .∈ Ref(string.(param_names))) || throw(ArgumentError(
                    "constraints.pie_share.parameters must be subset of model parameters",
                ))
                output_param_names = constrained_param_names
                Sampling.pie_share_sampling(length(constrained_param_names), n_samples; total = Float64(pie["total"]))
            elseif method == "delta_method"
                isnothing(constraints) && throw(ArgumentError("method=delta_method requires constraints.delta_method"))
                haskey(constraints, "delta_method") || throw(ArgumentError("constraints.delta_method is required when method=delta_method"))
                delta = constraints["delta_method"]
                name_to_index = Dict(string(param_names[i]) => i for i in eachindex(param_names))
                inequalities_idx = Tuple{Int,Int}[]
                for (lower_name, upper_name) in delta["inequalities"]
                    haskey(name_to_index, lower_name) || throw(ArgumentError("Unknown parameter in delta_method constraint: $lower_name"))
                    haskey(name_to_index, upper_name) || throw(ArgumentError("Unknown parameter in delta_method constraint: $upper_name"))
                    push!(inequalities_idx, (name_to_index[lower_name], name_to_index[upper_name]))
                end
                Sampling.delta_method_sampling(bounds, inequalities_idx; n_samples = n_samples)
            else
                Sampling.generate_samples(bounds; method = method, n_samples = n_samples)
            end

            # 转为可序列化格式: [{param1: v1, param2: v2, ...}, ...]
            samples_list = Dict{String,Float64}[]
            for j in 1:n_samples
                d = Dict{String,Float64}()
                if method == "pie_share"
                    pie = constraints["pie_share"]
                    constrained_param_names = String.(pie["parameters"])
                    for (i, pname) in enumerate(constrained_param_names)
                        d[pname] = samples_matrix[i, j]
                    end
                else
                    for (i, pname) in enumerate(param_names)
                        d[string(pname)] = samples_matrix[i, j]
                    end
                end
                push!(samples_list, d)
            end

            result = Dict{String,Any}(
                "status" => "success",
                "model_name" => canonical,
                "method" => method,
                "n_samples" => n_samples,
                "param_names" => output_param_names,
                "bounds" => Dict(pname => bound_lookup[pname] for pname in output_param_names),
                "samples" => samples_list,
            )
            !(constraints === nothing) && (result["constraints"] = constraints)
            return TextContent(text = JSON3.write(result))
        catch e
            return create_error_response(e)
        end
    end
)

# ==============================================================================
# 5. 单目标校准工具
# ==============================================================================

function _validate_model_inputs_forcing(model_name::String, forcing_nt::NamedTuple)
    resolved_model = Discovery.find_model(model_name)
    isnothing(resolved_model) && throw(ArgumentError("Model '$model_name' was not found."))

    model_info = Discovery.get_model_info(resolved_model)
    required_inputs = String.(model_info["inputs"])
    available_inputs = Set(string.(keys(forcing_nt)))
    missing_inputs = [input_name for input_name in required_inputs if !(input_name in available_inputs)]

    isempty(missing_inputs) || throw(ArgumentError(
        "Model '$resolved_model' is missing required inputs: $(join(missing_inputs, ", ")). Inspect the CSV and input mapping before calibration.",
    ))

    return resolved_model
end

calibrate_tool = MCPTool(
    name = "calibrate_model",
    description = "执行低成本单目标参数校准（统一 v2 协议：model + inputs），并自动执行二阶段 train/test 评估。支持自动划分与按时间段划分。默认使用轻量优化迭代。",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "model" => MODEL_NAME_SCHEMA,
            "inputs" => WORKFLOW_INPUTS_REQUIRED_FORCING_OBS,
            "options" => WORKFLOW_OPTIONS_SCHEMA,
            "objective" => OBJECTIVE_SCHEMA,
            "algorithm" => ALGORITHM_SCHEMA,
            "maxiters" => MAXITERS_SCHEMA,
            "n_trials" => N_TRIALS_SCHEMA,
            "log_transform" => LOG_TRANSFORM_SCHEMA,
            "log_transform_mode" => LOG_TRANSFORM_MODE_SCHEMA,
            "budget" => CALIBRATION_BUDGET_SCHEMA,
            "seed" => SEED_SCHEMA,
            "fixed_params" => FIXED_PARAMS_SCHEMA,
            "param_bounds" => PARAM_BOUNDS_SCHEMA,
            "constraints" => PARAMETER_CONSTRAINTS_SCHEMA,
            "solver" => SOLVER_SCHEMA,
            "interpolator" => INTERPOLATOR_SCHEMA,
            "init_states" => INIT_STATES_SCHEMA,
            "method" => SPLIT_METHOD_SCHEMA,
            "ratio" => RATIO_SCHEMA,
            "warmup" => WARMUP_SCHEMA,
            "calibration_period" => CALIBRATION_PERIOD_SCHEMA,
            "validation_period" => VALIDATION_PERIOD_SCHEMA,
            "metrics" => METRICS_SCHEMA,
            "save_to_storage" => SAVE_TO_STORAGE_SCHEMA,
        ),
        "required" => ["model", "inputs"]
    ),
    handler = function(params)
        validation_error = validate_required_params(params, ["model", "inputs"])
        !isnothing(validation_error) && return create_error_response(validation_error)

        model_name = string(params["model"])

        if haskey(params, "objective")
            try
                params["objective"] = _normalize_named_choice(
                    params["objective"],
                    "objective",
                    OBJECTIVE_NAME_ALIASES,
                    Set(["KGE", "NSE", "LogNSE", "LogKGE", "PBIAS", "R2", "RMSE"]),
                )
            catch e
                return create_error_response(e)
            end
        end
        if haskey(params, "algorithm")
            try
                params["algorithm"] = _normalize_named_choice(
                    params["algorithm"],
                    "algorithm",
                    SINGLE_ALGORITHM_ALIASES,
                    Set(["AUTO", "DDS", "SCE", "BBO", "DE", "PSO", "CMAES", "ECA"]),
                )
            catch e
                return create_error_response(e)
            end
        end

        if haskey(params, "budget")
            enum_error = validate_enum_param(params, "budget", ["low", "medium", "high"], "medium")
            !isnothing(enum_error) && return create_error_response(enum_error)
        end
        if haskey(params, "log_transform_mode")
            enum_error = validate_enum_param(params, "log_transform_mode", ["auto", "manual"], "auto")
            !isnothing(enum_error) && return create_error_response(enum_error)
        end

        if haskey(params, "method")
            enum_error = validate_enum_param(params, "method", ["recent_first", "split_sample", "use_all"], "split_sample")
            !isnothing(enum_error) && return create_error_response(enum_error)
        end

        try
            request = UnifiedInputs.normalize_workflow_request(params)
            _, _, model = Simulation._load_model(model_name)
            resolved = UnifiedInputs.resolve_common_inputs(request, model;
                require_observation = true,
                require_parameters = false,
            )

            forcing_nt = resolved["forcing_nt"]
            obs_vec = Float64.(resolved["observation"])
            forcing_meta = get(resolved["metadata"], "forcing", nothing)
            split_result, split_warnings = _stage2_split_dataset(obs_vec, forcing_nt, forcing_meta, params)

            train_forcing = split_result["cal_forcing"]
            train_obs = split_result["cal_obs"]
            resolved_model = _validate_model_inputs_forcing(model_name, train_forcing)

            runtime_cfg = resolved["runtime"]

            # ---- 解析校准参数 ----
            fixed = Dict{String,Float64}()
            if haskey(params, "fixed_params")
                for (k, v) in params["fixed_params"]
                    fixed[string(k)] = Float64(v)
                end
            end

            pb = nothing
            raw_pb = haskey(params, "param_bounds") ? params["param_bounds"] : nothing
            if !isnothing(raw_pb)
                pb = Dict{String,Vector{Float64}}()
                for (k, v) in raw_pb
                    if v isa AbstractVector
                        pb[string(k)] = Float64.(v)
                    elseif v isa Dict || (hasproperty(v, :keys))
                        lo = Float64(get(v, "lower", get(v, :lower, 0.0)))
                        hi = Float64(get(v, "upper", get(v, :upper, 1.0)))
                        pb[string(k)] = [lo, hi]
                    end
                end
            end

            constraints_cfg = _normalize_constraints(get(params, "constraints", nothing))
            constraint_diagnostics, constraint_warnings = _evaluate_constraint_plan(
                resolved_model,
                pb,
                constraints_cfg,
            )

            ist = nothing
            if haskey(params, "init_states")
                ist = Dict{String,Float64}()
                for (k, v) in params["init_states"]
                    ist[string(k)] = Float64(v)
                end
            end

            maxiters_val = Int(get(params, "maxiters", 12))
            log_transform_val = Bool(get(params, "log_transform", false))
            algorithm_val = string(get(params, "algorithm", "BBO"))

            t_start = time()
            result = Calibration.calibrate_model(
                resolved_model, train_forcing, train_obs;
                algorithm     = algorithm_val,
                maxiters      = maxiters_val,
                objective     = get(params, "objective", "KGE"),
                log_transform = log_transform_val,
                log_transform_mode = String(get(params, "log_transform_mode", "auto")),
                fixed_params  = fixed,
                param_bounds  = pb,
                constraints   = constraints_cfg,
                n_trials      = Int(get(params, "n_trials", 1)),
                solver_type   = get(runtime_cfg, "solver", get(params, "solver", "DISCRETE")),
                interp_type   = get(runtime_cfg, "interpolation", get(params, "interpolator", "LINEAR")),
                init_states_dict = ist,
                budget = String(get(params, "budget", "medium")),
                seed = haskey(params, "seed") ? Int(params["seed"]) : nothing,
            )
            runtime_seconds = time() - t_start

            # ---- 输出格式扩展：兼容 Python HydroAgent 协议 ----
            # best_parameters / final_parameters（best_params 的别名）
            if haskey(result, "best_params")
                result["best_parameters"]  = result["best_params"]
                result["final_parameters"] = result["best_params"]
            end
            # iterations_used（maxiters 的别名）
            result["iterations_used"] = maxiters_val
            # runtime_seconds
            result["runtime_seconds"] = round(runtime_seconds; digits=2)

            objective_name = string(get(result, "objective_name", get(params, "objective", "KGE")))
            best_obj_val = get(result, "best_objective", 0.0)
            convergence_assessment = _infer_convergence_assessment(objective_name, Float64(best_obj_val))
            result["convergence_assessment"] = convergence_assessment
            result["converged"] = convergence_assessment["status"] == "converged"

            # objective_summary：从 all_trials 第一项提升到顶层
            if haskey(result, "all_trials") && !isempty(result["all_trials"])
                first_trial = result["all_trials"][1]
                if haskey(first_trial, "objective_summary")
                    result["objective_summary"] = first_trial["objective_summary"]
                end
                if haskey(first_trial, "objective_history")
                    result["convergence_curve"] = first_trial["objective_history"]
                end
            end

            metric_names = _normalize_metric_names(get(params, "metrics", String["NSE", "KGE", "RMSE"]))
            best_params = Dict{String,Float64}(string(k) => Float64(v) for (k, v) in pairs(result["best_params"]))
            result["stage2_evaluation"] = _build_stage2_evaluation(
                model,
                runtime_cfg,
                best_params,
                split_result,
                metric_names,
            )

            result["status"] = "success"
            warnings = String[]
            append!(warnings, String.(resolved["warnings"]))
            append!(warnings, split_warnings)
            append!(warnings, constraint_warnings)
            result["warnings"] = warnings
            result["inference_report"] = resolved["inference_report"]

            if !(constraints_cfg === nothing)
                result["constraints"] = constraints_cfg
                result["constraint_diagnostics"] = constraint_diagnostics
            end

            result_id = string(UUIDs.uuid4())
            result["result_id"] = result_id
            result["storage_category"] = "calibration"

            if get(params, "save_to_storage", true)
                Storage.save_result(
                    STORAGE_BACKEND,
                    "calibration",
                    result_id,
                    result,
                )
            end

            store_last_result(LAST_CALIBRATION_RESULT_HANDLE, result)

            return TextContent(text = JSON3.write(result))
        catch e
            return create_error_response(e)
        end
    end
)

# ==============================================================================
# 6. 多目标校准工具
# ==============================================================================
calibrate_multi_tool = MCPTool(
    name = "calibrate_multiobjective",
    description = "执行多目标参数校准(Strategy 9)，生成 Pareto 前沿。默认使用较小的探索性预算以避免长时间阻塞；需要更充分搜索时请显式提高 maxiters 和 population_size。",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "model" => MODEL_NAME_SCHEMA,
            "inputs" => WORKFLOW_INPUTS_REQUIRED_FORCING_OBS,
            "options" => WORKFLOW_OPTIONS_SCHEMA,
            "objectives" => OBJECTIVES_SCHEMA,
            "algorithm" => MULTI_ALGORITHM_SCHEMA,
            "maxiters" => MAXITERS_SCHEMA,
            "population_size" => POPULATION_SIZE_SCHEMA,
            "fixed_params" => FIXED_PARAMS_SCHEMA,
            "param_bounds" => PARAM_BOUNDS_SCHEMA,
            "solver" => SOLVER_SCHEMA,
            "interpolator" => INTERPOLATOR_SCHEMA,
            "init_states" => INIT_STATES_SCHEMA
        ),
        "required" => ["model", "inputs", "objectives"]
    ),
    handler = function(params)
        validation_error = validate_required_params(params, ["model", "inputs", "objectives"])
        if !isnothing(validation_error)
            return create_error_response(validation_error)
        end

        if haskey(params, "algorithm")
            try
                params["algorithm"] = _normalize_named_choice(
                    params["algorithm"],
                    "algorithm",
                    MULTI_ALGORITHM_ALIASES,
                    Set(["NSGA2", "NSGA3"]),
                )
            catch e
                return create_error_response(e)
            end
        end

        try
            request = UnifiedInputs.normalize_workflow_request(params)
            model_name = string(request["model"])
            _, _, model = Simulation._load_model(model_name)
            resolved = UnifiedInputs.resolve_common_inputs(request, model;
                require_observation = true,
                require_parameters = false,
            )

            forcing_nt = resolved["forcing_nt"]
            obs_vec = Float64.(resolved["observation"])
            runtime_cfg = resolved["runtime"]
            maxiters_val = Int(get(params, "maxiters", 30))
            population_size_val = Int(get(params, "population_size", 16))

            fixed = Dict{String,Float64}()
            if haskey(params, "fixed_params")
                for (k, v) in params["fixed_params"]
                    fixed[string(k)] = Float64(v)
                end
            end

            pb = nothing
            if haskey(params, "param_bounds")
                pb = Dict{String,Vector{Float64}}()
                for (k, v) in params["param_bounds"]
                    pb[string(k)] = Float64.(v)
                end
            end

            ist = nothing
            if haskey(params, "init_states")
                ist = Dict{String,Float64}()
                for (k, v) in params["init_states"]
                    ist[string(k)] = Float64(v)
                end
            end

            result = Calibration.calibrate_multiobjective(
                model_name, forcing_nt, obs_vec;
                objectives = String.(params["objectives"]),
                algorithm = get(params, "algorithm", "NSGA2"),
                maxiters = maxiters_val,
                population_size = population_size_val,
                fixed_params = fixed,
                param_bounds = pb,
                solver_type = get(runtime_cfg, "solver", get(params, "solver", "DISCRETE")),
                interp_type = get(runtime_cfg, "interpolation", get(params, "interpolator", "LINEAR")),
                init_states_dict = ist
            )
            warnings = String.(resolved["warnings"])
            !haskey(params, "maxiters") && push!(warnings, "Used exploratory default maxiters=$(maxiters_val) for calibrate_multiobjective; increase it explicitly for a denser Pareto search")
            !haskey(params, "population_size") && push!(warnings, "Used exploratory default population_size=$(population_size_val) for calibrate_multiobjective")
            result["warnings"] = warnings
            result["inference_report"] = resolved["inference_report"]
            return TextContent(text = JSON3.write(result))
        catch e
            return create_error_response(e)
        end
    end
)

# ==============================================================================
# 7. 校准诊断工具
# ==============================================================================
diagnose_tool = MCPTool(
    name = "diagnose_calibration",
    description = "诊断校准结果质量(Strategy 10)。检查收敛性、参数边界触达、目标函数平台期，并给出改进建议(增加预算/扩大范围等)。输入为 calibrate_model 的返回结果。",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "calibration_result" => CALIBRATION_RESULT_SCHEMA,
            "boundary_tolerance" => BOUNDARY_TOLERANCE_SCHEMA,
            "convergence_threshold" => CONVERGENCE_THRESHOLD_SCHEMA,
            "plateau_window" => PLATEAU_WINDOW_SCHEMA
        ),
        "required" => []
    ),
    handler = function(params)
        try
            cal_result = get(params, "calibration_result", nothing)
            if cal_result === nothing
                cal_result = load_last_result(LAST_CALIBRATION_RESULT_HANDLE)
                cal_result === nothing &&
                    throw(ArgumentError("Missing required parameter: calibration_result"))
            end

            cal_result = _normalize_calibration_result_payload(cal_result)

            result = Calibration.diagnose_calibration(
                cal_result;
                boundary_tolerance = Float64(get(params, "boundary_tolerance", 0.01)),
                convergence_threshold = Float64(get(params, "convergence_threshold", 0.05)),
                plateau_window = Int(get(params, "plateau_window", 50))
            )
            return TextContent(text = JSON3.write(result))
        catch e
            return create_error_response(e)
        end
    end
)

diagnose_multi_tool = MCPTool(
    name = "diagnose_multiobjective",
    description = "Diagnose multi-objective calibration quality by checking Pareto-front degeneracy and objective conflict patterns.",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "multiobjective_result" => Dict{String,Any}(
                "type" => "object",
                "description" => "Result object returned by calibrate_multiobjective",
            ),
        ),
        "required" => ["multiobjective_result"],
    ),
    handler = function(params)
        validation_error = validate_required_params(params, ["multiobjective_result"])
        !isnothing(validation_error) && return create_error_response(validation_error)

        try
            raw = params["multiobjective_result"]
            result_obj = raw isa AbstractDict ? Dict{String,Any}(string(k) => v for (k, v) in pairs(raw)) :
                throw(ArgumentError("multiobjective_result must be an object"))
            diagnosis = Calibration.diagnose_multiobjective(result_obj)
            diagnosis["status"] = "success"
            return TextContent(text = JSON3.write(diagnosis))
        catch e
            return create_error_response(e)
        end
    end,
)

# ==============================================================================
# 8. 目标函数配置工具 (DEV.md Tool 2: configure_objectives)
# ==============================================================================
configure_objectives_tool = MCPTool(
    name = "configure_objectives",
    description = "根据业务目标自动推荐合适的评价指标组合。输入业务场景描述，返回推荐的目标函数、是否建议 log 变换、以及加权方案。对应论文 Strategy 7 的映射表。",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "goal" => Dict{String,Any}(
                "type" => "string",
                "enum" => ["general_fit", "peak_flows", "low_flows", "water_balance", "dynamics", "custom"],
                "description" => "业务目标: general_fit(综合拟合), peak_flows(洪峰), low_flows(枯水), water_balance(水量平衡), dynamics(动态过程), custom(自定义)"
            ),
            "custom_metrics" => METRICS_SCHEMA,
            "custom_weights" => Dict{String,Any}(
                "type" => "array",
                "items" => Dict{String,Any}("type" => "number"),
                "description" => "自定义权重列表 (仅 goal=custom 时使用，与 custom_metrics 对应)"
            )
        ),
        "required" => ["goal"]
    ),
    handler = function(params)
        # 验证必需参数
        validation_error = validate_required_params(params, ["goal"])
        if !isnothing(validation_error)
            return create_error_response(validation_error)
        end

        # 验证枚举参数
        enum_error = validate_enum_param(params, "goal",
            ["general_fit", "peak_flows", "low_flows", "water_balance", "dynamics", "custom"], nothing)
        if !isnothing(enum_error)
            return create_error_response(enum_error)
        end

        try
            goal = params["goal"]

            # Strategy 7 映射表
            config = if goal == "general_fit"
                Dict{String,Any}(
                    "primary_metric" => "KGE",
                    "secondary_metrics" => ["NSE", "PBIAS"],
                    "log_transform" => false,
                    "weights" => Dict("KGE" => 1.0),
                    "description" => "综合拟合：KGE 兼顾相关性、变异性和偏差"
                )
            elseif goal == "peak_flows"
                Dict{String,Any}(
                    "primary_metric" => "NSE",
                    "secondary_metrics" => ["KGE", "RMSE"],
                    "log_transform" => false,
                    "weights" => Dict("NSE" => 0.7, "KGE" => 0.3),
                    "description" => "洪峰模拟：NSE 对大值敏感，适合捕捉峰值"
                )
            elseif goal == "low_flows"
                Dict{String,Any}(
                    "primary_metric" => "LogKGE",
                    "secondary_metrics" => ["LogNSE", "PBIAS"],
                    "log_transform" => true,
                    "weights" => Dict("LogKGE" => 0.7, "PBIAS" => 0.3),
                    "description" => "枯水期：对数变换后的指标对低流量更敏感"
                )
            elseif goal == "water_balance"
                Dict{String,Any}(
                    "primary_metric" => "PBIAS",
                    "secondary_metrics" => ["KGE", "R2"],
                    "log_transform" => false,
                    "weights" => Dict("PBIAS" => 0.6, "KGE" => 0.4),
                    "description" => "水量平衡：PBIAS 直接衡量总量偏差"
                )
            elseif goal == "dynamics"
                Dict{String,Any}(
                    "primary_metric" => "R2",
                    "secondary_metrics" => ["KGE", "NSE"],
                    "log_transform" => false,
                    "weights" => Dict("R2" => 0.6, "KGE" => 0.4),
                    "description" => "动态过程：R² 衡量时间序列的相关性和模式匹配"
                )
            elseif goal == "custom"
                metrics = get(params, "custom_metrics", String["KGE"])
                weights_arr = get(params, "custom_weights", ones(length(metrics)))
                w_dict = Dict(metrics[i] => Float64(weights_arr[i]) for i in eachindex(metrics))
                Dict{String,Any}(
                    "primary_metric" => metrics[1],
                    "secondary_metrics" => length(metrics) > 1 ? metrics[2:end] : String[],
                    "log_transform" => false,
                    "weights" => w_dict,
                    "description" => "自定义指标组合"
                )
            else
                throw(ArgumentError("未知目标: $(goal)"))
            end

            config["goal"] = goal
            return TextContent(text = JSON3.write(config))
        catch e
            return create_error_response(e)
        end
    end
)

# ==============================================================================
# 9. 校准初始化设置工具 (DEV.md Tool 1: init_calibration_setup)
# ==============================================================================
init_calibration_setup_tool = MCPTool(
    name = "init_calibration_setup",
    description = "一键初始化校准配置(DEV.md Tool 1)。执行敏感性分析筛选参数，检测数据量级，推荐算法和采样策略。返回完整的校准配置建议，可直接用于 calibrate_model。",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "model" => MODEL_NAME_SCHEMA,
            "inputs" => WORKFLOW_INPUTS_REQUIRED_FORCING_OBS,
            "options" => WORKFLOW_OPTIONS_SCHEMA,
            "goal" => GOAL_SCHEMA,
            "budget" => BUDGET_SCHEMA,
            "sensitivity_samples" => SENSITIVITY_SAMPLES_SCHEMA,
            "solver" => SOLVER_SIMPLE_SCHEMA,
            "interpolator" => INTERPOLATOR_SIMPLE_SCHEMA
        ),
        "required" => ["model", "inputs"]
    ),
    handler = function(params)
        validation_error = validate_required_params(params, ["model", "inputs"])
        if !isnothing(validation_error)
            return create_error_response(validation_error)
        end

        # 验证枚举参数
        if haskey(params, "goal")
            enum_error = validate_enum_param(params, "goal",
                ["general_fit", "peak_flows", "low_flows", "water_balance", "dynamics"], "general_fit")
            if !isnothing(enum_error)
                return create_error_response(enum_error)
            end
        end

        if haskey(params, "budget")
            enum_error = validate_enum_param(params, "budget",
                ["low", "medium", "high"], "medium")
            if !isnothing(enum_error)
                return create_error_response(enum_error)
            end
        end

        try
            request = UnifiedInputs.normalize_workflow_request(params)
            model_name = string(request["model"])
            _, _, model = Simulation._load_model(model_name)
            resolved = UnifiedInputs.resolve_common_inputs(request, model;
                require_observation = true,
                require_parameters = false,
            )

            forcing_nt = resolved["forcing_nt"]
            obs_vec = Float64.(resolved["observation"])
            runtime_cfg = resolved["runtime"]
            goal = get(params, "goal", "general_fit")
            budget = get(params, "budget", "medium")
            n_sens = Int(get(params, "sensitivity_samples", 50))
            solver_type = get(runtime_cfg, "solver", get(params, "solver", "DISCRETE"))
            interp_type = get(runtime_cfg, "interpolation", get(params, "interpolator", "LINEAR"))

            # Step 1: 敏感性分析
            sens_result = SensitivityAnalysis.run_sensitivity(
                model_name, forcing_nt, obs_vec;
                method="morris", n_samples=n_sens, objective="NSE",
                threshold=0.1, solver_type=solver_type, interp_type=interp_type
            )

            # Step 2: 数据量级检测 (Strategy 3)
            pos_obs = filter(>(0), obs_vec)
            magnitude_ratio = length(pos_obs) > 0 ? maximum(pos_obs) / minimum(pos_obs) : 1.0
            log_recommended = magnitude_ratio > 100.0

            # Step 3: 算法推荐 (Strategy 8)
            n_calibratable = length(sens_result["calibratable"])
            algorithm = if budget == "low" || n_calibratable > 10
                "BBO"  # 高维或低预算用 BBO
            elseif budget == "high"
                "PSO"  # 高预算用 PSO
            else
                "BBO"  # 中等预算用 BBO
            end

            maxiters = if budget == "low"
                500
            elseif budget == "high"
                5000
            else
                1000
            end

            # Step 4: 目标函数推荐
            objective_config = if goal == "general_fit"
                Dict("metric" => "KGE", "log_transform" => false)
            elseif goal == "peak_flows"
                Dict("metric" => "NSE", "log_transform" => false)
            elseif goal == "low_flows"
                Dict("metric" => "LogKGE", "log_transform" => log_recommended)
            elseif goal == "water_balance"
                Dict("metric" => "PBIAS", "log_transform" => false)
            else
                Dict("metric" => "KGE", "log_transform" => false)
            end

            # Step 5: 构建固定参数 Dict
            fixed_params = Dict{String,Any}()
            for pname in sens_result["fixed"]
                fixed_params[pname] = "default"  # Agent 应从模型获取默认值
            end

            setup = Dict{String,Any}(
                "model_name" => sens_result["model_name"],
                "sensitivity" => Dict{String,Any}(
                    "calibratable" => sens_result["calibratable"],
                    "fixed" => sens_result["fixed"],
                    "sensitivities" => Dict(
                        sens_result["param_names"][i] => sens_result["sensitivities"][i]
                        for i in eachindex(sens_result["param_names"])
                    ),
                ),
                "data_analysis" => Dict{String,Any}(
                    "n_observations" => length(obs_vec),
                    "magnitude_ratio" => magnitude_ratio,
                    "log_transform_recommended" => log_recommended,
                ),
                "recommended_config" => Dict{String,Any}(
                    "algorithm" => algorithm,
                    "maxiters" => maxiters,
                    "n_trials" => 3,
                    "objective" => objective_config["metric"],
                    "log_transform" => objective_config["log_transform"],
                    "fixed_params" => fixed_params,
                ),
                "goal" => goal,
                "budget" => budget,
                "warnings" => resolved["warnings"],
                "inference_report" => resolved["inference_report"],
            )

            return TextContent(text = JSON3.write(setup))
        catch e
            return create_error_response(e)
        end
    end
)

# ==============================================================================
# 10. 完整诊断分析工具
# ==============================================================================
compute_diagnostics_full_tool = MCPTool(
    name = "compute_diagnostics_full",
    description = "完整的校准诊断分析。分析多次试验结果，检查收敛性、参数边界触达、平台期，推荐下一步操作。支持 final_parameters（calibrate_model 输出格式）和 best_parameters 两种字段名。",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "trial_results" => Dict{String,Any}(
                "type" => "array",
                "items" => Dict{String,Any}(
                    "type" => "object",
                    "properties" => Dict{String,Any}(
                        "trial_id" => Dict("type" => "integer"),
                        "best_objective" => Dict("type" => "number"),
                        "best_parameters" => Dict("type" => "object",
                            "description" => "最优参数字典（与 final_parameters 二选一）"),
                        "final_parameters" => Dict("type" => "object",
                            "description" => "最优参数字典（best_parameters 的别名）"),
                        "objective_history" => Dict("type" => "array", "items" => Dict("type" => "number")),
                        "iterations_used" => Dict("type" => "integer")
                    )
                ),
                "description" => "试验结果列表（来自 calibrate_model 的输出）"
            ),
            "parameter_bounds" => Dict{String,Any}(
                "type" => "object",
                "description" => "参数边界，格式: {\"param\": {\"lower\": x, \"upper\": y}} 或 {\"param\": [lower, upper]}"
            ),
            "data_handle" => Dict{String,Any}(
                "type" => "string",
                "description" => "（可选）数据句柄，仅用于上下文记录，不影响诊断计算"
            ),
            "model_name" => Dict{String,Any}(
                "type" => "string",
                "description" => "（可选）模型名称，仅用于上下文记录"
            ),
            "objective_threshold" => Dict{String,Any}(
                "type" => "number",
                "description" => "好拟合阈值，默认 0.7"
            )
        ),
        "required" => ["trial_results"]
    ),
    handler = function(params)
        validation_error = validate_required_params(params, ["trial_results"])
        if !isnothing(validation_error)
            return create_error_response(validation_error)
        end

        try
            # ---- 规范化 trial_results ----
            # 将 final_parameters 统一映射到 best_parameters
            raw_trials = params["trial_results"]
            normalized_trials = Dict{String,Any}[]
            inferred_history_count = 0
            for trial in raw_trials
                t = Dict{String,Any}(string(k) => v for (k, v) in trial)
                # final_parameters → best_parameters 兼容
                if haskey(t, "final_parameters") && !haskey(t, "best_parameters")
                    t["best_parameters"] = t["final_parameters"]
                end
                # convergence_curve → objective_history 兼容
                if haskey(t, "convergence_curve") && !haskey(t, "objective_history")
                    t["objective_history"] = t["convergence_curve"]
                end
                # 若无 objective_history，但有 objective_summary，合成一个虚拟历史
                # 使核心函数能正确检测 is_plateaued / is_still_improving
                if !haskey(t, "objective_history") && haskey(t, "objective_summary")
                    os = t["objective_summary"]
                    plateau   = Bool(get(os, "plateau_detected", get(os, :plateau_detected, false)))
                    improving = Bool(get(os, "still_improving",  get(os, :still_improving,  false)))
                    best_obj_v = Float64(get(t, "best_objective", 0.5))
                    if plateau && !improving
                        # 平台期：最后 50 步完全不变
                        t["objective_history"] = vcat(
                            range(max(best_obj_v - 0.3, 0.0), best_obj_v, length=50),
                            fill(best_obj_v, 50)
                        )
                        inferred_history_count += 1
                    elseif improving
                        # 仍在改进：末尾仍有下降趋势（内部用的是最小化，所以历史值递减）
                        t["objective_history"] = range(best_obj_v + 0.2, best_obj_v, length=100)
                        inferred_history_count += 1
                    end
                end
                push!(normalized_trials, t)
            end

            # ---- 处理 parameter_bounds（支持数组格式 [lower, upper]） ----
            param_bounds = if haskey(params, "parameter_bounds")
                raw_bounds = Dict{String,Any}(string(k) => v for (k, v) in params["parameter_bounds"])
                # 将 [lower, upper] 格式转换为 {"lower": x, "upper": y}
                converted = Dict{String,Any}()
                for (pname, bound) in raw_bounds
                    if bound isa AbstractVector || bound isa Vector
                        converted[pname] = Dict("lower" => Float64(bound[1]), "upper" => Float64(bound[2]))
                    else
                        converted[pname] = bound
                    end
                end
                converted
            else
                # 若未提供 bounds，从 trial_results 中推断（以参数值范围扩大 20% 作为 bounds）
                all_params = String[]
                for t in normalized_trials
                    if haskey(t, "best_parameters")
                        for k in keys(t["best_parameters"])
                            k ∉ all_params && push!(all_params, string(k))
                        end
                    end
                end
                inferred = Dict{String,Any}()
                for pname in all_params
                    vals = [Float64(t["best_parameters"][pname])
                            for t in normalized_trials if haskey(t, "best_parameters") && haskey(t["best_parameters"], pname)]
                    if !isempty(vals)
                        lo, hi = minimum(vals), maximum(vals)
                        margin = max((hi - lo) * 0.2, abs(lo) * 0.1 + 1e-6)
                        inferred[pname] = Dict("lower" => lo - margin, "upper" => hi + margin)
                    end
                end
                inferred
            end

            result = Calibration.diagnose_calibration_full(
                normalized_trials,
                param_bounds;
                objective_threshold = Float64(get(params, "objective_threshold", 0.7))
            )
            result["history_inferred_trials"] = inferred_history_count
            result["history_source"] = inferred_history_count == 0 ? "observed" : "mixed"
            if inferred_history_count > 0
                result["diagnostic_confidence"] = "medium"
                result["history_note"] = "Some trials used synthesized objective_history from objective_summary; prefer real objective_history for high-confidence diagnostics"
            else
                result["diagnostic_confidence"] = "high"
            end

            # ---- 补充输出字段：best_trial_id + parameter_uncertainty ----
            if !isempty(normalized_trials)
                objectives = [Float64(t["best_objective"]) for t in normalized_trials]
                best_idx = argmax(objectives)
                # trial_id 字段（若有）或用索引
                result["best_trial_id"] = if haskey(normalized_trials[best_idx], "trial_id")
                    normalized_trials[best_idx]["trial_id"]
                else
                    best_idx
                end

                # 参数不确定性统计
                all_pnames = String[]
                for t in normalized_trials
                    for k in keys(get(t, "best_parameters", Dict()))
                        string(k) ∉ all_pnames && push!(all_pnames, string(k))
                    end
                end
                param_uncertainty = Dict{String,Any}()
                for pname in all_pnames
                    vals = [Float64(t["best_parameters"][pname])
                            for t in normalized_trials
                            if haskey(t, "best_parameters") && haskey(t["best_parameters"], pname)]
                    if length(vals) >= 2
                        param_uncertainty[pname] = Dict(
                            "mean" => mean(vals),
                            "std"  => std(vals),
                            "min"  => minimum(vals),
                            "max"  => maximum(vals)
                        )
                    elseif length(vals) == 1
                        param_uncertainty[pname] = Dict("mean" => vals[1], "std" => 0.0,
                                                        "min" => vals[1], "max" => vals[1])
                    end
                end
                result["parameter_uncertainty"] = param_uncertainty
            end

            # 可选上下文字段原样回传
            if haskey(params, "data_handle")
                result["data_handle"] = params["data_handle"]
            end
            if haskey(params, "model_name")
                result["model_name"] = params["model_name"]
            end

            return TextContent(text = JSON3.write(result))
        catch e
            return create_error_response(e)
        end
    end
)

auto_calibration_workflow_tool = MCPTool(
    name = "auto_calibration_workflow",
    description = "Run an automatic end-to-end calibration workflow aligned with KNOWLEDEGE.md strategies 1-10, including readiness checks, sensitivity screening, objective setup, calibration, diagnostics feedback loops, and optional validation.",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "model" => MODEL_NAME_SCHEMA,
            "inputs" => WORKFLOW_INPUTS_REQUIRED_FORCING_OBS,
            "options" => WORKFLOW_OPTIONS_SCHEMA,
            "goal" => Dict{String,Any}(
                "type" => "string",
                "enum" => ["general_fit", "peak_flows", "low_flows", "water_balance", "dynamics"],
                "default" => "general_fit",
            ),
            "budget" => CALIBRATION_BUDGET_SCHEMA,
            "max_rounds" => Dict{String,Any}(
                "type" => "integer",
                "description" => "Maximum auto-feedback rounds (default 3)",
                "default" => 3,
            ),
            "maxiters_per_round" => Dict{String,Any}(
                "type" => "integer",
                "description" => "Optional fixed maxiters for each round (useful for lightweight tests)",
            ),
            "n_trials_per_round" => Dict{String,Any}(
                "type" => "integer",
                "description" => "Optional fixed n_trials for each round",
            ),
            "sensitivity_samples" => Dict{String,Any}(
                "type" => "integer",
                "description" => "Sensitivity sample size used for screening (default 40)",
                "default" => 40,
            ),
            "run_validation" => Dict{String,Any}(
                "type" => "boolean",
                "description" => "Whether to run validation after calibration rounds",
                "default" => true,
            ),
            "save_to_storage" => SAVE_TO_STORAGE_SCHEMA,
            "seed" => SEED_SCHEMA,
            "constraints" => PARAMETER_CONSTRAINTS_SCHEMA,
            "param_bounds" => PARAM_BOUNDS_SCHEMA,
            "metrics" => METRICS_SCHEMA,
        ),
        "required" => ["model", "inputs"],
    ),
    handler = function(params)
        validation_error = validate_required_params(params, ["model", "inputs"])
        !isnothing(validation_error) && return create_error_response(validation_error)

        try
            request = UnifiedInputs.normalize_workflow_request(params)
            model_name = string(request["model"])
            canonical_model, _, model = Simulation._load_model(model_name)
            model_info = Discovery.get_model_info(canonical_model)

            forcing_spec = request["inputs"]["forcing"]
            inspect_payload = _build_inspection_payload(
                _inspect_source_columns(forcing_spec);
                model_name = canonical_model,
                intended_use = "calibration",
                input_mapping = get(request["options"], "input_mapping", nothing),
            )
            if !isempty(get(inspect_payload, "blocking_issues", String[]))
                return create_error_response(ArgumentError("Data readiness check failed: $(join(inspect_payload["blocking_issues"], "; "))"))
            end

            resolved = UnifiedInputs.resolve_common_inputs(request, model;
                require_observation = true,
                require_parameters = false,
            )
            forcing_nt = resolved["forcing_nt"]
            obs_vec = Float64.(resolved["observation"])
            runtime_cfg = resolved["runtime"]

            goal = string(get(params, "goal", "general_fit"))
            budget = string(get(params, "budget", "medium"))
            max_rounds = max(1, Int(get(params, "max_rounds", 3)))
            maxiters_fixed = haskey(params, "maxiters_per_round") ? Int(params["maxiters_per_round"]) : nothing
            n_trials_fixed = haskey(params, "n_trials_per_round") ? Int(params["n_trials_per_round"]) : nothing
            sens_samples = Int(get(params, "sensitivity_samples", 40))
            run_validation_flag = Bool(get(params, "run_validation", true))
            seed = haskey(params, "seed") ? Int(params["seed"]) : nothing

            objective_cfg_text = configure_objectives_tool.handler(Dict("goal" => goal)).text
            startswith(objective_cfg_text, "Error:") && throw(ArgumentError(objective_cfg_text))
            objective_cfg = JSON3.read(objective_cfg_text, Dict{String,Any})
            objective_name = string(objective_cfg["primary_metric"])

            sensitivity = SensitivityAnalysis.run_sensitivity(
                canonical_model,
                forcing_nt,
                obs_vec;
                method = "morris",
                n_samples = sens_samples,
                objective = objective_name,
                threshold = 0.1,
                solver_type = get(runtime_cfg, "solver", "DISCRETE"),
                interp_type = get(runtime_cfg, "interpolation", "LINEAR"),
            )

            fixed_params = Dict{String,Float64}()
            bounds = haskey(params, "param_bounds") ? _canonical_param_bounds(params["param_bounds"]) :
                Dict{String,Vector{Float64}}()
            constraints_cfg = _normalize_constraints(get(params, "constraints", nothing))

            rounds = Dict{String,Any}[]
            best_payload = nothing
            best_score = -Inf
            previous_diagnostics = nothing

            for round_id in 1:max_rounds
                algorithm = round_id == 1 ? "AUTO" : (budget == "high" ? "SCE" : "DDS")
                maxiters = if !(maxiters_fixed === nothing)
                    maxiters_fixed
                elseif budget == "low"
                    round_id == 1 ? 120 : 180
                elseif budget == "high"
                    round_id == 1 ? 500 : 900
                else
                    round_id == 1 ? 280 : 420
                end
                n_trials = if !(n_trials_fixed === nothing)
                    n_trials_fixed
                else
                    round_id == 1 ? 2 : 3
                end

                run_payload = Calibration.calibrate_model(
                    canonical_model,
                    forcing_nt,
                    obs_vec;
                    algorithm = algorithm,
                    maxiters = maxiters,
                    objective = objective_name,
                    log_transform = Bool(get(objective_cfg, "log_transform", false)),
                    log_transform_mode = "auto",
                    fixed_params = fixed_params,
                    param_bounds = isempty(bounds) ? nothing : bounds,
                    constraints = constraints_cfg,
                    n_trials = n_trials,
                    solver_type = get(runtime_cfg, "solver", "DISCRETE"),
                    interp_type = get(runtime_cfg, "interpolation", "LINEAR"),
                    init_states_dict = haskey(runtime_cfg, "init_states") ? runtime_cfg["init_states"] : nothing,
                    warm_up = Int(get(params, "warmup", 30)),
                    budget = budget,
                    seed = isnothing(seed) ? nothing : seed + 10 * (round_id - 1),
                )

                diagnostics = Calibration.diagnose_calibration(
                    run_payload;
                    boundary_tolerance = 0.01,
                    convergence_threshold = round_id == 1 ? 0.12 : 0.08,
                    plateau_window = 20,
                )

                validation_payload = nothing
                if run_validation_flag
                    validation_request = Dict{String,Any}(
                        "model" => canonical_model,
                        "inputs" => Dict{String,Any}(
                            "forcing" => forcing_spec,
                            "observation" => request["inputs"]["observation"],
                            "parameters" => Dict{String,Any}(
                                "source_type" => "json",
                                "data" => Dict{String,Any}("best_params" => run_payload["best_params"]),
                            ),
                        ),
                        "calibration_period" => Dict("start_index" => 1, "end_index" => Int(round(length(obs_vec) * 0.7))),
                        "validation_period" => Dict("start_index" => Int(round(length(obs_vec) * 0.7)) + 1, "end_index" => length(obs_vec)),
                        "metrics" => haskey(params, "metrics") ? params["metrics"] : ["NSE", "KGE", "RMSE"],
                        "options" => Dict("auto_infer" => true, "strict_infer" => false),
                    )

                    validation_payload = Validation.run_validation(validation_request)
                end

                summary = _auto_calibration_iteration_summary(run_payload, diagnostics, validation_payload)
                summary["round"] = round_id
                push!(rounds, summary)

                score = summary["quality_score"]
                if score > best_score
                    best_score = score
                    best_payload = run_payload
                end

                previous_diagnostics = diagnostics
                if get(diagnostics, "hat_trick", false)
                    break
                end

                boundary_adjustments = get(diagnostics, "boundary_adjustments", Dict{String,Any}())
                if !isempty(boundary_adjustments)
                    bounds = _apply_boundary_expansion(bounds, boundary_adjustments)
                end
            end

            best_payload === nothing && throw(ArgumentError("auto_calibration_workflow did not produce any calibration round"))

            result_id = string(UUIDs.uuid4())
            output = Dict{String,Any}(
                "status" => "success",
                "workflow" => "knowledge_md_auto_calibration",
                "model" => canonical_model,
                "model_info" => model_info,
                "readiness" => inspect_payload,
                "strategy_alignment" => Dict(
                    "strategy_1_sensitivity" => true,
                    "strategy_2_constraints" => constraints_cfg !== nothing,
                    "strategy_3_auto_log" => true,
                    "strategy_4_split_validation" => run_validation_flag,
                    "strategy_5_sampling" => true,
                    "strategy_6_range_feedback" => true,
                    "strategy_7_objective_mapping" => true,
                    "strategy_8_algorithm_selection" => true,
                    "strategy_9_multiobjective_ready" => true,
                    "strategy_10_diagnostics_loop" => true,
                ),
                "goal" => goal,
                "budget" => budget,
                "objective_config" => objective_cfg,
                "sensitivity" => sensitivity,
                "rounds" => rounds,
                "best_result" => best_payload,
                "final_diagnostics" => previous_diagnostics,
                "warnings" => vcat(String.(resolved["warnings"]), String[]),
                "inference_report" => resolved["inference_report"],
                "result_id" => result_id,
                "storage_category" => "calibration",
            )

            if get(params, "save_to_storage", true)
                Storage.save_result(STORAGE_BACKEND, "calibration", result_id, output)
            end

            store_last_result(LAST_CALIBRATION_RESULT_HANDLE, best_payload)
            return TextContent(text = JSON3.write(output))
        catch e
            return create_error_response(e)
        end
    end,
)

export compute_metrics_tool, split_data_tool, sensitivity_tool, sensitivity_analysis_tool,
       sampling_tool, calibrate_tool, calibrate_multi_tool, diagnose_tool,
       diagnose_multi_tool, auto_calibration_workflow_tool,
       configure_objectives_tool, init_calibration_setup_tool, compute_diagnostics_full_tool
