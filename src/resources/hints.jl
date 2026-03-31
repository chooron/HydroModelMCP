using ModelContextProtocol

const LLM_HINT_SPECS = [
    Dict{String,Any}(
        "feature" => "simulation_v2",
        "title" => "Simulation v2 Accuracy Hints",
        "aliases" => [
            "simulation",
            "run_simulation",
            "simulate",
            "sim",
            "模拟",
            "径流模拟",
        ],
        "when_to_load" => [
            "Before calling run_simulation with workspace files",
            "When LLM output mixes legacy flat payloads with v2 inputs",
        ],
        "trigger_tools" => ["run_simulation", "compute_metrics", "inspect_hydro_data"],
        "required_prechecks" => [
            "inspect_hydro_data with intended_use=simulation",
        ],
        "recommended_sequence" => [
            "find_model",
            "get_model_info",
            "inspect_hydro_data",
            "run_simulation",
            "compute_metrics (if observation exists)",
        ],
        "payload_contract" => Dict(
            "required" => ["model", "inputs.forcing"],
            "optional" => ["inputs.parameters", "inputs.observation", "inputs.runtime", "options", "output"],
            "forbidden_legacy" => ["top-level source_type/path without inputs block"],
        ),
        "accuracy_tips" => [
            "Prefer canonical forcing names (P/T/Ep) or provide options.input_mapping",
            "Use explicit observation column when available to reduce inference ambiguity",
            "With options.strict_infer=true, ambiguous top candidates are rejected instead of auto-fallback",
        ],
    ),
    Dict{String,Any}(
        "feature" => "calibration_stage2",
        "title" => "Calibration Stage2 Accuracy Hints",
        "aliases" => [
            "calibration",
            "calibrate",
            "calibrate_model",
            "stage2",
            "train_test",
            "rate",
            "率定",
            "参数率定",
            "二阶段率定",
        ],
        "when_to_load" => [
            "Before calibrate_model execution",
            "When user asks for train/test split evaluation",
        ],
        "trigger_tools" => ["calibrate_model", "split_data", "diagnose_calibration"],
        "required_prechecks" => [
            "inspect_hydro_data with intended_use=calibration",
        ],
        "recommended_sequence" => [
            "configure_objectives",
            "calibrate_model",
            "diagnose_calibration",
        ],
        "payload_contract" => Dict(
            "required" => ["model", "inputs.forcing", "inputs.observation"],
            "split_auto" => ["method", "ratio", "warmup"],
            "split_period" => ["calibration_period", "validation_period"],
            "result_key_fields" => ["best_params", "stage2_evaluation", "warnings"],
        ),
        "lightweight_defaults" => Dict(
            "maxiters" => 12,
            "n_trials" => 1,
            "algorithm" => "BBO",
        ),
        "accuracy_tips" => [
            "Provide calibration_period and validation_period together",
            "Specify metrics for stage2 evaluation to align expected output",
        ],
    ),
    Dict{String,Any}(
        "feature" => "validation_v2",
        "title" => "Validation v2 Accuracy Hints",
        "aliases" => [
            "validation",
            "run_validation",
            "validate",
            "验证",
            "模型验证",
        ],
        "when_to_load" => [
            "Before run_validation",
            "When same-session parameter fallback may be triggered",
        ],
        "trigger_tools" => ["run_validation", "compute_metrics", "inspect_hydro_data"],
        "required_prechecks" => [
            "inspect_hydro_data with intended_use=validation",
        ],
        "recommended_sequence" => [
            "inspect_hydro_data",
            "run_validation",
            "compute_metrics",
        ],
        "payload_contract" => Dict(
            "required" => ["model", "inputs.forcing", "inputs.observation"],
            "recommended" => ["inputs.parameters"],
            "fallback_rule" => "Parameter omission is acceptable only with explicit same-session fallback diagnostics",
        ),
        "accuracy_tips" => [
            "Prefer explicit inputs.parameters for reproducibility",
            "Check warnings and inference_report before trusting metric interpretation",
        ],
    ),
    Dict{String,Any}(
        "feature" => "harness_fail_fast",
        "title" => "Harness Fail-Fast Orchestration Hints",
        "aliases" => [
            "harness",
            "mcp_harness",
            "fail_fast",
            "回归测试",
            "协议测试",
            "脚手架",
        ],
        "when_to_load" => [
            "When running MCP regression harness",
            "When user asks for protocol completeness checks",
        ],
        "trigger_tools" => [
            "find_model",
            "list_workspace_files",
            "inspect_hydro_data",
            "run_simulation",
            "calibrate_model",
            "run_validation",
            "clear_session_cache",
        ],
        "required_prechecks" => [
            "Always run inspect_hydro_data before downstream simulation/calibration/validation",
        ],
        "recommended_sequence" => [
            "find_model",
            "list_workspace_files",
            "inspect_hydro_data",
            "run_simulation",
            "calibrate_model",
            "run_validation",
            "clear_session_cache",
        ],
        "payload_contract" => Dict(
            "strict_shape" => "Prefer unified v2 request shape: model + inputs",
            "stop_rule" => "Stop downstream calls immediately on blocking errors when fail_fast=true",
            "cleanup" => "Call clear_session_cache once after successful suite completion",
        ),
        "accuracy_tips" => [
            "Record expected vs actual tool call sequence for each case",
            "Do not hide contract mismatch behind fallback behavior",
            "If protocol listing endpoints are inaccessible in your client, call list_mcp_surfaces as fallback",
        ],
    ),
]

function _normalize_hint_token(value::AbstractString)
    return lowercase(replace(strip(String(value)), r"[^0-9A-Za-z一-龥]+" => ""))
end

function _build_llm_hint_index(specs)
    feature_index = Dict{String,Dict{String,Any}}()
    alias_index = Dict{String,String}()

    for spec in specs
        feature_name = string(spec["feature"])
        feature_index[feature_name] = spec

        candidates = String[feature_name]
        append!(candidates, String.(get(spec, "aliases", Any[])))
        for alias in candidates
            token = _normalize_hint_token(alias)
            isempty(token) && continue
            alias_index[token] = feature_name
        end
    end

    return feature_index, alias_index
end

const LLM_HINT_FEATURE_INDEX, LLM_HINT_ALIAS_INDEX = _build_llm_hint_index(LLM_HINT_SPECS)

llm_hint_uri(feature::AbstractString) = "hydro://hints/$(String(feature))"

function resolve_llm_hint_feature(feature_token::AbstractString)
    normalized = _normalize_hint_token(feature_token)
    feature = get(LLM_HINT_ALIAS_INDEX, normalized, nothing)

    if isnothing(feature)
        available = join(sort!(collect(keys(LLM_HINT_FEATURE_INDEX))), ", ")
        throw(ArgumentError("Unknown hint feature '$feature_token'. Available: $available"))
    end

    return feature
end

function _llm_hint_summary(spec::Dict{String,Any})
    feature_name = string(spec["feature"])
    return Dict(
        "feature" => feature_name,
        "title" => string(spec["title"]),
        "aliases" => String.(get(spec, "aliases", Any[])),
        "trigger_tools" => String.(get(spec, "trigger_tools", Any[])),
        "uri" => llm_hint_uri(feature_name),
    )
end

function llm_hint_catalog_payload()
    features = [_llm_hint_summary(LLM_HINT_FEATURE_INDEX[name]) for name in sort!(collect(keys(LLM_HINT_FEATURE_INDEX)))]

    activation = Dict{String,Vector{String}}()
    for feature in features
        for tool_name in get(feature, "trigger_tools", String[])
            haskey(activation, tool_name) || (activation[tool_name] = String[])
            push!(activation[tool_name], feature["feature"])
        end
    end
    for value in values(activation)
        unique!(value)
        sort!(value)
    end

    return Dict(
        "features" => features,
        "count" => length(features),
        "uri_template" => "hydro://hints/{feature}",
        "tool_activation_map" => activation,
        "purpose" => "Lightweight LLM guidance to improve HydroModelMCP call accuracy.",
        "loading_model" => "Catalog is startup-lightweight; detailed hint payload loads on demand via template reads.",
    )
end

function llm_hint_payload(feature_token::AbstractString)
    feature = resolve_llm_hint_feature(feature_token)
    spec = LLM_HINT_FEATURE_INDEX[feature]

    return Dict(
        "feature" => feature,
        "requested_feature" => String(feature_token),
        "resolved_by_alias" => _normalize_hint_token(feature_token) != _normalize_hint_token(feature),
        "title" => string(spec["title"]),
        "when_to_load" => get(spec, "when_to_load", String[]),
        "trigger_tools" => get(spec, "trigger_tools", String[]),
        "required_prechecks" => get(spec, "required_prechecks", String[]),
        "recommended_sequence" => get(spec, "recommended_sequence", String[]),
        "payload_contract" => get(spec, "payload_contract", Dict{String,Any}()),
        "accuracy_tips" => get(spec, "accuracy_tips", String[]),
        "lightweight_defaults" => get(spec, "lightweight_defaults", Dict{String,Any}()),
        "source_note" => "Hints are distilled from harness-skill practices and are not a replacement for full skill loading.",
    )
end

const llm_hints_catalog_resource = MCPResource(
    uri = "hydro://guides/llm-hints",
    name = "LLM Accuracy Hints",
    title = "HydroModelMCP LLM Accuracy Hints",
    description = "Lightweight, alias-aware guidance to improve MCP call accuracy.",
    mime_type = "application/json",
    data_provider = llm_hint_catalog_payload,
)

function create_llm_hint_resources()
    return MCPResource[llm_hints_catalog_resource]
end

export create_llm_hint_resources,
    llm_hint_catalog_payload,
    llm_hint_payload,
    llm_hint_uri,
    llm_hints_catalog_resource,
    resolve_llm_hint_feature
