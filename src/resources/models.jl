using ModelContextProtocol

const MODEL_KNOWLEDGE_PATH = joinpath(@__DIR__, "model.json")

function _plain_json_value(value)
    if value isa JSON3.Object
        return Dict{String,Any}(String(key) => _plain_json_value(val) for (key, val) in pairs(value))
    elseif value isa JSON3.Array
        return Any[_plain_json_value(item) for item in value]
    end
    return value
end

function _normalize_model_token(value)
    return lowercase(replace(strip(String(value)), r"[^a-zA-Z0-9]+" => ""))
end

function _code_name_token(card::Dict{String,Any})
    model_name = card["model_name"]
    code_name = get(model_name, "code_name", "")
    isempty(code_name) && return nothing

    parts = split(String(code_name), "_")
    return length(parts) >= 3 ? lowercase(parts[3]) : nothing
end

function _knowledge_match_candidates(card::Dict{String,Any})
    model_name = card["model_name"]
    candidates = String[]

    code_token = _code_name_token(card)
    !isnothing(code_token) && push!(candidates, code_token)

    push!(candidates, _normalize_model_token(model_name["display_name"]))
    for alias in get(model_name, "aliases", Any[])
        push!(candidates, _normalize_model_token(alias))
    end

    return unique(filter(!isempty, candidates))
end

const MANUAL_KNOWLEDGE_MODEL_MAP = Dict(
    "us1" => "unitedstates",
    "xinanjiang" => "xaj",
)

function _load_model_knowledge_cards()
    parsed = JSON3.read(read(MODEL_KNOWLEDGE_PATH, String))
    return Dict{String,Any}[_plain_json_value(card) for card in parsed]
end

const MODEL_KNOWLEDGE_CARDS = _load_model_knowledge_cards()
const RUNTIME_MODEL_NAMES = Discovery.list_models()

function _build_runtime_knowledge_index(cards::Vector{Dict{String,Any}}, runtime_models::Vector{String})
    runtime_set = Set(runtime_models)
    index = Dict{String,Dict{String,Any}}()
    matched_card_ids = Set{String}()

    for card in cards
        for candidate in _knowledge_match_candidates(card)
            runtime_model = get(MANUAL_KNOWLEDGE_MODEL_MAP, candidate, candidate)
            runtime_model in runtime_set || continue
            haskey(index, runtime_model) && continue

            index[runtime_model] = Dict(
                "match_basis" => candidate,
                "card" => card,
            )
            push!(matched_card_ids, string(card["model_id"]))
        end
    end

    return index, matched_card_ids
end

const MODEL_KNOWLEDGE_INDEX, MATCHED_KNOWLEDGE_CARD_IDS =
    _build_runtime_knowledge_index(MODEL_KNOWLEDGE_CARDS, RUNTIME_MODEL_NAMES)

function has_model_knowledge(model_name::String)
    return haskey(MODEL_KNOWLEDGE_INDEX, model_name)
end

function model_knowledge_uri(model_name::String)
    return "hydro://models/$model_name/knowledge"
end

function model_knowledge_payload(model_name::String)
    payload = Dict{String,Any}(
        "runtime_model" => model_name,
        "resource_layer" => "supplementary_model_knowledge",
        "resource_uri" => model_knowledge_uri(model_name),
        "knowledge_card_available" => has_model_knowledge(model_name),
        "source_file" => "src/resources/model.json",
        "authority_note" => "This resource is supplementary knowledge only. Use discovery tools for runtime model metadata and contracts.",
    )

    if has_model_knowledge(model_name)
        entry = MODEL_KNOWLEDGE_INDEX[model_name]
        card = entry["card"]
        payload["match_basis"] = entry["match_basis"]
        payload["model_id"] = card["model_id"]
        payload["knowledge_card"] = card
    else
        payload["message"] = "No supplementary knowledge card was mapped for this runtime model."
    end

    return payload
end

const model_knowledge_index_resource = MCPResource(
    uri = "hydro://models/knowledge-index",
    name = "Model Knowledge Index",
    title = "Model Knowledge Index",
    description = "Supplementary knowledge-card coverage and resource URIs for runtime models.",
    mime_type = "application/json",
    data_provider = () -> Dict(
        "models" => [
            Dict(
                "name" => model_name,
                "knowledge_uri" => model_knowledge_uri(model_name),
                "knowledge_card_available" => has_model_knowledge(model_name),
            )
            for model_name in RUNTIME_MODEL_NAMES
        ],
        "count" => length(RUNTIME_MODEL_NAMES),
    ),
)

const model_knowledge_coverage_resource = MCPResource(
    uri = "hydro://models/knowledge-coverage",
    name = "Model Knowledge Coverage",
    title = "Model Knowledge Coverage",
    description = "Coverage report between runtime models and static supplementary knowledge cards.",
    mime_type = "application/json",
    data_provider = () -> Dict(
        "runtime_models_with_cards" => Base.sort([model for model in RUNTIME_MODEL_NAMES if has_model_knowledge(model)]),
        "runtime_models_without_cards" => Base.sort([model for model in RUNTIME_MODEL_NAMES if !has_model_knowledge(model)]),
        "knowledge_cards_without_runtime_model" => [
            Dict(
                "model_id" => card["model_id"],
                "display_name" => card["model_name"]["display_name"],
                "code_name" => get(card["model_name"], "code_name", ""),
            )
            for card in MODEL_KNOWLEDGE_CARDS
            if !(string(card["model_id"]) in MATCHED_KNOWLEDGE_CARD_IDS)
        ],
        "mapping_overrides" => copy(MANUAL_KNOWLEDGE_MODEL_MAP),
    ),
)

function create_model_knowledge_resources()
    return MCPResource[
        model_knowledge_index_resource,
        model_knowledge_coverage_resource,
    ]
end

export create_model_knowledge_resources,
    has_model_knowledge,
    model_knowledge_coverage_resource,
    model_knowledge_index_resource,
    model_knowledge_payload,
    model_knowledge_uri
