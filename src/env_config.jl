const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_DOTENV_PATH = joinpath(PROJECT_ROOT, ".env")
const DOTENV_OVERLAY_CACHE = Ref{Union{Nothing,Dict{String,String}}}(nothing)

function _dotenv_overlay(; dotenv_path::AbstractString = DEFAULT_DOTENV_PATH)
    resolved_path = normpath(abspath(String(dotenv_path)))
    cached = DOTENV_OVERLAY_CACHE[]
    if !(cached === nothing) && resolved_path == DEFAULT_DOTENV_PATH
        return cached
    end

    overlay = if isfile(resolved_path)
        Dict{String,String}(DotEnv.config(resolved_path; env = ENV, override = false).overlay)
    else
        Dict{String,String}()
    end

    if resolved_path == DEFAULT_DOTENV_PATH
        DOTENV_OVERLAY_CACHE[] = overlay
    end

    return overlay
end

function clear_dotenv_overlay_cache!()
    DOTENV_OVERLAY_CACHE[] = nothing
    return nothing
end

function get_config_env(name::AbstractString, default = nothing; dotenv_path::AbstractString = DEFAULT_DOTENV_PATH)
    env_value = get(ENV, String(name), nothing)
    if !(env_value === nothing)
        candidate = String(env_value)
        isempty(strip(candidate)) || return candidate
    end

    overlay = _dotenv_overlay(; dotenv_path)
    overlay_value = get(overlay, String(name), nothing)
    if !(overlay_value === nothing)
        candidate = String(overlay_value)
        isempty(strip(candidate)) || return candidate
    end

    return default
end

function parse_config_env_int(name::AbstractString, default::String; dotenv_path::AbstractString = DEFAULT_DOTENV_PATH)
    value = get_config_env(name, default; dotenv_path)
    try
        return parse(Int, String(value))
    catch
        throw(ArgumentError("Configuration variable $name must be an integer, got '$(value)'."))
    end
end
