"""
Session-scoped data handle management for HydroModelMCP.

Intermediate tool state is stored in a transient cache so tools can exchange
large payloads without writing temporary files. Redis is preferred when
available; otherwise the cache falls back to in-process memory.
"""

using Base64
using Serialization

const DATA_STORE = Dict{String,Any}()
const SESSION_CACHE_MODE = lowercase(get(ENV, "HYDRO_SESSION_CACHE_BACKEND", "auto"))
const SESSION_CACHE_HOST = get(ENV, "REDIS_HOST", "127.0.0.1")
const SESSION_CACHE_PORT = try
    parse(Int, get(ENV, "REDIS_PORT", "6379"))
catch
    6379
end
const SESSION_CACHE_TTL = try
    parse(Int, get(ENV, "HYDRO_SESSION_CACHE_TTL", "86400"))
catch
    86400
end
const SESSION_CACHE_PREFIX = get(
    ENV,
    "HYDRO_SESSION_CACHE_PREFIX",
    "hydro:session:" * string(UUIDs.uuid4()) * ":",
)
const SESSION_CACHE_BACKEND = Ref{Symbol}(:uninitialized)

const LAST_SIMULATION_RESULT_HANDLE = "__last_simulation_result__"
const LAST_CALIBRATION_RESULT_HANDLE = "__last_calibration_result__"

function _redis_connection()
    return RedisConnection(host = SESSION_CACHE_HOST, port = SESSION_CACHE_PORT)
end

function _detect_session_cache_backend()
    SESSION_CACHE_MODE in ("auto", "memory", "redis") ||
        throw(ArgumentError("HYDRO_SESSION_CACHE_BACKEND must be one of: auto, memory, redis"))

    if SESSION_CACHE_MODE == "memory"
        return :memory
    end

    try
        conn = _redis_connection()
        disconnect(conn)
        return :redis
    catch err
        SESSION_CACHE_MODE == "redis" &&
            @warn "Redis session cache unavailable; falling back to memory" exception = (err, catch_backtrace())
        return :memory
    end
end

function session_cache_backend()
    if SESSION_CACHE_BACKEND[] == :uninitialized
        SESSION_CACHE_BACKEND[] = _detect_session_cache_backend()
    end
    return SESSION_CACHE_BACKEND[]
end

session_cache_prefix() = SESSION_CACHE_PREFIX

function session_cache_info()
    info = Dict{String,Any}(
        "backend" => string(session_cache_backend()),
        "prefix" => session_cache_prefix(),
    )

    if session_cache_backend() == :redis
        info["host"] = SESSION_CACHE_HOST
        info["port"] = SESSION_CACHE_PORT
        info["ttl_seconds"] = SESSION_CACHE_TTL
    end

    return info
end

_session_cache_key(handle::String) = session_cache_prefix() * handle

function _serialize_cache_value(data::Any)
    io = IOBuffer()
    Serialization.serialize(io, data)
    return base64encode(take!(io))
end

function _deserialize_cache_value(payload::AbstractString)
    bytes = base64decode(payload)
    return Serialization.deserialize(IOBuffer(bytes))
end

function store_data(handle::String, data::Any)
    if session_cache_backend() == :redis
        conn = _redis_connection()
        try
            key = _session_cache_key(handle)
            payload = _serialize_cache_value(data)
            if SESSION_CACHE_TTL > 0
                setex(conn, key, SESSION_CACHE_TTL, payload)
            else
                set(conn, key, payload)
            end
        finally
            disconnect(conn)
        end
        return handle
    end

    DATA_STORE[handle] = data
    return handle
end

function get_data(handle::String)
    if session_cache_backend() == :redis
        conn = _redis_connection()
        try
            payload = get(conn, _session_cache_key(handle))
            isnothing(payload) && error("Data handle not found: $handle")
            return _deserialize_cache_value(payload)
        finally
            disconnect(conn)
        end
    end

    haskey(DATA_STORE, handle) || error("Data handle not found: $handle")
    return DATA_STORE[handle]
end

function has_data(handle::String)
    if session_cache_backend() == :redis
        conn = _redis_connection()
        try
            return !isnothing(get(conn, _session_cache_key(handle)))
        finally
            disconnect(conn)
        end
    end

    return haskey(DATA_STORE, handle)
end

function clear_data(handle::String)
    removed = false

    if session_cache_backend() == :redis
        conn = _redis_connection()
        try
            removed = del(conn, _session_cache_key(handle)) > 0
        finally
            disconnect(conn)
        end
    else
        removed = pop!(DATA_STORE, handle, nothing) !== nothing
    end

    return removed
end

function clear_all_data!()
    cleared_count = 0

    if session_cache_backend() == :redis
        conn = _redis_connection()
        try
            keys_list = keys(conn, session_cache_prefix() * "*")
            for key in keys_list
                cleared_count += del(conn, key)
            end
        finally
            disconnect(conn)
        end
    end

    cleared_count += length(DATA_STORE)
    empty!(DATA_STORE)
    return cleared_count
end

function list_handles()
    if session_cache_backend() == :redis
        conn = _redis_connection()
        try
            prefix = session_cache_prefix()
            keys_list = keys(conn, prefix * "*")
            return sort!(String[key[length(prefix)+1:end] for key in keys_list])
        finally
            disconnect(conn)
        end
    end

    return sort!(collect(keys(DATA_STORE)))
end

function get_data_info(handle::String)
    data = get_data(handle)
    info = Dict{String,Any}(
        "handle" => handle,
        "type" => string(typeof(data)),
        "cache_backend" => string(session_cache_backend()),
    )

    if data isa AbstractArray
        info["size"] = size(data)
        info["length"] = length(data)
        info["ndims"] = ndims(data)
    end

    return info
end
