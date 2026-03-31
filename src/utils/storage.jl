module Storage

using JSON3
using Dates
using Redis

export StorageBackend, RedisBackend, FileBackend
export save_result, load_result, list_results, delete_result, cleanup_expired

abstract type StorageBackend end

struct RedisBackend <: StorageBackend
    host::String
    port::Int
    prefix::String
    default_ttl::Union{Int,Nothing}
end

struct FileBackend <: StorageBackend
    base_path::String
    default_ttl::Union{Int,Nothing}
end

# Constructors with defaults
RedisBackend(host, port, prefix; ttl=604800) = RedisBackend(host, port, prefix, ttl)
FileBackend(base_path; ttl=604800) = FileBackend(base_path, ttl)

# ============================================================================
# Redis Backend Implementation
# ============================================================================

function save_result(backend::RedisBackend, category::String, id::String, data::Dict; ttl::Union{Int,Nothing}=nothing)
    conn = RedisConnection(host=backend.host, port=backend.port)
    try
        key = "$(backend.prefix):$(category):$(id)"
        json_str = JSON3.write(data)

        # Use provided TTL or backend default
        effective_ttl = isnothing(ttl) ? backend.default_ttl : ttl

        if isnothing(effective_ttl) || effective_ttl == 0
            # No expiration
            set(conn, key, json_str)
        else
            # Set with expiration
            setex(conn, key, effective_ttl, json_str)
        end
    finally
        disconnect(conn)
    end
end

function load_result(backend::RedisBackend, category::String, id::String)
    conn = RedisConnection(host=backend.host, port=backend.port)
    try
        key = "$(backend.prefix):$(category):$(id)"
        json_str = get(conn, key)

        if isnothing(json_str)
            error("Result not found: $(category)/$(id)")
        end

        return JSON3.read(json_str, Dict{String,Any})
    finally
        disconnect(conn)
    end
end

function list_results(backend::RedisBackend, category::String)
    conn = RedisConnection(host=backend.host, port=backend.port)
    try
        pattern = "$(backend.prefix):$(category):*"
        keys_list = keys(conn, pattern)

        # Extract IDs from keys
        prefix_len = length("$(backend.prefix):$(category):")
        return [k[prefix_len+1:end] for k in keys_list]
    finally
        disconnect(conn)
    end
end

function delete_result(backend::RedisBackend, category::String, id::String)
    conn = RedisConnection(host=backend.host, port=backend.port)
    try
        key = "$(backend.prefix):$(category):$(id)"
        del(conn, key)
    finally
        disconnect(conn)
    end
end

function cleanup_expired(backend::RedisBackend, category::String)
    # Redis handles expiration automatically, no manual cleanup needed
    return 0
end

# ============================================================================
# File Backend Implementation
# ============================================================================

function save_result(backend::FileBackend, category::String, id::String, data::Dict; ttl::Union{Int,Nothing}=nothing)
    # Create category directory if it doesn't exist
    category_dir = joinpath(backend.base_path, category)
    mkpath(category_dir)

    # Save data file
    data_path = joinpath(category_dir, "$(id).json")
    open(data_path, "w") do io
        JSON3.write(io, data)
    end

    # Save metadata file with timestamp
    meta_path = joinpath(category_dir, "$(id).meta.json")
    metadata = Dict(
        "created_at" => string(now()),
        "category" => category,
        "id" => id
    )
    open(meta_path, "w") do io
        JSON3.write(io, metadata)
    end
end

function load_result(backend::FileBackend, category::String, id::String)
    data_path = joinpath(backend.base_path, category, "$(id).json")

    if !isfile(data_path)
        error("Result not found: $(category)/$(id)")
    end

    return JSON3.read(read(data_path, String), Dict{String,Any})
end

function list_results(backend::FileBackend, category::String)
    # Cleanup expired results first
    cleanup_expired(backend, category)

    category_dir = joinpath(backend.base_path, category)

    if !isdir(category_dir)
        return String[]
    end

    # Find all .json files (excluding .meta.json)
    files = readdir(category_dir)
    result_ids = String[]

    for file in files
        if endswith(file, ".json") && !endswith(file, ".meta.json")
            # Remove .json extension to get ID
            push!(result_ids, file[1:end-5])
        end
    end

    return result_ids
end

function delete_result(backend::FileBackend, category::String, id::String)
    data_path = joinpath(backend.base_path, category, "$(id).json")
    meta_path = joinpath(backend.base_path, category, "$(id).meta.json")

    isfile(data_path) && rm(data_path)
    isfile(meta_path) && rm(meta_path)
end

function cleanup_expired(backend::FileBackend, category::String)
    # Skip cleanup if TTL is disabled
    if isnothing(backend.default_ttl) || backend.default_ttl == 0
        return 0
    end

    category_dir = joinpath(backend.base_path, category)

    if !isdir(category_dir)
        return 0
    end

    now_time = now()
    ttl_seconds = backend.default_ttl
    deleted_count = 0

    files = readdir(category_dir)

    for file in files
        if endswith(file, ".meta.json")
            meta_path = joinpath(category_dir, file)
            id = file[1:end-10]  # Remove .meta.json

            try
                metadata = JSON3.read(read(meta_path, String), Dict{String,Any})
                created_at = DateTime(metadata["created_at"])

                # Check if expired
                age_seconds = (now_time - created_at).value / 1000  # Convert milliseconds to seconds

                if age_seconds > ttl_seconds
                    # Delete both data and meta files
                    delete_result(backend, category, id)
                    deleted_count += 1
                end
            catch e
                # If metadata is corrupted, skip this file
                @warn "Failed to read metadata for $(id): $(e)"
            end
        end
    end

    return deleted_count
end

end # module Storage
