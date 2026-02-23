"""
Data handle management for HydroModelMCP.

Provides a global data storage system where Julia stores large arrays
and Python only passes lightweight handle references.

This prevents large data transmission across the Python/Julia boundary.
"""

# Global data storage
const DATA_STORE = Dict{String, Any}()

"""
    store_data(handle::String, data::Any) -> String

Store data in the global data store and return the handle.

# Arguments
- `handle::String`: Unique identifier for the data
- `data::Any`: Data to store (typically arrays, time series, etc.)

# Returns
- `String`: The handle (same as input)

# Example
```julia
handle = store_data("camels_5362000_streamflow", [1.0, 2.0, 3.0])
```
"""
function store_data(handle::String, data::Any)
    DATA_STORE[handle] = data
    return handle
end

"""
    get_data(handle::String) -> Any

Retrieve data from the global data store.

# Arguments
- `handle::String`: Data handle identifier

# Returns
- `Any`: The stored data

# Throws
- `ErrorException`: If handle not found

# Example
```julia
data = get_data("camels_5362000_streamflow")
```
"""
function get_data(handle::String)
    haskey(DATA_STORE, handle) || error("Data handle not found: $handle")
    return DATA_STORE[handle]
end

"""
    has_data(handle::String) -> Bool

Check if a data handle exists in the store.

# Arguments
- `handle::String`: Data handle identifier

# Returns
- `Bool`: true if handle exists, false otherwise
"""
function has_data(handle::String)
    return haskey(DATA_STORE, handle)
end

"""
    clear_data(handle::String) -> Bool

Remove data from the global data store.

# Arguments
- `handle::String`: Data handle identifier

# Returns
- `Bool`: true if data was removed, false if handle didn't exist
"""
function clear_data(handle::String)
    if haskey(DATA_STORE, handle)
        delete!(DATA_STORE, handle)
        return true
    end
    return false
end

"""
    clear_all_data!()

Clear all data from the global data store.
Useful for cleanup or testing.
"""
function clear_all_data!()
    empty!(DATA_STORE)
    return nothing
end

"""
    list_handles() -> Vector{String}

List all data handles currently in the store.

# Returns
- `Vector{String}`: Array of handle identifiers
"""
function list_handles()
    return collect(keys(DATA_STORE))
end

"""
    get_data_info(handle::String) -> Dict

Get information about stored data without retrieving it.

# Arguments
- `handle::String`: Data handle identifier

# Returns
- `Dict`: Information about the data (type, size, etc.)
"""
function get_data_info(handle::String)
    haskey(DATA_STORE, handle) || error("Data handle not found: $handle")
    data = DATA_STORE[handle]

    info = Dict{String, Any}(
        "handle" => handle,
        "type" => string(typeof(data))
    )

    # Add size information for arrays
    if data isa AbstractArray
        info["size"] = size(data)
        info["length"] = length(data)
        info["ndims"] = ndims(data)
    end

    return info
end
