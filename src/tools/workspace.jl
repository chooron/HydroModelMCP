using Dates
using ModelContextProtocol

list_workspace_files_tool = MCPTool(
    name = "list_workspace_files",
    description = "List files in a workspace directory without reading file contents.",
    input_schema = Dict(
        "type" => "object",
        "properties" => Dict(
            "directory" => Dict(
                "type" => "string",
                "description" => "Workspace-relative or absolute directory path within the workspace.",
            ),
            "extensions" => Dict(
                "type" => "array",
                "items" => Dict("type" => "string"),
                "description" => "Optional list of file extensions to include.",
            ),
            "include_size" => Dict("type" => "boolean", "default" => true),
            "include_modified" => Dict("type" => "boolean", "default" => true),
        ),
        "required" => ["directory"],
    ),
    handler = function(params)
        validation_error = validate_required_params(params, ["directory"])
        !isnothing(validation_error) && return create_error_response(validation_error)

        try
            directory = resolve_workspace_path(string(params["directory"]); must_exist = false)
            created_directory = false

            if !ispath(directory)
                mkpath(directory)
                created_directory = true
            end

            isdir(directory) || throw(ArgumentError("Path is not a directory: $directory"))

            include_size = Bool(get(params, "include_size", true))
            include_modified = Bool(get(params, "include_modified", true))
            extensions = haskey(params, "extensions") ? Set(lowercase.(String.(params["extensions"]))) : nothing

            files = Dict{String,Any}[]
            for entry in readdir(directory; join = true)
                isfile(entry) || continue

                ext = lowercase(replace(splitext(entry)[2], "." => ""))
                if !(extensions === nothing) && !(ext in extensions)
                    continue
                end

                info = Dict{String,Any}(
                    "name" => basename(entry),
                    "path" => entry,
                )
                include_size && (info["size_bytes"] = filesize(entry))
                include_modified && (info["modified"] = string(Dates.unix2datetime(mtime(entry))))
                push!(files, info)
            end

            sort!(files; by = item -> lowercase(String(item["name"])))

            return create_json_response(Dict(
                "status" => "success",
                "directory" => directory,
                "created_directory" => created_directory,
                "count" => length(files),
                "files" => files,
            ))
        catch e
            return create_error_response(e)
        end
    end,
)

export list_workspace_files_tool
