using ModelContextProtocol

clear_session_cache_tool = MCPTool(
    name = "clear_session_cache",
    description = "Clear transient session cache entries used for data handles. This removes Redis or in-memory intermediate state for the current MCP server session without deleting files in ./result.",
    input_schema = Dict(
        "type" => "object",
        "properties" => Dict(
            "include_handles" => Dict(
                "type" => "boolean",
                "description" => "Whether to return the cleared handle names.",
                "default" => true,
            ),
        ),
    ),
    handler = function(params)
        try
            include_handles = Bool(get(params, "include_handles", true))
            handles = list_handles()
            cleared_count = clear_all_data!()
            payload = Dict{String,Any}(
                "status" => "success",
                "backend" => session_cache_info()["backend"],
                "prefix" => session_cache_info()["prefix"],
                "cleared_count" => cleared_count,
            )
            include_handles && (payload["cleared_handles"] = handles)
            return create_json_response(payload)
        catch e
            return create_error_response(e)
        end
    end,
)

export clear_session_cache_tool
