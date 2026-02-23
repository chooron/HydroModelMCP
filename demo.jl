using ModelContextProtocol

# Tool with enum and array parameters
search_tool = MCPTool(
    name = "search",
    description = "Search with filters",
    input_schema = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "query" => Dict{String,Any}(
                "type" => "string",
                "description" => "Search query"
            ),
            "tags" => Dict{String,Any}(
                "type" => "array",
                "items" => Dict{String,Any}("type" => "string"),
                "description" => "Filter tags"
            ),
            "sort" => Dict{String,Any}(
                "type" => "string",
                "enum" => ["relevance", "date", "name"],
                "default" => "relevance"
            )
        ),
        "required" => ["query"]
    ),
    handler = function(params)
        query = params["query"]
        tags = get(params, "tags", String[])
        sort = get(params, "sort", "relevance")
        TextContent(text = "Searching '$query' with $(length(tags)) tags, sorted by $sort")
    end
)

server = mcp_server(
    name = "search-server",
    tools = search_tool
)
start!(server)