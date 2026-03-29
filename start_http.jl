project_file = normpath(joinpath(@__DIR__, "Project.toml"))
active_project = something(Base.active_project(), "")

if normpath(active_project) != project_file
    @warn "HydroModelMCP should usually be started with --project=$(abspath(@__DIR__)). Falling back to Pkg.activate because the active project does not match."
    import Pkg
    Pkg.activate(@__DIR__)
end

using DotEnv
DotEnv.config(joinpath(@__DIR__, ".env"))

using HydroModelMCP

host = get(ENV, "JULIA_HTTP_HOST", "127.0.0.1")
port = parse(Int, get(ENV, "JULIA_HTTP_PORT", "3000"))
allowed_origins = get(ENV, "JULIA_HTTP_ALLOWED_ORIGINS", "*")
origins = allowed_origins == "*" ? String[] : split(allowed_origins, ",") .|> strip

println("Starting HydroModelMCP HTTP server...")
println("  host: $host")
println("  port: $port")
println("  allowed_origins: $(isempty(origins) ? "*" : join(origins, ", "))")

HydroModelMCP.run_http_server(
    host = host,
    port = port,
    allowed_origins = origins,
)
