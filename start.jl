project_file = normpath(joinpath(@__DIR__, "Project.toml"))
active_project = something(Base.active_project(), "")

if normpath(active_project) != project_file
    @warn "HydroModelMCP should usually be started with --project=$(abspath(@__DIR__)). Falling back to Pkg.activate because the active project does not match."
    import Pkg
    Pkg.activate(@__DIR__)
end

using HydroModelMCP

HydroModelMCP.run_server()
