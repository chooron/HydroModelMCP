project_root = @__DIR__
project_file = normpath(joinpath(project_root, "Project.toml"))
active_project = something(Base.active_project(), "")
import Pkg

if normpath(active_project) != project_file
    @warn "HydroModelMCP should usually be started with --project=$(abspath(project_root)). Falling back to Pkg.activate because the active project does not match."
    Pkg.activate(project_root)
end

function append_default_depot_fallback!()
    custom_depot = get(ENV, "JULIA_DEPOT_PATH", "")
    isempty(custom_depot) && return false

    sep = Base.Filesystem.path_separator
    depots = filter(!isempty, split(custom_depot, sep))
    default_depot = normpath(joinpath(homedir(), ".julia"))

    if any(path -> normpath(path) == default_depot, depots)
        empty!(Base.DEPOT_PATH)
        append!(Base.DEPOT_PATH, depots)
        return false
    end

    push!(depots, default_depot)
    ENV["JULIA_DEPOT_PATH"] = join(depots, string(sep))

    empty!(Base.DEPOT_PATH)
    append!(Base.DEPOT_PATH, depots)

    println(stderr, "HydroModelMCP startup: appended default depot fallback at $default_depot")
    return true
end

function load_hydromodelmcp!()
    try
        @eval using HydroModelMCP
        return
    catch first_err
        first_msg = sprint(showerror, first_err)
        recoverable = first_err isa LoadError || occursin("failed to find source of parent package", first_msg)

        if recoverable && append_default_depot_fallback!()
            try
                @eval using HydroModelMCP
                return
            catch second_err
                println(stderr, "HydroModelMCP startup: dependency load failed even after depot fallback. Please run 'julia --project=. -e \"using Pkg; Pkg.instantiate(); Pkg.precompile()\"' once in this project.")
                rethrow(second_err)
            end
        end

        if recoverable
            println(stderr, "HydroModelMCP startup: dependency load failed. Please run 'julia --project=. -e \"using Pkg; Pkg.instantiate(); Pkg.precompile()\"' once in this project.")
        end
        rethrow(first_err)
    end
end

load_hydromodelmcp!()

HydroModelMCP.run_server()
