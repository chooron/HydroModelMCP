using Test
using CSV
using DataFrames
using StatsBase  # for Histogram in test_sampling.jl


include("../src/HydroModelMCP.jl")

@testset "test HydroModelMCP" begin
    @testset "test model discovery" begin
        include("test_discovery.jl")
    end

    @testset "test metrics" begin
        include("test_metrics.jl")
    end

    @testset "test data splitter" begin
        include("test_datasplitter.jl")
    end

    @testset "test sampling" begin
        include("test_sampling.jl")
    end

    @testset "test sensitivity analysis" begin
        include("test_sensitivity.jl")
    end

    @testset "test calibration" begin
        include("test_calibration.jl")
    end

    # @testset "test model simulation" begin
    #     include("test_simulation.jl")
    # end
end