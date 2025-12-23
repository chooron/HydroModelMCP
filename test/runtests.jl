using Test
using CSV
using DataFrames


include("../src/HydroModelMCP.jl")

@testset "test HydroModelMCP" begin 
    @testset "test model discovery" begin
        include("test_discovery.jl")
    end
    # @testset "test model simulation" begin
    #     include("test_simulation.jl")
    # end
end