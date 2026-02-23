# Example 3: Multi-objective Snow Modeling
# Scenario: Snow-dominated Catchment
# Demonstrates: Multi-objective calibration with parameter constraints and Pareto front

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using HydroModelMCP
using JSON3
using CSV
using DataFrames
using HydroModels
using HydroModelLibrary
using ComponentArrays
using DataInterpolations
using LinearAlgebra
using DotEnv
using NPZ
using Dates
using Statistics

println("=" ^ 80)
println("Example 3: Multi-objective Snow Modeling (Snow-dominated Catchment)")
println("=" ^ 80)
println()
println("This example demonstrates:")
println("  - Strategy 2: Parameter constraints (Delta Method)")
println("  - Strategy 5: Constrained sampling")
println("  - Strategy 9: Multi-objective calibration (NSGA2)")
println("  - Strategy 9: Pareto front degeneracy detection")
println()

# ==============================================================================
# Step 1: Data Loading from NPZ
# ==============================================================================
println("[Step 1] Loading data from CAMELS dataset...")
println()

# Load environment variables
DotEnv.load!(joinpath(@__DIR__, "..", ".env"))

# Load NPZ data
data_path = ENV["CAMESL_DATASET_PATH"]
if !isfile(data_path)
    error("CAMELS dataset not found at: $data_path\nPlease check your .env file.")
end

data = npzread(data_path)
println("  Dataset loaded: $data_path")

# Get dimensions
n_catchments = size(data["forcings"], 1)
n_timesteps = size(data["forcings"], 2)
println("  Number of catchments: $n_catchments")
println("  Number of timesteps: $n_timesteps")
println()

# Get gage IDs
gage_ids = 13011500

# Select a snow-dominated catchment (you can specify a particular gage_id)
# For snow-dominated characteristics, select a catchment at higher latitudes/elevations
target_gage_id = gage_ids[min(100, length(gage_ids))]  # Or specify your own gage_id
println("  Selected gage_id: $target_gage_id (for snow-dominated characteristics)")

# Find the index of the selected gage_id
catchment_idx = findfirst(==(target_gage_id), gage_ids)
if isnothing(catchment_idx)
    error("Gage ID $target_gage_id not found in dataset")
end
println("  Catchment index: $catchment_idx")
println()

# Build date index
full_dates = collect(Date(1980, 10, 1):Day(1):Date(2014, 9, 30))
dates = full_dates[1:n_timesteps]

# Define training and testing periods
train_start = Date(1989, 1, 1)
train_end = Date(1998, 12, 31)
test_start = Date(1999, 1, 1)
test_end = Date(2009, 12, 31)

# Find indices for train/test periods
train_idxs = findall(d -> d >= train_start && d <= train_end, dates)
test_idxs = findall(d -> d >= test_start && d <= test_end, dates)

println("  Training period: $train_start to $train_end ($(length(train_idxs)) days)")
println("  Testing period: $test_start to $test_end ($(length(test_idxs)) days)")
println()

# Extract forcing data for selected catchment
forcing_data = data["forcings"][catchment_idx, :, :]
target_data = data["target"][catchment_idx, :, 1]

# Get catchment area for unit conversion
area = data["attributes"][catchment_idx, 12]  # Area in km²
println("  Catchment area: $(round(area, digits=2)) km²")

# Convert streamflow from ft³/s to mm/day
# Formula: flow_mm_day = (10^3) * flow_ft3_s * 0.0283168 * 3600 * 24 / (area * 10^6)
# Simplifies to: flow_mm_day = flow_ft3_s * 2446.5792 / area
target_data_mm = target_data * (10^3) * 0.0283168 * 3600 * 24 / (area * (10^6))

# Create forcing NamedTuple
forcing_nt = (
    P = forcing_data[:, 1],      # Precipitation (mm/day)
    T = forcing_data[:, 2],      # Temperature (°C)
    Ep = forcing_data[:, 3]      # Potential evaporation (mm/day)
)
observed = target_data_mm  # Streamflow (mm/day)

# Check for NaN values
valid_mask = .!(isnan.(forcing_nt.P) .| isnan.(forcing_nt.T) .| isnan.(forcing_nt.Ep) .| isnan.(observed))
println("  Valid data points: $(sum(valid_mask)) / $(length(observed))")

# Basic statistics
valid_obs = observed[valid_mask]
if !isempty(valid_obs)
    println("  Flow statistics (valid data only):")
    println("    Mean: $(round(mean(valid_obs), digits=2)) mm/day")
    println("    Range: [$(round(minimum(valid_obs), digits=2)), $(round(maximum(valid_obs), digits=2))] mm/day")
else
    error("No valid data points found for selected catchment")
end
println("  Temperature range: [$(round(minimum(forcing_nt.T[valid_mask]), digits=1)), $(round(maximum(forcing_nt.T[valid_mask]), digits=1))] °C")
println()

# ==============================================================================
# Step 2: Model Information
# ==============================================================================
println("[Step 2] Getting GR4J+CemaNeige model information...")
println()

# NOTE: This assumes "gr4j_cemaNeige" or similar model name in HydroModelLibrary
# Adjust the model name based on your HydroModelLibrary implementation
model_name = "cemaneigegr4j"  # PLACEHOLDER: Update with actual model name

model_info = HydroModelMCP.Discovery.get_model_info(model_name)
println("  Model: $(model_info["model_name"])")
println("  Inputs: $(model_info["inputs"])")
println("  Outputs: $(model_info["outputs"])")
println()

# Get parameter details
params_info = HydroModelMCP.Discovery.get_parameters_detail(model_name)
println("  Parameters ($(length(params_info)) total):")
for param_info in params_info
    bounds = param_info["bounds"]
    if !isnothing(bounds)
        println("    $(param_info["name"]): [$(bounds[1]), $(bounds[2])] $(param_info["unit"])")
    end
end
println()

# ==============================================================================
# Step 3: Define Parameter Constraints (Strategy 2)
# ==============================================================================
println("[Step 3] Defining parameter constraints...")
println()

# For GR4J+CemaNeige, we might have constraints like:
# - CemaNeige threshold temperature constraints
# - Or other physical constraints between parameters
#
# Example: If parameters have ordering constraints (param_i < param_j)
# Define as: constraints = [(i, j)] where i and j are 1-based indices

# PLACEHOLDER: Define actual constraints based on your model
# For demonstration, we'll show the structure without specific constraints
constraints = []  # Empty if no constraints, or [(1, 3), (2, 4)] for example

if isempty(constraints)
    println("  No parameter constraints defined")
    println("  (Add constraints if your model requires parameter ordering)")
else
    println("  Constraints defined:")
    for (i, j) in constraints
        param_i = params_info[i]["name"]
        param_j = params_info[j]["name"]
        println("    $param_i < $param_j")
    end
end
println()

# ==============================================================================
# Step 4: Data Splitting
# ==============================================================================
println("[Step 4] Splitting data using predefined train/test periods...")
println()

# Use the predefined train/test indices
train_forcing = (
    P = forcing_nt.P[train_idxs],
    T = forcing_nt.T[train_idxs],
    Ep = forcing_nt.Ep[train_idxs]
)
test_forcing = (
    P = forcing_nt.P[test_idxs],
    T = forcing_nt.T[test_idxs],
    Ep = forcing_nt.Ep[test_idxs]
)
train_obs = observed[train_idxs]
test_obs = observed[test_idxs]

# Remove NaN values
train_valid = .!(isnan.(train_forcing.P) .| isnan.(train_forcing.T) .| isnan.(train_forcing.Ep) .| isnan.(train_obs))
train_forcing = (
    P = train_forcing.P[train_valid],
    T = train_forcing.T[train_valid],
    Ep = train_forcing.Ep[train_valid]
)
train_obs = train_obs[train_valid]

test_valid = .!(isnan.(test_forcing.P) .| isnan.(test_forcing.T) .| isnan.(test_forcing.Ep) .| isnan.(test_obs))
test_forcing = (
    P = test_forcing.P[test_valid],
    T = test_forcing.T[test_valid],
    Ep = test_forcing.Ep[test_valid]
)
test_obs = test_obs[test_valid]

println("  Training data: $(length(train_obs)) valid timesteps")
println("  Testing data: $(length(test_obs)) valid timesteps")

if length(train_obs) == 0 || length(test_obs) == 0
    error("No valid data in training or testing period. Please select a different catchment.")
end
println()

val_start, val_end = 1, length(test_obs)

# ==============================================================================
# Step 5: Multi-objective Calibration (Strategy 9)
# ==============================================================================
println("[Step 5] Running multi-objective calibration...")
println("  Objectives: NSE + LogNSE")
println("  Algorithm: NSGA2 (Non-dominated Sorting Genetic Algorithm II)")
println("  Population: 50")
println("  Iterations: 100")
println("  This will generate a Pareto front balancing high and low flows...")
println("  This may take 10-15 minutes...")
println()

multi_result = HydroModelMCP.Calibration.calibrate_multiobjective(
    model_name,
    train_forcing,
    train_obs;
    objectives=["NSE", "LogNSE"],
    algorithm="NSGA2",
    maxiters=1000,
    population_size=50,
    solver_type="ODE",
    interp_type="LINEAR"
)

println("  Multi-objective calibration complete!")
println("  Pareto front size: $(multi_result["n_solutions"]) solutions")
println()

# ==============================================================================
# Step 6: Pareto Front Diagnostics (Strategy 9)
# ==============================================================================
println("[Step 6] Analyzing Pareto front quality...")
println()

pareto_diagnostics = HydroModelMCP.Calibration.diagnose_multiobjective(
    multi_result
)

println("  Degeneracy Analysis:")
println("  -------------------")
println("  Status: $(pareto_diagnostics["degeneracy"]["status"])")
println()

if pareto_diagnostics["degeneracy"]["status"] == "point_degenerate"
    println("  ⚠ Point Degenerate: All solutions are nearly identical")
    println("    → Consider increasing population size or iterations")
elseif pareto_diagnostics["degeneracy"]["status"] == "line_degenerate"
    println("  ⚠ Line Degenerate: Objectives are highly correlated")
    if haskey(pareto_diagnostics, "objective_correlation")
        println("    Correlation: $(round(pareto_diagnostics["objective_correlation"], digits=4))")
    end
    println("    → Consider using different objectives")
elseif pareto_diagnostics["degeneracy"]["status"] == "normal"
    println("  ✓ Normal: Healthy Pareto front with diverse solutions")
else
    println("  ⚠ Insufficient solutions for analysis")
end
println()

println("  Objective Ranges:")
for (obj_name, range_info) in pareto_diagnostics["objective_ranges"]
    println("    $obj_name: [$(round(range_info["min"], digits=4)), $(round(range_info["max"], digits=4))]")
    println("      Range: $(round(range_info["range"], digits=4))")
end
println()

if !isempty(pareto_diagnostics["recommendations"])
    println("  Recommendations:")
    for rec in pareto_diagnostics["recommendations"]
        println("    - $rec")
    end
    println()
end

# ==============================================================================
# Step 7: Extract Representative Solutions
# ==============================================================================
println("[Step 7] Extracting representative solutions from Pareto front...")
println()

pareto_front = multi_result["pareto_front"]

# Find best NSE solution
best_nse_idx = argmax([sol["objectives"]["NSE"] for sol in pareto_front])
best_nse_solution = pareto_front[best_nse_idx]

# Find best LogNSE solution
best_lognse_idx = argmax([sol["objectives"]["LogNSE"] for sol in pareto_front])
best_lognse_solution = pareto_front[best_lognse_idx]

# Find balanced solution (closest to ideal point [1, 1])
distances = [sqrt((1 - sol["objectives"]["NSE"])^2 + (1 - sol["objectives"]["LogNSE"])^2)
             for sol in pareto_front]
balanced_idx = argmin(distances)
balanced_solution = pareto_front[balanced_idx]

println("  Representative Solutions:")
println("  ------------------------")
println()
println("  1. Best NSE (high flows):")
println("     NSE: $(round(best_nse_solution["objectives"]["NSE"], digits=4))")
println("     LogNSE: $(round(best_nse_solution["objectives"]["LogNSE"], digits=4))")
println()
println("  2. Best LogNSE (low flows):")
println("     NSE: $(round(best_lognse_solution["objectives"]["NSE"], digits=4))")
println("     LogNSE: $(round(best_lognse_solution["objectives"]["LogNSE"], digits=4))")
println()
println("  3. Balanced (compromise):")
println("     NSE: $(round(balanced_solution["objectives"]["NSE"], digits=4))")
println("     LogNSE: $(round(balanced_solution["objectives"]["LogNSE"], digits=4))")
println("     Distance to ideal: $(round(distances[balanced_idx], digits=4))")
println()

# ==============================================================================
# Step 8: Validate Representative Solutions
# ==============================================================================
println("[Step 8] Validating representative solutions...")
println()

# Load model
model_module = HydroModelLibrary.load_model(Symbol(model_name), reload=false)
model = Base.invokelatest(m -> m.model, model_module)
input_names = HydroModels.get_input_names(model)

hydro_config = HydroModels.HydroConfig(
    solver=HydroModels.ODESolver,
    interpolator=Val(HydroModels.LinearInterpolation)
)

val_input_matrix = stack([Float64.(test_forcing[n]) for n in input_names], dims=1)

# Helper function to run simulation
function run_solution(params_dict)
    params_nt = NamedTuple{Tuple(Symbol.(keys(params_dict)))}(values(params_dict))
    params_cv = ComponentVector(params=params_nt)
    result_matrix = model(val_input_matrix, params_cv; config=hydro_config)
    return result_matrix[end, :]
end

# Run all three solutions
sim_best_nse = run_solution(best_nse_solution["params"])
sim_best_lognse = run_solution(best_lognse_solution["params"])
sim_balanced = run_solution(balanced_solution["params"])

# Compute validation metrics
metrics_best_nse = HydroModelMCP.Metrics.compute_metrics(
    sim_best_nse, test_obs, ["NSE", "LogNSE", "KGE", "LogKGE", "RMSE", "PBIAS"]
)

metrics_best_lognse = HydroModelMCP.Metrics.compute_metrics(
    sim_best_lognse, test_obs, ["NSE", "LogNSE", "KGE", "LogKGE", "RMSE", "PBIAS"]
)

metrics_balanced = HydroModelMCP.Metrics.compute_metrics(
    sim_balanced, test_obs, ["NSE", "LogNSE", "KGE", "LogKGE", "RMSE", "PBIAS"]
)

println("  Validation Performance:")
println("  " * "-" ^ 70)
println("  Metric     | Best NSE    | Best LogNSE | Balanced    | Best Choice")
println("  " * "-" ^ 70)

metric_names = ["NSE", "LogNSE", "KGE", "LogKGE", "RMSE", "PBIAS"]
for metric in metric_names
    val_nse = get(metrics_best_nse, metric, NaN)
    val_lognse = get(metrics_best_lognse, metric, NaN)
    val_bal = get(metrics_balanced, metric, NaN)

    # Determine best (higher for NSE/KGE, lower for RMSE/PBIAS)
    if metric in ["RMSE", "PBIAS"]
        best_val = minimum([val_nse, val_lognse, val_bal])
    else
        best_val = maximum([val_nse, val_lognse, val_bal])
    end

    best_label = if val_nse == best_val
        "NSE"
    elseif val_lognse == best_val
        "LogNSE"
    else
        "Balanced"
    end

    println("  $(rpad(metric, 10)) | $(rpad(round(val_nse, digits=4), 11)) | $(rpad(round(val_lognse, digits=4), 11)) | $(rpad(round(val_bal, digits=4), 11)) | $best_label")
end
println("  " * "-" ^ 70)
println()

# ==============================================================================
# Step 9: Export Results
# ==============================================================================
println("[Step 9] Exporting results...")
println()

# Export Pareto front
pareto_df = DataFrame()
pareto_df[!, :solution_id] = 1:length(pareto_front)
pareto_df[!, :NSE] = [sol["objectives"]["NSE"] for sol in pareto_front]
pareto_df[!, :LogNSE] = [sol["objectives"]["LogNSE"] for sol in pareto_front]

# Add parameters
param_names = collect(keys(pareto_front[1]["params"]))
for pname in param_names
    pareto_df[!, Symbol(pname)] = [sol["params"][pname] for sol in pareto_front]
end

CSV.write(joinpath(@__DIR__, "example3_pareto_front.csv"), pareto_df)
println("  ✓ example3_pareto_front.csv")

# Export diagnostics
diagnostics_dict = Dict(
    "model" => model_name,
    "scenario" => "snow_dominated",
    "data_file" => data_path,
    "n_solutions" => multi_result["n_solutions"],
    "objectives" => ["NSE", "LogNSE"],
    "degeneracy" => pareto_diagnostics["degeneracy"],
    "objective_ranges" => pareto_diagnostics["objective_ranges"],
    "recommendations" => pareto_diagnostics["recommendations"],
    "representative_solutions" => Dict(
        "best_nse" => Dict(
            "objectives" => best_nse_solution["objectives"],
            "validation_metrics" => Dict(string(k) => v for (k, v) in metrics_best_nse if !startswith(string(k), "_"))
        ),
        "best_lognse" => Dict(
            "objectives" => best_lognse_solution["objectives"],
            "validation_metrics" => Dict(string(k) => v for (k, v) in metrics_best_lognse if !startswith(string(k), "_"))
        ),
        "balanced" => Dict(
            "objectives" => balanced_solution["objectives"],
            "validation_metrics" => Dict(string(k) => v for (k, v) in metrics_balanced if !startswith(string(k), "_"))
        )
    )
)

open(joinpath(@__DIR__, "example3_pareto_diagnostics.json"), "w") do io
    JSON3.pretty(io, diagnostics_dict)
end
println("  ✓ example3_pareto_diagnostics.json")

# Export time series for each solution
nse_df = DataFrame(
    date = dates[test_idxs][test_valid],
    time_step = 1:length(test_obs),
    observed = test_obs,
    simulated = sim_best_nse,
    residual = test_obs .- sim_best_nse
)
CSV.write(joinpath(@__DIR__, "example3_best_nse_timeseries.csv"), nse_df)
println("  ✓ example3_best_nse_timeseries.csv")

lognse_df = DataFrame(
    date = dates[test_idxs][test_valid],
    time_step = 1:length(test_obs),
    observed = test_obs,
    simulated = sim_best_lognse,
    residual = test_obs .- sim_best_lognse
)
CSV.write(joinpath(@__DIR__, "example3_best_lognse_timeseries.csv"), lognse_df)
println("  ✓ example3_best_lognse_timeseries.csv")

balanced_df = DataFrame(
    date = dates[test_idxs][test_valid],
    time_step = 1:length(test_obs),
    observed = test_obs,
    simulated = sim_balanced,
    residual = test_obs .- sim_balanced
)
CSV.write(joinpath(@__DIR__, "example3_balanced_timeseries.csv"), balanced_df)
println("  ✓ example3_balanced_timeseries.csv")

println()
println("=" ^ 80)
println("Example 3 Complete!")
println("=" ^ 80)
println()
println("Summary:")
println("  - Pareto front size: $(multi_result["n_solutions"]) solutions")
println("  - Degeneracy status: $(pareto_diagnostics["degeneracy"]["status"])")
println("  - Best NSE validation: $(round(metrics_best_nse["NSE"], digits=4))")
println("  - Best LogNSE validation: $(round(metrics_best_lognse["LogNSE"], digits=4))")
println("  - Balanced validation: NSE=$(round(metrics_balanced["NSE"], digits=4)), LogNSE=$(round(metrics_balanced["LogNSE"], digits=4))")
println()
println("Key Insight:")
println("  Multi-objective optimization reveals trade-offs between high and low flows.")
println("  The Pareto front shows that improving one objective may degrade the other.")
println("  Choose a solution based on your modeling priorities:")
println("    - Flood forecasting → Best NSE")
println("    - Drought analysis → Best LogNSE")
println("    - General purpose → Balanced")
println()
println("Next steps:")
println("  1. Visualize Pareto front (NSE vs LogNSE scatter plot)")
println("  2. Compare time series for different solutions")
println("  3. Review diagnostics in example3_pareto_diagnostics.json")
println()
