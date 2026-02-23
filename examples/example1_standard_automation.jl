# Example 1: Standard Automation Baseline
# Scenario: Humid, Perennial Catchment
# Demonstrates: Full lifecycle automation with quality diagnostics

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
using DotEnv
using NPZ
using Dates
using Statistics

println("="^80)
println("Example 1: Standard Automation Baseline (Humid, Perennial Catchment)")
println("="^80)
println()
println("This example demonstrates:")
println("  - Strategy 1: Global sensitivity analysis (Morris method)")
println("  - Strategy 4: Data splitting (split_sample, 70/30)")
println("  - Strategy 8: Single-objective calibration (BBO algorithm)")
println("  - Strategy 10: Calibration diagnostics (convergence, boundaries, plateau)")
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
println("  Available keys: $(keys(data))")

# Get dimensions
# Data structure: forcings(671 catchments, 12418 timesteps, 3 variables)
#                 target(671 catchments, 12418 timesteps, 1)
n_catchments = size(data["forcings"], 1)
n_timesteps = size(data["forcings"], 2)
n_variables = size(data["forcings"], 3)
println("  Number of catchments: $n_catchments")
println("  Number of timesteps: $n_timesteps")
println("  Number of forcing variables: $n_variables")
println()

# Get gage IDs
gage_ids = data["gage_ids"]
println("  Available gage IDs: $(length(gage_ids)) catchments")
println("  First 5 gage IDs: $(gage_ids[1:min(5, length(gage_ids))])")
println()

# Select a catchment by gage_id
# You can change this to any gage_id from the dataset
target_gage_id = 5362000  # Use first catchment, or specify your own gage_id
println("  Selected gage_id: $target_gage_id")

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
# forcings shape: (catchments, timesteps, variables)
# Variables: [precipitation, temperature, potential_evaporation]
forcing_data = data["forcings"][catchment_idx, :, :]  # (timesteps, variables)
target_data = data["target"][catchment_idx, :, 1]     # (timesteps,)

# Get catchment area for unit conversion
area = data["attributes"][catchment_idx, 12]  # Area in km²
println("  Catchment area: $(round(area, digits=2)) km²")

# Convert streamflow from ft³/s to mm/day
# Formula: flow_mm_day = (10^3) * flow_ft3_s * 0.0283168 * 3600 * 24 / (area * 10^6)
# Simplifies to: flow_mm_day = flow_ft3_s * 2446.5792 / area
target_data_mm = target_data * (10^3) * 0.0283168 * 3600 * 24 / (area * (10^6))

# Create forcing NamedTuple
forcing_nt = (
    P=forcing_data[:, 1],      # Precipitation (mm/day)
    T=forcing_data[:, 2],      # Temperature (°C)
    Ep=forcing_data[:, 3]      # Potential evaporation (mm/day)
)
observed = target_data_mm  # Streamflow (mm/day)

# Check for NaN values in all data
nan_in_P = sum(isnan.(forcing_nt.P))
nan_in_T = sum(isnan.(forcing_nt.T))
nan_in_Ep = sum(isnan.(forcing_nt.Ep))
nan_in_obs = sum(isnan.(observed))

println("  NaN values detected:")
println("    Precipitation: $nan_in_P")
println("    Temperature: $nan_in_T")
println("    Potential evaporation: $nan_in_Ep")
println("    Observed flow: $nan_in_obs")
println()

# Create valid mask (no NaN in any variable)
valid_mask = .!(isnan.(forcing_nt.P) .| isnan.(forcing_nt.T) .| isnan.(forcing_nt.Ep) .| isnan.(observed))
println("  Valid data points: $(sum(valid_mask)) / $(length(observed))")

# Basic statistics (only valid data)
valid_obs = observed[valid_mask]
if !isempty(valid_obs)
    println("  Flow statistics (valid data only):")
    println("    Mean: $(round(mean(valid_obs), digits=2)) mm/day")
    println("    Std: $(round(std(valid_obs), digits=2)) mm/day")
    println("    Range: [$(round(minimum(valid_obs), digits=2)), $(round(maximum(valid_obs), digits=2))] mm/day")
else
    error("No valid data points found for selected catchment")
end
println()

# ==============================================================================
# Step 2: Model Information
# ==============================================================================
println("[Step 2] Getting GR4J model information...")
println()

model_info = HydroModelMCP.Discovery.get_model_info("gr4j")
println("  Model: $(model_info["model_name"])")
println("  Inputs: $(model_info["inputs"])")
println("  Outputs: $(model_info["outputs"])")
println()

# Get parameter details
params_info = HydroModelMCP.Discovery.get_parameters_detail("gr4j")
println("  Parameters:")
for param_info in params_info
    bounds = param_info["bounds"]
    if !isnothing(bounds)
        println("    $(param_info["name"]): [$(bounds[1]), $(bounds[2])] $(param_info["unit"])")
    end
end
println()

# ==============================================================================
# Step 3: Data Splitting (Strategy 4)
# ==============================================================================
println("[Step 3] Splitting data using predefined train/test periods...")
println()

# Use the predefined train/test indices and filter for valid data
train_forcing = (
    P=forcing_nt.P[train_idxs],
    T=forcing_nt.T[train_idxs],
    Ep=forcing_nt.Ep[train_idxs]
)
test_forcing = (
    P=forcing_nt.P[test_idxs],
    T=forcing_nt.T[test_idxs],
    Ep=forcing_nt.Ep[test_idxs]
)
train_obs = observed[train_idxs]
test_obs = observed[test_idxs]

# Remove NaN values from training data
train_valid = .!(isnan.(train_forcing.P) .| isnan.(train_forcing.T) .| isnan.(train_forcing.Ep) .| isnan.(train_obs))
train_forcing = (
    P=train_forcing.P[train_valid],
    T=train_forcing.T[train_valid],
    Ep=train_forcing.Ep[train_valid]
)
train_obs = train_obs[train_valid]

# Remove NaN values from testing data
test_valid = .!(isnan.(test_forcing.P) .| isnan.(test_forcing.T) .| isnan.(test_forcing.Ep) .| isnan.(test_obs))
test_forcing = (
    P=test_forcing.P[test_valid],
    T=test_forcing.T[test_valid],
    Ep=test_forcing.Ep[test_valid]
)
test_obs = test_obs[test_valid]

println("  Training data: $(length(train_obs)) valid timesteps")
println("  Testing data: $(length(test_obs)) valid timesteps")

if length(train_obs) == 0 || length(test_obs) == 0
    error("No valid data in training or testing period. Please select a different catchment.")
end
println()

# Store indices for later use
cal_start, cal_end = 1, length(train_obs)
val_start, val_end = 1, length(test_obs)

# ==============================================================================
# Step 4: Sensitivity Analysis (Strategy 1)
# ==============================================================================
println("[Step 4] Running global sensitivity analysis (Morris method)...")
println("  This identifies which parameters are most important for calibration")
println()

sensitivity_result = HydroModelMCP.SensitivityAnalysis.run_sensitivity(
    "gr4j",
    train_forcing,
    train_obs;
    method="morris",
    n_samples=50,
    objective="KGE",
    threshold=0.1,
    solver_type="ODE",
    interp_type="LINEAR"
)

println("  Sensitivity analysis complete")
println("  Important parameters (sensitivity > threshold):")
for param_name in sensitivity_result["calibratable"]
    idx = findfirst(p -> p == param_name, sensitivity_result["param_names"])
    sensitivity = sensitivity_result["sensitivities"][idx]
    println("    $param_name: sensitivity = $(round(sensitivity, digits=4))")
end
println()

# Save sensitivity results
sensitivity_df = DataFrame(
    parameter=sensitivity_result["param_names"],
    sensitivity=sensitivity_result["sensitivities"]
)
CSV.write(joinpath(@__DIR__, "example1_sensitivity.csv"), sensitivity_df)
println("  Sensitivity results saved to: example1_sensitivity.csv")
println()

# ==============================================================================
# Step 5: Parameter Calibration (Strategy 8)
# ==============================================================================
println("[Step 5] Running parameter calibration...")
println("  Algorithm: BBO (Biogeography-Based Optimization)")
println("  Objective: KGE (Kling-Gupta Efficiency)")
println("  Iterations: 1000")
println("  Trials: 3 (for convergence diagnostics)")
println("  This may take several minutes...")
println()

calib_result = HydroModelMCP.Calibration.calibrate_model(
    "gr4j",
    train_forcing,
    train_obs;
    algorithm="BBO",
    maxiters=1000,
    n_trials=3,
    objective="KGE",
    solver_type="ODE",
    interp_type="LINEAR"
)

println("  Calibration complete!")
println()
println("  Best parameters:")
for (param, value) in pairs(calib_result["best_params"])
    println("    $param: $(round(value, digits=6))")
end
println()
println("  Training KGE: $(round(calib_result["best_objective"], digits=4))")
println()

# ==============================================================================
# Step 6: Calibration Diagnostics (Strategy 10)
# ==============================================================================
println("[Step 6] Running calibration diagnostics...")
println()

diagnostics = HydroModelMCP.Calibration.diagnose_calibration(
    calib_result;
    boundary_tolerance=0.01,
    convergence_threshold=0.05
)

println("  Diagnostic Results:")
println("  ------------------")
println("  1. Convergence Check:")
println("     Status: $(diagnostics["convergence"]["passed"] ? "PASS ✓" : "FAIL ✗")")
if haskey(diagnostics["convergence"], "cv")
    println("     CV: $(round(diagnostics["convergence"]["cv"], digits=4)) (threshold: 0.05)")
else
    println("     $(diagnostics["convergence"]["message"])")
end
println()

println("  2. Boundary Check:")
if isempty(diagnostics["boundaries"]["at_bound"])
    println("     Status: PASS ✓ (no parameters at bounds)")
else
    println("     Status: FAIL ✗")
    println("     Parameters at bounds: $(diagnostics["boundaries"]["at_bound"])")
end
println()

println("  3. Plateau Check:")
println("     Status: $(diagnostics["plateau"]["passed"] ? "PASS ✓" : "FAIL ✗")")
if haskey(diagnostics["plateau"], "best_worst_gap")
    println("     Best-worst gap: $(round(diagnostics["plateau"]["best_worst_gap"], digits=6))")
else
    println("     $(diagnostics["plateau"]["message"])")
end
println()

println("  4. Hat-trick (all checks pass):")
println("     Status: $(diagnostics["hat_trick"] ? "YES ✓" : "NO ✗")")
println()

if !isempty(diagnostics["recommendations"])
    println("  Recommendations:")
    for rec in diagnostics["recommendations"]
        println("    - $rec")
    end
    println()
end

# ==============================================================================
# Step 7: Validation Run
# ==============================================================================
println("[Step 7] Running validation on test period...")
println()

# Load model
model_module = HydroModelLibrary.load_model(:gr4j, reload=false)
model = Base.invokelatest(m -> m.model, model_module)
input_names = HydroModels.get_input_names(model)

# Prepare configuration
hydro_config = HydroModels.HydroConfig(
    solver=HydroModels.ODESolver,
    interpolator=Val(HydroModels.LinearInterpolation)
)

# Convert best parameters to ComponentVector
best_params_nt = NamedTuple{Tuple(Symbol.(keys(calib_result["best_params"])))}(values(calib_result["best_params"]))
best_params_cv = ComponentVector(params=best_params_nt)

# Run on calibration period
cal_input_matrix = stack([Float64.(train_forcing[n]) for n in input_names], dims=1)
cal_result_matrix = model(cal_input_matrix, best_params_cv; config=hydro_config)
cal_sim = cal_result_matrix[end, :]

# Run on validation period
val_input_matrix = stack([Float64.(test_forcing[n]) for n in input_names], dims=1)
val_result_matrix = model(val_input_matrix, best_params_cv; config=hydro_config)
val_sim = val_result_matrix[end, :]

# Compute metrics
cal_metrics = HydroModelMCP.Metrics.compute_metrics(
    cal_sim, train_obs,
    ["NSE", "KGE", "RMSE", "PBIAS", "R2"]
)

val_metrics = HydroModelMCP.Metrics.compute_metrics(
    val_sim, test_obs,
    ["NSE", "KGE", "RMSE", "PBIAS", "R2"]
)

println("  Calibration period metrics:")
for (metric, value) in cal_metrics
    if !startswith(string(metric), "_")  # Skip internal metrics
        println("    $metric: $(round(value, digits=4))")
    end
end
println()

println("  Validation period metrics:")
for (metric, value) in val_metrics
    if !startswith(string(metric), "_")
        println("    $metric: $(round(value, digits=4))")
    end
end
println()

# Performance drop analysis
nse_drop = cal_metrics["NSE"] - val_metrics["NSE"]
kge_drop = cal_metrics["KGE"] - val_metrics["KGE"]
println("  Performance drop (calibration → validation):")
println("    NSE: $(round(nse_drop, digits=4)) ($(round(abs(nse_drop/cal_metrics["NSE"])*100, digits=1))%)")
println("    KGE: $(round(kge_drop, digits=4)) ($(round(abs(kge_drop/cal_metrics["KGE"])*100, digits=1))%)")
println()

# ==============================================================================
# Step 8: Export Results
# ==============================================================================
println("[Step 8] Exporting results...")
println()

# Export summary JSON
result_dict = Dict(
    "model" => "gr4j",
    "scenario" => "humid_perennial",
    "data_file" => data_path,
    "data_length" => length(observed),
    "calibration_period" => "$cal_start:$cal_end",
    "validation_period" => "$val_start:$val_end",
    "best_parameters" => Dict(string(k) => v for (k, v) in pairs(calib_result["best_params"])),
    "calibration_metrics" => Dict(string(k) => v for (k, v) in cal_metrics if !startswith(string(k), "_")),
    "validation_metrics" => Dict(string(k) => v for (k, v) in val_metrics if !startswith(string(k), "_")),
    "diagnostics" => diagnostics,
    "sensitivity" => Dict(
        "param_names" => sensitivity_result["param_names"],
        "sensitivities" => sensitivity_result["sensitivities"],
        "calibratable" => sensitivity_result["calibratable"],
        "fixed" => sensitivity_result["fixed"]
    )
)

open(joinpath(@__DIR__, "example1_results.json"), "w") do io
    JSON3.pretty(io, result_dict)
end
println("  ✓ example1_results.json")

# Export calibration time series
cal_df = DataFrame(
    date=dates[train_idxs][train_valid],
    time_step=1:length(train_obs),
    observed=train_obs,
    simulated=cal_sim,
    residual=train_obs .- cal_sim
)
CSV.write(joinpath(@__DIR__, "example1_calibration_timeseries.csv"), cal_df)
println("  ✓ example1_calibration_timeseries.csv")

# Export validation time series
val_df = DataFrame(
    date=dates[test_idxs][test_valid],
    time_step=1:length(test_obs),
    observed=test_obs,
    simulated=val_sim,
    residual=test_obs .- val_sim
)
CSV.write(joinpath(@__DIR__, "example1_validation_timeseries.csv"), val_df)
println("  ✓ example1_validation_timeseries.csv")

println()
println("="^80)
println("Example 1 Complete!")
println("="^80)
println()
println("Summary:")
println("  - Calibration KGE: $(round(cal_metrics["KGE"], digits=4))")
println("  - Validation KGE: $(round(val_metrics["KGE"], digits=4))")
println("  - Hat-trick: $(diagnostics["hat_trick"] ? "YES ✓" : "NO ✗")")
println()
println("Next steps:")
println("  1. Review results in example1_results.json")
println("  2. Visualize time series using utils/plot_results.py")
println("  3. Check sensitivity analysis in example1_sensitivity.csv")
println()
