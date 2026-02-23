# Example 2: Data Distribution Challenge
# Scenario: Arid, Ephemeral Stream
# Demonstrates: Automatic log transformation detection and performance comparison

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

println("=" ^ 80)
println("Example 2: Data Distribution Challenge (Arid, Ephemeral Stream)")
println("=" ^ 80)
println()
println("This example demonstrates:")
println("  - Strategy 3: Automatic log transformation detection")
println("  - Strategy 3: Manual data transformation (log, Box-Cox)")
println("  - Strategy 7: Objective function selection (KGE vs LogKGE)")
println("  - Before/after comparison showing transformation benefits")
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
gage_ids = data["gage_ids"]

# Select an arid catchment (you can specify a particular gage_id)
# For arid characteristics, select a catchment with lower mean flow
target_gage_id = 8025500 # Or specify your own gage_id
println("  Selected gage_id: $target_gage_id (for arid characteristics)")

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

# ==============================================================================
# Step 2: Data Distribution Analysis (Strategy 3)
# ==============================================================================
println("[Step 2] Analyzing data distribution...")
println()

# Compute magnitude ratio (max/min for non-zero values)
non_zero_flow = observed[valid_mask .& (observed .> 0)]
if !isempty(non_zero_flow)
    magnitude_ratio = maximum(non_zero_flow) / minimum(non_zero_flow)
    println("  Magnitude ratio (max/min): $(round(magnitude_ratio, digits=2))")

    if magnitude_ratio > 100.0
        println("  ⚠ Large magnitude ratio detected (>100)")
        println("  → Log transformation is RECOMMENDED")
    else
        println("  ✓ Moderate magnitude ratio (<100)")
        println("  → Log transformation may not be necessary")
    end
else
    println("  ⚠ All flow values are zero - check data quality")
    magnitude_ratio = 1.0
end
println()

# Compute basic statistics
valid_obs = observed[valid_mask]
println("  Flow statistics:")
println("    Mean: $(round(mean(valid_obs), digits=2)) mm/day")
println("    Std: $(round(std(valid_obs), digits=2)) mm/day")
println("    CV: $(round(std(valid_obs)/mean(valid_obs), digits=2))")
println("    Zero-flow days: $(sum(valid_obs .== 0)) ($(round(sum(valid_obs .== 0)/length(valid_obs)*100, digits=1))%)")
println()

# ==============================================================================
# Step 3: Data Splitting
# ==============================================================================
println("[Step 3] Splitting data using predefined train/test periods...")
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
# Step 4: Calibration WITHOUT Log Transformation
# ==============================================================================
println("[Step 4] Calibration WITHOUT log transformation (baseline)...")
println("  Objective: KGE (original space)")
println("  This may perform poorly for low flows...")
println()

calib_original = HydroModelMCP.Calibration.calibrate_model(
    "gr4j",
    train_forcing,
    train_obs;
    algorithm="BBO",
    maxiters=100,
    objective="KGE",
    solver_type="ODE",
    interp_type="LINEAR"
)

println("  Calibration complete (original space)")
println("  Training KGE: $(round(calib_original["best_objective"], digits=4))")
println()

# ==============================================================================
# Step 5: Calibration WITH Log Transformation (Strategy 3)
# ==============================================================================
println("[Step 5] Calibration WITH log transformation...")
println("  Objective: LogKGE (log space)")
println("  This should better capture low flows...")
println()

calib_log = HydroModelMCP.Calibration.calibrate_model(
    "gr4j",
    train_forcing,
    train_obs;
    algorithm="BBO",
    maxiters=100,
    objective="LogKGE",
    solver_type="ODE",
    interp_type="LINEAR"
)

println("  Calibration complete (log space)")
println("  Training LogKGE: $(round(calib_log["best_objective"], digits=4))")
println()

# ==============================================================================
# Step 6: Validation and Comparison
# ==============================================================================
println("[Step 6] Validating both approaches...")
println()

# Load model
model_module = HydroModelLibrary.load_model(:gr4j, reload=false)
model = Base.invokelatest(m -> m.model, model_module)
input_names = HydroModels.get_input_names(model)

hydro_config = HydroModels.HydroConfig(
    solver=HydroModels.ODESolver,
    interpolator=Val(HydroModels.LinearInterpolation)
)

# Prepare validation input
val_input_matrix = stack([Float64.(test_forcing[n]) for n in input_names], dims=1)

# Run with original parameters
params_original_nt = NamedTuple{Tuple(Symbol.(keys(calib_original["best_params"])))}(values(calib_original["best_params"]))
params_original_cv = ComponentVector(params=params_original_nt)
result_original = model(val_input_matrix, params_original_cv; config=hydro_config)
sim_original = result_original[end, :]

# Run with log-transformed parameters
params_log_nt = NamedTuple{Tuple(Symbol.(keys(calib_log["best_params"])))}(values(calib_log["best_params"]))
params_log_cv = ComponentVector(params=params_log_nt)
result_log = model(val_input_matrix, params_log_cv; config=hydro_config)
sim_log = result_log[end, :]

# Compute comprehensive metrics for both
metrics_original = HydroModelMCP.Metrics.compute_metrics(
    sim_original, test_obs,
    ["NSE", "KGE", "LogNSE", "LogKGE", "RMSE", "PBIAS", "R2"]
)

metrics_log = HydroModelMCP.Metrics.compute_metrics(
    sim_log, test_obs,
    ["NSE", "KGE", "LogNSE", "LogKGE", "RMSE", "PBIAS", "R2"]
)

println("  Validation Results Comparison:")
println("  " * "-" ^ 60)
println("  Metric          | Original (KGE) | Log (LogKGE) | Improvement")
println("  " * "-" ^ 60)

metric_names = ["NSE", "KGE", "LogNSE", "LogKGE", "RMSE", "PBIAS", "R2"]
for metric in metric_names
    if haskey(metrics_original, metric) && haskey(metrics_log, metric)
        val_orig = metrics_original[metric]
        val_log = metrics_log[metric]

        # Calculate improvement (higher is better for NSE, KGE, R2; lower for RMSE, PBIAS)
        if metric in ["RMSE", "PBIAS"]
            improvement = val_orig - val_log  # Positive means log is better
            symbol = improvement > 0 ? "↓" : "↑"
        else
            improvement = val_log - val_orig  # Positive means log is better
            symbol = improvement > 0 ? "↑" : "↓"
        end

        println("  $(rpad(metric, 15)) | $(rpad(round(val_orig, digits=4), 14)) | $(rpad(round(val_log, digits=4), 12)) | $(round(improvement, digits=4)) $symbol")
    end
end
println("  " * "-" ^ 60)
println()

# Highlight key findings
println("  Key Findings:")
if metrics_log["LogKGE"] > metrics_original["LogKGE"]
    println("    ✓ Log transformation IMPROVED low-flow performance (LogKGE)")
else
    println("    ✗ Log transformation did NOT improve low-flow performance")
end

if metrics_log["KGE"] < metrics_original["KGE"]
    println("    ⚠ Trade-off: Slightly reduced overall fit (KGE)")
else
    println("    ✓ Maintained or improved overall fit (KGE)")
end
println()

# ==============================================================================
# Step 7: Manual Transformation Demonstration (Strategy 3)
# ==============================================================================
println("[Step 7] Demonstrating manual data transformation...")
println()

# Log transformation
transformed_log, params_log_transform = HydroModelMCP.Metrics.transform_data(
    train_obs,
    :log
)
println("  Log transformation:")
println("    Original range: [$(round(minimum(train_obs), digits=2)), $(round(maximum(train_obs), digits=2))]")
println("    Transformed range: [$(round(minimum(transformed_log), digits=2)), $(round(maximum(transformed_log), digits=2))]")
println("    Floor value: $(params_log_transform["floor"])")
println()

# Box-Cox transformation
transformed_boxcox, params_boxcox = HydroModelMCP.Metrics.transform_data(
    train_obs,
    :boxcox
)
println("  Box-Cox transformation:")
println("    Optimal lambda: $(round(params_boxcox["lambda"], digits=4))")
println("    Transformed range: [$(round(minimum(transformed_boxcox), digits=2)), $(round(maximum(transformed_boxcox), digits=2))]")
println()

# Demonstrate inverse transformation
recovered_log = HydroModelMCP.Metrics.inverse_transform_data(
    transformed_log,
    params_log_transform
)
recovery_error = maximum(abs.(recovered_log .- train_obs))
println("  Inverse transformation test:")
println("    Max recovery error: $(round(recovery_error, digits=10))")
println("    $(recovery_error < 1e-6 ? "✓" : "✗") Transformation is reversible")
println()

# ==============================================================================
# Step 8: Export Results
# ==============================================================================
println("[Step 8] Exporting results...")
println()

# Export comparison JSON
comparison_dict = Dict(
    "model" => "gr4j",
    "scenario" => "arid_ephemeral",
    "data_file" => data_path,
    "magnitude_ratio" => magnitude_ratio,
    "log_transform_recommended" => magnitude_ratio > 100.0,
    "original_calibration" => Dict(
        "objective" => "KGE",
        "parameters" => Dict(string(k) => v for (k, v) in pairs(calib_original["best_params"])),
        "validation_metrics" => Dict(string(k) => v for (k, v) in metrics_original if !startswith(string(k), "_"))
    ),
    "log_calibration" => Dict(
        "objective" => "LogKGE",
        "parameters" => Dict(string(k) => v for (k, v) in pairs(calib_log["best_params"])),
        "validation_metrics" => Dict(string(k) => v for (k, v) in metrics_log if !startswith(string(k), "_"))
    ),
    "transformation_methods" => Dict(
        "log" => params_log_transform,
        "boxcox" => params_boxcox
    )
)

open(joinpath(@__DIR__, "example2_comparison.json"), "w") do io
    JSON3.pretty(io, comparison_dict)
end
println("  ✓ example2_comparison.json")

# Export time series - original
original_df = DataFrame(
    date = dates[test_idxs][test_valid],
    time_step = 1:length(test_obs),
    observed = test_obs,
    simulated = sim_original,
    residual = test_obs .- sim_original
)
CSV.write(joinpath(@__DIR__, "example2_original_timeseries.csv"), original_df)
println("  ✓ example2_original_timeseries.csv")

# Export time series - log
log_df = DataFrame(
    date = dates[test_idxs][test_valid],
    time_step = 1:length(test_obs),
    observed = test_obs,
    simulated = sim_log,
    residual = test_obs .- sim_log
)
CSV.write(joinpath(@__DIR__, "example2_log_timeseries.csv"), log_df)
println("  ✓ example2_log_timeseries.csv")

# Export metrics comparison
metrics_comparison_df = DataFrame(
    metric = metric_names,
    original_KGE = [get(metrics_original, m, NaN) for m in metric_names],
    log_LogKGE = [get(metrics_log, m, NaN) for m in metric_names]
)
CSV.write(joinpath(@__DIR__, "example2_metrics_comparison.csv"), metrics_comparison_df)
println("  ✓ example2_metrics_comparison.csv")

println()
println("=" ^ 80)
println("Example 2 Complete!")
println("=" ^ 80)
println()
println("Summary:")
println("  - Original (KGE) validation: $(round(metrics_original["KGE"], digits=4))")
println("  - Log (LogKGE) validation: $(round(metrics_log["LogKGE"], digits=4))")
println("  - Log transformation recommended: $(magnitude_ratio > 100.0 ? "YES" : "NO")")
println()
println("Key Insight:")
if metrics_log["LogKGE"] > metrics_original["LogKGE"]
    println("  ✓ Log transformation successfully improved low-flow simulation")
    println("    This demonstrates Strategy 3: Data transformation for skewed distributions")
else
    println("  ⚠ Log transformation did not improve performance")
    println("    This may indicate the data is not sufficiently skewed")
end
println()
println("Next steps:")
println("  1. Review comparison in example2_comparison.json")
println("  2. Visualize both time series to see differences in low-flow periods")
println("  3. Compare metrics in example2_metrics_comparison.csv")
println()
