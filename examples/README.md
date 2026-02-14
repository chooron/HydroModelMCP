# HydroModelMCP Examples

This directory contains three standalone Julia examples demonstrating different hydrological modeling strategies and calibration techniques.

## Examples Overview

### Example 1: Standard Automation Baseline
- **File:** [example1_standard_automation.jl](example1_standard_automation.jl)
- **Scenario:** Humid, perennial catchment
- **Model:** GR4J (4 parameters)
- **Focus:** Full lifecycle automation with quality diagnostics
- **Strategies:** Sensitivity analysis, data splitting, single-objective calibration, diagnostics

### Example 2: Data Distribution Challenge
- **File:** [example2_data_transformation.jl](example2_data_transformation.jl)
- **Scenario:** Arid, ephemeral stream
- **Model:** GR4J
- **Focus:** Automatic log transformation detection and performance comparison
- **Strategies:** Data transformation, objective function selection, before/after comparison

### Example 3: Multi-objective Snow Modeling
- **File:** [example3_multiobjective_snow.jl](example3_multiobjective_snow.jl)
- **Scenario:** Snow-dominated catchment
- **Model:** GR4J + CemaNeige (6 parameters total)
- **Focus:** Pareto front generation with parameter constraints
- **Strategies:** Constrained sampling, multi-objective optimization, Pareto analysis

## Running Examples

Each example is self-contained and can be run independently:

```bash
# Navigate to examples directory
cd examples

# Run any example
julia --project=.. example1_standard_automation.jl
julia --project=.. example2_data_transformation.jl
julia --project=.. example3_multiobjective_snow.jl
```

**Required Julia Packages:**

The examples require the following packages (already in Project.toml):
- HydroModelMCP
- HydroModels
- HydroModelLibrary
- ComponentArrays
- DataInterpolations
- JSON3
- CSV
- DataFrames
- DotEnv
- NPZ
- Dates
- Statistics
- LinearAlgebra (Example 3 only)

Install dependencies:
```bash
julia --project=.. -e 'using Pkg; Pkg.instantiate()'
```

## Data Requirements

### Using CAMELS Dataset (NPZ Format)

All examples are configured to load data from the CAMELS dataset in NPZ format. The dataset path is specified in the `.env` file at the project root.

**Setup:**

1. Ensure you have the CAMELS dataset NPZ file
2. Update the `.env` file in the project root:
   ```
   CAMESL_DATASET_PATH = G:\Dataset\camels_dataset.npz
   ```
3. The examples will automatically load data from this file

**Dataset Structure:**

The NPZ file contains:
- `gage_ids`: Array of gage IDs (671 catchments)
- `forcings`: Array of shape (671 catchments, 12418 timesteps, 3 variables)
  - Variable 1: Precipitation (mm/day)
  - Variable 2: Temperature (°C)
  - Variable 3: Potential evaporation (mm/day)
- `target`: Array of shape (671 catchments, 12418 timesteps, 1)
  - Streamflow observations (mm/day)
- `attributes`: Catchment attributes (optional)

**Selecting Catchments by Gage ID:**

Each example selects a catchment using its gage_id. You can modify the `target_gage_id` variable in each script:

```julia
# Example: Select a specific gage_id
gage_ids = data["gage_ids"]
target_gage_id = "01013500"  # Replace with your desired gage_id

# Or use index-based selection
target_gage_id = gage_ids[1]  # First catchment
```

**Default Selections:**
- **Example 1** (Humid): `gage_ids[1]` (first catchment)
- **Example 2** (Arid): `gage_ids[50]` (50th catchment)
- **Example 3** (Snow): `gage_ids[100]` (100th catchment)

**Handling Missing Data (NaN):**

The examples automatically:
1. Detect NaN values in all forcing variables and observations
2. Filter out timesteps with any NaN values
3. Report the number of valid data points
4. Ensure sufficient data for calibration and validation

**Time Periods:**

All examples use consistent time periods:
- **Full dataset**: 1980-10-01 to 2014-09-30 (12418 days)
- **Training**: 1989-01-01 to 1998-12-31 (10 years)
- **Testing**: 1999-01-01 to 2009-12-31 (11 years)

### Alternative: Using CSV Files

If you prefer to use CSV files instead of NPZ, you can modify the data loading section in each script. The required CSV format is:

```csv
date,prcp(mm/day),tmean(C),pet(mm),flow(mm)
1989-01-01,5.2,10.3,2.1,3.5
1989-01-02,0.0,12.1,2.3,3.2
...
```

## Expected Outputs

### Example 1 Outputs:
- `example1_results.json` - Calibration summary and diagnostics
- `example1_calibration_timeseries.csv` - Simulated vs observed (calibration period)
- `example1_validation_timeseries.csv` - Simulated vs observed (validation period)
- `example1_sensitivity.csv` - Parameter sensitivity indices

### Example 2 Outputs:
- `example2_comparison.json` - Side-by-side comparison of original vs log-transformed
- `example2_original_timeseries.csv` - Results without transformation
- `example2_log_timeseries.csv` - Results with log transformation
- `example2_metrics_comparison.csv` - Performance metrics comparison

### Example 3 Outputs:
- `example3_pareto_front.csv` - All Pareto solutions (parameters + objectives)
- `example3_pareto_diagnostics.json` - Degeneracy analysis
- `example3_best_nse_timeseries.csv` - Simulation with best NSE solution
- `example3_best_lognse_timeseries.csv` - Simulation with best LogNSE solution
- `example3_balanced_timeseries.csv` - Simulation with balanced solution

## Visualizing Results

See [utils/README.md](utils/README.md) for plotting instructions using Python, Julia, or R.

Quick example with Python:

```bash
cd utils
python plot_results.py --example 1 --input ../example1_validation_timeseries.csv
```

## Calibration Strategies Demonstrated

These examples demonstrate the 10 calibration strategies from the HydroModelMCP framework:

| Strategy | Description | Example |
|----------|-------------|---------|
| 1. Sensitivity Analysis | Global sensitivity (Morris, Sobol) | Example 1 |
| 2. Parameter Constraints | Delta Method for inequality constraints | Example 3 |
| 3. Data Transformation | Log, Box-Cox transformations | Example 2 |
| 4. Data Splitting | Split-sample test (70/30) | Examples 1, 2 |
| 5. Sampling Methods | LHS, Sobol, constrained sampling | Example 3 |
| 7. Objective Functions | KGE, LogKGE, NSE, LogNSE | All examples |
| 8. Optimization Algorithms | BBO, NSGA2 | All examples |
| 9. Multi-objective | Pareto front generation | Example 3 |
| 10. Diagnostics | Convergence, boundaries, plateau | Examples 1, 3 |

## Adapting for Your Research

To adapt these examples for your own research:

1. **Prepare your data** in the required CSV format
2. **Update data paths** in the scripts
3. **Adjust model selection** if using different hydrological models
4. **Modify calibration settings**:
   - `maxiters` - Number of optimization iterations
   - `n_trials` - Number of independent calibration runs
   - `ratio` - Calibration/validation split ratio
   - `warmup` - Warmup period (days) to skip
5. **Select appropriate objectives** based on your research goals:
   - High flows: NSE, KGE
   - Low flows: LogNSE, LogKGE
   - Water balance: PBIAS
   - Dynamics: R²

## Model Requirements

### Example 1 & 2: GR4J
The GR4J model should be available in HydroModelLibrary. If not, add it following the HydroModelLibrary documentation.

### Example 3: GR4J + CemaNeige
This example requires the combined GR4J+CemaNeige model (6 parameters: 4 from GR4J + 2 from CemaNeige). Add this model to HydroModelLibrary before running Example 3.

## Troubleshooting

**Error: "Data file not found"**
- Update the `data_path` variable to point to your actual data file

**Error: "Model not found in HydroModelLibrary"**
- Ensure the required model is installed in HydroModelLibrary
- Check model name spelling (case-sensitive)

**Error: "Missing input variables"**
- Verify your CSV has all required columns: prcp(mm/day), tmean(C), pet(mm), flow(mm)
- Check column names match exactly (case-sensitive)

**Calibration takes too long**
- Reduce `maxiters` (e.g., from 100 to 50)
- Reduce `n_trials` (e.g., from 3 to 1)
- For Example 3, reduce `population_size` (e.g., from 50 to 30)

## References

For more information on the calibration strategies and framework design, see:

- Mai, J. (2023). Ten strategies towards successful calibration of environmental models. *Journal of Hydrology*.
- HydroModelMCP documentation: [DEV.md](../DEV.md)
- HydroModels.jl: https://github.com/chooron/HydroModels.jl
- HydroModelLibrary.jl: https://github.com/chooron/HydroModelLibrary.jl
