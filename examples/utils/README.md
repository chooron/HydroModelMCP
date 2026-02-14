# Visualization Utilities

This directory contains plotting utilities for visualizing HydroModelMCP example results.

## Python Plotting (plot_results.py)

The `plot_results.py` script provides automated visualization for all three examples.

### Requirements

```bash
pip install pandas matplotlib numpy
```

### Usage

#### Generate All Plots for an Example

```bash
# From examples/utils/ directory
python plot_results.py --example 1
python plot_results.py --example 2
python plot_results.py --example 3
```

This will generate all relevant plots for the specified example and save them as PNG files.

#### Plot Specific Files

```bash
# Time series plot
python plot_results.py --timeseries ../example1_validation_timeseries.csv --output validation.png

# Scatter plot (observed vs simulated)
python plot_results.py --scatter ../example1_validation_timeseries.csv --output scatter.png

# Sensitivity analysis
python plot_results.py --sensitivity ../example1_sensitivity.csv --output sensitivity.png

# Pareto front
python plot_results.py --pareto ../example3_pareto_front.csv --output pareto.png

# Compare two time series
python plot_results.py --comparison ../example2_original_timeseries.csv ../example2_log_timeseries.csv --output comparison.png
```

### Output Files

#### Example 1 Outputs:
- `example1_calibration_plot.png` - Calibration period time series
- `example1_validation_plot.png` - Validation period time series
- `example1_scatter_plot.png` - Observed vs simulated scatter
- `example1_sensitivity_plot.png` - Parameter sensitivity bar charts

#### Example 2 Outputs:
- `example2_comparison_plot.png` - Side-by-side comparison of original vs log-transformed
- `example2_original_scatter.png` - Scatter plot for original calibration
- `example2_log_scatter.png` - Scatter plot for log-transformed calibration

#### Example 3 Outputs:
- `example3_pareto_front_plot.png` - Pareto front visualization
- `example3_best_nse_plot.png` - Time series for best NSE solution
- `example3_best_lognse_plot.png` - Time series for best LogNSE solution
- `example3_balanced_plot.png` - Time series for balanced solution

## Julia Plotting (Plots.jl)

If you prefer Julia for visualization, here are examples using Plots.jl:

### Installation

```julia
using Pkg
Pkg.add("Plots")
Pkg.add("CSV")
Pkg.add("DataFrames")
```

### Time Series Plot

```julia
using CSV, DataFrames, Plots

# Load data
df = CSV.read("example1_validation_timeseries.csv", DataFrame)

# Create plot
plot(df.time_step, df.observed, label="Observed", linewidth=2, alpha=0.7)
plot!(df.time_step, df.simulated, label="Simulated", linewidth=2, alpha=0.7)
xlabel!("Time Step")
ylabel!("Flow (mm)")
title!("Example 1: Validation Period")

# Save
savefig("example1_validation_julia.png")
```

### Scatter Plot

```julia
using CSV, DataFrames, Plots

df = CSV.read("example1_validation_timeseries.csv", DataFrame)

# Scatter plot
scatter(df.observed, df.simulated, label="Data", alpha=0.5, markersize=3)

# 1:1 line
min_val = minimum([minimum(df.observed), minimum(df.simulated)])
max_val = maximum([maximum(df.observed), maximum(df.simulated)])
plot!([min_val, max_val], [min_val, max_val], label="1:1 line", linewidth=2, linestyle=:dash, color=:red)

xlabel!("Observed Flow (mm)")
ylabel!("Simulated Flow (mm)")
title!("Observed vs Simulated")

savefig("scatter_julia.png")
```

### Sensitivity Analysis

```julia
using CSV, DataFrames, Plots

df = CSV.read("example1_sensitivity.csv", DataFrame)

# Bar plot
bar(df.parameter, df.mu_star, orientation=:h, legend=false)
xlabel!("μ* (Elementary Effect Mean)")
ylabel!("Parameter")
title!("Parameter Sensitivity")

savefig("sensitivity_julia.png")
```

### Pareto Front

```julia
using CSV, DataFrames, Plots

df = CSV.read("example3_pareto_front.csv", DataFrame)

# Scatter plot
scatter(df.NSE, df.LogNSE, label="Pareto Front", markersize=5, alpha=0.7)
scatter!([1.0], [1.0], label="Ideal Point", markersize=10, color=:red, marker=:star)

xlabel!("NSE (High Flow Performance)")
ylabel!("LogNSE (Low Flow Performance)")
title!("Pareto Front")

savefig("pareto_julia.png")
```

## R Plotting (ggplot2)

For R users, here are examples using ggplot2:

### Installation

```r
install.packages(c("ggplot2", "readr"))
```

### Time Series Plot

```r
library(ggplot2)
library(readr)

# Load data
df <- read_csv("example1_validation_timeseries.csv")

# Create plot
ggplot(df, aes(x = time_step)) +
  geom_line(aes(y = observed, color = "Observed"), linewidth = 1) +
  geom_line(aes(y = simulated, color = "Simulated"), linewidth = 1) +
  labs(x = "Time Step", y = "Flow (mm)", title = "Example 1: Validation Period") +
  scale_color_manual(values = c("Observed" = "black", "Simulated" = "red")) +
  theme_minimal() +
  theme(legend.title = element_blank())

ggsave("example1_validation_r.png", width = 12, height = 6, dpi = 300)
```

### Scatter Plot

```r
library(ggplot2)
library(readr)

df <- read_csv("example1_validation_timeseries.csv")

# Calculate limits for 1:1 line
min_val <- min(c(df$observed, df$simulated))
max_val <- max(c(df$observed, df$simulated))

ggplot(df, aes(x = observed, y = simulated)) +
  geom_point(alpha = 0.5, size = 2) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", linewidth = 1) +
  labs(x = "Observed Flow (mm)", y = "Simulated Flow (mm)", title = "Observed vs Simulated") +
  coord_fixed() +
  theme_minimal()

ggsave("scatter_r.png", width = 8, height = 8, dpi = 300)
```

### Pareto Front

```r
library(ggplot2)
library(readr)

df <- read_csv("example3_pareto_front.csv")

ggplot(df, aes(x = NSE, y = LogNSE)) +
  geom_point(aes(color = solution_id), size = 3, alpha = 0.7) +
  geom_point(aes(x = 1.0, y = 1.0), color = "red", size = 5, shape = 8) +
  labs(x = "NSE (High Flow Performance)",
       y = "LogNSE (Low Flow Performance)",
       title = "Pareto Front",
       color = "Solution") +
  scale_color_viridis_c() +
  theme_minimal()

ggsave("pareto_r.png", width = 10, height = 8, dpi = 300)
```

## Custom Plotting

All example scripts export results in standard CSV format, making it easy to create custom visualizations with any tool:

### CSV File Structures

**Time Series Files** (`*_timeseries.csv`):
- `time_step`: Integer time step index
- `observed`: Observed flow values
- `simulated`: Simulated flow values
- `residual`: Observed - Simulated

**Sensitivity Files** (`*_sensitivity.csv`):
- `parameter`: Parameter name
- `mu_star`: Mean of absolute elementary effects
- `sigma`: Standard deviation of elementary effects

**Pareto Front Files** (`*_pareto_front.csv`):
- `solution_id`: Solution index
- `NSE`: NSE objective value
- `LogNSE`: LogNSE objective value
- Additional columns for each parameter

**Metrics Comparison Files** (`*_metrics_comparison.csv`):
- `metric`: Metric name
- Additional columns for each calibration approach

## Tips for Publication-Quality Figures

1. **Resolution**: Use at least 300 DPI for publications
2. **File Format**: PNG for presentations, PDF/EPS for publications
3. **Font Size**: Keep labels readable (10-12pt)
4. **Colors**: Use colorblind-friendly palettes
5. **Line Width**: 1-2pt for main lines, 0.5-1pt for grid lines
6. **Legends**: Place outside plot area if space allows
7. **Axis Labels**: Always include units

## Troubleshooting

**Python: "ModuleNotFoundError: No module named 'matplotlib'"**
```bash
pip install matplotlib pandas numpy
```

**Julia: "Package Plots not found"**
```julia
using Pkg
Pkg.add("Plots")
```

**R: "Error: package 'ggplot2' not found"**
```r
install.packages("ggplot2")
```

**Python: "FileNotFoundError"**
- Make sure you're running from the `examples/utils/` directory
- Or use `--dir` flag to specify the examples directory:
  ```bash
  python plot_results.py --example 1 --dir /path/to/examples
  ```

## Additional Resources

- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
- [Plots.jl Documentation](https://docs.juliaplots.org/stable/)
- [ggplot2 Documentation](https://ggplot2.tidyverse.org/)
