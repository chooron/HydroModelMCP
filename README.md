# HydroModelMCP: Hydrological Modeling Agent Interface

HydroModelMCP is a **Model Context Protocol (MCP)** server built in Julia. It acts as a bridge between Large Language Models (like Claude) and high-performance hydrological models (powered by `HydroModels.jl`).

This interface allows AI agents to autonomously discover model structures, query physical parameter bounds, execute complex hydrological simulations, and perform model calibration with persistent result storage.

## 🚀 Features

### 1. Model Discovery & Inspection (5 Tools)

* **`list_models`**: Lists all available hydrological models in the library.
* **`find_model`**: Searches for a specific model (supports fuzzy matching/case-insensitivity).
* **`get_model_info`**: Retrieves the general structure, inputs, and description of a model.
* **`get_model_variables`**: Returns detailed metadata about internal variables (physical meaning, units).
* **`get_model_parameters`**: Returns detailed parameter information, including **physical bounds** (min/max) and units.

### 2. Simulation Engine (3 Tools)

* **`run_simulation`**: The core execution engine.
  - **Dynamic Loading**: Instantiates models on the fly.
  - **Multi-Source Data**: Accepts forcing data from **CSV files**, **JSON objects**, or **Redis keys**.
  - **Auto-Configuration**: Handles parameter injection (or random generation if missing) and state initialization.
  - **Flexible Solvers**: Supports various ODE solvers and interpolation methods via Enums.

* **`run_ensemble_parameters`**: Batch simulation with multiple parameter sets for a single model.
  - **Parameter Uncertainty Analysis**: Runs the same model with different parameter combinations.
  - **Parallel Execution**: Leverages multi-threading for performance.
  - **Ensemble Statistics**: Computes mean and standard deviation across ensemble members.
  - **Monte Carlo Simulation**: Ideal for parameter uncertainty quantification and sensitivity visualization.
  - **Result Storage**: Automatically saves ensemble results with unique IDs.

* **`run_validation`**: Split-period validation for model evaluation.
  - **Dual-Period Execution**: Runs model on both calibration and validation periods.
  - **Performance Comparison**: Computes metrics for both periods to assess generalization.
  - **Flexible Time Specification**: Supports both date ranges and index ranges.
  - **Multiple Metrics**: Calculates NSE, KGE, RMSE, and other performance indicators.

### 3. Calibration Workflow (9 Tools)

* **`compute_metrics`**: Calculate performance metrics (NSE, KGE, LogNSE, LogKGE, PBIAS, R2, RMSE).
* **`split_data`**: Split time series data into calibration and validation sets (3 strategies).
* **`run_sensitivity`**: Perform global sensitivity analysis (Sobol, Morris, eFAST).
* **`generate_samples`**: Generate parameter samples (LHS, Sobol, Random).
* **`calibrate_model`**: Single-objective calibration (BBO, PSO, DE, CMAES, ECA).
* **`calibrate_multiobjective`**: Multi-objective calibration (NSGA2, NSGA3).
* **`diagnose_calibration`**: Diagnose calibration quality and suggest improvements.
* **`configure_objectives`**: Recommend objective functions based on calibration goals.
* **`init_calibration_setup`**: One-click calibration setup with automatic configuration.

### 4. MCP Resources (9 Resources)

Resources allow clients to query data without invoking tools, improving efficiency:

**Static Resources**:
* **`hydro://models/catalog`**: Complete model catalog with URIs.
* **`hydro://guides/algorithms`**: Algorithm selection guide based on problem characteristics.
* **`hydro://guides/objectives`**: Objective function selection guide for different calibration goals.
* **`hydro://calibration/results`**: List of all stored calibration results.

**Dynamic Resources** (URI Templates):
* **`hydro://models/{model_name}/info`**: Model-specific detailed information.
* **`hydro://models/{model_name}/parameters`**: Model-specific parameter bounds.
* **`hydro://models/{model_name}/variables`**: Model-specific variable metadata.
* **`hydro://calibration/results/{result_id}`**: Retrieve specific calibration result by ID.
* **`hydro://sensitivity/results/{result_id}`**: Retrieve specific sensitivity analysis result by ID.

### 5. Result Persistence

* **Automatic Storage**: Calibration and sensitivity analysis results are automatically saved with unique IDs.
* **Flexible Backends**: Support both file-based (development) and Redis (production) storage.
* **TTL Management**: Automatic cleanup of old results (default: 7 days, configurable).
* **Result Retrieval**: Access stored results via resources or tool responses.



## 🛠️ Prerequisites

* **Julia** (v1.11+)
* **Node.js & npm** (Required for the MCP Inspector/Debugging)
* **Redis** (Optional, only if using Redis storage backend)

## 📦 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/chooron/HydroModelMCP
cd HydroModelMCP
```

2. **Instantiate the Julia environment**:
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## ⚙️ Configuration

HydroModelMCP supports flexible configuration via environment variables:

### Storage Backend Configuration

**File Storage (Default)**:
```bash
# No configuration needed - uses file storage by default
# Storage path: ~/.hydro_mcp/storage
# TTL: 7 days (604800 seconds)
```

**Redis Storage**:
```bash
export STORAGE_BACKEND=redis
export REDIS_HOST=127.0.0.1
export REDIS_PORT=6379
export HYDRO_STORAGE_TTL=604800  # 7 days in seconds
```

**Custom File Storage Path**:
```bash
export STORAGE_BACKEND=file
export HYDRO_STORAGE_PATH=/path/to/storage
export HYDRO_STORAGE_TTL=604800
```

**Disable TTL (Permanent Storage)**:
```bash
export HYDRO_STORAGE_TTL=0  # Results never expire
```

### Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `STORAGE_BACKEND` | `file` | Storage backend type: `file` or `redis` |
| `HYDRO_STORAGE_PATH` | `~/.hydro_mcp/storage` | Base path for file storage |
| `HYDRO_STORAGE_TTL` | `604800` (7 days) | TTL in seconds, `0` = no expiration |
| `REDIS_HOST` | `127.0.0.1` | Redis server host |
| `REDIS_PORT` | `6379` | Redis server port |


## ▶️ Usage

### Option 1: Standard MCP Server (stdio)

Start the server with standard input/output transport:

```bash
julia --project=. start.jl
```

This mode is suitable for integration with MCP clients like Claude Desktop.

### Option 2: HTTP Server

Start the server with HTTP transport on port 3000:

```julia
julia --project=. -e 'using HydroModelMCP; HydroModelMCP.run_http_server()'
```

The server will be accessible at `http://127.0.0.1:3000` with Server-Sent Events (SSE) support.

### Option 3: Debugging with MCP Inspector (Recommended for Development)

You can use the official MCP Inspector to test tools interactively in your browser without connecting to a specific LLM client.

Run the following command in your terminal:

```bash
npx @modelcontextprotocol/inspector julia --project=. start.jl
```

* This will open a web interface (usually at `http://localhost:6274`).
* You can select tools like `run_simulation` or `list_models` and execute them to verify the Julia backend is working correctly.
* You can also browse available resources like `hydro://models/catalog`.

## 📖 Usage Examples

### Example 1: List Available Models

**Using Tool**:
```json
{
  "tool": "list_models"
}
```

**Using Resource** (more efficient):
```
GET hydro://models/catalog
```

Returns:
```json
{
  "models": [
    {
      "name": "exphydro",
      "info_uri": "hydro://models/exphydro/info",
      "params_uri": "hydro://models/exphydro/parameters"
    },
    ...
  ],
  "count": 15
}
```

### Example 2: Get Model Information

**Using Resource**:
```
GET hydro://models/hbv/info
```

Returns detailed model structure, inputs, outputs, and description.

### Example 3: Run Simulation

```json
{
  "tool": "run_simulation",
  "params": {
    "model_name": "exphydro",
    "forcing": {
      "source_type": "csv",
      "path": "data/test_forcing.csv"
    },
    "params": [0.5, 2.0, 0.1, 0.5, 0.5],
    "solver": "ODE",
    "interpolator": "LINEAR"
  }
}
```

### Example 4: Calibrate Model

```json
{
  "tool": "calibrate_model",
  "params": {
    "model_name": "hbv",
    "forcing": {
      "source_type": "csv",
      "path": "data/forcing.csv"
    },
    "observation": {
      "source_type": "csv",
      "path": "data/observation.csv"
    },
    "obs_column": "discharge",
    "objective": "KGE",
    "algorithm": "BBO",
    "maxiters": 1000
  }
}
```

Returns:
```json
{
  "result_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "timestamp": "2024-01-15T10:30:00",
  "optimal_params": [1.2, 3.4, 0.8, ...],
  "objective_value": 0.85,
  "convergence": "success"
}
```

The result is automatically saved and can be retrieved later:
```
GET hydro://calibration/results/a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

### Example 5: One-Click Calibration Setup

```json
{
  "tool": "init_calibration_setup",
  "params": {
    "model_name": "hbv",
    "forcing": {"source_type": "csv", "path": "data/forcing.csv"},
    "observation": {"source_type": "csv", "path": "data/obs.csv"},
    "obs_column": "discharge",
    "goal": "general_fit",
    "budget": "medium"
  }
}
```

Returns a complete recommended configuration including:
- Recommended objective function
- Suggested algorithm
- Optimal iteration count
- Data split strategy
- Sensitive parameters to focus on

### Example 6: Browse Calibration Results

**List all results**:
```
GET hydro://calibration/results
```

Returns:
```json
{
  "results": ["uuid1", "uuid2", "uuid3"],
  "count": 3
}
```

**Get specific result**:
```
GET hydro://calibration/results/uuid1
```

#### MCP Inspector — Interface

![MCP Inspector Interface](asserts/figure%201.png)

### MCP Inspector — Listing models

The Inspector calling the `list_models` tool to retrieve all available model names.

![List models example](asserts/figure%202.png)

### MCP Inspector — Running an `exphydro` simulation

The Inspector calling the `run_simulation` tool for the `exphydro` model using the input file `data/test_input.json`.

- Input file: `data/test_input.json`

![Run exphydro simulation (using data/test_input.json)](asserts/figure%203.png)

## 📂 Project Structure

```
HydroModelMCP/
├── src/
│   ├── core/                    # Core business logic
│   │   ├── dataloader.jl        # Multi-source data loading (CSV/JSON/Redis)
│   │   ├── metrics.jl           # Performance metrics calculation
│   │   ├── datasplitter.jl      # Data splitting strategies
│   │   ├── sampling.jl          # Parameter sampling methods
│   │   ├── simulation.jl        # Model simulation engine
│   │   ├── discovery.jl         # Model introspection
│   │   ├── sensitivity.jl       # Sensitivity analysis
│   │   ├── calibration.jl       # Calibration algorithms
│   │   └── storage.jl           # Result persistence (NEW)
│   ├── resources/               # MCP resource definitions (NEW)
│   │   ├── models.jl            # Model catalog resources
│   │   ├── calibration.jl       # Calibration result resources
│   │   ├── parameters.jl        # Parameter/objective guides
│   │   └── templates.jl         # URI templates and parsing
│   ├── tools/                   # MCP tool definitions
│   │   ├── discovery.jl         # Model discovery tools (5 tools)
│   │   ├── simulation.jl        # Simulation tool (1 tool)
│   │   └── calibration.jl       # Calibration tools (9 tools)
│   ├── prompts/                 # MCP prompt templates
│   │   └── experts.jl           # Expert guidance prompts
│   └── HydroModelMCP.jl         # Main module
├── test/                        # Unit tests
├── data/                        # Example data files
├── start.jl                     # Server entry point
├── Project.toml                 # Julia dependencies
└── README.md                    # This file
```

## 🗄️ Storage Structure

### File Backend (Default)

```
~/.hydro_mcp/storage/
├── calibration/
│   ├── {uuid}.json              # Calibration result data
│   ├── {uuid}.meta.json         # Metadata (created_at timestamp)
│   └── ...
├── sensitivity/
│   ├── {uuid}.json
│   ├── {uuid}.meta.json
│   └── ...
├── metrics/
└── splits/
```

### Redis Backend

```
hydro:calibration:{uuid}         # Auto-expires after TTL
hydro:sensitivity:{uuid}
hydro:metrics:{uuid}
hydro:splits:{uuid}
```

## 🔧 Advanced Features

### 1. Result Persistence

All calibration and sensitivity analysis results are automatically saved with:
- Unique UUID identifier
- Timestamp
- Complete parameter and metric information
- Automatic cleanup after TTL expiration (default: 7 days)

### 2. Resource-Based Data Access

Instead of calling tools repeatedly, clients can:
- Browse model catalog via `hydro://models/catalog`
- Query model details via `hydro://models/{model_name}/info`
- Access stored results via `hydro://calibration/results/{result_id}`
- Get algorithm recommendations via `hydro://guides/algorithms`

This reduces API calls and improves performance.

### 3. URI Template Support

The server advertises available resource templates, allowing clients to discover:
- What parameterized resources are available
- What parameters each resource accepts
- Expected MIME types and descriptions

### 4. Flexible Storage Backends

Switch between file-based (development) and Redis (production) storage without code changes:

```bash
# Development
julia --project=. start.jl

# Production with Redis
STORAGE_BACKEND=redis REDIS_HOST=prod-redis.example.com julia --project=. start.jl
```

## 🧪 Testing

Run unit tests:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Test storage layer:

```julia
using HydroModelMCP

# Create test backend
backend = HydroModelMCP.Storage.FileBackend("/tmp/hydro_test"; ttl=0)

# Save test data
test_data = Dict("model" => "hbv", "objective" => 0.85)
HydroModelMCP.Storage.save_result(backend, "calibration", "test-001", test_data)

# Load it back
loaded = HydroModelMCP.Storage.load_result(backend, "calibration", "test-001")
println("Loaded: ", loaded)

# List results
results = HydroModelMCP.Storage.list_results(backend, "calibration")
println("Results: ", results)
```

## 📊 Performance Considerations

### Resource vs Tool Calls

**Using Tools** (multiple API calls):
```
1. Call list_models → Get model names
2. Call get_model_info for each model → Get details
3. Call get_model_parameters for each model → Get bounds
```

**Using Resources** (single API call):
```
1. GET hydro://models/catalog → Get everything at once
```

Resources are cached and can be queried multiple times without re-computation.

### Storage Performance

**File Backend**:
- Fast for small-to-medium datasets
- Automatic cleanup on `list_results()` call
- No external dependencies

**Redis Backend**:
- High performance for concurrent access
- Automatic TTL expiration
- Requires Redis server

## 🤝 Integration with Claude Desktop

Add to your Claude Desktop MCP configuration (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "hydro-model": {
      "command": "julia",
      "args": ["--project=/path/to/HydroModelMCP", "/path/to/HydroModelMCP/start.jl"],
      "env": {
        "STORAGE_BACKEND": "file",
        "HYDRO_STORAGE_TTL": "604800"
      }
    }
  }
}
```

## 📚 Documentation

- [MCP Implementation Guide](MCP_IMPLEMENTATION_GUIDE.md) - Detailed implementation patterns
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md) - Feature summary (Chinese)
- [Knowledge Base](KNOWLEDEGE.md) - Domain knowledge and best practices
- [Development Guide](DEV.md) - Development workflow

## 🆕 What's New in v0.2.0

- ✅ **9 MCP Resources** for efficient data browsing
- ✅ **Result Persistence** with file/Redis backends
- ✅ **URI Template Support** for parameterized resources
- ✅ **TTL Management** for automatic cleanup
- ✅ **Standardized Error Handling** across all 15 tools
- ✅ **9 Calibration Tools** for complete workflow
- ✅ **Environment Variable Configuration** for flexible deployment

## 🐛 Troubleshooting

### Module fails to load

```bash
# Reinstall dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.resolve()'
```

### Redis connection errors

```bash
# Check Redis is running
redis-cli ping

# Use file backend instead
unset STORAGE_BACKEND
julia --project=. start.jl
```

### Storage permission errors

```bash
# Check storage directory permissions
mkdir -p ~/.hydro_mcp/storage
chmod 755 ~/.hydro_mcp/storage

# Or use custom path
export HYDRO_STORAGE_PATH=/tmp/hydro_storage
```

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Built on [HydroModels.jl](https://github.com/chooron/HydroModels.jl)
- Uses [ModelContextProtocol.jl](https://github.com/JuliaSMLM/ModelContextProtocol.jl)
- Inspired by the [Model Context Protocol](https://modelcontextprotocol.io/) specification