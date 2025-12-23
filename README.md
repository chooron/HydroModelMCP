# HydroModelMCP: Hydrological Modeling Agent Interface

HydroModelMCP is a **Model Context Protocol (MCP)** server built in Julia. It acts as a bridge between Large Language Models (like Claude) and high-performance hydrological models (powered by `HydroModels.jl`).

This interface allows AI agents to autonomously discover model structures, query physical parameter bounds, and execute complex hydrological simulations using data from various sources (CSV, JSON, Redis).

## 🚀 Features

The server exposes the following tools to the AI agent:

### 1. Model Discovery & Inspection

* **`list_models_tool`**: Lists all available hydrological models in the library.
* **`find_model_tool`**: Searches for a specific model (supports fuzzy matching/case-insensitivity).
* **`get_model_info_tool`**: Retrieves the general structure, inputs, and description of a model.
* **`get_model_variables_tool`**: Returns detailed metadata about internal variables (physical meaning, units).
* **`get_model_parameters_tool`**: Returns detailed parameter information, including **physical bounds** (min/max) and units. *Crucial for reasoning-based calibration.*

### 2. Simulation Engine

* **`simulation_tool`**: The core execution engine.
* **Dynamic Loading**: Instantiates models on the fly.
* **Multi-Source Data**: Accepts forcing data from **CSV files**, **JSON objects**, or **Redis keys**.
* **Auto-Configuration**: Handles parameter injection (or random generation if missing) and state initialization.
* **Flexible Solvers**: Supports various ODE solvers and interpolation methods via Enums.



## 🛠️ Prerequisites

* **Julia** (v1.11+)
* **Node.js & npm** (Required for the MCP Inspector/Debugging)

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


## ▶️ Usage

### Option 1: Debugging with MCP Inspector (Recommended)

You can use the official MCP Inspector to test tools interactively in your browser without connecting to a specific LLM client.

Run the following command in your terminal:

```bash
npx @modelcontextprotocol/inspector julia --project=. start.jl

```

* This will open a web interface (usually at `http://localhost:6274`).
* You can select tools like `run_simulation` or `list_models` and execute them to verify the Julia backend is working correctly.

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

* **`src/core/`**: Path configuration and core utilities.
* **`src/resources/`**: Core business logic (Simulation engine, Data loaders, Discovery logic).
* **`src/tools/`**: MCP Tool definitions (Schema mappings and Handlers).
* **`start.jl`**: Entry point for the MCP server.
* **`test/`**: Unit tests for data loading and model discovery.