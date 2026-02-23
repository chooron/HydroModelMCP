"""
RavenPy Tools Suite

Pure tool functions for hydrological modeling using RavenPy, SPOTPY, and HydroErr.
These functions contain the core logic and are registered as MCP tools by the
server in hydroagent.mcp.server.
"""

from .model_config_tools import (
    create_model_config,
    get_model_parameters,
    validate_model_parameters,
    PARAM_INFO,
)
from .data_extraction_tools import (
    extract_climate_data_from_nc,
    process_precipitation_data,
    process_temperature_data,
    extract_basin_properties_from_nc,
    compute_potential_evapotranspiration,
    quality_check_climate_data,
    create_raven_forcing_nc,
)
from .model_execution_tools import (
    run_simulation,
    run_ensemble,
    compute_objective_function,
    run_validation,
)
from .calibration_tools import (
    run_calibration,
    run_multi_objective_calibration,
    analyze_parameter_sensitivity,
    get_calibration_diagnostics,
)
from .analysis_tools import (
    compute_performance_metrics,
    analyze_flow_components,
    compute_flow_duration_curve,
    analyze_hydrograph_signatures,
    detect_peak_flows,
    analyze_recession_curves,
    compute_water_balance,
    compare_simulations,
    generate_diagnostic_plots,
    export_results_summary,
)

__version__ = "0.1.0"

__all__ = [
    # Model Config
    "create_model_config",
    "get_model_parameters",
    "validate_model_parameters",
    "PARAM_INFO",
    # Data Extraction
    "extract_climate_data_from_nc",
    "process_precipitation_data",
    "process_temperature_data",
    "extract_basin_properties_from_nc",
    "compute_potential_evapotranspiration",
    "quality_check_climate_data",
    "create_raven_forcing_nc",
    # Model Execution
    "run_simulation",
    "run_ensemble",
    "compute_objective_function",
    "run_validation",
    # Calibration
    "run_calibration",
    "run_multi_objective_calibration",
    "analyze_parameter_sensitivity",
    "get_calibration_diagnostics",
    # Analysis
    "compute_performance_metrics",
    "analyze_flow_components",
    "compute_flow_duration_curve",
    "analyze_hydrograph_signatures",
    "detect_peak_flows",
    "analyze_recession_curves",
    "compute_water_balance",
    "compare_simulations",
    "generate_diagnostic_plots",
    "export_results_summary",
]
