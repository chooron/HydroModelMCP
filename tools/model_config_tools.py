"""
RavenPy Model Configuration Tools

Provides functions for creating and managing RavenPy hydrological model
configurations. Supports GR4JCN, HMETS, MOHYSE, HBVEC, etc.
"""

from typing import Any
from pathlib import Path

from ravenpy.config import emulators, commands
from ravenpy import Emulator

from hydroagent.schemas import (
    CreateModelConfigInput,
    CreateModelConfigOutput,
    GetModelParametersInput,
    GetModelParametersOutput,
    ValidateModelParametersInput,
    ValidateModelParametersOutput,
    ToolResponse,
)


# Parameter reference data for supported models
PARAM_INFO = {
    "GR4JCN": {
        "params": [0.529, -3.396, 407.29, 1.072, 16.9, 0.947],
        "bounds": [
            [0.01, 2.5], [-15.0, 10.0], [10.0, 700.0],
            [0.0, 7.0], [0.0, 100.0], [0.0, 1.0],
        ],
        "names": [
            "GAMMA_SHAPE", "GAMMA_SCALE",
            "GAMMA_SHAPE2", "GAMMA_SCALE2",
            "MIN_MELT_FACTOR", "MAX_MELT_FACTOR",
            "DD_MELT_TEMP", "DD_AGGRADATION",
            "SNOW_SWI_MIN", "SNOW_SWI_MAX",
            "SWI_REDUCT_COEFF", "DD_REFREEZE_TEMP",
            "REFREEZE_FACTOR", "REFREEZE_EXP",
            "PET_CORRECTION", "HMETS_RUNOFF_COEFF",
            "PERC_COEFF", "BASEFLOW_COEFF_1",
            "BASEFLOW_COEFF_2", "TOPSOIL", "PHREATIC",
        ],
    },
    "MOHYSE": {
        "params": [
            1.0, 0.0464, 4.2952, 2.658, 0.4038,
            0.0621, 0.0273, 0.0453, 0.9039, 5.6167,
        ],
        "bounds": [
            [0.01, 20.0], [0.01, 1.0], [0.01, 20.0],
            [0.0, 5.0], [0.0, 0.99], [0.0, 0.99],
            [0.0, 0.1], [0.0, 0.1], [0.01, 1.5],
            [0.0, 15.0],
        ],
        "names": [f"par_x{i:02d}" for i in range(1, 11)],
    },
    "HBVEC": {
        "params": [
            0.05985, 4.07223, 2.00157, 0.03474,
            0.09985, 0.50605, 3.43849, 38.32455,
            0.46066, 0.06304, 2.27778, 4.87369,
            0.57188, 0.04506, 0.87761, 18.94145,
            2.03694, 0.44528, 0.67718, 1.14161,
            1.02428,
        ],
        "bounds": [
            [0.0, 0.1], [0.0, 8.0], [0.0, 7.0],
            [0.0, 0.1], [0.05, 0.15], [0.0, 1.0],
            [1.0, 6.0], [0.0, 100.0], [0.3, 1.0],
            [0.0, 0.3], [0.0, 7.0], [0.0, 7.0],
            [0.0, 1.0], [0.0, 0.3], [0.0, 1.0],
            [0.0, 30.0], [0.0, 4.0], [0.0, 1.0],
            [0.0, 1.0], [0.0, 3.0], [0.0, 2.0],
        ],
        "names": [f"par_x{i:02d}" for i in range(1, 22)],
    },
}


def create_model_config(input: CreateModelConfigInput) -> dict[str, Any]:
    """
    Create a RavenPy emulator configuration and write Raven input files.

    Args:
        input: CreateModelConfigInput with model configuration parameters

    Returns:
        Dictionary with status, message, and data fields
    """
    try:
        emulator_cls = getattr(emulators, input.model_type)
        gauge = commands.Gauge.from_nc(input.forcing_file)
        config = emulator_cls(
            params=input.params,
            StartDate=input.start_date,
            EndDate=input.end_date,
            Gauge=[gauge],
        )

        if input.workdir:
            wp = Path(input.workdir)
            wp.mkdir(parents=True, exist_ok=True)
            e = Emulator(config=config, workdir=str(wp))
        else:
            e = Emulator(config=config)

        output = CreateModelConfigOutput(
            model_type=input.model_type,
            workdir=str(e.workdir),
            num_parameters=len(input.params),
            parameters=input.params,
        )
        response = ToolResponse[CreateModelConfigOutput].success(
            data=output,
            message=f"Created {input.model_type} config"
        )
        return response.model_dump()
    except Exception as ex:
        response = ToolResponse[CreateModelConfigOutput].error(
            message=str(ex),
            error_type=type(ex).__name__
        )
        return response.model_dump()


def get_model_parameters(input: GetModelParametersInput) -> dict[str, Any]:
    """
    Get default parameters and bounds for a model.

    Args:
        input: GetModelParametersInput with model type and options

    Returns:
        Dictionary with status, message, and data fields
    """
    key = input.model_type
    if key not in PARAM_INFO:
        response = ToolResponse[GetModelParametersOutput].error(
            message=f"Unknown model: {input.model_type}. Supported: {list(PARAM_INFO)}",
            error_type="ValueError"
        )
        return response.model_dump()

    info = PARAM_INFO[key]
    params = {}
    for i, name in enumerate(info["names"]):
        d = {"default": info["params"][i], "index": i}
        if input.include_bounds and i < len(info["bounds"]):
            d["bounds"] = info["bounds"][i]
        params[name] = d

    output = GetModelParametersOutput(
        model_type=input.model_type,
        parameters=params,
        num_parameters=len(params),
        default_values=info["params"],
    )
    response = ToolResponse[GetModelParametersOutput].success(
        data=output,
        message=f"Retrieved parameters for {input.model_type}"
    )
    return response.model_dump()


def validate_model_parameters(input: ValidateModelParametersInput) -> dict[str, Any]:
    """
    Validate parameters against expected bounds.

    Args:
        input: ValidateModelParametersInput with model type and parameters

    Returns:
        Dictionary with status, message, and data fields
    """
    key = input.model_type
    if key not in PARAM_INFO:
        response = ToolResponse[ValidateModelParametersOutput].error(
            message=f"Unknown model: {input.model_type}",
            error_type="ValueError"
        )
        return response.model_dump()

    info = PARAM_INFO[key]
    errors = []
    warnings = []

    expected_n = len(info["params"])
    if len(input.parameters) != expected_n:
        errors.append(
            f"Expected {expected_n} params, got {len(input.parameters)}"
        )

    for i, val in enumerate(input.parameters):
        if i < len(info["bounds"]):
            lo, hi = info["bounds"][i]
            name = info["names"][i]
            if val < lo or val > hi:
                errors.append(f"{name}={val} outside [{lo}, {hi}]")

    output = ValidateModelParametersOutput(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )
    response = ToolResponse[ValidateModelParametersOutput].success(
        data=output,
        message="Parameter validation completed"
    )
    return response.model_dump()
