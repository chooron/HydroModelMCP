"""
RavenPy Model Execution Tools

Provides functions for executing RavenPy hydrological models, including
simulation runs, ensemble runs, and result retrieval.
"""

from typing import Any
from pathlib import Path
import numpy as np

from ravenpy import Emulator
from ravenpy.config import emulators, commands

import HydroErr as he

from hydroagent.schemas import (
    RunSimulationInput,
    RunSimulationOutput,
    RunEnsembleInput,
    RunEnsembleOutput,
    ComputeObjectiveFunctionInput,
    ComputeObjectiveFunctionOutput,
    RunValidationInput,
    RunValidationOutput,
    ToolResponse,
)
from hydroagent.schemas.tool_outputs import (
    SimulationResults,
    EnsembleMember,
    ValidationMetrics,
)


def run_simulation(input: RunSimulationInput) -> dict[str, Any]:
    """
    Run a single RavenPy model simulation.

    Args:
        input: RunSimulationInput with model configuration and parameters

    Returns:
        Dictionary with status, message, and data fields
    """
    try:
        emulator_cls = getattr(emulators, input.model_type.upper())
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

        output = e.run()

        q_sim = (
            output.hydrograph.q_sim.values.tolist()
            if hasattr(output.hydrograph, "q_sim") else []
        )
        q_obs = (
            output.hydrograph.q_obs.values.tolist()
            if hasattr(output.hydrograph, "q_obs") else []
        )

        results = SimulationResults(
            simulated_flow=q_sim,
            observed_flow=q_obs,
            n_timesteps=len(q_sim),
        )

        output_data = RunSimulationOutput(
            model_type=input.model_type,
            period={"start": input.start_date, "end": input.end_date},
            workdir=str(e.workdir),
            results=results,
        )

        response = ToolResponse[RunSimulationOutput].success(
            data=output_data,
            message="Simulation completed"
        )
        return response.model_dump()
    except Exception as ex:
        response = ToolResponse[RunSimulationOutput].error(
            message=str(ex),
            error_type=type(ex).__name__
        )
        return response.model_dump()


def run_ensemble(input: RunEnsembleInput) -> dict[str, Any]:
    """
    Run ensemble simulation with multiple parameter sets.

    Args:
        input: RunEnsembleInput with model configuration and parameter sets

    Returns:
        Dictionary with status, message, and data fields
    """
    try:
        emulator_cls = getattr(emulators, input.model_type.upper())
        gauge = commands.Gauge.from_nc(input.forcing_file)

        ensemble_members = []
        all_flows = []

        for i, p in enumerate(input.parameter_sets):
            try:
                config = emulator_cls(
                    params=p,
                    StartDate=input.start_date,
                    EndDate=input.end_date,
                    Gauge=[gauge],
                )
                mwd = None
                if input.workdir:
                    mwd = str(Path(input.workdir) / f"member_{i:03d}")
                e = Emulator(config=config, workdir=mwd)
                output = e.run()
                q = (
                    output.hydrograph.q_sim.values.tolist()
                    if hasattr(output.hydrograph, "q_sim") else []
                )

                member = EnsembleMember(
                    member_id=i,
                    parameters=p,
                    simulated_flow=q,
                )
                ensemble_members.append(member)
                all_flows.append(q)
            except Exception:
                # Skip failed members
                continue

        # Compute ensemble statistics
        if all_flows:
            flows_array = np.array(all_flows)
            ensemble_mean = np.mean(flows_array, axis=0).tolist()
            ensemble_std = np.std(flows_array, axis=0).tolist()
        else:
            ensemble_mean = []
            ensemble_std = []

        output_data = RunEnsembleOutput(
            model_type=input.model_type,
            n_members=len(ensemble_members),
            period={"start": input.start_date, "end": input.end_date},
            ensemble_members=ensemble_members,
            ensemble_mean=ensemble_mean,
            ensemble_std=ensemble_std,
        )

        response = ToolResponse[RunEnsembleOutput].success(
            data=output_data,
            message=f"Ensemble: {len(ensemble_members)}/{len(input.parameter_sets)} completed"
        )
        return response.model_dump()
    except Exception as ex:
        response = ToolResponse[RunEnsembleOutput].error(
            message=str(ex),
            error_type=type(ex).__name__
        )
        return response.model_dump()



def compute_objective_function(input: ComputeObjectiveFunctionInput) -> dict[str, Any]:
    """
    Compute objective function using HydroErr.

    Args:
        input: ComputeObjectiveFunctionInput with observed and simulated values

    Returns:
        Dictionary with status, message, and data fields
    """
    try:
        obs = np.array(input.observed)
        sim = np.array(input.simulated)

        # Remove NaN values
        mask = ~(np.isnan(obs) | np.isnan(sim))
        obs = obs[mask]
        sim = sim[mask]

        func = getattr(he, input.metric.lower(), None)
        if func is None:
            response = ToolResponse[ComputeObjectiveFunctionOutput].error(
                message=f"Metric '{input.metric}' not found in HydroErr",
                error_type="ValueError"
            )
            return response.model_dump()

        value = float(func(sim, obs))

        # Determine if metric should be maximized (NSE, KGE) or minimized (RMSE, MAE)
        maximize_metrics = ["nse", "kge", "kge_2009", "kge_2012", "r2"]
        is_maximize = any(m in input.metric.lower() for m in maximize_metrics)

        output_data = ComputeObjectiveFunctionOutput(
            metric=input.metric,
            value=value,
            is_maximize=is_maximize,
            n_points=len(obs),
        )

        response = ToolResponse[ComputeObjectiveFunctionOutput].success(
            data=output_data,
            message=f"{input.metric} = {value:.4f}"
        )
        return response.model_dump()
    except Exception as ex:
        response = ToolResponse[ComputeObjectiveFunctionOutput].error(
            message=str(ex),
            error_type=type(ex).__name__
        )
        return response.model_dump()


def run_validation(input: RunValidationInput) -> dict[str, Any]:
    """
    Run model on calibration and validation periods.

    Args:
        input: RunValidationInput with model configuration and periods

    Returns:
        Dictionary with status, message, and data fields
    """
    try:
        emulator_cls = getattr(emulators, input.model_type.upper())
        gauge = commands.Gauge.from_nc(input.forcing_file)

        def _run_period(sd: str, ed: str):
            cfg = emulator_cls(
                params=input.params,
                StartDate=sd,
                EndDate=ed,
                Gauge=[gauge],
            )
            e = Emulator(config=cfg)
            return e.run()

        def _compute_metrics(out):
            m = {}
            if not hasattr(out.hydrograph, "q_obs"):
                return m
            obs = out.hydrograph.q_obs.values
            sim = out.hydrograph.q_sim.values
            mask = ~(np.isnan(obs) | np.isnan(sim))
            obs = obs[mask]
            sim = sim[mask]
            for name in input.metrics:
                func = getattr(he, name.lower(), None)
                if func:
                    m[name] = float(func(sim, obs))
            return m

        cal_out = _run_period(
            input.calibration_period["start"],
            input.calibration_period["end"]
        )
        val_out = _run_period(
            input.validation_period["start"],
            input.validation_period["end"]
        )

        cal_metrics = _compute_metrics(cal_out)
        val_metrics = _compute_metrics(val_out)

        metrics = ValidationMetrics(
            calibration=cal_metrics,
            validation=val_metrics,
        )

        output_data = RunValidationOutput(
            model_type=input.model_type,
            parameters=input.params,
            metrics=metrics,
            calibration_period=input.calibration_period,
            validation_period=input.validation_period,
        )

        response = ToolResponse[RunValidationOutput].success(
            data=output_data,
            message="Validation completed"
        )
        return response.model_dump()
    except Exception as ex:
        response = ToolResponse[RunValidationOutput].error(
            message=str(ex),
            error_type=type(ex).__name__
        )
        return response.model_dump()