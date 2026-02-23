"""
RavenPy Calibration Tools

Provides functions for model calibration using SPOTPY optimization algorithms
(DDS, SCE-UA, PSO, etc.) with RavenPy emulators and HydroErr metrics.
"""

from typing import Any, Optional
import numpy as np
from pathlib import Path

import spotpy
from spotpy.parameter import Uniform
import HydroErr as he
from ravenpy import Emulator
from ravenpy.config import emulators, commands

from hydroagent.schemas import (
    RunCalibrationInput,
    RunCalibrationOutput,
    RunMultiObjectiveCalibrationInput,
    RunMultiObjectiveCalibrationOutput,
    RunSensitivityAnalysisInput,
    RunSensitivityAnalysisOutput,
    GetCalibrationDiagnosticsInput,
    GetCalibrationDiagnosticsOutput,
    ToolResponse,
)
from hydroagent.schemas.tool_outputs import (
    SensitivityIndices,
    ParetoSolution,
)
from hydroagent.schemas.common import Statistics


# ---------------------------------------------------------------------------
# Objective function helper
# ---------------------------------------------------------------------------
_MAXIMIZE_METRICS = {"nse", "kge_2009", "kge_2012", "r_squared", "pearson_r", "d"}
_MINIMIZE_METRICS = {"rmse", "mae", "mse", "mape", "me", "rmsle"}


def _compute_objective(sim, obs, metric: str) -> float:
    """Compute objective using HydroErr. Returns scalar."""
    func = getattr(he, metric.lower(), None)
    if func is None:
        raise ValueError(f"Unknown metric '{metric}' in HydroErr")
    return float(func(np.array(sim), np.array(obs)))


def _is_maximize(metric: str) -> bool:
    m = metric.lower()
    if m in _MAXIMIZE_METRICS:
        return True
    if m in _MINIMIZE_METRICS:
        return False
    return False  # default: minimize


# ---------------------------------------------------------------------------
# SPOTPY setup class that wraps a RavenPy emulator
# ---------------------------------------------------------------------------
class _RavenSpotpySetup:
    """Generic SPOTPY setup that drives any RavenPy emulator."""

    def __init__(
        self,
        model_type: str,
        forcing_file: str,
        start_date: str,
        end_date: str,
        observed: list[float],
        param_names: list[str],
        lower_bounds: list[float],
        upper_bounds: list[float],
        objective: str = "nse",
    ):
        self.model_type = model_type
        self.forcing_file = forcing_file
        self.start_date = start_date
        self.end_date = end_date
        self.observed = observed
        self.objective = objective
        self.maximize = _is_maximize(objective)

        self.params = []
        for name, lo, hi in zip(param_names, lower_bounds, upper_bounds):
            self.params.append(Uniform(name, low=lo, high=hi))

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self, vector):
        """Run RavenPy emulator with given parameter vector."""
        params_list = list(vector)
        emulator_cls = getattr(emulators, self.model_type)
        gauge = commands.Gauge.from_nc(self.forcing_file)
        config = emulator_cls(
            params=params_list,
            StartDate=self.start_date,
            EndDate=self.end_date,
            Gauge=[gauge],
        )
        e = Emulator(config=config)
        output = e.run()
        q_sim = output.hydrograph.q_sim.values
        q_sim = np.where(np.isnan(q_sim), 0.0, q_sim)
        return q_sim.tolist()

    def evaluation(self):
        return self.observed

    def objectivefunction(self, simulation, evaluation):
        sim = np.array(simulation)
        obs = np.array(evaluation)
        return _compute_objective(sim, obs, self.objective)


# ---------------------------------------------------------------------------
# Algorithm mapping
# ---------------------------------------------------------------------------
_ALGORITHM_MAP = {
    "DDS": spotpy.algorithms.dds,
    "SCEUA": spotpy.algorithms.sceua,
    "SCE-UA": spotpy.algorithms.sceua,
    "PSO": spotpy.algorithms.mc,
    "MC": spotpy.algorithms.mc,
    "LHS": spotpy.algorithms.lhs,
    "ROPE": spotpy.algorithms.rope,
    "DREAM": spotpy.algorithms.dream,
    "SA": spotpy.algorithms.sa,
    "MCMC": spotpy.algorithms.mcmc,
    "ABC": spotpy.algorithms.abc,
    "FSCABC": spotpy.algorithms.fscabc,
    "DEMCZ": spotpy.algorithms.demcz,
    "FAST": spotpy.algorithms.fast,
    "NSGAII": spotpy.algorithms.NSGAII,
    "NSGA-II": spotpy.algorithms.NSGAII,
    "PADDS": spotpy.algorithms.padds,
}


def _load_observed(path: str) -> list[float]:
    """Load observed streamflow from .npy, .csv, or .nc file."""
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p).tolist()
    elif p.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(p)
        return df.iloc[:, -1].values.tolist()
    elif p.suffix == ".nc":
        import xarray as xr
        ds = xr.open_dataset(p)
        for var in ["q_obs", "qobs", "discharge", "flow"]:
            if var in ds:
                return ds[var].values.tolist()
        first_var = list(ds.data_vars)[0]
        return ds[first_var].values.tolist()
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")


def _get_algorithm(name: str):
    key = name.upper().replace(" ", "")
    algo = _ALGORITHM_MAP.get(key)
    if algo is None:
        raise ValueError(
            f"Unknown algorithm '{name}'. Available: {list(_ALGORITHM_MAP.keys())}"
        )
    return algo


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------
def run_calibration(input: RunCalibrationInput) -> dict[str, Any]:
    """
    Run single-objective calibration with SPOTPY.

    Args:
        input: RunCalibrationInput with calibration configuration

    Returns:
        Dictionary with status, message, and data fields
    """
    try:
        obs = _load_observed(input.observed_file)

        setup = _RavenSpotpySetup(
            model_type=input.model_type,
            forcing_file=input.forcing_file,
            start_date=input.start_date,
            end_date=input.end_date,
            observed=obs,
            param_names=input.param_names,
            lower_bounds=input.lower_bounds,
            upper_bounds=input.upper_bounds,
            objective=input.objective,
        )

        algo_cls = _get_algorithm(input.algorithm.value)
        sampler = algo_cls(
            setup, dbname="calibration", dbformat="ram", random_state=None,
        )
        sampler.sample(input.budget)

        results = sampler.getdata()
        maximize = _is_maximize(input.objective)
        best = spotpy.analyser.get_best_parameterset(results, maximize=maximize)
        if maximize:
            _, val = spotpy.analyser.get_maxlikeindex(results)
        else:
            _, val = spotpy.analyser.get_minlikeindex(results)

        best_params = [float(best[0][i]) for i in range(len(input.param_names))]
        likes = list(results["like1"])

        # Get parameter history
        params_array = spotpy.analyser.get_parameters(results)
        param_history = [list(p) for p in params_array]

        output = RunCalibrationOutput(
            algorithm=input.algorithm.value,
            objective=input.objective,
            budget_used=len(likes),
            best_parameters=best_params,
            best_objective=float(val),
            objective_history=likes,
            parameter_history=param_history,
        )
        response = ToolResponse[RunCalibrationOutput].success(
            data=output,
            message=f"Calibration done: {input.objective} = {float(val):.4f}"
        )
        return response.model_dump()
    except Exception as ex:
        response = ToolResponse[RunCalibrationOutput].error(
            message=str(ex),
            error_type=type(ex).__name__
        )
        return response.model_dump()

# PLACEHOLDER_CAL_MORE


def run_multi_objective_calibration(input: RunMultiObjectiveCalibrationInput) -> dict[str, Any]:
    """
    Run multi-objective calibration (PADDS / NSGA-II).

    Args:
        input: RunMultiObjectiveCalibrationInput with calibration configuration

    Returns:
        Dictionary with status, message, and data fields
    """
    try:
        obs = _load_observed(input.observed_file)

        class _MultiObjSetup(_RavenSpotpySetup):
            def objectivefunction(self, simulation, evaluation):
                sim = np.array(simulation)
                ob = np.array(evaluation)
                return [_compute_objective(sim, ob, m) for m in input.objectives]

        setup = _MultiObjSetup(
            model_type=input.model_type,
            forcing_file=input.forcing_file,
            start_date=input.start_date,
            end_date=input.end_date,
            observed=obs,
            param_names=input.param_names,
            lower_bounds=input.lower_bounds,
            upper_bounds=input.upper_bounds,
            objective=input.objectives[0],
        )

        algo_cls = _get_algorithm(input.algorithm.value)
        sampler = algo_cls(setup, dbname="mo_cal", dbformat="ram")

        if input.algorithm.value.upper() in ("NSGAII", "NSGA-II"):
            sampler.sample(generations=input.budget, n_obj=len(input.objectives))
        else:
            sampler.sample(input.budget)

        results = sampler.getdata()
        params = spotpy.analyser.get_parameters(results)

        solutions = []
        for i in range(min(len(params), 50)):
            param_list = [float(params[i][j]) for j in range(len(input.param_names))]
            obj_dict = {}
            for oi, obj_name in enumerate(input.objectives):
                col = f"like{oi + 1}"
                if col in results.dtype.names:
                    obj_dict[obj_name] = float(results[col][i])

            solutions.append(ParetoSolution(
                parameters=param_list,
                objectives=obj_dict
            ))

        output = RunMultiObjectiveCalibrationOutput(
            algorithm=input.algorithm.value,
            objectives=input.objectives,
            budget_used=len(params),
            pareto_front=solutions,
            n_pareto_solutions=len(solutions),
        )
        response = ToolResponse[RunMultiObjectiveCalibrationOutput].success(
            data=output,
            message="Multi-objective calibration done"
        )
        return response.model_dump()
    except Exception as ex:
        response = ToolResponse[RunMultiObjectiveCalibrationOutput].error(
            message=str(ex),
            error_type=type(ex).__name__
        )
        return response.model_dump()

# PLACEHOLDER_CAL_FINAL


def analyze_parameter_sensitivity(input: RunSensitivityAnalysisInput) -> dict[str, Any]:
    """
    Global sensitivity analysis via SPOTPY FAST/eFAST.

    Args:
        input: RunSensitivityAnalysisInput with sensitivity analysis configuration

    Returns:
        Dictionary with status, message, and data fields
    """
    try:
        obs = _load_observed(input.observed_file)
        setup = _RavenSpotpySetup(
            model_type=input.model_type,
            forcing_file=input.forcing_file,
            start_date=input.start_date,
            end_date=input.end_date,
            observed=obs,
            param_names=input.param_names,
            lower_bounds=input.lower_bounds,
            upper_bounds=input.upper_bounds,
            objective="nse",  # Default objective for sensitivity analysis
        )

        if input.method.lower() == "efast":
            sampler = spotpy.algorithms.efast(
                setup, dbname="sa", dbformat="ram"
            )
        else:
            sampler = spotpy.algorithms.fast(
                setup, dbname="sa", dbformat="ram"
            )
        sampler.sample(input.n_samples)

        results = sampler.getdata()
        SI = spotpy.analyser.get_sensitivity_of_fast(results)

        indices = []
        for i, name in enumerate(input.param_names):
            if i < len(SI):
                indices.append(SensitivityIndices(
                    parameter_name=name,
                    first_order=float(SI[i]),
                    total_order=None
                ))
            else:
                indices.append(SensitivityIndices(
                    parameter_name=name,
                    first_order=0.0,
                    total_order=None
                ))

        # Rank parameters by sensitivity
        ranked = sorted(
            indices,
            key=lambda x: x.first_order if x.first_order is not None else 0,
            reverse=True,
        )
        most_sensitive = [idx.parameter_name for idx in ranked]

        output = RunSensitivityAnalysisOutput(
            method=input.method,
            n_samples=input.n_samples,
            sensitivity_indices=indices,
            most_sensitive=most_sensitive,
        )
        response = ToolResponse[RunSensitivityAnalysisOutput].success(
            data=output,
            message="Sensitivity analysis done"
        )
        return response.model_dump()
    except Exception as ex:
        response = ToolResponse[RunSensitivityAnalysisOutput].error(
            message=str(ex),
            error_type=type(ex).__name__
        )
        return response.model_dump()


def get_calibration_diagnostics(input: GetCalibrationDiagnosticsInput) -> dict[str, Any]:
    """
    Compute diagnostics from SPOTPY calibration results.

    Args:
        input: GetCalibrationDiagnosticsInput with diagnostics configuration

    Returns:
        Dictionary with status, message, and data fields
    """
    try:
        results = spotpy.analyser.load_csv_results(input.results_path)
        maximize = _is_maximize(input.objective_name)

        likes = results[input.objective_name]
        params = spotpy.analyser.get_parameters(results)

        # Get best results
        n_best = min(input.n_best, len(likes))
        if maximize:
            best_idx = np.argsort(likes)[-n_best:][::-1]
        else:
            best_idx = np.argsort(likes)[:n_best]

        best_likes = likes[best_idx]
        best_params = params[best_idx]

        # Best overall
        if maximize:
            _, best_val = spotpy.analyser.get_maxlikeindex(results)
        else:
            _, best_val = spotpy.analyser.get_minlikeindex(results)

        best_param_set = best_params[0].tolist()

        # Objective statistics
        obj_stats = Statistics(
            mean=float(np.mean(best_likes)),
            std=float(np.std(best_likes)),
            min=float(np.min(best_likes)),
            max=float(np.max(best_likes)),
            count=len(best_likes)
        )

        # Parameter ranges in best results
        param_ranges = {}
        for i in range(best_params.shape[1]):
            param_col = best_params[:, i]
            param_ranges[f"param_{i}"] = {
                "min": float(np.min(param_col)),
                "max": float(np.max(param_col)),
                "mean": float(np.mean(param_col)),
                "std": float(np.std(param_col))
            }

        output = GetCalibrationDiagnosticsOutput(
            n_results=len(likes),
            n_best=n_best,
            best_objective=float(best_val),
            best_parameters=best_param_set,
            objective_statistics=obj_stats,
            parameter_ranges=param_ranges,
        )
        response = ToolResponse[GetCalibrationDiagnosticsOutput].success(
            data=output,
            message="Diagnostics computed"
        )
        return response.model_dump()
    except Exception as ex:
        response = ToolResponse[GetCalibrationDiagnosticsOutput].error(
            message=str(ex),
            error_type=type(ex).__name__
        )
        return response.model_dump()
