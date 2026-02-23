"""
RavenPy Analysis and Visualization Tools

Provides functions for analyzing model results, computing performance metrics,
generating visualizations, and producing diagnostic plots.
"""

from typing import Any
import json
import numpy as np
from pathlib import Path

import HydroErr as he

from hydroagent.schemas import (
    ComputePerformanceMetricsInput,
    ComputePerformanceMetricsOutput,
    AnalyzeFlowComponentsInput,
    AnalyzeFlowComponentsOutput,
    FlowComponent,
    ComputeFlowDurationCurveInput,
    ComputeFlowDurationCurveOutput,
    AnalyzeHydrographSignaturesInput,
    AnalyzeHydrographSignaturesOutput,
    DetectPeakFlowsInput,
    DetectPeakFlowsOutput,
    PeakFlow,
    AnalyzeRecessionCurveInput,
    AnalyzeRecessionCurveOutput,
    RecessionSegment,
    ComputeWaterBalanceInput,
    ComputeWaterBalanceOutput,
    CompareSimulationsInput,
    CompareSimulationsOutput,
    GenerateDiagnosticPlotsInput,
    GenerateDiagnosticPlotsOutput,
    ExportResultsInput,
    ExportResultsOutput,
    ToolResponse,
)


def compute_performance_metrics(input: ComputePerformanceMetricsInput) -> dict[str, Any]:
    """
    Compute performance metrics using HydroErr.

    Args:
        input: ComputePerformanceMetricsInput with observed, simulated, and metrics

    Returns:
        Dictionary with status, message, and data fields
    """
    try:
        obs = np.array(input.observed)
        sim = np.array(input.simulated)
        results = {}
        for m in input.metrics:
            func = getattr(he, m.lower(), None)
            if func is not None:
                results[m] = float(func(sim, obs))
            else:
                results[m] = None

        output = ComputePerformanceMetricsOutput(
            metrics=results,
            n_points=len(obs),
            interpretation=_interpret_metrics(results),
        )
        response = ToolResponse[ComputePerformanceMetricsOutput].success(
            data=output,
            message="Performance metrics computed"
        )
        return response.model_dump()
    except Exception as e:
        response = ToolResponse[ComputePerformanceMetricsOutput].error(
            message=str(e),
            error_type=type(e).__name__
        )
        return response.model_dump()


def analyze_flow_components(input: AnalyzeFlowComponentsInput) -> dict[str, Any]:
    """
    Analyze flow components from RavenPy output.

    Args:
        input: AnalyzeFlowComponentsInput with workdir

    Returns:
        Dictionary with status, message, and data fields
    """
    try:
        import xarray as xr
        hydro_path = Path(input.workdir) / "output" / "Hydrographs.nc"
        if not hydro_path.exists():
            for f in Path(input.workdir).rglob("*ydrograph*.nc"):
                hydro_path = f
                break

        ds = xr.open_dataset(str(hydro_path))
        components = {}

        total = None
        for var in ds.data_vars:
            vals = ds[var].values
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            components[var] = FlowComponent(
                mean=float(np.mean(vals)),
                std=float(np.std(vals)),
                min=float(np.min(vals)),
                max=float(np.max(vals)),
                total=float(np.sum(vals)),
            )
            if "sim" in var.lower():
                total = np.sum(vals)

        if total and total > 0:
            for var in components:
                vol = components[var].total
                components[var].fraction_of_total = float(vol / total)

        output = AnalyzeFlowComponentsOutput(components=components)
        response = ToolResponse[AnalyzeFlowComponentsOutput].success(
            data=output,
            message="Flow components analyzed"
        )
        return response.model_dump()
    except Exception as e:
        response = ToolResponse[AnalyzeFlowComponentsOutput].error(
            message=str(e),
            error_type=type(e).__name__
        )
        return response.model_dump()


def compute_flow_duration_curve(input: ComputeFlowDurationCurveInput) -> dict[str, Any]:
    """
    Compute flow duration curve.

    Args:
        input: ComputeFlowDurationCurveInput with flow_data and exceedance_probs

    Returns:
        Dictionary with status, message, and data fields
    """
    try:
        flows = np.array(input.flow_data)
        flows = flows[~np.isnan(flows)]
        sorted_flows = np.sort(flows)[::-1]
        n = len(sorted_flows)

        # Use default exceedance probabilities if not provided
        exceedance_probs = input.exceedance_probs or [
            0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99,
        ]

        fdc_dict = {}
        flow_values = []
        for prob in exceedance_probs:
            idx = min(int(prob * n), n - 1)
            fdc_dict[f"Q{int(prob * 100)}"] = float(sorted_flows[idx])
            flow_values.append(float(sorted_flows[idx]))

        # Extract specific quantiles for the output model
        q50 = fdc_dict.get("Q50", float(np.median(flows)))
        q95 = fdc_dict.get("Q95", float(np.percentile(flows, 5)))  # 95% exceedance = 5th percentile
        q05 = fdc_dict.get("Q5", float(np.percentile(flows, 95)))  # 5% exceedance = 95th percentile

        from hydroagent.schemas.tool_outputs import FlowDurationCurve
        fdc = FlowDurationCurve(
            exceedance_probabilities=[p * 100 for p in exceedance_probs],
            flow_values=flow_values,
            q50=q50,
            q95=q95,
            q05=q05,
        )

        output = ComputeFlowDurationCurveOutput(
            fdc=fdc,
            flow_regime=_classify_flow_regime(fdc_dict),
        )
        response = ToolResponse[ComputeFlowDurationCurveOutput].success(
            data=output,
            message="Flow duration curve computed"
        )
        return response.model_dump()
    except Exception as e:
        response = ToolResponse[ComputeFlowDurationCurveOutput].error(
            message=str(e),
            error_type=type(e).__name__
        )
        return response.model_dump()

# PLACEHOLDER_ANALYSIS_MORE


def analyze_hydrograph_signatures(input: AnalyzeHydrographSignaturesInput) -> dict[str, Any]:
    """
    Compute hydrograph signatures.

    Args:
        input: AnalyzeHydrographSignaturesInput with flow_data

    Returns:
        Dictionary with status, message, and data fields
    """
    try:
        flow = np.array(input.flow_data, dtype=float)

        # Compute baseflow index if requested
        baseflow_index = None
        if input.compute_baseflow:
            bf = _baseflow_filter(flow)
            baseflow_index = float(np.sum(bf) / np.sum(flow)) if np.sum(flow) > 0 else 0

        # Compute flashiness index if requested
        flashiness_index = None
        if input.compute_flashiness:
            flashiness_index = _flashiness_index(flow)

        # Compute mean flow and coefficient of variation
        mean_flow = float(np.mean(flow))
        cv_flow = float(np.std(flow) / mean_flow) if mean_flow > 0 else 0

        # Compute Q95/Q5 ratio
        q95 = float(np.percentile(flow, 5))  # 95% exceedance = 5th percentile
        q5 = float(np.percentile(flow, 95))  # 5% exceedance = 95th percentile
        q_ratio = float(q5 / q95) if q95 > 0 else None

        from hydroagent.schemas.tool_outputs import HydrographSignatures
        signatures = HydrographSignatures(
            baseflow_index=baseflow_index,
            flashiness_index=flashiness_index,
            mean_flow=mean_flow,
            cv_flow=cv_flow,
            q_ratio=q_ratio,
        )

        output = AnalyzeHydrographSignaturesOutput(signatures=signatures)
        response = ToolResponse[AnalyzeHydrographSignaturesOutput].success(
            data=output,
            message="Hydrograph signatures computed"
        )
        return response.model_dump()
    except Exception as e:
        response = ToolResponse[AnalyzeHydrographSignaturesOutput].error(
            message=str(e),
            error_type=type(e).__name__
        )
        return response.model_dump()


def detect_peak_flows(input: DetectPeakFlowsInput) -> dict[str, Any]:
    """
    Detect peak flow events.

    Args:
        input: DetectPeakFlowsInput with flow_data, threshold_percentile, and min_separation_days

    Returns:
        Dictionary with status, message, and data fields
    """
    try:
        flows = np.array(input.flow_data)
        threshold = float(np.percentile(flows, input.threshold_percentile))

        peaks = []
        for i in range(1, len(flows) - 1):
            if flows[i] > threshold and flows[i] >= flows[i - 1] and flows[i] >= flows[i + 1]:
                if not peaks or i - peaks[-1].index >= input.min_separation_days:
                    date = input.dates[i] if input.dates and i < len(input.dates) else None
                    peaks.append(PeakFlow(
                        index=i,
                        value=float(flows[i]),
                        date=date,
                    ))

        # Sort by flow value descending
        peaks.sort(key=lambda p: p.value, reverse=True)

        output = DetectPeakFlowsOutput(
            n_peaks=len(peaks),
            peaks=peaks[:20],  # Return top 20 peaks
            threshold_value=threshold,
        )
        response = ToolResponse[DetectPeakFlowsOutput].success(
            data=output,
            message=f"Detected {len(peaks)} peaks"
        )
        return response.model_dump()
    except Exception as e:
        response = ToolResponse[DetectPeakFlowsOutput].error(
            message=str(e),
            error_type=type(e).__name__
        )
        return response.model_dump()

# PLACEHOLDER_ANALYSIS_FINAL


def analyze_recession_curves(input: AnalyzeRecessionCurveInput) -> dict[str, Any]:
    """
    Analyze recession curve characteristics.

    Args:
        input: AnalyzeRecessionCurveInput with flow_data and min_recession_length

    Returns:
        Dictionary with status, message, and data fields
    """
    try:
        flows = np.array(input.flow_data)
        recessions = []
        start = None

        for i in range(1, len(flows)):
            if flows[i] < flows[i - 1]:
                if start is None:
                    start = i - 1
            else:
                if start is not None and (i - start) >= input.min_recession_length:
                    seg = flows[start:i]
                    q0 = float(seg[0])
                    qt = float(seg[-1])
                    k = (qt / q0) ** (1.0 / len(seg)) if q0 > 0 else 0
                    recessions.append(RecessionSegment(
                        start_index=start,
                        end_index=i - 1,
                        recession_coefficient=float(k),
                        duration_days=i - start,
                    ))
                start = None

        ks = [r.recession_coefficient for r in recessions]
        output = AnalyzeRecessionCurveOutput(
            n_segments=len(recessions),
            segments=recessions[:10],  # Return top 10 segments
            mean_recession_coefficient=float(np.mean(ks)) if ks else 0,
        )
        response = ToolResponse[AnalyzeRecessionCurveOutput].success(
            data=output,
            message=f"Analyzed {len(recessions)} recessions"
        )
        return response.model_dump()
    except Exception as e:
        response = ToolResponse[AnalyzeRecessionCurveOutput].error(
            message=str(e),
            error_type=type(e).__name__
        )
        return response.model_dump()


def compute_water_balance(input: ComputeWaterBalanceInput) -> dict[str, Any]:
    """
    Compute water balance components.

    Args:
        input: ComputeWaterBalanceInput with workdir and optional precipitation/observed_flow

    Returns:
        Dictionary with status, message, and data fields
    """
    try:
        import xarray as xr

        # Try to load data from workdir
        workdir_path = Path(input.workdir)

        # Look for hydrograph file
        hydro_path = workdir_path / "output" / "Hydrographs.nc"
        if not hydro_path.exists():
            for f in workdir_path.rglob("*ydrograph*.nc"):
                hydro_path = f
                break

        # Look for water storage file
        storage_path = workdir_path / "output" / "WaterStorage.nc"
        if not storage_path.exists():
            for f in workdir_path.rglob("*aterStorage*.nc"):
                storage_path = f
                break

        # Extract data
        ds_hydro = xr.open_dataset(str(hydro_path)) if hydro_path.exists() else None
        ds_storage = xr.open_dataset(str(storage_path)) if storage_path.exists() else None

        # Get precipitation (from input or forcing file)
        if input.precipitation:
            total_p = float(np.sum(input.precipitation))
        else:
            # Try to extract from forcing or other sources
            total_p = 0.0

        # Get ET from storage file
        total_et = 0.0
        if ds_storage and 'ET' in ds_storage.data_vars:
            et_vals = ds_storage['ET'].values
            et_vals = et_vals[~np.isnan(et_vals)]
            total_et = float(np.sum(et_vals))

        # Get streamflow from hydrograph file
        total_q = 0.0
        if ds_hydro:
            for var in ds_hydro.data_vars:
                if 'sim' in var.lower() or 'flow' in var.lower():
                    q_vals = ds_hydro[var].values
                    q_vals = q_vals[~np.isnan(q_vals)]
                    total_q = float(np.sum(q_vals))
                    break

        # Compute storage change
        ds = total_p - total_et - total_q

        # Compute balance error as percentage
        balance_error = (abs(ds) / total_p * 100) if total_p > 0 else 0

        from hydroagent.schemas.tool_outputs import WaterBalance
        water_balance = WaterBalance(
            precipitation=total_p,
            evapotranspiration=total_et,
            streamflow=total_q,
            storage_change=ds,
            balance_error=float(balance_error),
        )

        output = ComputeWaterBalanceOutput(water_balance=water_balance)
        response = ToolResponse[ComputeWaterBalanceOutput].success(
            data=output,
            message="Water balance computed"
        )
        return response.model_dump()
    except Exception as e:
        response = ToolResponse[ComputeWaterBalanceOutput].error(
            message=str(e),
            error_type=type(e).__name__
        )
        return response.model_dump()

# PLACEHOLDER_ANALYSIS_LAST


def compare_simulations(input: CompareSimulationsInput) -> dict[str, Any]:
    """
    Compare multiple simulations against observed data.

    Args:
        input: CompareSimulationsInput with simulations, observed, and metrics

    Returns:
        Dictionary with status, message, and data fields
    """
    try:
        from hydroagent.schemas.tool_outputs import SimulationComparison
        from hydroagent.schemas.common import Statistics

        obs = np.array(input.observed) if input.observed else None
        comparisons = []

        for sim_name, sim_vals in input.simulations.items():
            sim = np.array(sim_vals)

            # Compute metrics if observed data is provided
            metrics_dict = {}
            if obs is not None:
                for m in input.metrics:
                    func = getattr(he, m.lower(), None)
                    if func:
                        metrics_dict[m] = float(func(sim, obs))

            # Compute statistics for this simulation
            stats = Statistics(
                mean=float(np.mean(sim)),
                min=float(np.min(sim)),
                max=float(np.max(sim)),
                std=float(np.std(sim)),
                count=len(sim),
            )

            comparisons.append(SimulationComparison(
                name=sim_name,
                metrics=metrics_dict,
                statistics=stats,
            ))

        # Determine best simulation based on first metric
        best_sim = ""
        if comparisons and input.metrics and obs is not None:
            first_metric = input.metrics[0].lower()
            # Metrics to maximize
            maximize_metrics = {"nse", "kge_2009", "kge_2012", "r_squared", "pearson_r"}

            valid_comparisons = [c for c in comparisons if first_metric in c.metrics]
            if valid_comparisons:
                if first_metric in maximize_metrics:
                    best_sim = max(valid_comparisons, key=lambda c: c.metrics[first_metric]).name
                else:
                    best_sim = min(valid_comparisons, key=lambda c: c.metrics[first_metric]).name

        output = CompareSimulationsOutput(
            n_simulations=len(input.simulations),
            comparisons=comparisons,
            best_simulation=best_sim,
        )
        response = ToolResponse[CompareSimulationsOutput].success(
            data=output,
            message=f"Compared {len(input.simulations)} simulations"
        )
        return response.model_dump()
    except Exception as e:
        response = ToolResponse[CompareSimulationsOutput].error(
            message=str(e),
            error_type=type(e).__name__
        )
        return response.model_dump()


def generate_diagnostic_plots(input: GenerateDiagnosticPlotsInput) -> dict[str, Any]:
    """
    Generate diagnostic plots using matplotlib.

    Args:
        input: GenerateDiagnosticPlotsInput with observed, simulated, output_dir, and plot_types

    Returns:
        Dictionary with status, message, and data fields
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime

        out = Path(input.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        obs = np.array(input.observed)
        sim = np.array(input.simulated)
        dates = None
        if input.dates:
            dates = [datetime.fromisoformat(t) for t in input.dates]

        plots = {}

        if "hydrograph" in input.plot_types:
            fig, ax = plt.subplots(figsize=(12, 4))
            if dates:
                ax.plot(dates, obs, "b-", label="Observed", lw=1)
                ax.plot(dates, sim, "r-", label="Simulated", lw=1, alpha=0.8)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                fig.autofmt_xdate()
            else:
                ax.plot(obs, "b-", label="Observed", lw=1)
                ax.plot(sim, "r-", label="Simulated", lw=1, alpha=0.8)
            ax.set_ylabel("Flow (m³/s)")
            ax.legend()
            p = out / "hydrograph.png"
            fig.savefig(str(p), dpi=200, bbox_inches="tight")
            plt.close(fig)
            plots["hydrograph"] = str(p)

        if "scatter" in input.plot_types:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(obs, sim, s=3, alpha=0.5)
            lims = [min(obs.min(), sim.min()), max(obs.max(), sim.max())]
            ax.plot(lims, lims, "k--", lw=0.8)
            ax.set_xlabel("Observed")
            ax.set_ylabel("Simulated")
            ax.set_title("Scatter Plot")
            p = out / "scatter.png"
            fig.savefig(str(p), dpi=200, bbox_inches="tight")
            plt.close(fig)
            plots["scatter"] = str(p)

        if "residuals" in input.plot_types:
            fig, ax = plt.subplots(figsize=(12, 3))
            residuals = sim - obs
            if dates:
                ax.bar(dates, residuals, width=1, color="grey", alpha=0.7)
                fig.autofmt_xdate()
            else:
                ax.bar(range(len(residuals)), residuals, width=1, color="grey", alpha=0.7)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_ylabel("Residual (m³/s)")
            p = out / "residuals.png"
            fig.savefig(str(p), dpi=200, bbox_inches="tight")
            plt.close(fig)
            plots["residuals"] = str(p)

        if "flow_duration" in input.plot_types or "fdc" in input.plot_types:
            fig, ax = plt.subplots(figsize=(8, 5))
            obs_sorted = np.sort(obs)[::-1]
            sim_sorted = np.sort(sim)[::-1]
            n = len(obs_sorted)
            exc = np.arange(1, n + 1) / n * 100
            ax.semilogy(exc, obs_sorted, "b-", label="Obs")
            ax.semilogy(exc, sim_sorted, "r-", label="Sim")
            ax.set_xlabel("Exceedance (%)")
            ax.set_ylabel("Flow (m³/s)")
            ax.legend()
            ax.set_title("Flow Duration Curve")
            p = out / "flow_duration.png"
            fig.savefig(str(p), dpi=200, bbox_inches="tight")
            plt.close(fig)
            plots["flow_duration"] = str(p)

        output = GenerateDiagnosticPlotsOutput(
            output_dir=input.output_dir,
            plots_generated=list(plots.keys()),
            plot_paths=plots,
        )
        response = ToolResponse[GenerateDiagnosticPlotsOutput].success(
            data=output,
            message=f"Generated {len(plots)} plots"
        )
        return response.model_dump()
    except Exception as e:
        response = ToolResponse[GenerateDiagnosticPlotsOutput].error(
            message=str(e),
            error_type=type(e).__name__
        )
        return response.model_dump()


def export_results_summary(input: ExportResultsInput) -> dict[str, Any]:
    """
    Export results summary to file.

    Args:
        input: ExportResultsInput with results, output_path, and format

    Returns:
        Dictionary with status, message, and data fields
    """
    try:
        p = Path(input.output_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        n_records = 0
        if input.format == "json":
            with open(p, "w") as f:
                json.dump(input.results, f, indent=2, default=str)
            # Count records (top-level keys or items)
            if isinstance(input.results, dict):
                n_records = len(input.results)
            elif isinstance(input.results, list):
                n_records = len(input.results)
        elif input.format == "csv":
            import pandas as pd
            df = pd.json_normalize(input.results)
            df.to_csv(p, index=False)
            n_records = len(df)
        else:
            response = ToolResponse[ExportResultsOutput].error(
                message=f"Unsupported format: {input.format}",
                error_type="ValueError"
            )
            return response.model_dump()

        output = ExportResultsOutput(
            output_path=str(p),
            format=input.format,
            n_records=n_records,
        )
        response = ToolResponse[ExportResultsOutput].success(
            data=output,
            message=f"Exported to {p}"
        )
        return response.model_dump()
    except Exception as e:
        response = ToolResponse[ExportResultsOutput].error(
            message=str(e),
            error_type=type(e).__name__
        )
        return response.model_dump()


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def _interpret_metrics(metrics: dict) -> dict[str, str]:
    interp = {}
    nse = metrics.get("nse")
    if nse is not None:
        if nse > 0.75:
            interp["nse"] = "Very good"
        elif nse > 0.65:
            interp["nse"] = "Good"
        elif nse > 0.5:
            interp["nse"] = "Satisfactory"
        else:
            interp["nse"] = "Unsatisfactory"

    kge = metrics.get("kge_2009") or metrics.get("kge_2012")
    if kge is not None:
        if kge > 0.75:
            interp["kge"] = "Very good"
        elif kge > 0.5:
            interp["kge"] = "Good"
        else:
            interp["kge"] = "Needs improvement"
    return interp


def _classify_flow_regime(fdc: dict) -> str:
    q5 = fdc.get("Q5")
    q95 = fdc.get("Q95")
    if q5 is not None and q95 is not None and q95 > 0:
        ratio = q5 / q95
        if ratio > 20:
            return "flashy"
        elif ratio > 10:
            return "moderately_variable"
        else:
            return "stable"
    return "unknown"


def _baseflow_filter(q: np.ndarray, alpha: float = 0.925) -> np.ndarray:
    """Lyne-Hollick recursive digital filter for baseflow."""
    qf = np.zeros_like(q)
    qf[0] = q[0]
    for i in range(1, len(q)):
        qf[i] = alpha * qf[i - 1] + (1 + alpha) / 2 * (q[i] - q[i - 1])
        qf[i] = max(0, qf[i])
    baseflow = q - qf
    baseflow = np.maximum(baseflow, 0)
    return baseflow


def _flashiness_index(q: np.ndarray) -> float:
    """Richards-Baker flashiness index."""
    total = np.sum(q)
    if total == 0:
        return 0.0
    return float(np.sum(np.abs(np.diff(q))) / total)
