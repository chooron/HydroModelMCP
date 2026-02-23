"""
RavenPy Data Extraction and Preprocessing Tools

Provides functions for extracting, processing, and preparing hydrological data
for RavenPy models including climate data processing and forcing file creation.
"""

from typing import Any
import numpy as np
from pathlib import Path

from hydroagent.schemas import (
    ExtractClimateDataInput,
    ExtractClimateDataOutput,
    ProcessPrecipitationInput,
    ProcessPrecipitationOutput,
    ProcessTemperatureInput,
    ProcessTemperatureOutput,
    ExtractBasinPropertiesInput,
    ExtractBasinPropertiesOutput,
    ComputePETInput,
    ComputePETOutput,
    QualityCheckClimateDataInput,
    QualityCheckClimateDataOutput,
    CreateForcingFileInput,
    CreateForcingFileOutput,
    ToolResponse,
    Statistics,
)


def extract_climate_data_from_nc(input: ExtractClimateDataInput) -> dict[str, Any]:
    """
    Extract climate data from a NetCDF file.

    Args:
        input: ExtractClimateDataInput with nc_path, variables, and date filters

    Returns:
        Dictionary with status, message, and data fields
    """
    try:
        import xarray as xr

        ds = xr.open_dataset(input.nc_path)
        if input.start_date and input.end_date:
            ds = ds.sel(time=slice(input.start_date, input.end_date))

        summary = {}
        for var in input.variables:
            if var in ds:
                vals = ds[var].values.flatten()
                vals = vals[~np.isnan(vals)]
                summary[var] = {
                    "mean": float(np.mean(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "std": float(np.std(vals)),
                    "n_records": len(vals),
                }
            else:
                summary[var] = {"error": "not found"}

        time_vals = ds.time.values if "time" in ds.dims else []

        output = ExtractClimateDataOutput(
            nc_path=input.nc_path,
            n_timesteps=len(time_vals),
            variables_found=[v for v in input.variables if v in ds],
            data_summary=summary,
        )
        response = ToolResponse[ExtractClimateDataOutput].success(
            data=output,
            message="Climate data extracted"
        )
        return response.model_dump()
    except Exception as e:
        response = ToolResponse[ExtractClimateDataOutput].error(
            message=str(e),
            error_type=type(e).__name__
        )
        return response.model_dump()


def process_precipitation_data(input: ProcessPrecipitationInput) -> dict[str, Any]:
    """
    Process and quality-check precipitation data.

    Args:
        input: ProcessPrecipitationInput with data path and processing options

    Returns:
        Dictionary with status, message, and data fields
    """
    try:
        import pandas as pd

        p = Path(input.data_path)
        if p.suffix == ".nc":
            import xarray as xr
            ds = xr.open_dataset(input.data_path)
            for var in ["pr", "precip", "precipitation", "PRECIP"]:
                if var in ds:
                    series = ds[var].to_series()
                    break
            else:
                first = list(ds.data_vars)[0]
                series = ds[first].to_series()
        elif p.suffix == ".csv":
            df = pd.read_csv(input.data_path, parse_dates=[0], index_col=0)
            series = df.iloc[:, 0]
        else:
            response = ToolResponse[ProcessPrecipitationOutput].error(
                message=f"Unsupported file format: {p.suffix}",
                error_type="ValueError"
            )
            return response.model_dump()

        missing_before = int(series.isna().sum())

        agg_map = {"daily": "D", "hourly": "h", "monthly": "ME"}
        if input.aggregation in agg_map:
            series = series.resample(agg_map[input.aggregation]).sum()

        if input.fill_missing:
            series = series.interpolate(method=input.method)

        vals = series.dropna().values

        output = ProcessPrecipitationOutput(
            data_path=input.data_path,
            aggregation=input.aggregation,
            statistics={
                "total_records": len(series),
                "missing_before": missing_before,
                "missing_after": int(series.isna().sum()),
                "mean": float(np.mean(vals)),
                "max": float(np.max(vals)),
                "total": float(np.sum(vals)),
                "unit": "mm/d",
            },
        )
        response = ToolResponse[ProcessPrecipitationOutput].success(
            data=output,
            message="Precipitation data processed"
        )
        return response.model_dump()
    except Exception as e:
        response = ToolResponse[ProcessPrecipitationOutput].error(
            message=str(e),
            error_type=type(e).__name__
        )
        return response.model_dump()


def process_temperature_data(input: ProcessTemperatureInput) -> dict[str, Any]:
    """
    Process temperature data.

    Args:
        input: ProcessTemperatureInput with data path and conversion options

    Returns:
        Dictionary with status, message, and data fields
    """
    try:
        import pandas as pd

        p = Path(input.data_path)
        if p.suffix == ".nc":
            import xarray as xr
            ds = xr.open_dataset(input.data_path)
            tmin = tmax = tmean = None
            for var in ds.data_vars:
                vl = var.lower()
                if "min" in vl:
                    tmin = ds[var].values.flatten()
                elif "max" in vl:
                    tmax = ds[var].values.flatten()
                elif "mean" in vl or "avg" in vl:
                    tmean = ds[var].values.flatten()
        elif p.suffix == ".csv":
            df = pd.read_csv(input.data_path, parse_dates=[0], index_col=0)
            cols = df.columns.str.lower()
            tmin = tmax = tmean = None
            for i, c in enumerate(cols):
                if "min" in c:
                    tmin = df.iloc[:, i].values
                elif "max" in c:
                    tmax = df.iloc[:, i].values
                elif "mean" in c or "avg" in c:
                    tmean = df.iloc[:, i].values
        else:
            response = ToolResponse[ProcessTemperatureOutput].error(
                message=f"Unsupported file format: {p.suffix}",
                error_type="ValueError"
            )
            return response.model_dump()

        def _convert(arr, to):
            if arr is None:
                return None
            arr = arr.astype(float)
            if np.nanmean(arr) > 200 and to == "celsius":
                arr = arr - 273.15
            elif np.nanmean(arr) < 100 and to == "kelvin":
                arr = arr + 273.15
            return arr

        tmin = _convert(tmin, input.convert_to)
        tmax = _convert(tmax, input.convert_to)
        tmean = _convert(tmean, input.convert_to)

        mean_computed = False
        if input.compute_mean and tmean is None and tmin is not None and tmax is not None:
            tmean = (tmin + tmax) / 2.0
            mean_computed = True

        stats = {}
        for name, arr in [("tmin", tmin), ("tmax", tmax), ("tmean", tmean)]:
            if arr is not None:
                clean = arr[~np.isnan(arr)]
                stats[name] = {
                    "mean": float(np.mean(clean)),
                    "min": float(np.min(clean)),
                    "max": float(np.max(clean)),
                    "unit": input.convert_to,
                }

        output = ProcessTemperatureOutput(
            data_path=input.data_path,
            unit=input.convert_to,
            statistics=stats,
            mean_computed=mean_computed,
        )
        response = ToolResponse[ProcessTemperatureOutput].success(
            data=output,
            message="Temperature data processed"
        )
        return response.model_dump()
    except Exception as e:
        response = ToolResponse[ProcessTemperatureOutput].error(
            message=str(e),
            error_type=type(e).__name__
        )
        return response.model_dump()


def extract_basin_properties_from_nc(input: ExtractBasinPropertiesInput) -> dict[str, Any]:
    """
    Extract basin properties from a RavenPy-compatible NetCDF.

    Args:
        input: ExtractBasinPropertiesInput with nc_path and optional property filter

    Returns:
        Dictionary with status, message, and data fields
    """
    try:
        import xarray as xr
        ds = xr.open_dataset(input.nc_path)

        props = {}
        for var in ds.data_vars:
            val = ds[var].values
            if val.ndim == 0:
                props[var] = float(val)
            elif val.size == 1:
                props[var] = float(val.flat[0])

        for attr in ds.attrs:
            props[f"attr_{attr}"] = ds.attrs[attr]

        # Filter properties if specified
        if input.properties:
            props = {k: v for k, v in props.items() if k in input.properties}

        output = ExtractBasinPropertiesOutput(
            nc_path=input.nc_path,
            properties=props,
            n_properties=len(props),
        )
        response = ToolResponse[ExtractBasinPropertiesOutput].success(
            data=output,
            message="Basin properties extracted"
        )
        return response.model_dump()
    except Exception as e:
        response = ToolResponse[ExtractBasinPropertiesOutput].error(
            message=str(e),
            error_type=type(e).__name__
        )
        return response.model_dump()


def compute_potential_evapotranspiration(input: ComputePETInput) -> dict[str, Any]:
    """
    Compute PET using Oudin or Hargreaves method.

    Args:
        input: ComputePETInput with temperature data, latitude, and method

    Returns:
        Dictionary with status, message, and data fields
    """
    try:
        from datetime import datetime, timedelta

        tmin = np.array(input.tasmin)
        tmax = np.array(input.tasmax)
        tmean = (tmin + tmax) / 2.0
        n = len(tmean)

        # Determine start date for day-of-year calculation
        if input.dates and len(input.dates) > 0:
            start_date = input.dates[0]
        else:
            start_date = "2000-01-01"

        d0 = datetime.fromisoformat(start_date)
        doys = np.array([
            (d0 + timedelta(days=i)).timetuple().tm_yday for i in range(n)
        ])

        lat_rad = np.radians(input.latitude)
        dr = 1 + 0.033 * np.cos(2 * np.pi * doys / 365)
        delta = 0.409 * np.sin(2 * np.pi * doys / 365 - 1.39)
        ws = np.arccos(-np.tan(lat_rad) * np.tan(delta))
        Ra = (
            24 * 60 / np.pi * 0.0820 * dr
            * (ws * np.sin(lat_rad) * np.sin(delta)
               + np.cos(lat_rad) * np.cos(delta) * np.sin(ws))
        )

        if input.method == "oudin":
            pet = np.where(tmean > -5, Ra / (100 * 2.45) * (tmean + 5) / 100, 0.0)
        elif input.method == "hargreaves":
            tdiff = np.maximum(tmax - tmin, 0)
            pet = 0.0023 * Ra / 2.45 * np.sqrt(tdiff) * (tmean + 17.8)
            pet = np.maximum(pet, 0)
        else:
            response = ToolResponse[ComputePETOutput].error(
                message=f"Unknown method: {input.method}",
                error_type="ValueError"
            )
            return response.model_dump()

        output = ComputePETOutput(
            method=input.method,
            pet_values=pet.tolist(),
            statistics=Statistics(
                mean=float(np.mean(pet)),
                std=float(np.std(pet)),
                min=float(np.min(pet)),
                max=float(np.max(pet)),
                count=len(pet),
            ),
        )
        response = ToolResponse[ComputePETOutput].success(
            data=output,
            message=f"PET computed via {input.method}"
        )
        return response.model_dump()
    except Exception as e:
        response = ToolResponse[ComputePETOutput].error(
            message=str(e),
            error_type=type(e).__name__
        )
        return response.model_dump()


def quality_check_climate_data(input: QualityCheckClimateDataInput) -> dict[str, Any]:
    """
    Quality checks on climate data arrays.

    Args:
        input: QualityCheckClimateDataInput with climate data and check options

    Returns:
        Dictionary with status, message, and data fields
    """
    try:
        errors = []
        warnings = []
        checks_performed = []

        # Extract data arrays
        pr = np.array(input.data.get("precipitation", []))
        tmin = np.array(input.data.get("temperature_min", []))
        tmax = np.array(input.data.get("temperature_max", []))

        # Check array lengths
        if len(pr) > 0 and len(tmin) > 0 and len(tmax) > 0:
            if len(pr) != len(tmin) or len(pr) != len(tmax):
                errors.append("Array lengths do not match")
            checks_performed.append("array_length_consistency")

        # Check negative precipitation
        if input.check_negative_precip and len(pr) > 0:
            n_neg = int(np.sum(pr < 0))
            if n_neg > 0:
                errors.append(f"{n_neg} negative precipitation values")
            checks_performed.append("negative_precipitation")

        # Check temperature range
        if input.check_temp_range and len(tmin) > 0 and len(tmax) > 0:
            n_inv = int(np.sum(tmin > tmax))
            if n_inv > 0:
                warnings.append(f"{n_inv} days where Tmin > Tmax")
            checks_performed.append("temperature_range")

        # Check missing values
        if input.check_missing:
            if len(pr) > 0:
                n_nan_pr = int(np.sum(np.isnan(pr)))
                if n_nan_pr > 0:
                    warnings.append(f"{n_nan_pr} NaN in precipitation")
            if len(tmin) > 0 or len(tmax) > 0:
                n_nan_t = int(np.sum(np.isnan(tmin)) + np.sum(np.isnan(tmax)))
                if n_nan_t > 0:
                    warnings.append(f"{n_nan_t} NaN in temperature")
            checks_performed.append("missing_values")

        # Check extreme values
        if len(pr) > 0:
            n_extreme = int(np.sum(pr > 200))
            if n_extreme > 0:
                warnings.append(f"{n_extreme} days with precip > 200 mm/d")
            checks_performed.append("extreme_values")

        output = QualityCheckClimateDataOutput(
            checks_performed=checks_performed,
            errors=errors,
            warnings=warnings,
            is_valid=len(errors) == 0,
        )
        response = ToolResponse[QualityCheckClimateDataOutput].success(
            data=output,
            message="Quality check completed"
        )
        return response.model_dump()
    except Exception as e:
        response = ToolResponse[QualityCheckClimateDataOutput].error(
            message=str(e),
            error_type=type(e).__name__
        )
        return response.model_dump()


def create_raven_forcing_nc(input: CreateForcingFileInput) -> dict[str, Any]:
    """
    Create a Raven-compatible NetCDF forcing file.

    Args:
        input: CreateForcingFileInput with climate data and metadata

    Returns:
        Dictionary with status, message, and data fields
    """
    try:
        import xarray as xr
        import pandas as pd

        n = len(input.pr)
        times = pd.to_datetime(input.dates)

        data_vars = {
            "pr": (["time", "station"], np.array(input.pr).reshape(-1, 1)),
            "tasmin": (["time", "station"], np.array(input.tasmin).reshape(-1, 1)),
            "tasmax": (["time", "station"], np.array(input.tasmax).reshape(-1, 1)),
        }

        elevation = input.elevation if input.elevation is not None else 0.0

        ds = xr.Dataset(
            data_vars,
            coords={
                "time": times,
                "station": [0],
                "lat": ("station", [input.latitude]),
                "lon": ("station", [input.longitude]),
                "elevation": ("station", [elevation]),
            },
        )

        ds["pr"].attrs = {"units": "mm/d", "long_name": "Precipitation"}
        ds["tasmin"].attrs = {"units": "degC", "long_name": "Min Temperature"}
        ds["tasmax"].attrs = {"units": "degC", "long_name": "Max Temperature"}

        p = Path(input.output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(str(p))

        output = CreateForcingFileOutput(
            output_path=str(p),
            n_timesteps=n,
            variables=list(data_vars.keys()),
            date_range={
                "start": str(times[0].date()),
                "end": str(times[-1].date()),
            },
        )
        response = ToolResponse[CreateForcingFileOutput].success(
            data=output,
            message=f"Forcing file created: {p}"
        )
        return response.model_dump()
    except Exception as e:
        response = ToolResponse[CreateForcingFileOutput].error(
            message=str(e),
            error_type=type(e).__name__
        )
        return response.model_dump()
