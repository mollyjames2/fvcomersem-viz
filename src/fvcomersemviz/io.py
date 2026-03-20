"""I/O helpers.

Implement:
 - discover_paths(base_dir, file_pattern) -> list[str]
 - load_from_base(base_dir, file_pattern) -> xr.Dataset
 - filter_time(ds, months=None, years=None, start_date=None, end_date=None)
 - eval_group_or_var(ds, name, groups) -> xr.DataArray   # optional centralised helper
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any
import warnings
import re
import numpy as np
import pandas as pd
import xarray as xr


# --------------------------
# Temporal resampling helpers
# --------------------------

# Approximate timedelta for each pandas resample alias, used to detect when
# the requested averaging period is finer than the data's actual time step.
_ALIAS_APPROX_TIMEDELTA: dict = {
    "h":  pd.Timedelta("1h"),
    "D":  pd.Timedelta("1D"),
    "W":  pd.Timedelta("7D"),
    "MS": pd.Timedelta("28D"),   # conservative lower bound (shortest month)
    "YS": pd.Timedelta("365D"),
}

_AVERAGE_BY_ALIASES: dict = {
    # hourly
    "h": "h", "hour": "h", "hours": "h", "hourly": "h", "1h": "h",
    # daily
    "d": "D", "day": "D", "days": "D", "daily": "D", "1d": "D",
    # weekly
    "w": "W", "week": "W", "weeks": "W", "weekly": "W",
    # monthly
    "m": "MS", "ms": "MS", "month": "MS", "months": "MS", "monthly": "MS",
    # yearly
    "y": "YS", "ys": "YS", "year": "YS", "years": "YS", "yearly": "YS", "annual": "YS",
}


def normalise_average_by(average_by: str) -> str:
    """
    Normalise a human-friendly frequency string to a pandas resample alias.

    Accepted values (case-insensitive): ``"hour"``, ``"day"``, ``"week"``,
    ``"month"``, ``"year"`` and common aliases such as ``"daily"``,
    ``"monthly"``, ``"annual"``, ``"D"``, ``"MS"``, ``"YS"``, etc.

    Parameters
    ----------
    average_by : str
        Human-friendly period name (case-insensitive). Accepted values:
        ``"hour"`` / ``"hourly"``; ``"day"`` / ``"daily"``; ``"week"`` /
        ``"weekly"``; ``"month"`` / ``"monthly"``; ``"year"`` / ``"yearly"``
        / ``"annual"``; or any matching pandas alias (``"h"``, ``"D"``,
        ``"MS"``, ``"YS"``).

    Returns
    -------
    str
        A pandas-compatible resample frequency alias (e.g. ``"D"``, ``"MS"``,
        ``"YS"``).

    Raises
    ------
    ValueError
        If *average_by* is not recognised.
    """
    key = average_by.strip().lower()
    alias = _AVERAGE_BY_ALIASES.get(key)
    if alias is None:
        raise ValueError(
            f"average_by={average_by!r} not recognised. "
            "Use one of: hour, day, week, month, year (or aliases daily, monthly, annual, …)."
        )
    return alias


def resample_time(
    ds: xr.Dataset,
    freq: str,
    method: str = "mean",
) -> xr.Dataset:
    """
    Resample the time-varying variables in *ds* to a coarser frequency.

    Static variables (those without a ``time`` dimension) are preserved
    unchanged so that grid coordinates, bathymetry, connectivity, etc. are
    not affected.

    Parameters
    ----------
    ds : xr.Dataset
        Source dataset with a ``time`` dimension.
    freq : str
        Target frequency — any value accepted by :func:`normalise_average_by`
        (e.g. ``"month"``, ``"day"``, ``"MS"``).
    method : {"mean", "sum", "median"}, default "mean"
        Aggregation applied within each period.

    Returns
    -------
    xr.Dataset
        Dataset whose time-varying variables are resampled; static variables
        and non-time coordinates are re-attached unchanged. If *freq* is finer
        than the data's own time step, the dataset is returned **unchanged** and
        a :class:`UserWarning` is issued.
    """
    freq_alias = normalise_average_by(freq)

    # Warn if the requested period is clearly finer than the data time step.
    # A factor-of-2 tolerance avoids false positives when target and data periods
    # are approximately equal (e.g. monthly data averaged by month: 28D alias vs
    # ~30D actual step, or yearly data averaged by year on a leap year).
    if "time" in ds.coords and ds.sizes.get("time", 0) > 1:
        times = pd.DatetimeIndex(ds["time"].values)
        median_dt = pd.to_timedelta(np.median(np.diff(times.asi8)), unit="ns")
        target_dt = _ALIAS_APPROX_TIMEDELTA.get(freq_alias)
        if target_dt is not None and target_dt < median_dt / 2:
            warnings.warn(
                f"average_by={freq!r} requests ~{target_dt} period means but the "
                f"data time step is ~{median_dt}. Resampling to a finer resolution "
                f"than the data produces mostly NaN. Skipping average_by.",
                UserWarning,
                stacklevel=3,
            )
            return ds

    time_vars = [v for v in ds.data_vars if "time" in ds[v].dims]
    static_vars = [v for v in ds.data_vars if "time" not in ds[v].dims]

    if not time_vars:
        return ds

    rs = ds[time_vars].resample(time=freq_alias)
    if method == "mean":
        ds_rs = rs.mean()
    elif method == "sum":
        ds_rs = rs.sum()
    elif method == "median":
        ds_rs = rs.median()
    else:
        raise ValueError(f"method={method!r} not supported. Use: mean, sum, median.")

    # Re-attach static data variables
    for v in static_vars:
        ds_rs[v] = ds[v]

    # Re-attach static (non-time) coordinates
    for c in ds.coords:
        if c not in ds_rs.coords and "time" not in ds[c].dims:
            ds_rs = ds_rs.assign_coords({c: ds[c]})

    return ds_rs


# --------------------------
# Time filtering
# --------------------------
def filter_time(
    ds: xr.Dataset,
    months: Optional[Iterable[int]] = None,
    years: Optional[Iterable[int]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_var: str = "time",
    average_by: Optional[str] = None,
) -> xr.Dataset:
    """Return a time-filtered (and optionally resampled) view of *ds*.

    Filtering is applied first (months/years/date bounds), then if *average_by*
    is given the result is passed through :func:`resample_time` so that each
    period is represented by its mean.

    Parameters
    ----------
    ds : xr.Dataset
    months : iterable of int, optional
    years : iterable of int, optional
    start_date, end_date : str, optional
        Inclusive ISO date bounds.
    time_var : str, default "time"
    average_by : str, optional
        Temporal averaging period applied **after** the time filter.
        Resamples the filtered dataset to period means using
        ``xr.Dataset.resample().mean()``. Static variables (coordinates,
        mesh geometry) are excluded from the resampling so grid arrays are
        preserved unchanged.
        Accepted values (case-insensitive): ``"hour"`` / ``"hourly"``,
        ``"day"`` / ``"daily"``, ``"week"`` / ``"weekly"``,
        ``"month"`` / ``"monthly"``, ``"year"`` / ``"yearly"`` /
        ``"annual"``. Default ``None`` (no averaging).

    Works whether ``time`` is a dimension or only a coordinate.
    """
    if time_var not in ds.dims and time_var not in ds.coords:
        # nothing to filter
        return ds

    # Get a robust DatetimeIndex from the coordinate
    try:
        # xarray >= 0.20 has DataArray.to_index()
        tindex = pd.DatetimeIndex(ds[time_var].to_index())
    except Exception:
        # fallback: use the raw values
        tindex = pd.DatetimeIndex(pd.to_datetime(ds[time_var].values))

    mask = np.ones(tindex.shape, dtype=bool)

    if months is not None:
        months = np.asarray(list(months), dtype=int)
        mask &= np.isin(tindex.month, months)

    if years is not None:
        years = np.asarray(list(years), dtype=int)
        mask &= np.isin(tindex.year, years)

    if start_date is not None:
        mask &= tindex >= pd.to_datetime(start_date)

    if end_date is not None:
        mask &= tindex <= pd.to_datetime(end_date)

    mask = mask.ravel()

    # If time is a dimension, boolean-index it; if it's only a coord, select by labels
    if time_var in ds.dims:
        ds_out = ds.isel({time_var: mask})
    else:
        ds_out = ds.sel({time_var: tindex[mask]})

    if average_by is not None:
        ds_out = resample_time(ds_out, average_by)

    return ds_out


# --------------------------
# Group / variable resolver (optional but recommended to centralize)
# --------------------------
def eval_group_or_var(ds: xr.Dataset, name: str, groups: Optional[Dict[str, Any]]) -> xr.DataArray:
    """
    Return a DataArray for a native variable or a compound 'group' variable.
    - list/tuple: summed elementwise
    - str expression: evaluated in the dataset namespace
    """
    if name in ds:
        return ds[name]

    if not groups:
        raise KeyError(f"Variable '{name}' not found and no groups provided.")

    spec = groups.get(name)
    if spec is None:
        raise KeyError(f"'{name}' not found and not defined in groups.")

    if isinstance(spec, (list, tuple)):
        missing = [v for v in spec if v not in ds]
        if missing:
            raise KeyError(f"Missing members for group '{name}': {missing}")
        out = None
        for v in spec:
            out = ds[v] if out is None else (out + ds[v])
        return out

    if isinstance(spec, str):
        local = {k: ds[k] for k in ds.variables}
        try:
            return eval(spec, {"__builtins__": {}}, local)
        except NameError as e:
            raise KeyError(f"Group expression for '{name}' references unknown name: {e}") from e

    raise TypeError("Group definitions must be list/tuple (sum) or str (expression).")


# --------------------------
# Path discovery
# --------------------------
def discover_paths(base_dir: str, file_pattern: str) -> List[str]:
    files = sorted(str(p) for p in Path(base_dir).glob(file_pattern))
    if not files:
        warnings.warn(f"No files matched {file_pattern!r} in {base_dir!r}")
    return files


# --------------------------
# FVCOM time coercion helpers
# --------------------------
_UNIT_MAP = {
    "days": "D",
    "day": "D",
    "hours": "h",
    "hour": "h",
    "hr": "h",
    "hrs": "h",
    "minutes": "m",
    "minute": "m",
    "min": "m",
    "seconds": "s",
    "second": "s",
    "sec": "s",
    "secs": "s",
    "milliseconds": "ms",
    "millisecond": "ms",
    "msec": "ms",
    "msecs": "ms",
    "ms": "ms",
    "microseconds": "us",
    "microsecond": "us",
    "usec": "us",
    "usecs": "us",
    "μs": "us",
}


def _parse_origin_from_units(units: str, default: str = "1858-11-17 00:00:00") -> pd.Timestamp:
    """
    Parse an origin datetime out of a CF-style units string:
        "<unit> since YYYY-MM-DD[ HH:MM:SS]"
    Falls back to the provided default if no date is found.
    """
    if not units:
        return pd.Timestamp(default)
    m = re.search(
        r"since\s+(\d{1,4}-\d{1,2}-\d{1,2})(?:[ T](\d{1,2}:\d{2}:\d{2}))?",
        units,
        flags=re.IGNORECASE,
    )
    if not m:
        return pd.Timestamp(default)
    date_str = m.group(1)
    time_str = m.group(2) or "00:00:00"
    return pd.to_datetime(f"{date_str} {time_str}")


def _decode_numeric_time(values: np.ndarray, units_attr: str) -> np.ndarray:
    """
    Decode numeric time given a units string like 'days since YYYY-MM-DD ...'
    Returns numpy datetime64[ns] array.
    """
    units_attr = (units_attr or "").strip().lower()
    # Extract base unit word (e.g., 'days', 'hours', 'msec', ...)
    m = re.match(r"([a-zμ]+)", units_attr)
    base = _UNIT_MAP.get(m.group(1), None) if m else None
    origin = _parse_origin_from_units(units_attr)

    if base is None:
        # Unknown unit; fall back to seconds
        base = "s"

    td = pd.to_timedelta(values, unit=base)
    dt = origin + td
    return dt.to_numpy(dtype="datetime64[ns]")


def coerce_time(ds: xr.Dataset) -> xr.Dataset:
    """
    Build a proper datetime64 'time' coordinate from typical FVCOM patterns:
      - 'Itime' (integer days since <origin>) + 'Itime2' (milliseconds since 00:00:00)
      - OR a numeric 'time' with CF-like 'units' attribute (e.g., 'days since YYYY-MM-DD')
    Leaves datetime64 times unchanged. If nothing to do, returns ds.
    """
    # Already datetime64?
    if "time" in ds and np.issubdtype(ds["time"].dtype, np.datetime64):
        return ds

    # Case A: Itime + Itime2
    # io.py (inside coerce_time, Case A)
    if "Itime" in ds and "Itime2" in ds:
        it = np.asarray(ds["Itime"].values)
        it2 = np.asarray(ds["Itime2"].values)
        units = str(ds["Itime"].attrs.get("units", "")).lower()
        origin = _parse_origin_from_units(units, default="1858-11-17 00:00:00")

        # Try common encodings for Itime2
        # 1) milliseconds
        try:
            dt = (
                pd.to_datetime(origin)
                + pd.to_timedelta(it, unit="D")
                + pd.to_timedelta(it2, unit="ms")
            )
            return ds.assign_coords(time=("time", dt.to_numpy(dtype="datetime64[ns]")))
        except Exception:
            pass
        # 2) microseconds
        try:
            dt = (
                pd.to_datetime(origin)
                + pd.to_timedelta(it, unit="D")
                + pd.to_timedelta(it2, unit="us")
            )
            return ds.assign_coords(time=("time", dt.to_numpy(dtype="datetime64[ns]")))
        except Exception:
            pass
        # 3) fractional day
        try:
            dt = pd.to_datetime(origin) + pd.to_timedelta(it + it2, unit="D")
            return ds.assign_coords(time=("time", dt.to_numpy(dtype="datetime64[ns]")))
        except Exception:
            pass

    # Case B: a numeric 'time' variable with units
    if "time" in ds and np.issubdtype(ds["time"].dtype, np.number):
        units = str(ds["time"].attrs.get("units", "")).lower()
        try:
            dt = _decode_numeric_time(ds["time"].values, units)
            return ds.assign_coords(time=("time", dt))
        except Exception:
            pass  # fall through

    return ds


# --------------------------
# Dataset loader
# --------------------------


def _ensure_vertical_coords(ds: xr.Dataset) -> xr.Dataset:
    """
    Ensure 'siglev' and 'siglay' exist as 1-D coordinates on their own dims.
    If shipped as 2-D (e.g., (siglev, nele)), reduce over non-vertical dims.
    """
    for name in ("siglev", "siglay"):
        if name not in ds:
            continue
        v = ds[name]
        # already (name,) -> just mark as coord
        if v.ndim == 1 and v.dims == (name,):
            ds = ds.set_coords(name)
            continue
        # has vertical dim + others -> reduce others
        if name in v.dims:
            other_dims = [d for d in v.dims if d != name]
            v1 = v
            try:
                for d in other_dims:
                    v1 = v1.isel({d: 0})
            except Exception:
                for d in other_dims:
                    v1 = v1.mean(d)
            if v1.ndim == 1 and v1.dims == (name,):
                ds[name] = v1
                ds = ds.set_coords(name)
    return ds


def _ensure_index_coords(ds: xr.Dataset) -> xr.Dataset:
    for dim in ("node", "nele"):
        if dim in ds.dims and dim not in ds.coords:
            ds = ds.assign_coords({dim: (dim, np.arange(ds.sizes[dim]))})
    return ds


def _preprocess_one(ds: xr.Dataset) -> xr.Dataset:
    ds = _ensure_vertical_coords(ds)
    ds = _ensure_index_coords(ds)
    return ds


def load_from_base(base_dir: str, file_pattern: str) -> xr.Dataset:
    paths = discover_paths(base_dir, file_pattern)

    def _open(engine: str) -> xr.Dataset:
        print(f"[io] Trying engine='{engine}' for open_mfdataset …")
        warnings.filterwarnings(
            "ignore",
            message="numpy.ndarray size changed",
            category=RuntimeWarning,
        )
        return xr.open_mfdataset(
            paths,
            combine="nested",
            concat_dim="time",
            decode_times=False,  # we coerce below
            engine=engine,  # 'netcdf4' (C lib), 'h5netcdf' (HDF5), 'scipy' (netCDF3)
            chunks={"time": 168},  # dask-friendly weekly-ish chunks
            parallel=False,  # parallel=True deadlocks with O(1000) files on threaded scheduler
            preprocess=_preprocess_one,
            data_vars="minimal",
            coords="minimal",
            compat="override",
            join="override",  # siglev/siglay coords may vary across files
            combine_attrs="override",
        )

    # Prefer 'netcdf4' for NetCDF4/HDF5 files (FVCOM output); scipy is NetCDF3 only
    for engine in ("netcdf4", "h5netcdf", "scipy"):
        try:
            ds = _open(engine)
            print(f"[io] Using engine='{engine}'.")
            break
        except Exception as e:
            print(f"[io] Engine '{engine}' failed: {e}")
    else:
        # Final fallback: sequential open with netcdf4
        print("[io] Falling back to sequential open with engine='netcdf4' …")
        datasets = []
        for p in paths:
            dsi = xr.open_dataset(p, decode_times=False, engine="netcdf4")
            dsi = _preprocess_one(dsi)
            datasets.append(dsi)
        ds = xr.concat(
            datasets,
            dim="time",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            combine_attrs="override",
        )

    # Build clean datetime64 time (handles Itime/Itime2 or numeric time+units)
    ds = coerce_time(ds)

    # Normalize lon/lonc to [-180, 180]
    for name in ("lon", "lonc"):
        if name in ds:
            v = ds[name]
            v = xr.where(v > 180, v - 360, v)
            v = xr.where(v < -180, v + 360, v)
            ds[name] = v

    return ds


# --------------------------
# Depth definer
# --------------------------


def ensure_z_from_sigma(
    ds: xr.Dataset,
    *,
    eta_candidates=("zeta", "ssh", "eta"),
    depth_candidates=("h", "H", "depth", "bathymetry"),
    sigma_name: str = "siglay",
    out_node: str = "z",
    out_elem: str = "z_nele",
    compute_elements: bool = True,
    verbose: bool = True,
) -> xr.Dataset:
    """
    Ensure a physical vertical coordinate in meters (negative downward) exists.

    Creates:
      - node z:   ds[out_node]  with dims e.g. ('time','siglay','node')
      - elem z:   ds[out_elem]  with dims e.g. ('time','siglay','nele') if compute_elements and 'nv' exists

    Formula (FVCOM-style terrain-following):
        z = zeta + (zeta + h) * sigma

    Requirements for node z:
      - a 1-D sigma coord array in ds[sigma_name] with dim 'siglay'
      - free surface at nodes (one of eta_candidates) with dims incl. 'time' and 'node'
      - bathymetry at nodes (one of depth_candidates) with dim 'node'
    """
    if sigma_name not in ds or "siglay" not in ds[sigma_name].dims:
        raise ValueError(f"Sigma coord '{sigma_name}' with dim 'siglay' is required.")

    eta_var = next((v for v in eta_candidates if v in ds), None)
    h_var = next((v for v in depth_candidates if v in ds), None)
    if eta_var is None:
        raise ValueError(
            f"None of eta candidates {eta_candidates} found (need free-surface at nodes)."
        )
    if h_var is None:
        raise ValueError(
            f"None of depth candidates {depth_candidates} found (need bathymetry at nodes)."
        )

    sig = ds[sigma_name]  # ('siglay',)
    zeta = ds[eta_var]  # ('time','node') or broadcastable to that
    h = ds[h_var]  # ('node',)

    # Node-based z: xarray broadcasts across time/siglay/node (lazy if dask)
    z_node = zeta + (zeta + h) * sig
    z_node = z_node.transpose(*[d for d in ("time", "siglay", "node") if d in z_node.dims])
    ds[out_node] = z_node
    if verbose:
        print(f"[io] Added node vertical coord '{out_node}' with dims {tuple(z_node.dims)}")

    # Optional: element-based z via nv mean of node values
    if compute_elements and "nv" in ds and "nele" in ds.dims:
        nv = np.asarray(ds["nv"].values)
        tri = (nv.T if nv.ndim == 2 and nv.shape[0] == 3 else nv) - 1
        tri = tri.astype(int)
        # skip if any node index is out of range for this ds
        if tri.min() < 0 or tri.max() >= ds.sizes.get("node", 0):
            if verbose:
                print(
                    "[io] Skipping element z: 'nv' references node indices outside current dataset."
                )
        else:
            tri_da = xr.DataArray(tri.astype(int), dims=("nele", "three"))
            zeta_e = zeta.isel(node=tri_da).mean("three")
            h_e = h.isel(node=tri_da).mean("three")
            z_elem = zeta_e + (zeta_e + h_e) * sig
            z_elem = z_elem.transpose(*[d for d in ("time", "siglay", "nele") if d in z_elem.dims])
            ds[out_elem] = z_elem
            if verbose:
                print(
                    f"[io] Added element vertical coord '{out_elem}' with dims {tuple(z_elem.dims)}"
                )

    return ds
