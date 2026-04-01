# fvcomersemviz/plots/bars.py
from __future__ import annotations

"""
Bar-chart summaries (means with uncertainty) for FVCOM-ERSEM outputs.

Goal
----
A single function, plot_bars(), that can aggregate by x/hue/facet across a flexible
set of dimensions:
  - identity dims: variable (or grouped variables), region, station, depth
  - time dims: day, month, year
  - derived time dim: season (defined at runtime via a dict like {"Spring": [3,4,5], ...})

Key design choice (scalability)
-------------------------------
This implementation does NOT build a giant "tidy" table for the whole dataset.
It aggregates per (series unit) and updates per-bin running stats (n, sum, sumsq),
so memory is O(number_of_bins), not O(number_of_time_samples).

Conventions followed
--------------------
- explicit function arguments (no global state)
- time filtering via fvcomersemviz.io.filter_time
- grouped variables via fvcomersemviz.utils.resolve_da_with_depth (uses eval_group_or_var)
- depth handling via fvcomersemviz.utils.resolve_da_with_depth
- region masking via fvcomersemviz.regions polygon mask helpers
- output naming via fvcomersemviz.utils.file_prefix/out_dir

Notes
-----
- "season" is the only "derived" time dimension here (month -> season mapping).
- month/day/year are direct time-derived dimensions.
- "depth" grouping only works if you pass multiple depths (list/tuple).

Stations
--------
Stations are specified as:
    stations=[("StationA", lat, lon), ("StationB", lat, lon), ...]

We select the nearest node (if variable has 'node' dim) or nearest element center
(if variable has 'nele' dim and ds has 'lonc'/'latc').

Uncertainty
-----------
- error="sd"   : mean +/- 1 sample standard deviation across time samples
- error="ci95" : 95% CI for the mean across time samples

"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from ..io import filter_time
from ..regions import (
    element_mask_from_node_mask,
    polygon_from_csv_boundary,
    polygon_mask,
    polygon_mask_from_shapefile,
)
from ..utils import (
    build_time_window_label,
    depth_tag,
    file_prefix,
    out_dir,
    resolve_da_with_depth,
)


# -----------------------
# small utilities
# -----------------------
def _vprint(verbose: bool, *args, **kwargs) -> None:
    if verbose:
        print(*args, **kwargs)


def _t_critical_975(df: int) -> float:
    """Return t critical value for 95% CI (two-sided), p=0.975.
    Uses scipy if available; otherwise falls back to 1.96.
    """
    if df <= 0:
        return 1.96
    try:
        from scipy.stats import t as student_t  # type: ignore

        return float(student_t.ppf(0.975, df))
    except Exception:
        return 1.96


def _ensure_time_index(da_1d: xr.DataArray) -> pd.DatetimeIndex:
    if "time" not in da_1d.dims:
        raise ValueError("Expected a 'time' dimension after reduction/selection.")
    return pd.DatetimeIndex(da_1d["time"].to_index())


def _month_name(m: int) -> str:
    return [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ][m - 1]


def _month_levels() -> List[str]:
    return [_month_name(i) for i in range(1, 13)]


def _labels_month(t: pd.DatetimeIndex) -> np.ndarray:
    names = np.asarray(_month_levels(), dtype=object)
    return names[t.month - 1]


def _labels_year(t: pd.DatetimeIndex) -> np.ndarray:
    return t.year.astype(int).astype(str).to_numpy()


def _labels_day(t: pd.DatetimeIndex) -> np.ndarray:
    # day-of-month: "1".."31"
    return t.day.astype(int).astype(str).to_numpy()


def _labels_season(t: pd.DatetimeIndex, seasons: Dict[str, Sequence[int]]) -> np.ndarray:
    # seasons: label -> months
    m2s: Dict[int, str] = {}
    for label, months in seasons.items():
        for m in months:
            m2s[int(m)] = str(label)
    return np.array([m2s.get(int(m), "Unknown") for m in t.month], dtype=object)


def _pick_color_map(hue_levels: List[str], hue_colors: Optional[Any]) -> Dict[str, Any]:
    """Return mapping hue_level -> matplotlib color.

    hue_colors can be:
      - None: use default color cycle in order
      - list/tuple: used in order of hue_levels
      - dict: explicit mapping
    """
    if hue_colors is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if not cycle:
            cycle = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
        return {h: cycle[i % len(cycle)] for i, h in enumerate(hue_levels)}

    if isinstance(hue_colors, dict):
        return {h: hue_colors.get(h, None) for h in hue_levels}

    if isinstance(hue_colors, (list, tuple)):
        return {h: hue_colors[i % len(hue_colors)] for i, h in enumerate(hue_levels)}

    raise TypeError("hue_colors must be None, a dict, or a list/tuple")


def _space_mean_over_dims(
    da: xr.DataArray, ds: xr.Dataset, *, verbose: bool = False
) -> xr.DataArray:
    """Mean across all non-time dims; area-weight with art1 if possible.

    Mirrors the behavior used in other plotters: reduce everything except time.
    """
    if "time" in da.dims:
        space_dims = [d for d in da.dims if d != "time"]
    else:
        space_dims = list(da.dims)

    if not space_dims:
        return da

    if "art1" in ds and all(d in ds["art1"].dims for d in space_dims):
        w = ds["art1"]
        # Align weights to da using label selection when possible.
        for d in space_dims:
            if d in w.dims and d in da.dims and w.sizes.get(d) != da.sizes.get(d):
                if d in w.coords and d in da.coords and w[d].ndim == 1 and da[d].ndim == 1:
                    try:
                        w = w.sel({d: da[d]})
                    except Exception:
                        _vprint(
                            verbose,
                            f"[space-mean] Failed to align 'art1' on dim '{d}', using simple mean",
                        )
                        return da.mean(space_dims, skipna=True)
                else:
                    return da.mean(space_dims, skipna=True)
        num = (da * w).sum(space_dims, skipna=True)
        den = w.sum(space_dims, skipna=True)
        return num / den

    return da.mean(space_dims, skipna=True)


def _space_mean_keep_siglay(
    da: xr.DataArray, ds: xr.Dataset, *, verbose: bool = False
) -> xr.DataArray:
    """Area-weighted mean over horizontal space dims only; preserves 'siglay'.

    Used when depth='depth_avg' so that siglay variance is captured by the
    running stats rather than being collapsed before accumulation.
    """
    space_dims = [d for d in da.dims if d not in ("time", "siglay")]
    if not space_dims:
        return da

    if "art1" in ds and all(d in ds["art1"].dims for d in space_dims):
        w = ds["art1"]
        for d in space_dims:
            if d in w.dims and d in da.dims and w.sizes.get(d) != da.sizes.get(d):
                if d in w.coords and d in da.coords and w[d].ndim == 1 and da[d].ndim == 1:
                    try:
                        w = w.sel({d: da[d]})
                    except Exception:
                        _vprint(
                            verbose,
                            f"[space-mean-siglay] Failed to align 'art1' on dim '{d}', using simple mean",
                        )
                        return da.mean(space_dims, skipna=True)
                else:
                    return da.mean(space_dims, skipna=True)
        num = (da * w).sum(space_dims, skipna=True)
        den = w.sum(space_dims, skipna=True)
        return num / den

    return da.mean(space_dims, skipna=True)


# -----------------------
# running stats (streaming)
# -----------------------
@dataclass
class _RunningStats:
    n: int = 0
    s1: float = 0.0
    s2: float = 0.0

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return
        self.n += int(x.size)
        self.s1 += float(x.sum())
        self.s2 += float((x * x).sum())

    def mean(self) -> float:
        return self.s1 / self.n if self.n > 0 else float("nan")

    def sd(self) -> float:
        if self.n <= 1:
            return 0.0
        var = (self.s2 - (self.s1 * self.s1) / self.n) / (self.n - 1)
        return float(math.sqrt(max(var, 0.0)))

    def err(self, error: str) -> float:
        if error == "sd":
            return self.sd()
        # ci95
        if self.n <= 1:
            return float("nan")
        sd = self.sd()
        se = sd / math.sqrt(self.n)
        return _t_critical_975(self.n - 1) * se


# -----------------------
# grouping helpers
# -----------------------
_ID_DIMS = {"variable", "region", "station", "depth"}
_TIME_DIMS = {"day", "month", "year", "season"}
_SUPPORTED_DIMS = _ID_DIMS | _TIME_DIMS


def _is_supported_dim(name: Optional[str]) -> bool:
    return (name is None) or (name in _SUPPORTED_DIMS)


def _dim_levels_from_data(
    dim: str,
    *,
    seasons: Dict[str, Sequence[int]],
    month_levels: List[str],
    years: Optional[List[str]] = None,
) -> Optional[List[str]]:
    if dim == "month":
        return month_levels
    if dim == "season":
        return list(seasons.keys())
    if dim == "year":
        return years  # can be None -> inferred
    # day/variable/region/station/depth inferred from data
    return None


def _extract_station_index(
    ds: xr.Dataset,
    *,
    lon: float,
    lat: float,
    kind: str,
) -> int:
    """Return nearest index for a station for node- or element-centered fields."""
    if kind == "node":
        if "lon" not in ds or "lat" not in ds:
            raise ValueError("Station selection requires ds['lon'] and ds['lat'] for node fields.")
        x = np.asarray(ds["lon"].values).ravel()
        y = np.asarray(ds["lat"].values).ravel()
    elif kind == "nele":
        if "lonc" not in ds or "latc" not in ds:
            raise ValueError(
                "Station selection requires ds['lonc'] and ds['latc'] for element fields."
            )
        x = np.asarray(ds["lonc"].values).ravel()
        y = np.asarray(ds["latc"].values).ravel()
    else:
        raise ValueError("kind must be 'node' or 'nele'")

    # simple nearest in lon/lat degrees (usually fine for small regions)
    dx = x - float(lon)
    dy = y - float(lat)
    d2 = dx * dx + dy * dy
    return int(np.argmin(d2))


def _split_identity_and_dispatch(
    *,
    split_by: Optional[str],
    ds: xr.Dataset,
    variables: List[str],
    depth_list: List[Any],
    regions: Optional[List[Tuple[str, Dict[str, Any]]]],
    stations: Optional[List[Tuple[str, float, float]]],
    station_groups: Optional[Dict[str, List[Tuple[str, float, float]]]],
    call_fn,
    base_kwargs: Dict[str, Any],
) -> bool:
    """
    Split plotting calls by identity dimensions only.

    Returns True if splitting was performed (caller should return).
    """

    if split_by is None:
        return False

    if split_by == "variable":
        for v in variables:
            call_fn(ds, variables=[v], **base_kwargs)
        return True

    if split_by == "depth":
        for d in depth_list:
            call_fn(ds, depth=d, **base_kwargs)
        return True

    if split_by == "region":
        if regions is None:
            raise ValueError("split_by='region' requires regions=...")
        for r in regions:
            call_fn(ds, regions=[r], **base_kwargs)
        return True

    if split_by == "station":
        if stations is None and station_groups is None:
            raise ValueError("split_by='station' requires stations= or station_groups=...")
        if stations is not None:
            for s in stations:
                call_fn(ds, stations=[s], **base_kwargs)
        else:
            for group_name, grp_stations in station_groups.items():
                call_fn(ds, station_groups={group_name: grp_stations}, **base_kwargs)
        return True

    raise ValueError("split_by must be one of: None, 'variable', 'depth', 'region', 'station'")


# -----------------------
# main function
# -----------------------
def plot_bars(
    ds: xr.Dataset,
    variables: List[str],
    *,
    # spatial grouping
    regions: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
    stations: Optional[List[Tuple[str, float, float]]] = None,  # (name, lat, lon)
    station_groups: Optional[Dict[str, List[Tuple[str, float, float]]]] = None,  # {label: [(name, lat, lon), ...]}
    depth: Any = "surface",
    # time filtering (applied before aggregation)
    months: Optional[List[int]] = None,
    years: Optional[List[int]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    # season definition (runtime)
    seasons: Optional[Dict[str, Sequence[int]]] = None,
    # grouped variables
    groups: Optional[Dict[str, Any]] = None,
    split_by: Optional[str] = None,
    # plot grammar
    facet_by: Optional[str] = None,
    x_by: str = "region",
    hue_by: Optional[str] = None,
    # style
    hue_colors: Optional[Any] = None,
    error: str = "ci95",  # "sd" | "ci95"
    dpi: int = 150,
    figsize: Tuple[float, float] = (12, 8),
    ncols: Optional[int] = None,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    # output
    base_dir: str = ".",
    figures_root: str = ".",
    verbose: bool = False,
    average_by: Optional[str] = None,
) -> None:
    """Create bar-chart summaries with uncertainty.

    You can aggregate by:
      - variable (including grouped variables via `groups`)
      - region (requires `regions=...`)
      - station (requires `stations=...`)
      - depth (requires multiple depths via depth=[...])
      - day/month/year (from time coordinate)
      - season (derived from month using runtime `seasons` mapping)

    Parameters
    ----------
    ds : xr.Dataset
        FVCOM/ERSEM outputs, with a 'time' coordinate.
    variables : list[str]
        Variables or group names (resolved via resolve_da_with_depth + groups).
    regions : optional list[(name, spec)]
        Polygon specs: {"shapefile": ..., ...} or {"csv_boundary": ..., ...}.
    stations : optional list[(name, lat, lon)]
        Point stations (nearest node/element).
    depth : Any
        Either a single depth selector OR a list/tuple of selectors to enable depth grouping.
    facet_by, x_by, hue_by : str | None
        Dimension names. Must be in:
          {"variable","region","station","depth","day","month","year","season"}.
    error : "sd" | "ci95"
        Error bar definition.
    average_by : str, optional
        Temporal averaging period applied before aggregation. Resamples the
        time-filtered dataset to period means via ``xr.Dataset.resample().mean()``.
        Accepted values: ``"hour"``, ``"day"``, ``"week"``, ``"month"``,
        ``"year"`` (and common variants such as ``"daily"``, ``"monthly"``).
        Default ``None`` (no averaging).
    """

    if not _is_supported_dim(facet_by):
        raise ValueError(f"facet_by must be one of {sorted(_SUPPORTED_DIMS)} or None")
    if not _is_supported_dim(x_by):
        raise ValueError(f"x_by must be one of {sorted(_SUPPORTED_DIMS)}")
    if not _is_supported_dim(hue_by):
        raise ValueError(f"hue_by must be one of {sorted(_SUPPORTED_DIMS)} or None")
    if error not in ("sd", "ci95"):
        raise ValueError("error must be 'sd' or 'ci95'")

    # guard: ambiguous spatial grouping
    spatial_dims_used = {d for d in [facet_by, x_by, hue_by] if d in ("region", "station")}
    if "region" in spatial_dims_used and "station" in spatial_dims_used:
        raise ValueError(
            "Do not mix 'region' and 'station' within one plot (choose one spatial grouping)."
        )

    if x_by == "region" and regions is None:
        raise ValueError("x_by='region' requires regions=...")
    if x_by == "station" and stations is None and station_groups is None:
        raise ValueError("x_by='station' requires stations= or station_groups=...")

    if hue_by == "region" and regions is None:
        raise ValueError("hue_by='region' requires regions=...")
    if hue_by == "station" and stations is None and station_groups is None:
        raise ValueError("hue_by='station' requires stations= or station_groups=...")

    if facet_by == "region" and regions is None:
        raise ValueError("facet_by='region' requires regions=...")
    if facet_by == "station" and stations is None and station_groups is None:
        raise ValueError("facet_by='station' requires stations= or station_groups=...")

    # seasons default (runtime overridable)
    if seasons is None:
        seasons = {
            "Winter": [12, 1, 2],
            "Spring": [3, 4, 5],
            "Summer": [6, 7, 8],
            "Autumn": [9, 10, 11],
        }

    # depths: allow list to enable depth grouping
    depth_list: List[Any]
    if isinstance(depth, (list, tuple)):
        depth_list = list(depth)
    else:
        depth_list = [depth]

        # --------------------------------------------------
    # Split into multiple figures by identity dimension
    # --------------------------------------------------
    if _split_identity_and_dispatch(
        split_by=split_by,
        ds=ds,
        variables=variables,
        depth_list=depth_list,
        regions=regions,
        stations=stations,
        station_groups=station_groups,
        call_fn=plot_bars,
        base_kwargs=dict(
            regions=regions,
            stations=stations,
            station_groups=station_groups,
            depth=depth,
            months=months,
            years=years,
            start_date=start_date,
            end_date=end_date,
            seasons=seasons,
            groups=groups,
            facet_by=facet_by,
            x_by=x_by,
            hue_by=hue_by,
            hue_colors=hue_colors,
            error=error,
            dpi=dpi,
            figsize=figsize,
            ncols=ncols,
            title=title,
            ylabel=ylabel,
            base_dir=base_dir,
            figures_root=figures_root,
            verbose=verbose,
        ),
    ):
        return

    # output tags
    label = build_time_window_label(months, years, start_date, end_date)
    prefix = file_prefix(base_dir)
    outdir = out_dir(base_dir, figures_root)
    os.makedirs(outdir, exist_ok=True)

    # time-filter dataset first (keeps everything consistent)
    ds_t = filter_time(ds, months, years, start_date, end_date, average_by=average_by)

    # precompute region masks once
    region_names: List[str] = []
    region_masks: Dict[str, xr.DataArray] = {}
    region_masks_elem: Dict[str, xr.DataArray] = {}
    if regions is not None:
        if "lon" not in ds_t or "lat" not in ds_t:
            raise ValueError("Dataset must contain 'lon' and 'lat' for region masking.")
        for region_name, spec in regions:
            if "shapefile" in spec:
                m_nodes = polygon_mask_from_shapefile(
                    ds_t,
                    spec["shapefile"],
                    name_field=spec.get("name_field"),
                    name_equals=spec.get("name_equals"),
                )
            elif "csv_boundary" in spec:
                poly = polygon_from_csv_boundary(
                    spec["csv_boundary"],
                    lon_col=spec.get("lon_col", "lon"),
                    lat_col=spec.get("lat_col", "lat"),
                )
                m_nodes = polygon_mask(ds_t, poly)
            else:
                raise ValueError("Region spec must have 'shapefile' or 'csv_boundary'.")
            region_names.append(region_name)
            region_masks[region_name] = m_nodes
            if "nv" in ds_t:
                region_masks_elem[region_name] = element_mask_from_node_mask(m_nodes, ds_t["nv"])

    # station names (individual stations or group labels)
    station_names: List[str] = []
    if stations is not None:
        station_names = [s[0] for s in stations]
    elif station_groups is not None:
        station_names = list(station_groups.keys())

    # years ordering (optional)
    years_in_data: Optional[List[str]] = None
    if "time" in ds_t:
        try:
            t_all = pd.DatetimeIndex(ds_t["time"].to_index())
            years_in_data = sorted({str(int(y)) for y in t_all.year})
        except Exception:
            years_in_data = None

    month_levels = _month_levels()

    # helper: resolve series for a given var/depth and optionally region/station
    def _series_for_domain(var: str, depth_sel: Any) -> xr.DataArray:
        keep = depth_sel == "depth_avg"
        da = resolve_da_with_depth(ds_t, var, depth=depth_sel, groups=groups, verbose=verbose, keep_siglay=keep)
        if keep:
            da_1d = _space_mean_keep_siglay(da, ds_t, verbose=verbose)
            sq = [d for d in da_1d.dims if d not in ("time", "siglay") and da_1d.sizes[d] == 1]
            da_1d = da_1d.squeeze(sq) if sq else da_1d
        else:
            da_1d = _space_mean_over_dims(da, ds_t, verbose=verbose).squeeze()
        if "time" not in da_1d.dims:
            if "time" in da.dims:
                da_1d = da_1d.expand_dims(time=da["time"])
            else:
                raise ValueError("Result has no time dimension")
        return da_1d

    def _series_for_region(var: str, depth_sel: Any, region_name: str) -> xr.DataArray:
        keep = depth_sel == "depth_avg"
        da = resolve_da_with_depth(ds_t, var, depth=depth_sel, groups=groups, verbose=verbose, keep_siglay=keep)
        if "node" in da.dims:
            da = da.where(region_masks[region_name])
        elif "nele" in da.dims:
            if region_name not in region_masks_elem:
                raise ValueError("Element-centered variable but 'nv' missing for element mask")
            da = da.where(region_masks_elem[region_name])
        else:
            _vprint(
                verbose, f"[bars] Variable '{var}' has no node/nele dims; region masking skipped"
            )

        if keep:
            da_1d = _space_mean_keep_siglay(da, ds_t, verbose=verbose)
            sq = [d for d in da_1d.dims if d not in ("time", "siglay") and da_1d.sizes[d] == 1]
            da_1d = da_1d.squeeze(sq) if sq else da_1d
        else:
            da_1d = _space_mean_over_dims(da, ds_t, verbose=verbose).squeeze()
        if "time" not in da_1d.dims:
            if "time" in da.dims:
                da_1d = da_1d.expand_dims(time=da["time"])
            else:
                raise ValueError("Result has no time dimension")
        return da_1d

    def _series_for_station(
        var: str, depth_sel: Any, station_name: str, lon: float, lat: float
    ) -> xr.DataArray:
        keep = depth_sel == "depth_avg"
        da = resolve_da_with_depth(ds_t, var, depth=depth_sel, groups=groups, verbose=verbose, keep_siglay=keep)

        if "node" in da.dims:
            idx = _extract_station_index(ds_t, lon=lon, lat=lat, kind="node")
            da_s = da.isel(node=idx)
        elif "nele" in da.dims:
            idx = _extract_station_index(ds_t, lon=lon, lat=lat, kind="nele")
            da_s = da.isel(nele=idx)
        else:
            raise ValueError(
                f"Station selection not supported for variable '{var}' (no node/nele dim)."
            )

        if keep:
            sq = [d for d in da_s.dims if d not in ("time", "siglay") and da_s.sizes[d] == 1]
            da_s = da_s.squeeze(sq) if sq else da_s
        else:
            da_s = da_s.squeeze()
        if "time" not in da_s.dims:
            if "time" in da.dims:
                da_s = da_s.expand_dims(time=da["time"])
            else:
                raise ValueError("Result has no time dimension")
        return da_s

    # -----------------------
    # Build running stats bins
    # key is (facet_level, x_level, hue_level)
    # -----------------------
    stats: Dict[Tuple[str, str, str], _RunningStats] = {}

    # Determine which dims are used (for label computation)
    used_dims = [d for d in [facet_by, x_by, hue_by] if d is not None]

    # Precompute "levels" for identity dims (these become axis categories)
    def _levels_for_dim(dim: str) -> List[str]:
        if dim == "variable":
            return list(variables)
        if dim == "region":
            if regions is None:
                return ["domain"]
            return list(region_names)
        if dim == "station":
            if stations is None and station_groups is None:
                return []
            return list(station_names)
        if dim == "depth":
            return [depth_tag(d) for d in depth_list]
        if dim == "month":
            return month_levels
        if dim == "season":
            return list(seasons.keys())
        if dim == "year":
            if years_in_data is not None:
                return years_in_data
            return []
        if dim == "day":
            # inferred; empty means infer later
            return []
        raise ValueError(f"Unsupported dim: {dim}")

    # Identify dims that are identity vs time dims
    def _label_array_for_time_dim(dim: str, t: pd.DatetimeIndex) -> np.ndarray:
        if dim == "month":
            return _labels_month(t)
        if dim == "year":
            return _labels_year(t)
        if dim == "day":
            return _labels_day(t)
        if dim == "season":
            return _labels_season(t, seasons)
        raise ValueError(f"Not a time dim: {dim}")

    def _level_for_identity_dim(dim: str, ctx: Dict[str, str]) -> str:
        # ctx contains scalar strings for identity dims
        if dim not in ctx:
            raise KeyError(f"Missing identity dim '{dim}' in context.")
        return ctx[dim]

    # Iterate over "series units" and stream-update bins
    # series unit choices:
    # - if regions provided: each var * depth * region
    # - else if stations provided: each var * depth * station
    # - else: each var * depth * domain
    for var in variables:
        for depth_sel in depth_list:
            depth_level = depth_tag(depth_sel)

            if regions is not None:
                series_iter: Iterable[Tuple[Dict[str, str], xr.DataArray]] = []
                tmp: List[Tuple[Dict[str, str], xr.DataArray]] = []
                for region_name in region_names:
                    ctx = {
                        "variable": str(var),
                        "region": str(region_name),
                        "depth": str(depth_level),
                    }
                    s = _series_for_region(var, depth_sel, region_name)
                    tmp.append((ctx, s))
                series_iter = tmp

            elif station_groups is not None:
                tmp_grp: List[Tuple[Dict[str, str], xr.DataArray]] = []
                for group_name, grp_stations in station_groups.items():
                    for station_name, lat, lon in grp_stations:
                        ctx = {
                            "variable": str(var),
                            "station": str(group_name),
                            "depth": str(depth_level),
                        }
                        s = _series_for_station(var, depth_sel, station_name, lon, lat)
                        tmp_grp.append((ctx, s))
                series_iter = tmp_grp

            elif stations is not None:
                tmp2: List[Tuple[Dict[str, str], xr.DataArray]] = []
                for station_name, lat, lon in stations:
                    ctx = {
                        "variable": str(var),
                        "station": str(station_name),
                        "depth": str(depth_level),
                    }
                    s = _series_for_station(var, depth_sel, station_name, lon, lat)
                    tmp2.append((ctx, s))
                series_iter = tmp2

            else:
                ctx = {"variable": str(var), "region": "domain", "depth": str(depth_level)}
                s = _series_for_domain(var, depth_sel)
                series_iter = [(ctx, s)]

            for ctx, s in series_iter:
                t = _ensure_time_index(s)
                vals = np.asarray(s.values, dtype=float)

                # When depth_avg is used, siglay is preserved and vals may be
                # 2D (time, siglay). Flatten to 1D by repeating each time label
                # once per depth layer so every (time, depth) sample lands in
                # the correct time bin and contributes to the variance estimate.
                if vals.ndim > 1:
                    n_time = len(t)
                    n_depth = vals.size // n_time
                    t = pd.DatetimeIndex(np.repeat(t, n_depth))
                    vals = vals.reshape(n_time, n_depth).ravel()

                # Build time-label arrays only for time dims that are actually used
                time_labels: Dict[str, np.ndarray] = {}
                for d in used_dims:
                    if d in _TIME_DIMS:
                        time_labels[d] = _label_array_for_time_dim(d, t)

                # Determine scalar levels for identity dims (for this series)
                id_levels: Dict[str, str] = {}
                for d in used_dims:
                    if d in _ID_DIMS:
                        id_levels[d] = _level_for_identity_dim(d, ctx)

                # Now group by the time labels (if any); identity dims are constant
                # We do this per series unit to keep memory bounded.
                time_dims_used = [d for d in used_dims if d in _TIME_DIMS]

                if not time_dims_used:
                    # No time grouping at all: everything goes into one bin
                    facet_level = (
                        _level_for_identity_dim(facet_by, id_levels)
                        if facet_by in _ID_DIMS
                        else "all"
                    )
                    x_level = (
                        _level_for_identity_dim(x_by, id_levels) if x_by in _ID_DIMS else "all"
                    )
                    hue_level = (
                        _level_for_identity_dim(hue_by, id_levels) if hue_by in _ID_DIMS else "_"
                    )
                    key = (str(facet_level), str(x_level), str(hue_level))
                    stats.setdefault(key, _RunningStats()).update(vals)
                    continue

                # Build per-sample labels for the three roles, to create bins
                # Each role is either scalar (identity) or per-sample (time).
                def _role_labels(role_dim: Optional[str]) -> Union[str, np.ndarray]:
                    if role_dim is None:
                        return "_"
                    if role_dim in _ID_DIMS:
                        return id_levels[role_dim]
                    return time_labels[role_dim]

                facet_lab = _role_labels(facet_by) if facet_by is not None else "all"
                x_lab = _role_labels(x_by)
                hue_lab = _role_labels(hue_by) if hue_by is not None else "_"

                # Normalize to arrays (same length) for grouping
                # Scalars become repeated arrays to simplify grouping code.
                def _to_array(x: Union[str, np.ndarray]) -> np.ndarray:
                    if isinstance(x, np.ndarray):
                        return x
                    return np.asarray([x] * len(t), dtype=object)

                facet_arr = _to_array(facet_lab)
                x_arr = _to_array(x_lab)
                hue_arr = _to_array(hue_lab)

                # Drop NaNs before grouping to avoid skewing counts
                m = np.isfinite(vals)
                if not np.any(m):
                    continue
                facet_arr = facet_arr[m]
                x_arr = x_arr[m]
                hue_arr = hue_arr[m]
                v = vals[m]

                # Group indices by (facet, x, hue)
                # Using pandas MultiIndex for robust grouping on object arrays.
                mi = pd.MultiIndex.from_arrays(
                    [facet_arr, x_arr, hue_arr], names=["facet", "x", "hue"]
                )
                # factorize returns codes per sample and uniques per group
                codes, uniques = pd.factorize(mi)

                for gi, u in enumerate(uniques):
                    sel = codes == gi
                    key = (str(u[0]), str(u[1]), str(u[2]))
                    stats.setdefault(key, _RunningStats()).update(v[sel])

    # If no data, exit early
    if not stats:
        _vprint(verbose, "[bars] No data after filtering/grouping; nothing to plot.")
        return

    # -----------------------
    # Determine facet/x/hue levels for plotting order
    # -----------------------
    def _infer_levels(role_dim: Optional[str], role_name: str) -> List[str]:
        if role_dim is None:
            return ["all"] if role_name == "facet" else ["_"]

        # explicit levels for some dims
        explicit = _dim_levels_from_data(
            role_dim,
            seasons=seasons,
            month_levels=month_levels,
            years=years_in_data,
        )
        # preserve user-supplied region / variable order
        if explicit is None and role_dim == "region":
            explicit = region_names
        if explicit is None and role_dim == "variable":
            explicit = list(variables)

        # role-specific present levels:
        idx = {"facet": 0, "x": 1, "hue": 2}[role_name]
        present = sorted({k[idx] for k in stats.keys()})

        # apply explicit ordering if available
        if explicit is not None and len(explicit) > 0:
            present_set = set(present)
            ordered = [v for v in explicit if v in present_set]
            # include any leftovers at the end
            leftovers = [v for v in present if v not in set(ordered)]
            return ordered + leftovers

        # otherwise use present (sorted)
        return present

    facet_levels = _infer_levels(facet_by, "facet")
    x_levels = _infer_levels(x_by, "x")
    hue_levels = _infer_levels(hue_by, "hue") if hue_by is not None else ["_"]

    # Default facet layout
    if ncols is None:
        ncols = min(4, max(1, len(facet_levels)))
    ncols_use = int(ncols)
    nrows_use = int(math.ceil(len(facet_levels) / ncols_use))

    fig, axes = plt.subplots(nrows_use, ncols_use, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel().tolist()

    # Color mapping for hue (ignore "_" if present)
    hue_for_colors = [h for h in hue_levels if h != "_"]
    hue_color_map = _pick_color_map(hue_for_colors, hue_colors)

    # -----------------------
    # Plot each facet
    # -----------------------
    for ax_i, facet_level in enumerate(facet_levels):
        ax = axes_flat[ax_i]
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", alpha=0.25)

        if hue_by is None:
            bar_width = 0.7
            x_pos = np.arange(len(x_levels), dtype=float)

            y = []
            e = []
            for xl in x_levels:
                rs = stats.get((facet_level, xl, "_"), _RunningStats())
                y.append(rs.mean())
                e.append(rs.err(error))

            ax.bar(x_pos, y, width=bar_width)
            ax.errorbar(x_pos, y, yerr=e, fmt="none", capsize=3, color="black")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_levels, rotation=45, ha="right")

        else:
            n_h = len([h for h in hue_levels if h != "_"])
            n_h = max(1, n_h)
            group_width = 0.8
            bar_width = group_width / float(n_h)
            x_pos = np.arange(len(x_levels), dtype=float)

            j = 0
            for hue_level in hue_levels:
                if hue_level == "_":
                    continue
                offset = (j - (n_h - 1) / 2.0) * bar_width
                xpos_j = x_pos + offset

                y = []
                e = []
                for xl in x_levels:
                    rs = stats.get((facet_level, xl, hue_level), _RunningStats())
                    y.append(rs.mean())
                    e.append(rs.err(error))

                color = hue_color_map.get(hue_level, None)
                ax.bar(xpos_j, y, width=bar_width * 0.95, label=hue_level, color=color)
                ax.errorbar(xpos_j, y, yerr=e, fmt="none", capsize=3, color="black")
                j += 1

            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_levels, rotation=45, ha="right")
            ax.legend(fontsize=9)

        # facet title
        if facet_by is None:
            ax.set_title("all")
        else:
            ax.set_title(f"{facet_by}: {facet_level}")

        # y label
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        else:
            # keep it simple; variable labels can be handled by title
            ax.set_ylabel("Mean")

    # Hide unused axes
    for j in range(len(facet_levels), len(axes_flat)):
        axes_flat[j].axis("off")

    # Figure title / output naming
    # If multiple depths, don't bake a single depth tag into title; include "multi-depth".
    depth_part = depth_tag(depth_list[0]) if len(depth_list) == 1 else "multi-depth"
    if title is None:
        fig_title = f"{', '.join(variables)} ({depth_part}, {label})"
    else:
        fig_title = title
    fig.suptitle(fig_title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # filename
    # filename
    if len(variables) == 1:
        var_part = variables[0]
    else:
        var_part = f"vars-{len(variables)}"

    fname = f"{prefix}__Bars__{var_part}__{depth_part}__{label}"

    fname += f"__x-{x_by}"
    if hue_by is not None:
        fname += f"__hue-{hue_by}"
    if facet_by is not None:
        fname += f"__facet-{facet_by}"
    fname += ".png"

    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    _vprint(verbose, f"[bars] Saved: {path}")


# -----------------------
# box + whisker plots (streaming with reservoir sampling)
# -----------------------
@dataclass
class _Reservoir:
    """Reservoir sampler to keep at most k samples per bin (memory safety)."""

    k: int
    n_seen: int = 0
    data: List[float] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.data is None:
            self.data = []

    def update(self, x: np.ndarray, rng: np.random.Generator) -> None:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return

        for v in x:
            self.n_seen += 1
            if len(self.data) < self.k:
                self.data.append(float(v))
            else:
                # replace with decreasing probability
                j = int(rng.integers(0, self.n_seen))
                if j < self.k:
                    self.data[j] = float(v)

    def values(self) -> np.ndarray:
        if not self.data:
            return np.asarray([], dtype=float)
        return np.asarray(self.data, dtype=float)


def plot_box(
    ds: xr.Dataset,
    variables: List[str],
    *,
    # spatial grouping
    regions: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
    stations: Optional[List[Tuple[str, float, float]]] = None,  # (name, lat, lon)
    station_groups: Optional[Dict[str, List[Tuple[str, float, float]]]] = None,  # {label: [(name, lat, lon), ...]}
    depth: Any = "surface",
    # time filtering (applied before aggregation)
    months: Optional[List[int]] = None,
    years: Optional[List[int]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    # season definition (runtime)
    seasons: Optional[Dict[str, Sequence[int]]] = None,
    # grouped variables
    groups: Optional[Dict[str, Any]] = None,
    split_by: Optional[str] = None,
    # plot grammar
    facet_by: Optional[str] = None,
    x_by: str = "region",
    hue_by: Optional[str] = None,
    # style
    hue_colors: Optional[Any] = None,
    dpi: int = 150,
    figsize: Tuple[float, float] = (12, 8),
    ncols: Optional[int] = None,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    # NEW: per-variable y-axis labels when facet_by == "variable"
    ylabels: Optional[Dict[str, str]] = None,
    # sampling / memory safety
    max_samples_per_bin: int = 20000,
    random_seed: int = 0,
    # output
    base_dir: str = ".",
    figures_root: str = ".",
    verbose: bool = False,
    average_by: Optional[str] = None,
) -> None:
    """Create box-and-whisker summaries across time samples.

    This uses reservoir sampling per (facet, x, hue) bin so you can plot boxplots
    without storing every time sample for massive datasets.

    Parameters
    ----------
    average_by : str, optional
        Temporal averaging period applied before aggregation. Resamples the
        time-filtered dataset to period means via ``xr.Dataset.resample().mean()``.
        Accepted values: ``"hour"``, ``"day"``, ``"week"``, ``"month"``,
        ``"year"`` (and common variants such as ``"daily"``, ``"monthly"``).
        Default ``None`` (no averaging).

    Notes
    -----
    - Per-facet y-axis labeling when facet_by == "variable" via ylabels mapping.
    - No "Unknown" / white boxes: if grouping by season, samples not mapped to a season
      are dropped; hue plotting uses only explicit hue levels.
    """

    if not _is_supported_dim(facet_by):
        raise ValueError(f"facet_by must be one of {sorted(_SUPPORTED_DIMS)} or None")
    if not _is_supported_dim(x_by):
        raise ValueError(f"x_by must be one of {sorted(_SUPPORTED_DIMS)}")
    if not _is_supported_dim(hue_by):
        raise ValueError(f"hue_by must be one of {sorted(_SUPPORTED_DIMS)} or None")

    spatial_dims_used = {d for d in [facet_by, x_by, hue_by] if d in ("region", "station")}
    if "region" in spatial_dims_used and "station" in spatial_dims_used:
        raise ValueError(
            "Do not mix 'region' and 'station' within one plot (choose one spatial grouping)."
        )

    if x_by == "region" and regions is None:
        raise ValueError("x_by='region' requires regions=...")
    if x_by == "station" and stations is None and station_groups is None:
        raise ValueError("x_by='station' requires stations= or station_groups=...")

    if hue_by == "region" and regions is None:
        raise ValueError("hue_by='region' requires regions=...")
    if hue_by == "station" and stations is None and station_groups is None:
        raise ValueError("hue_by='station' requires stations= or station_groups=...")

    if facet_by == "region" and regions is None:
        raise ValueError("facet_by='region' requires regions=...")
    if facet_by == "station" and stations is None and station_groups is None:
        raise ValueError("facet_by='station' requires stations= or station_groups=...")

    if seasons is None:
        seasons = {
            "Winter": [12, 1, 2],
            "Spring": [3, 4, 5],
            "Summer": [6, 7, 8],
            "Autumn": [9, 10, 11],
        }

    depth_list: List[Any]
    if isinstance(depth, (list, tuple)):
        depth_list = list(depth)
    else:
        depth_list = [depth]

        # --------------------------------------------------
    # Split into multiple figures by identity dimension
    # --------------------------------------------------
    if _split_identity_and_dispatch(
        split_by=split_by,
        ds=ds,
        variables=variables,
        depth_list=depth_list,
        regions=regions,
        stations=stations,
        station_groups=station_groups,
        call_fn=plot_box,
        base_kwargs=dict(
            regions=regions,
            stations=stations,
            station_groups=station_groups,
            depth=depth,
            months=months,
            years=years,
            start_date=start_date,
            end_date=end_date,
            seasons=seasons,
            groups=groups,
            facet_by=facet_by,
            x_by=x_by,
            hue_by=hue_by,
            hue_colors=hue_colors,
            dpi=dpi,
            figsize=figsize,
            ncols=ncols,
            title=title,
            ylabel=ylabel,
            ylabels=ylabels,
            max_samples_per_bin=max_samples_per_bin,
            random_seed=random_seed,
            base_dir=base_dir,
            figures_root=figures_root,
            verbose=verbose,
        ),
    ):
        return

    label = build_time_window_label(months, years, start_date, end_date)
    prefix = file_prefix(base_dir)
    outdir = out_dir(base_dir, figures_root)
    os.makedirs(outdir, exist_ok=True)

    ds_t = filter_time(ds, months, years, start_date, end_date, average_by=average_by)

    # precompute region masks once
    region_names: List[str] = []
    region_masks: Dict[str, xr.DataArray] = {}
    region_masks_elem: Dict[str, xr.DataArray] = {}
    if regions is not None:
        if "lon" not in ds_t or "lat" not in ds_t:
            raise ValueError("Dataset must contain 'lon' and 'lat' for region masking.")
        for region_name, spec in regions:
            if "shapefile" in spec:
                m_nodes = polygon_mask_from_shapefile(
                    ds_t,
                    spec["shapefile"],
                    name_field=spec.get("name_field"),
                    name_equals=spec.get("name_equals"),
                )
            elif "csv_boundary" in spec:
                poly = polygon_from_csv_boundary(
                    spec["csv_boundary"],
                    lon_col=spec.get("lon_col", "lon"),
                    lat_col=spec.get("lat_col", "lat"),
                )
                m_nodes = polygon_mask(ds_t, poly)
            else:
                raise ValueError("Region spec must have 'shapefile' or 'csv_boundary'.")
            region_names.append(region_name)
            region_masks[region_name] = m_nodes
            if "nv" in ds_t:
                region_masks_elem[region_name] = element_mask_from_node_mask(m_nodes, ds_t["nv"])

    station_names: List[str] = []
    if stations is not None:
        station_names = [s[0] for s in stations]
    elif station_groups is not None:
        station_names = list(station_groups.keys())

    years_in_data: Optional[List[str]] = None
    if "time" in ds_t:
        try:
            t_all = pd.DatetimeIndex(ds_t["time"].to_index())
            years_in_data = sorted({str(int(y)) for y in t_all.year})
        except Exception:
            years_in_data = None

    month_levels = _month_levels()

    rng = np.random.default_rng(int(random_seed))

    def _series_for_domain(var: str, depth_sel: Any) -> xr.DataArray:
        da = resolve_da_with_depth(ds_t, var, depth=depth_sel, groups=groups, verbose=verbose)
        da_1d = _space_mean_over_dims(da, ds_t, verbose=verbose).squeeze()
        if "time" not in da_1d.dims:
            if "time" in da.dims:
                da_1d = da_1d.expand_dims(time=da["time"])
            else:
                raise ValueError("Result has no time dimension")
        return da_1d

    def _series_for_region(var: str, depth_sel: Any, region_name: str) -> xr.DataArray:
        da = resolve_da_with_depth(ds_t, var, depth=depth_sel, groups=groups, verbose=verbose)
        if "node" in da.dims:
            da = da.where(region_masks[region_name])
        elif "nele" in da.dims:
            if region_name not in region_masks_elem:
                raise ValueError("Element-centered variable but 'nv' missing for element mask")
            da = da.where(region_masks_elem[region_name])
        else:
            _vprint(
                verbose, f"[box] Variable '{var}' has no node/nele dims; region masking skipped"
            )
        da_1d = _space_mean_over_dims(da, ds_t, verbose=verbose).squeeze()
        if "time" not in da_1d.dims:
            if "time" in da.dims:
                da_1d = da_1d.expand_dims(time=da["time"])
            else:
                raise ValueError("Result has no time dimension")
        return da_1d

    def _series_for_station(
        var: str, depth_sel: Any, station_name: str, lon: float, lat: float
    ) -> xr.DataArray:
        da = resolve_da_with_depth(ds_t, var, depth=depth_sel, groups=groups, verbose=verbose)
        if "node" in da.dims:
            idx = _extract_station_index(ds_t, lon=lon, lat=lat, kind="node")
            da_s = da.isel(node=idx)
        elif "nele" in da.dims:
            idx = _extract_station_index(ds_t, lon=lon, lat=lat, kind="nele")
            da_s = da.isel(nele=idx)
        else:
            raise ValueError(
                f"Station selection not supported for variable '{var}' (no node/nele dim)."
            )
        da_s = da_s.squeeze()
        if "time" not in da_s.dims:
            if "time" in da.dims:
                da_s = da_s.expand_dims(time=da["time"])
            else:
                raise ValueError("Result has no time dimension")
        return da_s

    used_dims = [d for d in [facet_by, x_by, hue_by] if d is not None]

    def _label_array_for_time_dim(dim: str, t: pd.DatetimeIndex) -> np.ndarray:
        if dim == "month":
            return _labels_month(t)
        if dim == "year":
            return _labels_year(t)
        if dim == "day":
            return _labels_day(t)
        if dim == "season":
            # IMPORTANT: do not create "Unknown" bins
            # return np.nan for unmapped months so we can drop them
            m2s: Dict[int, str] = {}
            for lab, mons in seasons.items():
                for mm in mons:
                    m2s[int(mm)] = str(lab)
            out = np.empty(len(t), dtype=object)
            for i, mm in enumerate(t.month):
                out[i] = m2s.get(int(mm), None)
            return out
        raise ValueError(f"Not a time dim: {dim}")

    # Reservoirs per bin
    bins: Dict[Tuple[str, str, str], _Reservoir] = {}

    for var in variables:
        for depth_sel in depth_list:
            depth_level = depth_tag(depth_sel)

            if regions is not None:
                tmp: List[Tuple[Dict[str, str], xr.DataArray]] = []
                for region_name in region_names:
                    ctx = {
                        "variable": str(var),
                        "region": str(region_name),
                        "depth": str(depth_level),
                    }
                    s = _series_for_region(var, depth_sel, region_name)
                    tmp.append((ctx, s))
                series_iter = tmp

            elif station_groups is not None:
                tmp_grp2: List[Tuple[Dict[str, str], xr.DataArray]] = []
                for group_name, grp_stations in station_groups.items():
                    for station_name, lat, lon in grp_stations:
                        ctx = {
                            "variable": str(var),
                            "station": str(group_name),
                            "depth": str(depth_level),
                        }
                        s = _series_for_station(var, depth_sel, station_name, lon, lat)
                        tmp_grp2.append((ctx, s))
                series_iter = tmp_grp2

            elif stations is not None:
                tmp2: List[Tuple[Dict[str, str], xr.DataArray]] = []
                for station_name, lat, lon in stations:
                    ctx = {
                        "variable": str(var),
                        "station": str(station_name),
                        "depth": str(depth_level),
                    }
                    s = _series_for_station(var, depth_sel, station_name, lon, lat)
                    tmp2.append((ctx, s))
                series_iter = tmp2

            else:
                ctx = {"variable": str(var), "region": "domain", "depth": str(depth_level)}
                s = _series_for_domain(var, depth_sel)
                series_iter = [(ctx, s)]

            for ctx, s in series_iter:
                t = _ensure_time_index(s)
                vals = np.asarray(s.values, dtype=float)

                # When depth_avg is used, siglay is preserved and vals may be
                # 2D (time, siglay). Flatten to 1D by repeating each time label
                # once per depth layer so every (time, depth) sample lands in
                # the correct time bin and contributes to the variance estimate.
                if vals.ndim > 1:
                    n_time = len(t)
                    n_depth = vals.size // n_time
                    t = pd.DatetimeIndex(np.repeat(t, n_depth))
                    vals = vals.reshape(n_time, n_depth).ravel()

                time_labels: Dict[str, np.ndarray] = {}
                for d in used_dims:
                    if d in _TIME_DIMS:
                        time_labels[d] = _label_array_for_time_dim(d, t)

                # identity levels for this series
                id_levels: Dict[str, str] = {}
                for d in used_dims:
                    if d in _ID_DIMS:
                        if d not in ctx:
                            raise KeyError(f"Missing identity dim '{d}' in context.")
                        id_levels[d] = ctx[d]

                def _role_labels(role_dim: Optional[str]) -> Union[str, np.ndarray]:
                    if role_dim is None:
                        return "_"
                    if role_dim in _ID_DIMS:
                        return id_levels[role_dim]
                    return time_labels[role_dim]

                facet_lab = _role_labels(facet_by) if facet_by is not None else "all"
                x_lab = _role_labels(x_by)
                hue_lab = _role_labels(hue_by) if hue_by is not None else "_"

                def _to_array(x: Union[str, np.ndarray]) -> np.ndarray:
                    if isinstance(x, np.ndarray):
                        return x
                    return np.asarray([x] * len(t), dtype=object)

                facet_arr = _to_array(facet_lab)
                x_arr = _to_array(x_lab)
                hue_arr = _to_array(hue_lab)

                # numeric finite mask
                m = np.isfinite(vals)
                if not np.any(m):
                    continue

                # drop unmapped season labels (None) for any role that uses season
                # (prevents "Unknown" / white boxes)
                if facet_by == "season":
                    m = m & np.not_equal(facet_arr, None)
                if x_by == "season":
                    m = m & np.not_equal(x_arr, None)
                if hue_by == "season":
                    m = m & np.not_equal(hue_arr, None)

                if not np.any(m):
                    continue

                facet_arr = facet_arr[m]
                x_arr = x_arr[m]
                hue_arr = hue_arr[m]
                v = vals[m]

                mi = pd.MultiIndex.from_arrays(
                    [facet_arr, x_arr, hue_arr], names=["facet", "x", "hue"]
                )
                codes, uniques = pd.factorize(mi)

                for gi, u in enumerate(uniques):
                    sel = codes == gi
                    key = (str(u[0]), str(u[1]), str(u[2]))
                    if key not in bins:
                        bins[key] = _Reservoir(k=int(max_samples_per_bin))
                    bins[key].update(v[sel], rng)

    if not bins:
        _vprint(verbose, "[box] No data after filtering/grouping; nothing to plot.")
        return

    def _infer_levels(role_dim: Optional[str], role_name: str) -> List[str]:
        if role_dim is None:
            return ["all"] if role_name == "facet" else ["_"]

        explicit = _dim_levels_from_data(
            role_dim,
            seasons=seasons,
            month_levels=month_levels,
            years=years_in_data,
        )
        # preserve user-supplied region / variable order
        if explicit is None and role_dim == "region":
            explicit = region_names
        if explicit is None and role_dim == "variable":
            explicit = list(variables)

        idx = {"facet": 0, "x": 1, "hue": 2}[role_name]
        present = sorted({k[idx] for k in bins.keys()})

        if explicit is not None and len(explicit) > 0:
            present_set = set(present)
            ordered = [v for v in explicit if v in present_set]
            leftovers = [v for v in present if v not in set(ordered)]
            return ordered + leftovers

        return present

    facet_levels = _infer_levels(facet_by, "facet")
    x_levels = _infer_levels(x_by, "x")
    hue_levels = _infer_levels(hue_by, "hue") if hue_by is not None else ["_"]

    if ncols is None:
        ncols = min(4, max(1, len(facet_levels)))
    ncols_use = int(ncols)
    nrows_use = int(math.ceil(len(facet_levels) / ncols_use))

    fig, axes = plt.subplots(nrows_use, ncols_use, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel().tolist()

    hue_no_blank = [h for h in hue_levels if h != "_"]
    hue_color_map = _pick_color_map(hue_no_blank, hue_colors)

    lineprops = {"color": "black"}
    medianprops = {"color": "black"}
    flierprops = {"markeredgecolor": "black"}

    for ax_i, facet_level in enumerate(facet_levels):
        ax = axes_flat[ax_i]
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", alpha=0.25)

        if hue_by is None:
            positions = np.arange(len(x_levels), dtype=float)
            data = [bins.get((facet_level, xl, "_"), _Reservoir(k=1)).values() for xl in x_levels]

            ax.boxplot(
                data,
                positions=positions,
                widths=0.6,
                patch_artist=True,
                boxprops={"facecolor": "white", "edgecolor": "black"},
                whiskerprops=lineprops,
                capprops=lineprops,
                medianprops=medianprops,
                flierprops=flierprops,
            )

            ax.set_xticks(positions)
            ax.set_xticklabels(x_levels, rotation=45, ha="right")

        else:
            n_h = max(1, len(hue_no_blank))
            group_width = 0.8
            box_width = group_width / float(n_h)
            x_pos = np.arange(len(x_levels), dtype=float)

            for j, hue_level in enumerate(hue_no_blank):
                offset = (j - (n_h - 1) / 2.0) * box_width
                positions = x_pos + offset
                data = [
                    bins.get((facet_level, xl, hue_level), _Reservoir(k=1)).values()
                    for xl in x_levels
                ]

                color = hue_color_map.get(hue_level, None)
                if color is None:
                    # If user passed hue_by but didn't provide a color for a hue, keep it white
                    # but this should not happen for "Unknown" now (we drop those samples).
                    color = "white"

                ax.boxplot(
                    data,
                    positions=positions,
                    widths=box_width * 0.9,
                    patch_artist=True,
                    boxprops={"facecolor": color, "edgecolor": "black"},
                    whiskerprops=lineprops,
                    capprops=lineprops,
                    medianprops=medianprops,
                    flierprops=flierprops,
                )

            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_levels, rotation=45, ha="right")

            handles = []
            labels = []
            for hue_level in hue_no_blank:
                c = hue_color_map.get(hue_level, None)
                if c is None:
                    continue
                handles.append(plt.Line2D([0], [0], color=c, linewidth=8))
                labels.append(hue_level)
            if handles:
                ax.legend(handles, labels, fontsize=9)

        if facet_by is None:
            ax.set_title("all")
        else:
            ax.set_title(f"{facet_by}: {facet_level}")

        # Y-axis labeling:
        # - if ylabel explicitly provided, use it for all panels
        # - else if facet_by == "variable" and ylabels provided, use ylabels[facet_level]
        # - else fall back to "Value"
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        else:
            if facet_by == "variable" and ylabels is not None:
                ax.set_ylabel(ylabels.get(str(facet_level), str(facet_level)))
            else:
                ax.set_ylabel("Value")

    for j in range(len(facet_levels), len(axes_flat)):
        axes_flat[j].axis("off")

    depth_part = depth_tag(depth_list[0]) if len(depth_list) == 1 else "multi-depth"
    if title is None:
        fig_title = f"{', '.join(variables)} ({depth_part}, {label})"
    else:
        fig_title = title
    fig.suptitle(fig_title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if len(variables) == 1:
        var_part = variables[0]
    else:
        var_part = f"vars-{len(variables)}"

    fname = f"{prefix}__Box__{var_part}__{depth_part}__{label}"

    fname += f"__x-{x_by}"
    if hue_by is not None:
        fname += f"__hue-{hue_by}"
    if facet_by is not None:
        fname += f"__facet-{facet_by}"
    fname += ".png"

    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    _vprint(verbose, f"[box] Saved: {path}")
