# fvcomersemviz/plots/composition.py
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List, Sequence
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from ..io import filter_time
from ..regions import apply_scope
from ..utils import (
    out_dir, file_prefix,
    select_depth, build_time_window_label,
    resolve_available_vars, sum_over_all_dims,
)
from ..plot import stacked_fraction_bars


def _fractional_breakdown(ds_depth: xr.Dataset, var_names: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    """Compute pooled sum for each var, then normalize to fractions."""
    labels: List[str] = list(var_names)
    totals = []
    for v in var_names:
        if v not in ds_depth:
            totals.append(np.nan)
        else:
            totals.append(sum_over_all_dims(ds_depth[v]))
    totals = np.array(totals, dtype=float)

    if not np.isfinite(totals).any():
        return np.full(len(labels), np.nan), labels

    totals[~np.isfinite(totals)] = 0.0
    s = totals.sum()
    if s <= 0:
        return np.zeros_like(totals), labels
    return (totals / s), labels


def _depth_average_dataset(ds_scoped: xr.Dataset, *, verbose: bool = False) -> xr.Dataset:
    """
    Try select_depth(..., 'depth_avg'); if unsupported, compute an unweighted mean over 'siglay'.
    If 'siglay' is absent, return ds_scoped unchanged.
    """
    try:
        return select_depth(ds_scoped, "depth_avg", verbose=verbose)
    except Exception:
        # manual averaging across layers
        if "siglay" in ds_scoped.dims:
            return ds_scoped.mean("siglay", skipna=True)
        return ds_scoped


# =============================================================================
# Public API (single-axes figures)
# =============================================================================

def composition_surface_bottom(
    ds: xr.Dataset,
    *,
    months: Optional[List[int]] = None,
    years: Optional[List[int]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    region: Optional[Tuple[str, Dict[str, Any]]] = None,
    station: Optional[Tuple[str, float, float]] = None,  # (name, lat, lon)
    base_dir: str,
    figures_root: str,
    phyto_vars: Sequence[str],
    zoo_vars: Sequence[str],
    dpi: int = 150,
    figsize: Tuple[float, float] = (9, 6),
    verbose: bool = False,
) -> None:
    """
    Plot a single stacked-bar figure comparing **surface vs bottom composition**
    for phytoplankton and zooplankton.

    The figure has **four bars**:
      [Surface Phyto] [Bottom Phyto] [Surface Zoo] [Bottom Zoo]

    For each bar, variables in the corresponding group are pooled (summed) over
    the selected time window and scope, then normalized so the bar **sums to 1.0**.
    Missing variables are tolerated and contribute 0 when normalizing.

    Workflow
    --------
    1. Apply time filters via ``filter_time``.
    2. Apply spatial scope via ``apply_scope`` (domain / region / station).
    3. Resolve available variables with ``resolve_available_vars``.
    4. Select depths: ``select_depth(..., "surface")`` and ``select_depth(..., "bottom")``.
    5. Compute pooled fractions with ``_fractional_breakdown``.
    6. Render stacked bars using ``stacked_fraction_bars`` and save a PNG.

    Parameters
    ----------
    ds : xr.Dataset
        Source dataset with the variables listed in ``phyto_vars`` / ``zoo_vars``.
    months, years : list[int], optional
        Time filters (calendar months 1–12; year integers). Can be combined.
    start_date, end_date : str, optional
        Inclusive date bounds "YYYY-MM-DD".
    region : (str, dict), optional
        Region scope as ``(region_name, spec)`` accepted by ``apply_scope``.
    station : (str, float, float), optional
        Station scope as ``(name, lat, lon)``; selects nearest column.
    base_dir : str
        Run root; used by ``file_prefix(base_dir)`` for filenames.
    figures_root : str
        Root directory where the PNG is written.
    phyto_vars : Sequence[str]
        Candidate phytoplankton variable names; only those present are used.
    zoo_vars : Sequence[str]
        Candidate zooplankton variable names; only those present are used.
    dpi : int, default 150
        Output PNG resolution.
    figsize : (float, float), default (9, 6)
        Figure size in inches.
    verbose : bool, default False
        Print progress (including saved path).

    Returns
    -------
    None
        Saves a PNG named like:
        ``<prefix>__Composition__SurfBottom__<ScopeTag>__<TimeTag>.png``

    Notes
    -----
    - Title annotates who (Domain/Region/Station) and when (time label).
    - Bars convey **relative dominance**; pair with separate totals if absolute
      magnitudes matter.
    """
    # time & scope
    ds_t = filter_time(ds, months=months, years=years, start_date=start_date, end_date=end_date)
    ds_scoped = apply_scope(ds_t, region=region, station=station, verbose=verbose)

    # variables present
    phyto_vars = resolve_available_vars(ds_scoped, phyto_vars)
    zoo_vars   = resolve_available_vars(ds_scoped,   zoo_vars)

    # depth slices
    ds_surf = select_depth(ds_scoped, "surface", verbose=verbose)
    ds_bott = select_depth(ds_scoped, "bottom",  verbose=verbose)

    # pooled fractions
    phyto_s, phyto_labels = _fractional_breakdown(ds_surf, phyto_vars)
    phyto_b, _            = _fractional_breakdown(ds_bott, phyto_vars)
    zoo_s,   zoo_labels   = _fractional_breakdown(ds_surf, zoo_vars)
    zoo_b,   _            = _fractional_breakdown(ds_bott, zoo_vars)

    # figure – single axes with four bars
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    who = "Domain" if region is None and station is None else (
        (region[0] if region is not None else f"Station {station[0]}")
    )
    when = build_time_window_label(months, years, start_date, end_date)
    fig.suptitle(f"Phyto / Zoo Composition - {who} - {when}", fontsize=12)

    bars = [phyto_s, phyto_b, zoo_s, zoo_b]
    labels_per_bar = [phyto_labels, phyto_labels, zoo_labels, zoo_labels]
    bar_names = ["Surface Phyto", "Bottom Phyto", "Surface Zoo", "Bottom Zoo"]

    stacked_fraction_bars(
        ax,
        bars,
        labels_per_bar,
        bar_names=bar_names,
        y_label="Fraction of group",
        show_legend=True,
        xtick_rotation=45.0,
    )

    # save
    out_root = out_dir(base_dir, figures_root)
    outdir = out_root
    os.makedirs(outdir, exist_ok=True)
    prefix = file_prefix(base_dir)
    scope_tag = who.replace(" ", "_")
    fname = f"{prefix}__Composition__SurfBottom__{scope_tag}__{when}.png"
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"[composition] saved {path}")


def composition_depth_average_single(
    ds: xr.Dataset,
    *,
    months: Optional[List[int]] = None,
    years: Optional[List[int]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    region: Optional[Tuple[str, Dict[str, Any]]] = None,
    station: Optional[Tuple[str, float, float]] = None,
    base_dir: str,
    figures_root: str,
    phyto_vars: Sequence[str],
    zoo_vars: Sequence[str],
    dpi: int = 150,
    figsize: Tuple[float, float] = (7, 5),
    verbose: bool = False,
) -> None:
    """
    Plot a single stacked-bar figure showing **depth-averaged composition**
    for phytoplankton and zooplankton.

    The figure has **two bars**:
      [Phyto (depth-avg)] [Zoo (depth-avg)]

    Depth-averaging behavior:
      - Tries ``select_depth(..., "depth_avg")`` (may be thickness-weighted if supported).
      - If not supported, falls back to **unweighted mean over ``siglay``** (if present).

    Workflow
    --------
    1. Apply time filters and spatial scope (as in the surface/bottom function).
    2. Resolve available variables for phyto and zoo groups.
    3. Produce a depth-averaged dataset (weighted if supported; else unweighted).
    4. Compute pooled fractions via ``_fractional_breakdown`` for each group.
    5. Render **two** stacked bars with legend outside and save a PNG.

    Parameters
    ----------
    ds : xr.Dataset
        Source dataset.
    months, years : list[int], optional
        Time filters (calendar-based).
    start_date, end_date : str, optional
        Inclusive date bounds "YYYY-MM-DD".
    region : (str, dict), optional
        Region scope accepted by ``apply_scope``.
    station : (str, float, float), optional
        Station scope (nearest column).
    base_dir : str
        Run root for filename prefixing.
    figures_root : str
        Output root directory.
    phyto_vars : Sequence[str]
        Candidate phytoplankton variable names; subset used if present.
    zoo_vars : Sequence[str]
        Candidate zooplankton variable names; subset used if present.
    dpi : int, default 150
        Output PNG resolution.
    figsize : (float, float), default (7, 5)
        Figure size in inches.
    verbose : bool, default False
        Print progress (including saved path).

    Returns
    -------
    None
        Saves a PNG named like:
        ``<prefix>__Composition__DepthAvg__<ScopeTag>__<TimeTag>.png``

    Notes
    -----
    - If neither ``select_depth("depth_avg")`` nor a ``siglay`` dimension is available,
      the function uses the dataset unchanged (no vertical aggregation).
    - Fractions sum to 1.0 within each bar; missing variables are tolerated.
    """
    # time & scope
    ds_t = filter_time(ds, months=months, years=years, start_date=start_date, end_date=end_date)
    ds_scoped = apply_scope(ds_t, region=region, station=station, verbose=verbose)

    # variables present
    phyto_vars = resolve_available_vars(ds_scoped, phyto_vars)
    zoo_vars   = resolve_available_vars(ds_scoped,   zoo_vars)

    # depth-average (weighted if supported; else manual mean over siglay)
    def _depth_average_dataset(ds_scoped: xr.Dataset) -> xr.Dataset:
        try:
            return select_depth(ds_scoped, "depth_avg", verbose=verbose)
        except Exception:
            return ds_scoped.mean("siglay", skipna=True) if "siglay" in ds_scoped.dims else ds_scoped

    ds_avg = _depth_average_dataset(ds_scoped)

    # pooled fractions
    phyto_f, phyto_labels = _fractional_breakdown(ds_avg, phyto_vars)
    zoo_f,   zoo_labels   = _fractional_breakdown(ds_avg, zoo_vars)

    # figure – single axes with two bars
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    who = "Domain" if region is None and station is None else (
        (region[0] if region is not None else f"Station {station[0]}")
    )
    when = build_time_window_label(months, years, start_date, end_date)
    fig.suptitle(f"Depth-averaged Composition - {who} - {when}", fontsize=12)

    bars = [phyto_f, zoo_f]
    labels_per_bar = [phyto_labels, zoo_labels]
    bar_names = ["Phyto (avg)", "Zoo (avg)"]

    stacked_fraction_bars(
        ax,
        bars,
        labels_per_bar,
        bar_names=bar_names,
        y_label="Fraction of group",
        show_legend=True,
        xtick_rotation=0.0,
        bar_width=0.30,          # thinner bars
        legend_outside=True,     # legend docked to the right
    )

    # save
    out_root = out_dir(base_dir, figures_root)
    outdir = out_root
    os.makedirs(outdir, exist_ok=True)
    prefix = file_prefix(base_dir)
    scope_tag = who.replace(" ", "_")
    fname = f"{prefix}__Composition__DepthAvg__{scope_tag}__{when}.png"
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"[composition] saved {path}")

def composition_at_depth_single(
    ds: xr.Dataset,
    *,
    z_level: float,                    # target depth (m; negative down)
    tol: float = 0.75,                 # +/- m tolerance for nearest layer
    months: Optional[List[int]] = None,
    years: Optional[List[int]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    region: Optional[Tuple[str, Dict[str, Any]]] = None,
    station: Optional[Tuple[str, float, float]] = None,
    base_dir: str,
    figures_root: str,
    phyto_vars: Sequence[str],
    zoo_vars: Sequence[str],
    dpi: int = 150,
    figsize: Tuple[float, float] = (7, 5),
    verbose: bool = False,
) -> None:
    """
    Plot a single stacked-bar figure showing **composition at a fixed absolute depth**.

    The figure has **two bars** at the requested depth:
      [Phyto (z)] [Zoo (z)]

    Depth selection uses the absolute-depth dict API:
      ``select_depth(ds_scoped, {"z_m": z_level, "tol": tol})``

    Workflow
    --------
    1. Apply time filters and spatial scope (domain / region / station).
    2. Resolve available variables for phyto and zoo groups.
    3. Slice the dataset at absolute depth ``z_level`` (m; negative down) within tolerance ``tol``.
    4. Compute pooled fractions via ``_fractional_breakdown``.
    5. Render **two** stacked bars with legend outside and save a PNG.

    Parameters
    ----------
    ds : xr.Dataset
        Source dataset.
    z_level : float
        Absolute depth in meters (negative downward), e.g., ``-10.0`` for 10 m below surface.
    tol : float, default 0.75
        Tolerance (± meters) for selecting the nearest layer to ``z_level``.
    months, years : list[int], optional
        Time filters (calendar-based).
    start_date, end_date : str, optional
        Inclusive date bounds "YYYY-MM-DD".
    region : (str, dict), optional
        Region scope accepted by ``apply_scope``.
    station : (str, float, float), optional
        Station scope (nearest column).
    base_dir : str
        Run root for filename prefixing.
    figures_root : str
        Output root directory.
    phyto_vars : Sequence[str]
        Candidate phytoplankton variable names; subset used if present.
    zoo_vars : Sequence[str]
        Candidate zooplankton variable names; subset used if present.
    dpi : int, default 150
        PNG resolution.
    figsize : (float, float), default (7, 5)
        Figure size in inches.
    verbose : bool, default False
        Print progress (including saved path).

    Returns
    -------
    None
        Saves a PNG named like:
        ``<prefix>__Composition__z<abs(z_level):.1f>m__<ScopeTag>__<TimeTag>.png``

    Raises
    ------
    RuntimeError
        If absolute-depth selection fails (e.g., dict API unsupported or no layer within tolerance).

    Notes
    -----
    - Negative depths indicate meters **below** the surface.
    - Fractions sum to 1.0 within each bar; missing variables are tolerated.
    - Use this view to highlight community structure at ecologically relevant depths
      (e.g., DCM or sampling depths).
    """
    # time & scope
    ds_t = filter_time(ds, months=months, years=years, start_date=start_date, end_date=end_date)
    ds_scoped = apply_scope(ds_t, region=region, station=station, verbose=verbose)

    # variables present
    phyto_vars = resolve_available_vars(ds_scoped, phyto_vars)
    zoo_vars   = resolve_available_vars(ds_scoped,   zoo_vars)

    # slice at absolute depth (dict API)
    try:
        ds_z = select_depth(ds_scoped, {"z_m": float(z_level), "tol": float(tol)}, verbose=verbose)
    except Exception as e:
        raise RuntimeError(
            f"[composition] absolute-depth selection failed at z={z_level} m: {e}. "
            "Ensure select_depth supports {'z_m': ...} requests."
        )

    # pooled fractions
    phyto_f, phyto_labels = _fractional_breakdown(ds_z, phyto_vars)
    zoo_f,   zoo_labels   = _fractional_breakdown(ds_z, zoo_vars)

    # figure – single axes with two bars
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    who = "Domain" if region is None and station is None else (
        (region[0] if region is not None else f"Station {station[0]}")
    )
    when = build_time_window_label(months, years, start_date, end_date)
    fig.suptitle(f"Composition at z={z_level:.2f} m - {who} - {when}", fontsize=12)

    bars = [phyto_f, zoo_f]
    labels_per_bar = [phyto_labels, zoo_labels]
    bar_names = ["Phyto (z)", "Zoo (z)"]

    stacked_fraction_bars(
        ax,
        bars,
        labels_per_bar,
        bar_names=bar_names,
        y_label="Fraction of group",
        show_legend=True,
        xtick_rotation=0.0,
        bar_width=0.30,          # thinner bars
        legend_outside=True,     # legend docked to the right
    )

    # save
    out_root = out_dir(base_dir, figures_root)
    outdir = out_root
    os.makedirs(outdir, exist_ok=True)
    prefix = file_prefix(base_dir)
    scope_tag = who.replace(" ", "_")
    fname = f"{prefix}__Composition__z{abs(z_level):.1f}m__{scope_tag}__{when}.png"
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"[composition] saved {path}")

