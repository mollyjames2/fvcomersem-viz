# fvcomersemviz/plots/composition.py
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List, Sequence
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

from ..io import filter_time, eval_group_or_var
from ..regions import apply_scope, build_region_masks, apply_prebuilt_mask
from ..utils import (
    out_dir,
    file_prefix,
    select_depth,
    build_time_window_label,
    resolve_available_vars,
    sum_over_all_dims,
    select_da_by_z,
    depth_tag,
)

from ..plot import stacked_fraction_bars


def _fractional_breakdown(
    ds_depth: xr.Dataset, var_names: Sequence[str]
) -> Tuple[np.ndarray, List[str]]:
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
        Time filters (calendar months 1-12; year integers). Can be combined.
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
    zoo_vars = resolve_available_vars(ds_scoped, zoo_vars)

    # depth slices
    ds_surf = select_depth(ds_scoped, "surface", verbose=verbose)
    ds_bott = select_depth(ds_scoped, "bottom", verbose=verbose)

    # pooled fractions
    phyto_s, phyto_labels = _fractional_breakdown(ds_surf, phyto_vars)
    phyto_b, _ = _fractional_breakdown(ds_bott, phyto_vars)
    zoo_s, zoo_labels = _fractional_breakdown(ds_surf, zoo_vars)
    zoo_b, _ = _fractional_breakdown(ds_bott, zoo_vars)

    # figure - single axes with four bars
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    who = (
        "Domain"
        if region is None and station is None
        else (region[0] if region is not None else f"Station {station[0]}")
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
    zoo_vars = resolve_available_vars(ds_scoped, zoo_vars)

    # depth-average (weighted if supported; else manual mean over siglay)
    def _depth_average_dataset(ds_scoped: xr.Dataset) -> xr.Dataset:
        try:
            return select_depth(ds_scoped, "depth_avg", verbose=verbose)
        except Exception:
            return (
                ds_scoped.mean("siglay", skipna=True) if "siglay" in ds_scoped.dims else ds_scoped
            )

    ds_avg = _depth_average_dataset(ds_scoped)

    # pooled fractions
    phyto_f, phyto_labels = _fractional_breakdown(ds_avg, phyto_vars)
    zoo_f, zoo_labels = _fractional_breakdown(ds_avg, zoo_vars)

    # figure - single axes with two bars
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    who = (
        "Domain"
        if region is None and station is None
        else (region[0] if region is not None else f"Station {station[0]}")
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
        bar_width=0.30,  # thinner bars
        legend_outside=True,  # legend docked to the right
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
    z_level: float,  # target depth (m; negative down)
    tol: float = 0.75,  # +/- m tolerance for nearest layer
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
    zoo_vars = resolve_available_vars(ds_scoped, zoo_vars)

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
    zoo_f, zoo_labels = _fractional_breakdown(ds_z, zoo_vars)

    # figure - single axes with two bars
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    who = (
        "Domain"
        if region is None and station is None
        else (region[0] if region is not None else f"Station {station[0]}")
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
        bar_width=0.30,  # thinner bars
        legend_outside=True,  # legend docked to the right
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


def composition_fraction_timeseries(
    ds: xr.Dataset,
    *,
    phyto_vars: Sequence[str],
    zoo_vars: Sequence[str],
    # scope
    scope: str = "domain",  # 'domain' | 'region' | 'station'
    regions: Optional[Sequence[Tuple[str, Dict[str, Any]]]] = None,  # [(name, spec), ...]
    stations: Optional[Sequence[Tuple[str, float, float]]] = None,  # [(name, lat, lon), ...]
    # time/depth filters
    depth: Any = "surface",
    months: Optional[Sequence[int]] = None,
    years: Optional[Sequence[int]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    # output
    base_dir: str = "",
    figures_root: str = "",
    # style
    show_std_band: bool = True,  # ±1σ shading across space
    colors: Optional[Dict[str, Any]] = None,  # optional per-variable color map
    linewidth: float = 2.0,
    alpha_band: float = 0.20,
    figsize_per_panel: Tuple[float, float] = (10, 3.2),  # width, height per panel
    dpi: int = 150,
    verbose: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Time-resolved community composition (fractions) for Phytoplankton and Zooplankton.

    This routine produces **two figures** (one for PHYTO, one for ZOO). Each figure contains
    1..N stacked panels depending on `scope`:
        • scope='domain'  → 1 panel labelled "Domain"
        • scope='region'  → one panel per region in `regions` (node/element masks respected)
        • scope='station' → one panel per station in `stations` (nearest column)

    For every panel and timestep, variables in the corresponding group are divided by the
    **group total** (sum across all provided variables that are actually present) to yield
    per-variable fractions in [0,1]. Fractions are then reduced **across space** (nodes or
    elements, as appropriate) to produce a **mean line** and, optionally, a **±1σ band**
    representing spatial variability.

    Depth and time filters are applied **once** up front:
      `select_depth(ds, depth)` → `filter_time(... months/years/start/end ...)`.
    For absolute-z requests (e.g., -10, {"z_m": -10}), a per-variable refinement is attempted
    via `select_da_by_z(...)` so the correct vertical level is extracted for each DataArray.

    Region handling is center-aware: node-centered and element-centered variables are masked
    against prebuilt region masks (`mask_nodes`, `mask_elems`) using their own space dims.

    The function saves PNGs whose filenames encode group, scope, depth tag, and time window,
    and returns the two output paths.

    Parameters
    ----------
    ds : xr.Dataset
        Source dataset containing the variables listed in `phyto_vars` / `zoo_vars`. Must include
        the grid space dims (e.g., 'node' and/or 'nele') and a 'time' dimension for plotted fields.
    phyto_vars : Sequence[str]
        Candidate variable names for the phytoplankton group. Only variables present (and resolvable
        via `eval_group_or_var` if imported in your project) are used per panel.
    zoo_vars : Sequence[str]
        Candidate variable names for the zooplankton group. Same presence/resolve logic as above.

    scope : {'domain', 'region', 'station'}, default 'domain'
        What each panel represents.
          - 'domain' : one panel averaged over the full domain.
          - 'region' : one panel for each entry in `regions`; masks are node/element aware.
          - 'station': one panel for each entry in `stations`; nearest column is selected.

    regions : Sequence[Tuple[str, Dict[str, Any]]], optional
        Regions as `(name, spec)` tuples, required for `scope='region'`. The `spec` should be
        compatible with your region utilities (e.g., shapefile or CSV boundary). Masks are built
        once from the **windowed dataset** and then applied center-aware per DataArray.
    stations : Sequence[Tuple[str, float, float]], optional
        Stations as `(name, lat, lon)` tuples, required for `scope='station'`. The nearest column
        is selected for each variable.

    depth : Any, default 'surface'
        Vertical selector passed to `select_depth`, e.g.:
          'surface' | 'bottom' | 'depth_avg' | int sigma index | float sigma in [-1,0]
          absolute depths (meters, negative down): -10.0, ('z_m', -10.0), {'z_m': -10.0}
        If absolute depth is requested, the function refines each DataArray with `select_da_by_z(...)`.

    months, years : Optional[Sequence[int]]
        Calendar filters. Examples: months=[4,5,6,7,8,9,10], years=[2018].
    start_date, end_date : Optional[str]
        Inclusive date bounds in "YYYY-MM-DD" format. May be used with or without `months`/`years`.

    base_dir : str, default ""
        Model run root; used only to form an output filename prefix via `file_prefix(base_dir)`.
    figures_root : str, default ""
        Root directory where figure files are saved (a 'composition' subfolder may be created by
        your `out_dir` policy).

    show_std_band : bool, default True
        If True, shade the ±1σ interval across space around the mean fraction line in each panel.
        At stations (single point), the band collapses to zero width.
    colors : dict, optional
        Optional mapping from variable name → matplotlib color (e.g., {'P1_c':'#b3de69'}). Any
        variable not listed uses the current matplotlib color cycle to ensure consistent colors
        across panels.
    linewidth : float, default 2.0
        Line width for fraction curves.
    alpha_band : float, default 0.20
        Opacity for the ±1σ shading band.
    figsize_per_panel : (float, float), default (10, 3.2)
        Size in inches used to scale the figure height as `n_panels * figsize_per_panel[1]`.
    dpi : int, default 150
        Output figure resolution.
    verbose : bool, default False
        If True, print progress and skip reasons (unresolvable variables, empty masks, etc.).

    Returns
    -------
    (phyto_path, zoo_path) : Tuple[Optional[str], Optional[str]]
        Full paths to the saved PNGs for the PHYTO and ZOO figures, respectively. A value can be
        `None` if that group had nothing to plot (e.g., all variables missing after masking).

    Behavior & Implementation Notes
    -------------------------------
    • **Safe division**: Fractions are computed with a mask-first strategy to avoid divide warnings:
        valid = isfinite(total) & (total > 0)
        frac  = (var.where(valid)) / (total.where(valid))
      This prevents Dask from dividing by zero/NaN and eliminates runtime warnings.

    • **Space reduction**: For each time step, fractions are averaged over all remaining space dims
      (everything except 'time'); the standard deviation over those dims is used for the ±1σ band.

    • **Region masking**: Masks are built once per region (on the windowed dataset) and applied to
      each DataArray using its native center (node vs element). If a mask selects nothing for a
      region, that panel is skipped.

    • **Variable resolution**: Variables are taken directly from the dataset when present; otherwise
      the function attempts `eval_group_or_var(ds, name)` if available in your project to support
      aliases or expressions. Only variables that (a) resolve and (b) contain a 'time' dimension
      are kept.

    • **Output filenames**: Encoded with group, scope, depth tag, and time window:
        <prefix>__CompositionTS__<Group>__<ScopeTag>__<DepthTag>__<TimeLabel>.png
      where:
        <prefix>   = file_prefix(base_dir)
        <ScopeTag> = 'Domain' | 'Regions-N' | 'Stations-N'
        <DepthTag> = from utils.depth_tag(depth)
        <TimeLabel>= from utils.build_time_window_label(...)

    Examples
    --------
    >>> composition_fraction_timeseries(
    ...     ds, phyto_vars=['P1_c','P2_c','P4_c'], zoo_vars=['Z4_c','Z5_c','Z6_c'],
    ...     scope='region', regions=[('West', specW), ('Central', specC), ('East', specE)],
    ...     depth='surface', months=[4,5,6,7,8,9,10], years=[2018],
    ...     base_dir=BASE_DIR, figures_root=FIG_DIR,
    ...     colors={'P1_c':'#a6cee3','P2_c':'#1f78b4','P4_c':'#b2df8a'}
    ... )
    """

    # -------- helpers --------
    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    def _space_dims(da: xr.DataArray) -> List[str]:
        # everything except time (siglay should be removed by depth selection)
        return [d for d in da.dims if d != "time"]

    def _apply_abs_z_if_needed(da: xr.DataArray, ds_for_z: xr.Dataset) -> xr.DataArray:
        # refine to absolute depth if requested
        try:
            if isinstance(depth, (float, np.floating)) and not (-1.0 <= float(depth) <= 0.0):
                return select_da_by_z(da, ds_for_z, float(depth), verbose=verbose)
            if isinstance(depth, tuple) and len(depth) > 0 and depth[0] == "z_m":
                return select_da_by_z(da, ds_for_z, float(depth[1]), verbose=verbose)
            if isinstance(depth, dict) and "z_m" in depth:
                return select_da_by_z(da, ds_for_z, float(depth["z_m"]), verbose=verbose)
        except Exception as e:
            _log(f"[composition/abs-z] skipping abs-z refinement: {e}")
        return da

    def _safe_fraction(num: xr.DataArray, den: xr.DataArray) -> xr.DataArray:
        """
        Compute num/den WITHOUT triggering dask divide warnings.
        We first mask both arrays where denominator is invalid, then divide.
        """
        valid = np.isfinite(den) & (den > 0)
        den_clean = den.where(valid)  # NaN where invalid → won't be used in division
        num_clean = num.where(valid)  # keep numerator only where denom valid
        return num_clean / den_clean  # division only happens on valid cells

    def _mask_da(
        da: xr.DataArray,
        mask_nodes: Optional[np.ndarray],
        mask_elems: Optional[np.ndarray],
    ) -> xr.DataArray:
        """Apply prebuilt region mask to a DA, honoring its space center."""
        if mask_nodes is None and mask_elems is None:
            return da
        return apply_prebuilt_mask(da, mask_nodes, mask_elems)

    def _fraction_timeseries_for_group(
        ds_scoped: xr.Dataset,
        var_names: Sequence[str],
        *,
        mask_nodes: Optional[np.ndarray] = None,
        mask_elems: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Tuple[pd.DatetimeIndex, np.ndarray, Optional[np.ndarray]]]]:
        """
        For a list of variables at a given scope, return:
            { var_name: (time_index, mean_fraction[t], std_fraction[t]) }
        If nothing usable, returns None.
        """
        # keep only available variables
        var_names = resolve_available_vars(ds_scoped, var_names)
        if not var_names:
            return None

        das: List[xr.DataArray] = []
        names_in: List[str] = []
        for v in var_names:
            try:
                da = (
                    ds_scoped[v] if v in ds_scoped else eval_group_or_var(ds_scoped, v, groups=None)
                )
            except Exception as e:
                _log(f"[composition] '{v}' not resolvable here: {e}")
                continue

            da = _apply_abs_z_if_needed(da, ds_scoped)
            da = _mask_da(da, mask_nodes, mask_elems)

            if "time" not in da.dims:
                _log(f"[composition] '{v}' has no 'time'; skipping.")
                continue

            # skip if everything is masked after region mask
            if not np.isfinite(da).any():
                _log(f"[composition] '{v}' has no finite data after mask; skipping.")
                continue

            das.append(da)
            names_in.append(v)

        if not das:
            return None

        # Align to common support (time/space)
        try:
            das = xr.align(*das, join="inner")
        except Exception as e:
            _log(f"[composition] alignment failed: {e}")
            return None

        # total across variables (per time × space)
        total = None
        for da in das:
            total = da if total is None else (total + da)

        # if totally empty after masking, bail
        if not np.isfinite(total).any():
            _log("[composition] total is all-NaN in this scope; skipping panel.")
            return None

        out: Dict[str, Tuple[pd.DatetimeIndex, np.ndarray, Optional[np.ndarray]]] = {}
        for v, da in zip(names_in, das):
            frac = _safe_fraction(da, total)

            # if a var is still entirely NaN after safe division, skip it
            if not np.isfinite(frac).any():
                _log(f"[composition] '{v}' has no finite fractions; skipping.")
                continue

            sdims = _space_dims(frac)
            if sdims:
                mean = frac.mean(dim=sdims, skipna=True)
                std = frac.std(dim=sdims, skipna=True) if show_std_band else None
            else:
                # single point (e.g., station)
                mean = frac
                std = xr.zeros_like(frac) if show_std_band else None

            t = pd.to_datetime(np.atleast_1d(mean["time"].values))
            out[v] = (
                t,
                np.asarray(mean.values, dtype=float),
                (None if std is None else np.asarray(std.values, dtype=float)),
            )

        return out or None

    def _panel_title(scope_kind: str, label: str, group_label: str) -> str:
        left = "Domain" if scope_kind == "domain" else label
        return f"{group_label} composition {left} averaged"

    def _draw_figure(
        group_label: str,
        panels: Sequence[Tuple[str, xr.Dataset, Optional[np.ndarray], Optional[np.ndarray]]],
        group_vars: Sequence[str],
        fig_tag_bits: str,
    ) -> Optional[str]:
        # build data for all panels
        payload: List[
            Tuple[
                str,
                Dict[str, Tuple[pd.DatetimeIndex, np.ndarray, Optional[np.ndarray]]],
            ]
        ] = []
        for name, ds_scoped, m_nodes, m_elems in panels:
            res = _fraction_timeseries_for_group(
                ds_scoped, group_vars, mask_nodes=m_nodes, mask_elems=m_elems
            )
            if res is None:
                _log(f"[composition/{group_label}] nothing to plot for panel '{name}'.")
                continue
            payload.append((name, res))

        if not payload:
            return None

        # figure size scales with number of panels
        n_pan = len(payload)
        fig_h = max(2.6, n_pan * figsize_per_panel[1])
        fig_w = figsize_per_panel[0]
        fig, axes = plt.subplots(
            n_pan, 1, figsize=(fig_w, fig_h), sharex=True, constrained_layout=True
        )
        if n_pan == 1:
            axes = [axes]  # type: ignore[assignment]

        # color palette per variable (stable across panels)
        cycle = plt.rcParams.get("axes.prop_cycle", None)
        default_colors = (
            cycle.by_key().get("color", [f"C{i}" for i in range(10)])
            if cycle is not None
            else [f"C{i}" for i in range(10)]
        )
        palette: Dict[str, Any] = {}
        for i, v in enumerate(group_vars):
            palette[v] = (
                colors.get(v) if colors and v in colors else default_colors[i % len(default_colors)]
            )

        # plot each panel
        for ax, (panel_name, res) in zip(axes, payload):
            # union of time for x-limits (guard empty)
            t_candidates: List[pd.DatetimeIndex] = [
                t for (_v, (t, _m, _s)) in res.items() if len(t) > 0
            ]
            if t_candidates:
                t0 = min(ti[0] for ti in t_candidates)
                t1 = max(ti[-1] for ti in t_candidates)
                ax.set_xlim(t0, t1)

            handles, labels = [], []
            for v in group_vars:
                if v not in res:
                    continue
                t, mean, std = res[v]
                (line,) = ax.plot(t, mean, lw=linewidth, color=palette[v], label=v, zorder=3)
                if show_std_band and std is not None:
                    lo = np.maximum(0.0, mean - std)
                    hi = np.minimum(1.0, mean + std)
                    ax.fill_between(
                        t,
                        lo,
                        hi,
                        color=palette[v],
                        alpha=alpha_band,
                        linewidth=0,
                        zorder=2,
                    )
                handles.append(line)
                labels.append(v)

            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Fraction (0–1)")
            ax.set_title(_panel_title(scope_norm, panel_name, group_label), loc="center")

            if ax is axes[0] and handles:
                ax.legend(handles=handles, labels=labels, loc="upper right", frameon=False)

        axes[-1].set_xlabel("Time (yyyy-mm)")

        # save
        outdir = out_dir(base_dir, figures_root)
        os.makedirs(outdir, exist_ok=True)
        prefix = file_prefix(base_dir)
        when = build_time_window_label(months, years, start_date, end_date)
        dtag = depth_tag(depth)
        fname = f"{prefix}__CompositionTS__{group_label}__{fig_tag_bits}__{dtag}__{when}.png"
        path = os.path.join(outdir, fname)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        _log(f"[composition/{group_label}] saved {path}")
        return path

    # -------- select depth + time window once --------
    ds_depth = select_depth(ds, depth, verbose=verbose)
    ds_t = filter_time(
        ds_depth, months=months, years=years, start_date=start_date, end_date=end_date
    )

    # -------- panels by scope --------
    scope_norm = scope.strip().lower()
    if scope_norm not in ("domain", "region", "station"):
        raise ValueError("scope must be 'domain', 'region', or 'station'")

    panels_phyto: List[Tuple[str, xr.Dataset, Optional[np.ndarray], Optional[np.ndarray]]] = []
    panels_zoo: List[Tuple[str, xr.Dataset, Optional[np.ndarray], Optional[np.ndarray]]] = []

    if scope_norm == "domain":
        scoped = apply_scope(ds_t, region=None, station=None, verbose=verbose)
        panels_phyto.append(("Domain", scoped, None, None))
        panels_zoo.append(("Domain", scoped, None, None))
        fig_tag = "Domain"

    elif scope_norm == "region":
        if not regions:
            raise ValueError("scope='region' requires a non-empty `regions=[(name, spec), ...]`")

        for name, spec in regions:
            # Build masks ON THE SAME windowed dataset
            try:
                mask_nodes, mask_elems = build_region_masks(ds_t, (name, spec), verbose=verbose)
            except Exception as e:
                _log(f"[composition/region] '{name}' mask failed: {e}; skipping.")
                continue

            # Quick emptiness check: if neither mask selects anything, skip
            sel_nodes = int(mask_nodes.sum()) if (mask_nodes is not None and mask_nodes.size) else 0
            sel_elems = int(mask_elems.sum()) if (mask_elems is not None and mask_elems.size) else 0
            if sel_nodes == 0 and sel_elems == 0:
                _log(f"[composition/region] '{name}' mask selects nothing; skipping.")
                continue

            # Keep original ds_t; masking happens per-DA (center-aware)
            panels_phyto.append((name, ds_t, mask_nodes, mask_elems))
            panels_zoo.append((name, ds_t, mask_nodes, mask_elems))

        fig_tag = f"Regions-{len(panels_phyto)}"

    else:  # station
        if not stations:
            raise ValueError(
                "scope='station' requires a non-empty `stations=[(name, lat, lon), ...]`"
            )
        for name, lat, lon in stations:
            scoped = apply_scope(ds_t, region=None, station=(name, lat, lon), verbose=verbose)
            panels_phyto.append((name, scoped, None, None))
            panels_zoo.append((name, scoped, None, None))
        fig_tag = f"Stations-{len(stations)}"

    # ensure we only try to draw variables that exist somewhere (keep full lists; per-panel filters happen inside)
    phyto_vars = list(phyto_vars)
    zoo_vars = list(zoo_vars)

    # -------- render both figures --------
    phyto_path = _draw_figure("Phyto", panels_phyto, phyto_vars, fig_tag_bits=fig_tag)
    zoo_path = _draw_figure("Zoo", panels_zoo, zoo_vars, fig_tag_bits=fig_tag)
    return phyto_path, zoo_path
