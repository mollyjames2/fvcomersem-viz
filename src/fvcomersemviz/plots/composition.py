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
    ONE figure, ONE axes: 4 bars: [Surface Phyto, Bottom Phyto, Surface Zoo, Bottom Zoo].
    """
    # 1) time & scope
    ds_t = filter_time(ds, months=months, years=years, start_date=start_date, end_date=end_date)
    ds_scoped = apply_scope(ds_t, region=region, station=station, verbose=verbose)

    # 2) variables present
    phyto_vars = resolve_available_vars(ds_scoped, phyto_vars)
    zoo_vars   = resolve_available_vars(ds_scoped,   zoo_vars)

    # 3) depth slices
    ds_surf = select_depth(ds_scoped, "surface", verbose=verbose)
    ds_bott = select_depth(ds_scoped, "bottom",  verbose=verbose)

    # 4) pooled fractions
    phyto_s, phyto_labels = _fractional_breakdown(ds_surf, phyto_vars)
    phyto_b, _            = _fractional_breakdown(ds_bott, phyto_vars)
    zoo_s,   zoo_labels   = _fractional_breakdown(ds_surf, zoo_vars)
    zoo_b,   _            = _fractional_breakdown(ds_bott, zoo_vars)

    # 5) figure – single axes with four bars
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

    # 6) save
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
    ONE figure, ONE axes: 2 bars: [Phyto (depth-avg), Zoo (depth-avg)].
    Depth-average is computed across ALL sigma layers (weighted by 'layer_thickness'
    if select_depth('depth_avg') supports it; else simple mean over 'siglay').
    """
    # 1) time & scope
    ds_t = filter_time(ds, months=months, years=years, start_date=start_date, end_date=end_date)
    ds_scoped = apply_scope(ds_t, region=region, station=station, verbose=verbose)

    # 2) variables present
    phyto_vars = resolve_available_vars(ds_scoped, phyto_vars)
    zoo_vars   = resolve_available_vars(ds_scoped,   zoo_vars)

    # 3) depth-average (weighted if supported; else manual mean over siglay)
    def _depth_average_dataset(ds_scoped: xr.Dataset) -> xr.Dataset:
        try:
            return select_depth(ds_scoped, "depth_avg", verbose=verbose)
        except Exception:
            return ds_scoped.mean("siglay", skipna=True) if "siglay" in ds_scoped.dims else ds_scoped

    ds_avg = _depth_average_dataset(ds_scoped)

    # 4) pooled fractions
    phyto_f, phyto_labels = _fractional_breakdown(ds_avg, phyto_vars)
    zoo_f,   zoo_labels   = _fractional_breakdown(ds_avg, zoo_vars)

    # 5) figure – single axes with two bars
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

    # 6) save
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
    ONE figure, ONE axes: 2 bars at a selected absolute depth: [Phyto (z), Zoo (z)].
    """
    # 1) time & scope
    ds_t = filter_time(ds, months=months, years=years, start_date=start_date, end_date=end_date)
    ds_scoped = apply_scope(ds_t, region=region, station=station, verbose=verbose)

    # 2) variables present
    phyto_vars = resolve_available_vars(ds_scoped, phyto_vars)
    zoo_vars   = resolve_available_vars(ds_scoped,   zoo_vars)

    # 3) slice at absolute depth (dict API)
    try:
        ds_z = select_depth(ds_scoped, {"z_m": float(z_level), "tol": float(tol)}, verbose=verbose)
    except Exception as e:
        raise RuntimeError(
            f"[composition] absolute-depth selection failed at z={z_level} m: {e}. "
            "Ensure select_depth supports {'z_m': ...} requests."
        )

    # 4) pooled fractions
    phyto_f, phyto_labels = _fractional_breakdown(ds_z, phyto_vars)
    zoo_f,   zoo_labels   = _fractional_breakdown(ds_z, zoo_vars)

    # 5) figure – single axes with two bars
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

    # 6) save
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

