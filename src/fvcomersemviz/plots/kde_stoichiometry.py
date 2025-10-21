# fvcomersemviz/plots/kde_stoichiometry.py
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List, Literal
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter

from ..io import filter_time, eval_group_or_var
from ..regions import build_region_masks, apply_prebuilt_mask
from ..utils import (
    out_dir, file_prefix,
    robust_clims, style_get,
    select_depth, build_time_window_label,
    align_flatten_pair,
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _vprint(verbose: bool, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)

def _space_dim(da: xr.DataArray) -> Optional[str]:
    """Return 'node' or 'nele' if present, else None."""
    if "node" in da.dims: return "node"
    if "nele" in da.dims: return "nele"
    return None

def _kde2d(
    x: np.ndarray, y: np.ndarray,
    *,
    grids: int = 100,
    bw_method: Optional[float | str] = "scott",
    xpad: float = 0.05, ypad: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, Y, Z) for density on a regular grid using Gaussian KDE."""
    if x.size < 2 or y.size < 2:
        raise ValueError("Need at least 2 points to estimate KDE.")
    # bounds with a small padding
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    xrng = xmax - xmin or 1.0
    yrng = ymax - ymin or 1.0
    xmin -= xpad * xrng
    xmax += xpad * xrng
    ymin -= ypad * yrng
    ymax += ypad * yrng

    xi = np.linspace(xmin, xmax, grids)
    yi = np.linspace(ymin, ymax, grids)
    X, Y = np.meshgrid(xi, yi, indexing="xy")

    kde = gaussian_kde(np.vstack([x, y]), bw_method=bw_method)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    return X, Y, Z

def _kde2d_hist(
    x: np.ndarray, y: np.ndarray,
    *, grids: int = 100, sigma: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Approximate KDE by 2D histogram then Gaussian blur (σ in bins).
    Extremely fast for large n.
    """
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    xrng = xmax - xmin or 1.0
    yrng = ymax - ymin or 1.0
    # pad 5%
    xmin -= 0.05 * xrng; xmax += 0.05 * xrng
    ymin -= 0.05 * yrng; ymax += 0.05 * yrng

    H, xe, ye = np.histogram2d(x, y, bins=grids, range=[[xmin, xmax], [ymin, ymax]])
    Z = gaussian_filter(H.T, sigma=sigma, mode="nearest")
    # bin centers
    Xc = 0.5 * (xe[:-1] + xe[1:])
    Yc = 0.5 * (ye[:-1] + ye[1:])
    X, Y = np.meshgrid(Xc, Yc, indexing="xy")
    return X, Y, Z

def _panel_title(group: str, side: str, ratio: str, var: str) -> str:
    # side: 'surface' or 'bottom'; ratio: 'NC' or 'PC'
    return f"{group} {ratio}:C vs {var} — {side}"

# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------
def kde_stoichiometry_2x2(
    ds: xr.Dataset,
    *,
    group: str,                    # e.g., "P5"
    variable: str,                 # native var or group name; resolved via groups
    region: Optional[Tuple[str, Dict[str, Any]]] = None,  # (name, spec) or None for full domain
    months: Optional[List[int]] = None,
    years: Optional[List[int]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    base_dir: str,
    figures_root: str,
    groups: Optional[Dict[str, Any]] = None,
    dpi: int = 150,
    figsize: Tuple[float, float] = (11, 9),
    cmap: str = "viridis",
    grids: int = 100,               # smaller default → faster
    bw_method: Optional[float | str] = "scott",  # ignored if method="hist"
    min_samples: int = 200,         # minimum finite pairs to attempt a panel
    scatter_underlay: int = 0,      # 0 = no scatter; else plot up to N random points under the density
    verbose: bool = False,
    styles: Optional[Dict[str, Dict[str, Any]]] = None,
    # --- new fast-path controls ---
    method: Literal["kde", "hist"] = "kde",
    sample_max: Optional[int] = 200_000,
    hist_sigma: float = 1.2,
    random_seed: Optional[int] = 12345,
) -> None:
    """
    Render a 2×2 **stoichiometry vs variable** density figure for a functional group,
    optionally scoped by region and time filters. Each panel pools all available
    (time × space) samples after depth selection (surface/bottom) and masking, then
    plots a 2D density estimate.
    
    Figure layout
    -------------
    [ (0,0) ] Surface:  N:C  vs `variable`
    [ (0,1) ] Surface:  P:C  vs `variable`
    [ (1,0) ] Bottom :  N:C  vs `variable`
    [ (1,1) ] Bottom :  P:C  vs `variable`
    
    Where the stoichiometric ratios are taken from native fields:
        N:C → f"{group}_NC"
        P:C → f"{group}_PC"
    
    Workflow
    --------
    1) Time-filter once with `filter_time` → `ds_t`.
    2) Build surface and bottom slices via `select_depth(ds_t, "surface"|"bottom")`.
    3) Build region masks once via `build_region_masks(ds, region)` (node/element aware).
    4) For each side (surface/bottom), resolve:
         - `variable` via `eval_group_or_var(ds_depth, variable, groups)`
         - `NC` as f"{group}_NC"
         - `PC` as f"{group}_PC"
       Then apply prebuilt region masks to each DataArray.
    5) For each panel, align and flatten paired arrays with `align_flatten_pair(...)`,
       optionally subsampling up to `sample_max`. If finite pair count < `min_samples`,
       the panel is skipped.
    6) Compute 2D density:
         - `method="kde"` → Gaussian KDE via `_kde2d` (bandwidth from `bw_method`)
         - `method="hist"` → Fast smoothed 2D histogram via `_kde2d_hist`
    7) Plot density with `pcolormesh`. Optionally underlay up to `scatter_underlay`
       random points to show support. Apply per-variable style overrides (cmap, vmin, vmax).
    8) Save one PNG for the 2×2 figure.
    
    Parameters
    ----------
    ds : xr.Dataset
        Source dataset with required fields for the chosen `group` and `variable`,
        plus grid coordinates for masking if `region` is provided.
    group : str
        Functional group tag used to derive stoichiometric names, e.g. `"P5"` →
        `"P5_NC"` and `"P5_PC"`.
    variable : str
        Target variable/expression to compare against (y-axis). Resolved via
        `eval_group_or_var` using `groups` if provided.
    region : (str, dict), optional
        Spatial scope as `(name, spec)`. `spec` is accepted by `build_region_masks`
        (e.g., shapefile or CSV boundary). If `None`, the full domain is used.
    months, years : list[int], optional
        Calendar filters: months 1–12 and/or integer years. Can be combined.
    start_date, end_date : str, optional
        Inclusive date bounds "YYYY-MM-DD".
    base_dir : str
        Run root; used by `file_prefix(base_dir)` to form the filename stem.
    figures_root : str
        Output root directory (created if missing).
    groups : dict, optional
        Global alias/expression mapping for resolving `variable` via `eval_group_or_var`.
    dpi : int, default 150
        Output PNG resolution.
    figsize : (float, float), default (11, 9)
        Figure size in inches for the 2×2 layout.
    cmap : str, default "viridis"
        Default colormap for density; can be overridden via `styles[variable]["cmap"]`.
    grids : int, default 100
        Grid resolution per axis for KDE/hist density (higher → smoother/denser but slower).
    bw_method : float or {"scott","silverman"}, default "scott"
        Bandwidth for KDE when `method="kde"`. Ignored for `method="hist"`.
    min_samples : int, default 200
        Minimum number of finite (x,y) pairs required to render a panel.
    scatter_underlay : int, default 0
        If >0, plots up to this many random (x,y) points beneath the density for context.
    verbose : bool, default False
        Print progress, skip reasons, and panel diagnostics.
    styles : dict, optional
        Per-variable style overrides (keyed by `variable`). Recognized keys:
          - "cmap": str or Colormap
          - "vmin": float (used for y-axis limit of `variable`)
          - "vmax": float (used for y-axis limit of `variable`)
    method : {"kde","hist"}, default "kde"
        Density backend:
          - "kde": Gaussian KDE (highest quality; slowest on very large samples).
          - "hist": Smoothed 2D histogram (≈10–100× faster; good for big data).
    sample_max : int, optional (default 200_000)
        If provided, randomly subsample at most this many finite pairs before density
        estimation (improves performance on very large datasets).
    hist_sigma : float, default 1.2
        Gaussian smoothing (in grid cells) applied to the 2D histogram when `method="hist"`.
    random_seed : int, optional (default 12345)
        Seed for reproducible subsampling and underlay selection.
    
    Returns
    -------
    None
        Saves a single PNG named like:
        ``<prefix>__KDE-Stoich__<group>__<variable>__<RegionTag>__<TimeLabel>.png``
    
    Notes
    -----
    - Panels are independently skipped if centers cannot be aligned, alignment fails,
      or there are fewer than `min_samples` valid pairs.
    - `vmin`/`vmax` from `styles` (for `variable`) are applied to the **y-axis** range.
    - Use `method="hist"` for very large datasets; combine with `sample_max` for speed.
    """

    outdir = out_dir(base_dir, figures_root)
    prefix = file_prefix(base_dir)
    label = build_time_window_label(months, years, start_date, end_date)
    region_tag = (region[0] if region else "Domain")

    # variables to get
    nc_name = f"{group}_NC"
    pc_name = f"{group}_PC"

    # time filter only once (depth selection applied per side)
    ds_t = filter_time(ds, months=months, years=years, start_date=start_date, end_date=end_date)

    # depth slices
    ds_surf = select_depth(ds_t, "surface", verbose=verbose)
    ds_bott = select_depth(ds_t, "bottom",  verbose=verbose)

    # Precompute masks ONCE
    mask_nodes, mask_elems = build_region_masks(ds, region, verbose=verbose)

    # resolve arrays (center-agnostic; will align later)
    def _get_triple(ds_depth: xr.Dataset, side: str):
        # resolve group/named variable
        try:
            var_da = eval_group_or_var(ds_depth, variable, groups)
        except Exception as e:
            raise KeyError(f"[kde/{side}] cannot resolve variable '{variable}': {e}")

        try:
            nc_da = eval_group_or_var(ds_depth, nc_name, groups=None)  # stoich should be native
        except Exception as e:
            raise KeyError(f"[kde/{side}] cannot resolve '{nc_name}': {e}")

        try:
            pc_da = eval_group_or_var(ds_depth, pc_name, groups=None)
        except Exception as e:
            raise KeyError(f"[kde/{side}] cannot resolve '{pc_name}': {e}")

        # Apply region mask (node/element aware) using prebuilt masks
        var_da = apply_prebuilt_mask(var_da, mask_nodes, mask_elems)
        nc_da  = apply_prebuilt_mask(nc_da,  mask_nodes, mask_elems)
        pc_da  = apply_prebuilt_mask(pc_da,  mask_nodes, mask_elems)
        return var_da, nc_da, pc_da

    # surface & bottom triplets
    var_s, nc_s, pc_s = _get_triple(ds_surf, "surface")
    var_b, nc_b, pc_b = _get_triple(ds_bott, "bottom")

    # sanity: space centers must match within each pair for alignment to be meaningful
    def _common_center(a: xr.DataArray, b: xr.DataArray) -> Optional[str]:
        sa = _space_dim(a); sb = _space_dim(b)
        return sa if sa == sb else None

    # Layout: (0,0)=surf NC; (0,1)=surf PC; (1,0)=bott NC; (1,1)=bott PC
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

    panels = [
        ("surface", "NC", nc_s, var_s, axes[0, 0]),
        ("surface", "PC", pc_s, var_s, axes[0, 1]),
        ("bottom",  "NC", nc_b, var_b, axes[1, 0]),
        ("bottom",  "PC", pc_b, var_b, axes[1, 1]),
    ]

    any_plotted = False
    rng = np.random.default_rng(random_seed)

    for side, ratio, x_da, y_da, ax in panels:
        # Make sure we can align over time & space
        center = _common_center(x_da, y_da)
        if center is None and (_space_dim(x_da) is not None or _space_dim(y_da) is not None):
            _vprint(verbose, f"[kde/{side}] center mismatch; skipping panel ({ratio}).")
            ax.axis("off")
            continue

        # Pool all time×space samples (with optional subsampling)
        try:
            x, y = align_flatten_pair(x_da, y_da, sample_max=sample_max, rng=rng)
        except Exception as e:
            _vprint(verbose, f"[kde/{side}] alignment failed: {e}")
            ax.axis("off")
            continue

        if x.size < min_samples:
            _vprint(verbose, f"[kde/{side}] only {x.size} finite pairs; need >= {min_samples}. Skipping.")
            ax.axis("off")
            continue

        try:
            if method == "kde":
                X, Y, Z = _kde2d(x, y, grids=grids, bw_method=bw_method)
            else:
                X, Y, Z = _kde2d_hist(x, y, grids=grids, sigma=hist_sigma)
        except Exception as e:
            _vprint(verbose, f"[kde/{side}] density failed: {e}")
            ax.axis("off")
            continue

        # If vmin/vmax specified in styles for this variable, use them for y-axis ticks/limits
        vmin_style = style_get(variable, styles, "vmin", None)
        vmax_style = style_get(variable, styles, "vmax", None)

        # Plot density
        pcm = ax.pcolormesh(X, Y, Z, shading="auto", cmap=style_get(variable, styles, "cmap", cmap))
        cb = plt.colorbar(pcm, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label("Density")

        # Optional sparse scatter to hint the support
        if scatter_underlay and x.size > 0:
            sel = rng.choice(x.size, size=min(scatter_underlay, x.size), replace=False)
            ax.plot(x[sel], y[sel], ".", ms=1.2, alpha=0.25, zorder=2, color="k")

        # Titles & labels
        ax.set_title(_panel_title(group, side, ratio, variable))
        ax.set_xlabel(f"{group} {ratio}:C")
        ax.set_ylabel(variable)

        # Nice limits
        if vmin_style is not None and vmax_style is not None:
            ax.set_ylim(vmin_style, vmax_style)

        any_plotted = True

    # If nothing plotted, bail with an informative message instead of saving a blank figure
    if not any_plotted:
        plt.close(fig)
        _vprint(verbose, "[kde] No panels had sufficient data; nothing saved.")
        return

    fname = f"{prefix}__KDE-Stoich__{group}__{variable}__{region_tag}__{label}.png"
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    _vprint(verbose, f"[kde] saved {path}")
