# fvcomersemviz/plots/curves.py
# General x–y “curves” diagnostics: binned median+IQR or raw scatter,
# using the package’s time filters, scope (domain/region/station), and depth selection.

from __future__ import annotations
from typing import Dict, Any, Optional, List, Sequence, Tuple

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from ..io import filter_time, eval_group_or_var
from ..regions import apply_scope
from ..utils import select_depth, align_flatten_pair

__all__ = ["build_curve_data", "plot_curves"]


# ---------------------------------------------------------------------
# Tolerant resolution helpers (variables / groups / expressions)
# ---------------------------------------------------------------------

def _normalize_key(s: str) -> str:
    """Casefold and strip underscores/dashes/spaces to ease matching."""
    return "".join(ch for ch in s.casefold() if ch not in "_- ")


def _lookup_var_tolerant(ds: xr.Dataset, name: str) -> Optional[str]:
    """
    Return the dataset key that best matches `name` under tolerant rules, or None.
    Rules: exact, case-insensitive, then normalized (case/underscore/dash/space-insensitive).
    """
    if name in ds.data_vars:
        return name
    # exact case-insensitive
    for k in ds.data_vars:
        if k.lower() == name.lower():
            return k
    # normalized match
    norm_target = _normalize_key(name)
    for k in ds.data_vars:
        if _normalize_key(k) == norm_target:
            return k
    return None


def _eval_any(
    ds: xr.Dataset,
    expr: str,
    groups: Optional[Dict[str, Any]] = None,
    *,
    spec_aliases: Optional[Dict[str, str]] = None,
):
    """
    Resolve an expression or name, tolerantly:
      1) apply per-spec aliases (if provided)
      2) try eval_group_or_var (supports groups & algebra)
      3) if it looks like a single token and failed, try tolerant var lookup in ds
         and retry eval_group_or_var with the mapped key
    """
    # 1) alias
    if spec_aliases and expr in spec_aliases:
        expr = spec_aliases[expr]
    # 2) groups/expressions
    try:
        return eval_group_or_var(ds, expr, groups)
    except KeyError:
        # 3) only try tolerant mapping if it's a single token (not an expression)
        if any(ch in expr for ch in "+-*/()[] "):
            raise
        mapped = _lookup_var_tolerant(ds, expr)
        if mapped is None:
            raise
        return eval_group_or_var(ds, mapped, groups)


# ---------------------------------------------------------------------
# Binning helper
# ---------------------------------------------------------------------

def _bin_xy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bins: int = 40,
    agg: str = "median",
    min_count: int = 10,
    iqr: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Equal-width x-binning with robust aggregations.
    Returns dict with x_mid, y_val, y_lo, y_hi, n (per-bin counts).
    """
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size == 0:
        z = np.array([])
        return dict(x_mid=z, y_val=z, y_lo=z, y_hi=z, n=z)

    xmin = np.nanmin(x); xmax = np.nanmax(x)
    if xmin == xmax:
        # Degenerate: single bin with all y
        val = np.nanmedian(y) if agg == "median" else np.nanmean(y)
        lo, hi = (np.nanpercentile(y, [25, 75]) if iqr else (np.nan, np.nan))
        return dict(
            x_mid=np.asarray([xmin]),
            y_val=np.asarray([val]),
            y_lo=np.asarray([lo]),
            y_hi=np.asarray([hi]),
            n=np.asarray([y.size]),
        )

    edges = np.linspace(xmin, xmax, n_bins + 1)
    mids  = 0.5 * (edges[:-1] + edges[1:])
    idx   = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1)

    y_val = np.full(n_bins, np.nan)
    y_lo  = np.full(n_bins, np.nan)
    y_hi  = np.full(n_bins, np.nan)
    n     = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        sel = (idx == b)
        nb  = int(sel.sum())
        n[b] = nb
        if nb < min_count:
            continue
        yb = y[sel]
        if agg == "median":
            y_val[b] = np.nanmedian(yb)
        elif agg == "mean":
            y_val[b] = np.nanmean(yb)
        elif agg.startswith("p"):
            # percentile, e.g. "p90"
            try:
                p = float(agg[1:])
            except Exception:
                p = 50.0
            y_val[b] = np.nanpercentile(yb, p)
        else:
            y_val[b] = np.nanmedian(yb)
        if iqr:
            y_lo[b], y_hi[b] = np.nanpercentile(yb, [25, 75])

    return dict(x_mid=mids, y_val=y_val, y_lo=y_lo, y_hi=y_hi, n=n)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def build_curve_data(
    ds: xr.Dataset,
    spec: Dict[str, Any],
    *,
    groups: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Resolve one curve described by `spec` into plottable arrays.

    Spec fields:
      name   : str (for legend)
      x,y    : str — resolvable by eval_group_or_var (supports GROUPS & algebra)
      filters: {months, years, start, end, where}
      depth  : "surface"|"bottom"|"depth_avg"|float sigma|{"z_m":-10}
      scope  : {"region": (name, spec)} | {"station": (name, lat, lon)} | {}
      bin    : {"x_bins":40, "agg":"median"|"mean"|"pXX", "min_count":10, "iqr":True}
      scatter: {"alpha":0.15, "s":4} (used if no "bin")
      style  : matplotlib line/scatter kwargs
      aliases: optional dict of per-spec name aliases, e.g. {"PAR":"light_parEIR"}
    """
    # 1) time filter
    fil = spec.get("filters", {})
    ds_t = filter_time(
        ds,
        months=fil.get("months"),
        years=fil.get("years"),
        start_date=fil.get("start"),
        end_date=fil.get("end"),
    )

    # 2) spatial scope (domain / region / station)
    sc = spec.get("scope", {})
    ds_s = apply_scope(ds_t, region=sc.get("region"), station=sc.get("station"), verbose=verbose)

    # 3) depth selection (supports "depth_avg" and deferred absolute-z via utils.select_depth)
    depth = spec.get("depth")
    ds_d = select_depth(ds_s, depth, verbose=verbose) if depth is not None else ds_s

    # 4) resolve x/y (groups & expressions allowed; tolerant names)
    try:
        x_da = _eval_any(ds_d, spec["x"], groups, spec_aliases=spec.get("aliases"))
    except Exception as e:
        raise KeyError(f"[curves] cannot resolve x='{spec['x']}': {e}")
    try:
        y_da = _eval_any(ds_d, spec["y"], groups, spec_aliases=spec.get("aliases"))
    except Exception as e:
        raise KeyError(f"[curves] cannot resolve y='{spec['y']}': {e}")

    # 5) optional predicate
    where_expr = fil.get("where")
    if where_expr:
        try:
            mask = _eval_any(ds_d, where_expr, groups, spec_aliases=spec.get("aliases"))
            x_da = x_da.where(mask)
            y_da = y_da.where(mask)
        except Exception as e:
            if verbose:
                print(f"[curves] where='{where_expr}' failed: {e}; ignoring predicate.")

    # 6) flatten to aligned numpy 1-D arrays
    x, y = align_flatten_pair(x_da, y_da)

    # 7) binned or scatter output
    style = spec.get("style", {})
    if "bin" in spec and spec["bin"]:
        cfg = spec["bin"]
        out = _bin_xy(
            x, y,
            n_bins=int(cfg.get("x_bins", 40)),
            agg=str(cfg.get("agg", "median")),
            min_count=int(cfg.get("min_count", 10)),
            iqr=bool(cfg.get("iqr", True)),
        )
        return dict(kind="binned", data=out, style=style)
    else:
        scfg = spec.get("scatter", {})
        return dict(kind="scatter",
                    data=dict(x=x, y=y, s=scfg.get("s", 4), alpha=scfg.get("alpha", 0.15)),
                    style=style)


def plot_curves(
    specs: Sequence[Dict[str, Any]],
    *,
    ds: xr.Dataset,
    groups: Optional[Dict[str, Any]] = None,
    # axis labels / legend
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    show_legend: bool = True,
    legend_outside: bool = True,
    legend_fontsize: int = 8,
    verbose: bool = False,
    # figure + saving
    base_dir: str,
    figures_root: str,
    stem: Optional[str] = None,      # optional override; otherwise auto-built with scope/depth/time tags
    dpi: int = 150,
    figsize: Tuple[float, float] = (7.2, 4.6),
    constrained_layout: bool = True,
) -> str:
    """
    Build a figure, render multiple curves (binned-with-IQR or scatter), and ALWAYS save a PNG.

    Saved path:
        <FIG_DIR>/<basename(BASE_DIR)>/<subdir>/<prefix>__Curves__<ScopeTag>__<DepthTag>__<TimeLabel>__<Content>.png

    Subdir logic:
        - If FVCOM_PLOT_SUBDIR is set (e.g., "project" or ""), it is respected.
        - Otherwise defaults to "curves" so outputs go to ".../curves/".

    Stem auto-build (when `stem` is None):
        - ScopeTag  : "Domain" | "Region-<Name>" | "Station-<Name>" | "MultiScope"
        - DepthTag  : from utils.depth_tag(depth) per spec, or "AllDepth" if depth not specified; "MixedDepth" if specs differ
        - TimeLabel : from utils.build_time_window_label(...) per spec; "MixedTime" if specs differ
        - Content   : "<X>_vs_<Y>" (from first spec), plus "Ncurves" if more than one spec

    Returns:
        str : Full file path of the saved PNG.
    """
    import os
    from ..utils import out_dir, file_prefix, build_time_window_label, depth_tag

    # -------- helpers for tag derivation
    def _scope_tag(spec: Dict[str, Any]) -> str:
        sc = spec.get("scope", {}) or {}
        if "region" in sc and sc["region"]:
            try:
                return f"Region-{sc['region'][0]}"
            except Exception:
                return "Region"
        if "station" in sc and sc["station"]:
            try:
                return f"Station-{sc['station'][0]}"
            except Exception:
                return "Station"
        return "Domain"

    def _depth_tag_from_spec(spec: Dict[str, Any]) -> str:
        d = spec.get("depth", None)
        return depth_tag(d) if d is not None else "AllDepth"

    def _time_label_from_spec(spec: Dict[str, Any]) -> str:
        fil = spec.get("filters", {}) or {}
        return build_time_window_label(
            fil.get("months"), fil.get("years"), fil.get("start"), fil.get("end")
        )

    def _all_equal(vals: List[str]) -> bool:
        return len(vals) <= 1 or all(v == vals[0] for v in vals)

    # -------- build fig/axes internally
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=constrained_layout)

    # color cycle
    cycle = plt.rcParams.get("axes.prop_cycle", None)
    colors = (cycle.by_key().get("color", [f"C{i}" for i in range(10)] )
              if cycle is not None else [f"C{i}" for i in range(10)])

    handles: List[Any] = []
    labels:  List[str] = []

    for i, spec in enumerate(specs):
        res = build_curve_data(ds, spec, groups=groups, verbose=verbose)

        # default style, overridable by spec["style"]
        sty = dict(color=colors[i % len(colors)], lw=2, label=spec.get("name", f"curve {i+1}"))
        sty.update(res.get("style", {}))
        lab = sty.pop("label", spec.get("name", f"curve {i+1}"))

        if res["kind"] == "binned":
            d = res["data"]
            have_band = ("y_lo" in d and "y_hi" in d and np.isfinite(d["y_lo"]).any())
            if have_band:
                ax.fill_between(d["x_mid"], d["y_lo"], d["y_hi"],
                                color=sty.get("color", None), alpha=0.15, linewidth=0)
            h = ax.plot(d["x_mid"], d["y_val"], **sty)[0]
        else:
            d = res["data"]
            h = ax.scatter(d["x"], d["y"], s=d["s"], alpha=d["alpha"],
                           color=sty.get("color", None))

        handles.append(h)
        labels.append(lab)

    # axis labels (fallback from first spec)
    if xlabel is None and specs:
        xlabel = specs[0].get("x_label", specs[0].get("x", None))
    if ylabel is None and specs:
        ylabel = specs[0].get("y_label", specs[0].get("y", None))
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    # legend
    if show_legend and handles:
        if legend_outside:
            ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                      borderaxespad=0.0, frameon=False, fontsize=legend_fontsize)
        else:
            ax.legend(handles, labels, loc="best", frameon=False, fontsize=legend_fontsize)

    # -------- ALWAYS SAVE (FIG_DIR + subdir logic)
    user_defined = "FVCOM_PLOT_SUBDIR" in os.environ
    if not user_defined:
        os.environ["FVCOM_PLOT_SUBDIR"] = "curves"  # default subfolder for this plot type

    try:
        out_folder = out_dir(base_dir, figures_root)
    finally:
        if not user_defined:
            os.environ.pop("FVCOM_PLOT_SUBDIR", None)

    os.makedirs(out_folder, exist_ok=True)

    # auto-build stem if not supplied
    if stem is None:
        scope_tags = [_scope_tag(s) for s in specs] if specs else ["Domain"]
        depth_tags = [_depth_tag_from_spec(s) for s in specs] if specs else ["AllDepth"]
        time_labels= [_time_label_from_spec(s) for s in specs] if specs else ["AllTime"]

        scope_tag_final = scope_tags[0] if _all_equal(scope_tags) else "MultiScope"
        depth_tag_final = depth_tags[0] if _all_equal(depth_tags) else "MixedDepth"
        time_label_final= time_labels[0] if _all_equal(time_labels) else "MixedTime"

        content_bits: List[str] = []
        if specs:
            xk = specs[0].get("x"); yk = specs[0].get("y")
            if xk and yk:
                content_bits.append(f"{xk}_vs_{yk}")
        if len(specs) > 1:
            content_bits.append(f"{len(specs)}curves")
        content = "__".join(content_bits) if content_bits else "Curves"

        stem = f"{scope_tag_final}__{depth_tag_final}__{time_label_final}__{content}"

    fname = f"{file_prefix(base_dir)}__Curves__{stem}.png"
    path = os.path.join(out_folder, fname)

    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
