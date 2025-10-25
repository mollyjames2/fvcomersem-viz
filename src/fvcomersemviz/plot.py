# fvcomersemviz/plot.py

"""
Console + plotting helpers for demos/runners.

This module centralizes:
- pretty printing (hr, info, bullet, kv)
- progress bar registration (Dask)
- file discovery summaries
- plotting wrapper (passes verbose=True when supported)
- dataset summary printer (dims/coords/time)
- path existence checks for region specs
- sample output listing

Keep these functions generic so any example script can reuse them.
"""

from __future__ import annotations
from typing import Any, Dict, List, Sequence, Tuple
import os
import glob
import textwrap
import numpy as np
import pandas as pd
import xarray as xr
import inspect
import contextlib
import matplotlib.pyplot as plt

# Optional imports are local (only for nicer UX); module works without them.
try:
    import dask  # type: ignore  # noqa: F401

    _HAS_DASK = True
except Exception:
    _HAS_DASK = False

# Local package import for file discovery
from .io import discover_paths


# ---------------------------
# Pretty printing utilities
# ---------------------------
def hr(char: str = "=", width: int = 78) -> str:
    """Horizontal rule."""
    return char * width


def info(title: str) -> None:
    """Section header."""
    print()
    print(hr("="))
    print(title)
    print(hr("-"))


def bullet(msg: str, indent: int = 2) -> None:
    """Indented, wrapped bullet text."""
    pad = " " * indent
    for line in textwrap.dedent(str(msg)).rstrip().splitlines():
        print(pad + line)


def kv(label: str, value: Any) -> None:
    """Key: Value printing with basic alignment."""
    print(f"  - {label:<18} {value}")


# ---------------------------
# Dask progress bar
# ---------------------------
def try_register_progress_bar(show: bool = True) -> None:
    """
    Enable Dask progress bar if available (no-op if dask is absent or show=False).
    """
    if not show or not _HAS_DASK:
        return
    try:
        from dask.diagnostics import ProgressBar  # type: ignore

        ProgressBar().register()
        bullet("Dask progress bar: enabled")
    except Exception as e:
        bullet(f"Dask progress bar: unavailable ({e})")


# ---------------------------
# File discovery summary
# ---------------------------
def list_files(base_dir: str, pattern: str) -> List[str]:
    """Use package discovery to list files; warns if none are found."""
    return discover_paths(base_dir, pattern)


def summarize_files(files: List[str]) -> None:
    """Print a succinct summary (head/tail) of matched files."""
    if not files:
        bullet("No files matched. Double-check BASE_DIR and FILE_PATTERN.")
        return
    kv("Matched files", len(files))
    head = files[:3]
    tail = files[-3:] if len(files) > 3 else []
    for p in head:
        bullet(f"• {p}")
    if tail:
        bullet("…")
        for p in tail:
            bullet(f"• {p}")


# ---------------------------
# plotting wrapper
# ---------------------------


def plot_call(fn, *, verbose: bool = False, **kwargs):
    """
    Call a plotting function.

    Behavior:
      - `verbose` controls whether print() output from the function is shown.
        • verbose=False -> suppress stdout/stderr during the call
        • verbose=True  -> show stdout/stderr
      - If the target function has a `verbose` kwarg, we pass this same value.
        If it doesn't, we just silence/allow prints as requested.
    """
    has_verbose = "verbose" in inspect.signature(fn).parameters
    if has_verbose:
        kwargs["verbose"] = verbose
    else:
        # don't accidentally pass an unknown kwarg
        kwargs.pop("verbose", None)

    if verbose:
        # Show prints
        return fn(**kwargs)

    # Suppress prints from inside fn (stdout + stderr)
    with contextlib.ExitStack() as stack:
        with open(os.devnull, "w") as devnull:
            stack.enter_context(contextlib.redirect_stdout(devnull))
            stack.enter_context(contextlib.redirect_stderr(devnull))
            return fn(**kwargs)


# ---------------------------
# Dataset + output summaries
# ---------------------------
def print_dataset_summary(ds: xr.Dataset) -> None:
    """Print core info: dims/coords/time coverage and presence of key vars."""
    kv("Dimensions", dict(ds.sizes))
    present_coords = [
        c
        for c in [
            "time",
            "lon",
            "lat",
            "lonc",
            "latc",
            "siglay",
            "siglev",
            "nele",
            "node",
        ]
        if c in ds
    ]
    kv("Key coords", present_coords)
    for c in ["art1", "Itime", "Itime2", "nv"]:
        kv(f"Has '{c}'", c in ds)

    if "time" in ds:
        try:
            t = pd.to_datetime(ds["time"].values)
            kv("Time start", str(pd.Timestamp(t[0])))
            kv("Time end", str(pd.Timestamp(t[-1])))
            kv("Timesteps", t.size)
        except Exception as e:
            kv("Time coverage", f"unavailable ({e})")
    else:
        kv("Time coverage", "no 'time' coordinate found")


def ensure_paths_exist(regions: List[Tuple[str, Dict[str, Any]]]) -> None:
    """
    Warn (do not fail) if shapefiles/CSVs referenced in region specs are missing.
    """
    for name, spec in regions:
        if "shapefile" in spec:
            path = spec["shapefile"]
            if not os.path.exists(path):
                bullet(f"[warn] Region '{name}': shapefile not found: {path}")
        if "csv_boundary" in spec:
            path = spec["csv_boundary"]
            if not os.path.exists(path):
                bullet(f"[warn] Region '{name}': CSV boundary not found: {path}")


def sample_output_listing(fig_folder: str, prefix: str) -> None:
    """List a few generated figure paths to show success."""
    pattern = os.path.join(fig_folder, f"{prefix}__*__Timeseries.png")
    files = sorted(glob.glob(pattern))
    kv("Figures created", len(files))
    for p in files[:5]:
        bullet(f"• {p}")
    if len(files) > 5:
        bullet("…")


def stacked_fraction_bar(
    ax,
    fractions,
    labels,
    *,
    title: str | None = None,
    colors: list[str] | None = None,
    y_label: str = "Fraction of group",
    show_legend: bool = True,
    legend_loc: str = "upper right",
    legend_fontsize: int = 8,
    annotate: bool = False,
    annotate_min: float = 0.03,  # annotate segments ≥ 3%
    annotate_fmt: str = "{:.1f}%",
    clip_to_1: bool = True,
):
    """
    Draw a single stacked bar whose segments sum to ~1.0, labeled with names and (optionally) percentages.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    fractions : Sequence[float]
        Segment heights (typically fractions in [0,1]); NaNs/inf treated as 0.
    labels : Sequence[str]
        Segment labels (same length as fractions).
    title : str, optional
        Axes title.
    colors : list[str], optional
        Optional list of matplotlib color specs; cycles if shorter than fractions.
    y_label : str
        Y-axis label.
    show_legend : bool
        Whether to show a legend with segment labels and percentages.
    legend_loc : str
        Legend location string.
    legend_fontsize : int
        Legend font size.
    annotate : bool
        If True, writes percentage text on top of segments ≥ `annotate_min`.
    annotate_min : float
        Minimum segment height to annotate (in fraction units).
    annotate_fmt : str
        Formatter for percentages, e.g., "{:.1f}%".
    clip_to_1 : bool
        If True, y-limits are [0,1]; else max(1, sum(fractions)).

    Returns
    -------
    None
    """

    f = np.asarray(list(fractions), dtype=float)
    f[~np.isfinite(f)] = 0.0
    labs = list(labels)

    # bar config
    x = [0]
    bottom = 0.0
    handles = []

    for i, (h, lab) in enumerate(zip(f, labs)):
        color = None if colors is None else colors[i % len(colors)]
        h = float(h) if np.isfinite(h) and h > 0 else 0.0
        rects = ax.bar(x, [h], bottom=bottom, color=color, label=lab if not show_legend else None)
        handles.append(rects[0])
        if annotate and h >= annotate_min:
            pct_txt = annotate_fmt.format(h * 100.0)
            ax.text(
                x[0],
                bottom + h / 2.0,
                pct_txt,
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
        bottom += h

    # y-limits, labels, cosmetics
    if clip_to_1:
        ax.set_ylim(0, 1)
    else:
        ax.set_ylim(0, max(1.0, bottom))

    ax.set_xticks([])
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)

    if show_legend:
        # Build legend entries with percentages
        pct_labels = []
        for h, lab in zip(f, labs):
            pct = 0.0 if not np.isfinite(h) else h * 100.0
            pct_labels.append(f"{lab} ({pct:.1f}%)")
        ax.legend(handles, pct_labels, loc=legend_loc, fontsize=legend_fontsize, frameon=False)


def stacked_fraction_bars(
    ax,
    bars: list[Sequence[float]],
    labels_per_bar: list[Sequence[str]],
    *,
    bar_names: list[str],
    colors: list[str] | None = None,  # optional palette list; used if provided
    y_label: str = "Fraction",
    show_legend: bool = True,
    legend_loc: str = "upper right",
    legend_fontsize: int = 8,
    legend_outside: bool = True,  # NEW: place legend outside to avoid overlap
    annotate: bool = False,
    annotate_min: float = 0.03,  # annotate segments ≥ 3%
    annotate_fmt: str = "{:.1f}%",
    clip_to_1: bool = True,
    xtick_rotation: float = 45.0,
    bar_width: float = 0.3,  # thinner bars by default
) -> None:
    """
    Draw multiple stacked bars on a single axes with consistent colors per label.

    bars           : list of fraction sequences (one per bar)
    labels_per_bar : list of label sequences (same shape as `bars`)
    bar_names      : x tick labels for each bar
    legend_outside : if True, dock legend outside the right edge to avoid overlap
    """

    n = len(bars)
    if not (len(labels_per_bar) == n and len(bar_names) == n):
        raise ValueError("bars, labels_per_bar, and bar_names must have the same length")

    # Collect unique labels in first-seen order
    unique_labels: list[str] = []
    for labs in labels_per_bar:
        for lab in labs:
            if lab not in unique_labels:
                unique_labels.append(lab)

    # Palette: provided or current mpl cycle; fall back to C0..C9
    if colors is None:
        try:
            cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        except Exception:
            cycle = []
        if not cycle:
            cycle = [f"C{i}" for i in range(10)]
    else:
        cycle = list(colors)

    label_to_color = {lab: cycle[i % len(cycle)] for i, lab in enumerate(unique_labels)}

    # Plot bars
    x = np.arange(n, dtype=float)
    handles = {}
    for i, (fracs, labs) in enumerate(zip(bars, labels_per_bar)):
        f = np.asarray(list(fracs), dtype=float)
        f[~np.isfinite(f)] = 0.0
        bottom = 0.0
        for h, lab in zip(f, labs):
            h = float(h) if np.isfinite(h) and h > 0 else 0.0
            if h <= 0:
                continue
            color = label_to_color.get(lab)
            rects = ax.bar([x[i]], [h], width=bar_width, bottom=bottom, color=color, label=lab)
            if lab not in handles:
                handles[lab] = rects[0]  # one handle per label for legend
            if annotate and h >= annotate_min:
                ax.text(
                    x[i],
                    bottom + h / 2.0,
                    annotate_fmt.format(h * 100.0),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
            bottom += h

    # Axis cosmetics
    ax.set_xticks(x, bar_names, rotation=xtick_rotation, ha="right")
    if clip_to_1:
        ax.set_ylim(0, 1)
    else:
        ymax = max(1.0, ax.get_ylim()[1])
        ax.set_ylim(0, ymax)
    if y_label:
        ax.set_ylabel(y_label)

    # Legend (outside by default to avoid overlap)
    if show_legend and handles:
        ordered_labels = [lab for lab in unique_labels if lab in handles]
        if legend_outside:
            # dock outside the axes on the right; constrained_layout or bbox_inches='tight' will handle spacing
            ax.legend(
                [handles[lab] for lab in ordered_labels],
                ordered_labels,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                fontsize=legend_fontsize,
                frameon=False,
            )
        else:
            ax.legend(
                [handles[lab] for lab in ordered_labels],
                ordered_labels,
                loc=legend_loc,
                fontsize=legend_fontsize,
                frameon=False,
            )


__all__ = [
    "hr",
    "info",
    "bullet",
    "kv",
    "try_register_progress_bar",
    "list_files",
    "summarize_files",
    "print_dataset_summary",
    "ensure_paths_exist",
    "sample_output_listing",
    "stacked_fraction_bar",
    "stacked_fraction_bars",
]
