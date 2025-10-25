#!/usr/bin/env python3
# examples/plot_kde.py
"""
=============================

A narrated demo of **2x2 Gaussian KDE (density) plots** that relate an ERSEM
functional group's nutrient stoichiometry (N:C, P:C) to any chosen model
variable (e.g. biomass, Chl, DOC).

Each figure shows:

    ┌──────────────────────────────┬──────────────────────────────┐
    │ Surface  N:C vs variable     │ Surface  P:C vs variable     │
    ├──────────────────────────────┼──────────────────────────────┤
    │ Bottom   N:C vs variable     │ Bottom   P:C vs variable     │
    └──────────────────────────────┴──────────────────────────────┘

All time x spatial samples inside the chosen **region** and **time window**
are pooled. Panels are skipped automatically if insufficient finite data exist.

This script demonstrates:
  • Domain-wide and region-specific stoichiometry-variable relationships
  • Different time windows (e.g. JJA, Apr-Oct)
  • Support for composites (e.g. phyto, chl) via GROUPS definitions
  • Automatic saving of figures under FIG_DIR/<basename(BASE_DIR)>/kde_stoichiometry/

Run:
    pip install -e .
    python examples/plot_kde.py
"""

from __future__ import annotations
import os
import sys
import warnings
import matplotlib

matplotlib.use("Agg", force=True)  # headless backend for batch runs

import xarray as xr
from fvcomersemviz.io import load_from_base
from fvcomersemviz.plot import (
    hr,
    info,
    kv,
    bullet,
    list_files,
    summarize_files,
    print_dataset_summary,
    plot_call,
    sample_output_listing,
)
from fvcomersemviz.utils import out_dir, file_prefix
from fvcomersemviz.plots.kde_stoichiometry import kde_stoichiometry_2x2

# ---------------------------------------------------------------------
# Project paths (EDIT THESE)
# ---------------------------------------------------------------------
BASE_DIR = "/data/proteus1/scratch/yli/project/lake_erie/output_updated_river_var"
FILE_PATTERN = "erie_00??.nc"
FIG_DIR = "/data/proteus1/scratch/moja/projects/Lake_Erie/fviz-plots/"

# ---------------------------------------------------------------------
# Variable groups / composites (optional)
# ---------------------------------------------------------------------
GROUPS = {
    "DOC": "R1_c + R2_c + R3_c + T1_30d_c + T2_30d_c",
    "phyto": ["P1_c", "P2_c", "P4_c", "P5_c"],
    "zoo": ["Z4_c", "Z5_c", "Z6_c"],
    "chl": "P1_Chl + P2_Chl + P4_Chl + P5_Chl",
}

# ---------------------------------------------------------------------
# Regions (OPTIONAL shapefiles)
# ---------------------------------------------------------------------
REGIONS = [
    ("Central", {"shapefile": "../data/shapefiles/central_basin_single.shp"}),
    ("East", {"shapefile": "../data/shapefiles/east_basin_single.shp"}),
]

# ---------------------------------------------------------------------
# Time windows (examples)
# ---------------------------------------------------------------------
MONTHS_JJA = [6, 7, 8]  # Jun-Aug
MONTHS_APR_OCT = [4, 5, 6, 7, 8, 9, 10]  # Apr-Oct
YEARS_2018 = [2018]

# ---------------------------------------------------------------------
# Per-variable style overrides (optional)
# ---------------------------------------------------------------------
PLOT_STYLES = {
    # e.g. "P5_c": {"cmap": "magma", "vmin": 0.0, "vmax": 100.0},
}

# Common fast options for all calls
FAST = dict(
    method="kde",  # density approximation
    sample_max=150_000,  # cap pairs; visually identical, much faster
    hist_sigma=1.2,  # gentle blur (in bins)
    grids=100,  # smaller grid is faster
    bw_method="scott",  # ignored when method="hist"
    verbose=False,  # quiet logs
)


def main():
    if not os.environ.get("PYTHONWARNINGS"):
        warnings.filterwarnings("default")
    xr.set_options(use_new_combine_kwarg_defaults=True)

    print(hr("="))
    print("KDE Stoichiometry examples (2x2 density: N:C/P:C x variable)")
    print(hr("="))

    # Discover & load
    info(" Discovering files")
    files = list_files(BASE_DIR, FILE_PATTERN)
    summarize_files(files)
    if not files:
        print("No files found; abort.")
        sys.exit(2)

    info(" Loading dataset")
    ds = load_from_base(BASE_DIR, FILE_PATTERN)
    print_dataset_summary(ds)

    # Where output will go (informational)
    out_folder = out_dir(BASE_DIR, FIG_DIR)
    prefix = file_prefix(BASE_DIR)
    kv("Figure folder", out_folder)
    kv("Filename prefix", prefix)

    # ===============================================================
    # 1) DOMAIN • JJA • group=P5 • variable=P5_c
    # ===============================================================
    info(" Example 1: DOMAIN • JJA • group=P5 • variable=P5_c")
    bullet(
        "Panels: [surf N:C vs P5_c | surf P:C vs P5_c; bottom N:C vs P5_c | bottom P:C vs P5_c]"
    )
    plot_call(
        kde_stoichiometry_2x2,
        ds=ds,
        group="P5",
        variable="P5_c",
        region=None,
        months=MONTHS_JJA,
        years=None,
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        groups=GROUPS,
        min_samples=200,
        scatter_underlay=800,
        styles=PLOT_STYLES,
        **FAST,
    )

    # ===============================================================
    # 2) REGION • Apr-Oct 2018 • group=P5 • variable=phyto (composite)
    # ===============================================================
    if REGIONS:
        info(" Example 2: REGION(Central) • Apr-Oct 2018 • group=P5 • variable=phyto")
        bullet("Pools all samples inside polygon; skips panels lacking finite data.")
        plot_call(
            kde_stoichiometry_2x2,
            ds=ds,
            group="P5",
            variable="phyto",
            region=REGIONS[0],
            months=MONTHS_APR_OCT,
            years=YEARS_2018,
            base_dir=BASE_DIR,
            figures_root=FIG_DIR,
            groups=GROUPS,
            min_samples=200,
            scatter_underlay=1200,
            styles=PLOT_STYLES,
            **FAST,
        )

    # ===============================================================
    # 3) DOMAIN • full run • group=P5 • variable=chl (composite)
    # ===============================================================
    info(" Example 3: DOMAIN • full run • group=P5 • variable=chl")
    plot_call(
        kde_stoichiometry_2x2,
        ds=ds,
        group="P5",
        variable="chl",
        region=None,
        months=None,
        years=None,
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        groups=GROUPS,
        min_samples=300,
        scatter_underlay=1500,
        styles=PLOT_STYLES,
        **FAST,
    )

    # ===============================================================
    # 4) REGION COMPARISON • JJA 2018 • group=P5 • variable=P5_c
    # ===============================================================
    if len(REGIONS) >= 2:
        info(" Example 4: REGION COMPARISON • JJA 2018 • group=P5 • variable=P5_c")
        for reg in REGIONS[:2]:
            bullet(f"Region: {reg[0]}")
            plot_call(
                kde_stoichiometry_2x2,
                ds=ds,
                group="P5",
                variable="P5_c",
                region=reg,
                months=MONTHS_JJA,
                years=YEARS_2018,
                base_dir=BASE_DIR,
                figures_root=FIG_DIR,
                groups=GROUPS,
                min_samples=180,
                scatter_underlay=800,
                styles=PLOT_STYLES,
                **FAST,
            )

    # -----------------------------
    # Output recap: list a few PNGs
    # -----------------------------
    info(" Output recap")
    bullet(
        "Figures are named like:\n"
        "  <basename(BASE_DIR)>__KDE-Stoich__<group>__<variable>__<Region>__<TimeLabel>.png"
    )
    bullet(f"Listing a few outputs in: {out_folder}")
    sample_output_listing(out_folder, prefix)

    print(hr("="))
    print("Done")
    print(hr("="))


if __name__ == "__main__":
    main()
