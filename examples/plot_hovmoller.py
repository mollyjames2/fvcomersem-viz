#!/usr/bin/env python3
# examples/hovmoller_examples.py

from __future__ import annotations
import os
import sys
import warnings
import matplotlib
matplotlib.use("Agg", force=True)  # headless backend for batch runs

import numpy as np
import xarray as xr
from matplotlib.colors import LogNorm

from fvcomersemviz.io import load_from_base
from fvcomersemviz.plot import (
    hr, info, kv, bullet,
    list_files, summarize_files, print_dataset_summary, plot_call
)
from fvcomersemviz.utils import out_dir, file_prefix
from fvcomersemviz.plots.hovmoller import station_hovmoller

# ---------------------------------------------------------------------
# Project paths (EDIT THESE)
# ---------------------------------------------------------------------
BASE_DIR     = "/data/proteus1/scratch/yli/project/lake_erie/output_updated_river_var"
FILE_PATTERN = "erie_00??.nc"
FIG_DIR = "/data/proteus1/scratch/moja/projects/Lake_Erie/fviz-plots/"

# ---------------------------------------------------------------------
# Variable groups / composites (optional)
# ---------------------------------------------------------------------
GROUPS = {
    "DOC":   "R1_c + R2_c + R3_c + T1_30d_c + T2_30d_c",
    "phyto": ["P1_c", "P2_c", "P4_c", "P5_c"],
    "zoo":   ["Z4_c", "Z5_c", "Z6_c"],
    "chl":   "P1_Chl + P2_Chl + P4_Chl + P5_Chl",
}

# ---------------------------------------------------------------------
# Stations (name, lat, lon)
# ---------------------------------------------------------------------
STATIONS = [
    ("WE12", 41.90, -83.10),
    ("WE13", 41.80, -83.20),
]

# Example time window
MONTHS = [4, 5, 6, 7, 8, 9, 10]   # Apr-Oct
YEARS  = [2018]

# If your station_hovmoller supports per-variable styles, define them here
# and pass `styles=PLOT_STYLES` to plot_call (lines are included but commented).

PLOT_STYLES = {
     "chl":   {"cmap": "Greens", "vmin": 0.0, "vmax": 5.0},
     "DOC":   {"cmap": "viridis"},
     "zoo":   {"cmap": "PuBu"},  # or "norm": LogNorm(1e-4, 1e0)
 }

def main():
    if not os.environ.get("PYTHONWARNINGS"):
        warnings.filterwarnings("default")

    print(hr("=")); print("Hovmoller examples"); print(hr("="))

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
    # 1) Sigma-coordinate Hovmoller at WE12 for chlorophyll
    #    Entire model run; automatic color limits (robust).
    # ===============================================================
    info(" Example 1: WE12  chl (sigma), full run")
    plot_call(
        station_hovmoller,
        ds=ds,
        variables=["chl"],
        stations=[STATIONS[0]],
        axis="sigma",
        base_dir=BASE_DIR, figures_root=FIG_DIR, groups=GROUPS,
        styles=PLOT_STYLES, 
    )

    # ===============================================================
    # 2) Absolute-depth Hovmoller at WE12 for DOC
    #    Apr-Oct 2018, explicit z grid from -20 m to surface.
    # ===============================================================
    info(" Example 2: WE12  DOC (z), Apr-Oct 2018")
    z_levels = np.linspace(-20.0, 0.0, 60)
    plot_call(
        station_hovmoller,
        ds=ds,
        variables=["DOC"],
        stations=[STATIONS[0]],
        axis="z",
        z_levels=z_levels,                 # omit to auto-build from the column
        months=MONTHS, years=YEARS,
        base_dir=BASE_DIR, figures_root=FIG_DIR, groups=GROUPS,
        styles=PLOT_STYLES,
    )

    # ===============================================================
    # 3) Another station/variable: WE13, zooplankton on sigma
    #    Same time window, default colormap & robust limits.
    # ===============================================================
    info(" Example 3: WE13  zoo (sigma), Apr-Oct 2018")
    plot_call(
        station_hovmoller,
        ds=ds,
        variables=["zoo"],
        stations=[STATIONS[1]],
        axis="sigma",
        months=MONTHS, years=YEARS,
        base_dir=BASE_DIR, figures_root=FIG_DIR, groups=GROUPS,
        styles=PLOT_STYLES,
    )

    print(hr("=")); print("Done"); print(hr("="))

if __name__ == "__main__":
    xr.set_options(use_new_combine_kwarg_defaults=True)
    main()
