#!/usr/bin/env python3
# examples/plot_maps.py

from __future__ import annotations
import os
from matplotlib.colors import LogNorm

import numpy as np
import xarray as xr

from fvcomersemviz.io import load_from_base
from fvcomersemviz.plot import (
    plot_call, info, hr, kv, bullet,
    list_files, summarize_files, print_dataset_summary,
)
from fvcomersemviz.utils import out_dir, file_prefix
from fvcomersemviz.plots.maps import domain_map, region_map
import matplotlib
matplotlib.use("Agg", force=True)  # headless backend

# ---------------------------------------------------------------------
# Project paths (EDIT)
# ---------------------------------------------------------------------
BASE_DIR     = "/data/proteus1/scratch/yli/project/lake_erie/output_updated_river_var"
FILE_PATTERN = "erie_00??.nc"
FIG_DIR = "/data/proteus1/scratch/moja/projects/Lake_Erie/fviz-plots/"

# ---------------------------------------------------------------------
# Variable groups (same idea as the timeseries runner)
# ---------------------------------------------------------------------
GROUPS = {
    "DOC":   "R1_c + R2_c + R3_c + T1_30d_c + T2_30d_c",
    "phyto": ["P1_c", "P2_c", "P4_c", "P5_c"],
    "zoo":   ["Z4_c", "Z5_c", "Z6_c"],
    "chl":   "P1_Chl + P2_Chl + P4_Chl + P5_Chl",
}

# ---------------------------------------------------------------------
# Region specs 
# ---------------------------------------------------------------------
REGIONS = [
    ("Central", {
        "shapefile": "../data/shapefiles/central_basin_single.shp"
    }),
    ("East", {
        "shapefile": "../data/shapefiles/east_basin_single.shp"
    }),
    ("West", {
        "shapefile": "../data/shapefiles/west_basin_single.shp"
    }),
 
# csv example
#    ("West", {
#        "csv_boundary": "/data/proteus1/backup/rito/Models/FVCOM/fvcom-projects/erie/python/postprocessing/west_stations.csv",
#        "lon_col": "lon", 
#        "lat_col": "lat",
#        "convex_hull": True,   # <- wrap points
#        # "sort": "auto",      # (use this if your CSV is a boundary but unordered)
#    }),
]   


# ---------------------------------------------------------------------
# Per-variable plotting styles (maps use cmap/vmin/vmax/norm if provided)
# ---------------------------------------------------------------------
PLOT_STYLES = {
    "temp":   {"cmap": "coolwarm"},
    "DOC":   {"cmap": "viridis"},
    "chl":   {"cmap": "Greens", "vmin": 0.0, "vmax": 5.0},         # fixed limits
    "phyto": {"cmap": "YlGn"},
    "zoo":   {"cmap": "PuBu", "norm": LogNorm(1e-4, 1e0)},          # example log scaling
}

# Convenience time windows
APR_OCT = [4, 5, 6, 7, 8, 9, 10]
YEAR_2018 = [2018]

def main():
    print(hr("="))
    print("fvcomersemviz: Maps Runner")
    print(hr("="))

    # Discover files & load dataset
    info(" Discovering files")
    files = list_files(BASE_DIR, FILE_PATTERN)
    summarize_files(files)
    if not files:
        print("No files found; abort.")
        return

    info(" Loading dataset")
    ds = load_from_base(BASE_DIR, FILE_PATTERN)
    print_dataset_summary(ds)

    out_folder = out_dir(BASE_DIR, FIG_DIR)
    prefix     = file_prefix(BASE_DIR)
    kv("Figure folder", out_folder)
    kv("Filename prefix", prefix)

    # ===============================================================
    # Examples
    # ===============================================================

    # Domain, SURFACE, time-mean, per-var cmap from styles
    bullet("\n Domain mean maps at SURFACE (DOC, chl) - colour-by-var")
    plot_call(
        domain_map,
        ds=ds,
        variables=["DOC", "aice"],
        depth="surface",
        base_dir=BASE_DIR, figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,     # <- per-variable cmap/vmin/vmax/log
        grid_on=True,           # <- draw triangular grid overlay
        months=[7],             # July across all years
        dpi=150, figsize=(8, 6),
        verbose=False,
    )

    # Domain, BOTTOM, instantaneous at two timestamps
    bullet("\n Domain instantaneous maps at BOTTOM (phyto) - two timestamps")
    plot_call(
        domain_map,
        ds=ds,
        variables=["phyto"],
        depth="bottom",
        at_times=["2018-06-15 00:00", "2018-09-01 12:00"],  # nearest match in data
        time_method="nearest",
        base_dir=BASE_DIR, figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
        grid_on=False,
        dpi=150, figsize=(8, 6),
        verbose=False,
    )

    # Domain, absolute z depth (e.g., z = -8 m), Apr-Oct 2018
    bullet("\n Domain mean at ABSOLUTE depth z = -8 m (phyto), Apr-Oct 2018")
    plot_call(
        domain_map,
        ds=ds,
        variables=["phyto"],
        depth=-8.0,                 # absolute depth in metres (negative downward)
        years=YEAR_2018,
        months=APR_OCT,
        base_dir=BASE_DIR, figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
        grid_on=True,
        dpi=150, figsize=(8, 6),
        verbose=False,
    )

    # Region (Central), DEPTH-AVG, time-mean with grid overlay
    bullet("\n Region=CENTRAL, depth-averaged, time-mean (zoo with log norm)")
    plot_call(
        region_map,
        ds=ds,
        variables=["zoo"],
        regions=[REGIONS[0]],       # Central
        depth="depth_avg",
        base_dir=BASE_DIR, figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
        grid_on=True,
        dpi=150, figsize=(8, 6),
        verbose=False,
    )

    # Region (West), sigma selection examples
    bullet("\n Region=WEST, sigma layer index k=5 (DOC) & sigma value s=-0.7 (chl)")
    # k = 5 (sigma index)
    plot_call(
        region_map,
        ds=ds,
        variables=["DOC"],
        regions=[REGIONS[2]],       # West
        depth=5,                    # == ("siglay_index", 5)
        base_dir=BASE_DIR, figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
        grid_on=False,
        dpi=150, figsize=(8, 6),
        verbose=False,
    )
    # s = -0.7 (sigma value)
    plot_call(
        region_map,
        ds=ds,
        variables=["chl"],
        regions=[REGIONS[2]],
        depth=-0.7,                 # == ("sigma", -0.7)
        base_dir=BASE_DIR, figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
        grid_on=False,
        dpi=150, figsize=(8, 6),
        verbose=False,
    )

    #  Region (East), absolute z instantaneous
    bullet("\n Region=EAST, ABSOLUTE z = -15 m instantaneous (DOC)")
    plot_call(
        region_map,
        ds=ds,
        variables=["DOC"],
        regions=[REGIONS[1]],       # East
        depth=("z_m", -15.0),       # explicit tuple form
        at_time="2018-08-15 00:00",
        base_dir=BASE_DIR, figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
        grid_on=True,
        dpi=150, figsize=(8, 6),
        verbose=False,
    )

    #  Region (East), 2D var
    bullet("\n Region=EAST, plotting a 2d var (aice)")
    plot_call(
        region_map,
        ds=ds,
        variables=["aice"],
        regions=[REGIONS[1]],       # East
        at_time="2018-08-15 00:00",
        base_dir=BASE_DIR, figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
        grid_on=True,
        dpi=150, figsize=(8, 6),
        verbose=False,
    )

    

    print("\nDone. Maps written under:", out_folder)

if __name__ == "__main__":
    xr.set_options(use_new_combine_kwarg_defaults=True)
    main()
