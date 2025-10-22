#!/usr/bin/env python3
# examples/plot_curves.py
"""
Curves diagnostics demo
-----------------------

This script demonstrates how to use `fvcomersemviz.plots.curves.plot_curves`
to make general x-y diagnostics from FVCOM-ERSEM output.

Examples included:
  1) Light vs Chlorophyll - surface - JJA 2018 - Central vs East (binned)
  2) Temperature vs Total Phytoplankton C - depth_avg - Apr-Oct 2018 - Domain (binned)
  3) Temperature vs P5_Cfix (rate) - depth_avg - Apr-Oct 2018 - Domain (binned)

Edit BASE_DIR / FILE_PATTERN / FIG_DIR and region shapefile paths before running.

Run:
    pip install -e .
    python examples/plot_curves.py
"""

from __future__ import annotations
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg", force=True)  # headless / batch-safe

from fvcomersemviz.io import load_from_base
from fvcomersemviz.plot import (
    hr, info, kv, bullet,
    list_files, summarize_files, print_dataset_summary,
    ensure_paths_exist,
)
from fvcomersemviz.utils import out_dir, file_prefix
from fvcomersemviz.plots.curves import plot_curves

# ----------------------------------------------------------------------------- #
# Project paths (EDIT THESE)
# ----------------------------------------------------------------------------- #
BASE_DIR     = "/data/proteus1/scratch/yli/project/lake_erie/output_updated_river_var"
FILE_PATTERN = "erie_00??.nc"
FIG_DIR      = "/data/proteus1/scratch/moja/projects/Lake_Erie/fviz-plots/"

# ----------------------------------------------------------------------------- #
# Regions & Stations (EDIT shapefile paths if needed)
# ----------------------------------------------------------------------------- #
REGIONS = [
    ("Central", {"shapefile": "../data/shapefiles/central_basin_single.shp"}),
    ("East",    {"shapefile": "../data/shapefiles/east_basin_single.shp"}),
]
STATION = ("WE12", 41.90, -83.10)  # (name, lat, lon) - not used below, but handy

# ----------------------------------------------------------------------------- #
# Time windows
# ----------------------------------------------------------------------------- #
MONTHS_JJA     = [6, 7, 8]              # Jun-Aug
MONTHS_APR_OCT = [4, 5, 6, 7, 8, 9, 10] # Apr-Oct
YEARS_2018     = [2018]

# ----------------------------------------------------------------------------- #
# GROUPS (aliases, composites, derived metrics)
#   Keep expressions here; refer to them by name in specs.
# ----------------------------------------------------------------------------- #
GROUPS = {
    # Aliases
    "PAR": "light_parEIR",
    "DIN": "N3_n + N4_n",

    # Composites present in the files
    "chl_total":       "P1_Chl + P2_Chl + P4_Chl",
    "phyto_c_total":   "P1_c  + P2_c  + P4_c",

    # Derived metric (rate available in files for P5 only)
    "P5_spec_prod": "P5_Cfix / (P5_c + 1e-12)",

    # Predicates for 'where'
    "PAR_pos": "light_parEIR > 0",
}

def main():
    if not os.environ.get("PYTHONWARNINGS"):
        warnings.filterwarnings("default")

    print(hr("=")); print("Curves Diagnostics Examples"); print(hr("="))

    # Discover & load
    info(" Discovering files")
    files = list_files(BASE_DIR, FILE_PATTERN)
    summarize_files(files)
    if not files:
        print("No files found; abort.")
        sys.exit(2)

    ensure_paths_exist(REGIONS)

    info(" Loading dataset")
    ds = load_from_base(BASE_DIR, FILE_PATTERN)
    print_dataset_summary(ds)

    # Where outputs will go (informational)
    prefix = file_prefix(BASE_DIR)
    base_out = out_dir(BASE_DIR, FIG_DIR)  # root; plot_curves decides subdir ("curves" by default)
    kv("Base figure folder", base_out)
    kv("Filename prefix", prefix)

    # ========================================================================= #
    # 1) Light vs Chlorophyll - surface - JJA 2018 - Central vs East (binned)
    # ========================================================================= #
    info(" Example 1: Light vs Chlorophyll - surface - JJA 2018 - Central vs East")
    bullet("Binned median with IQR shading; daylight only; one curve per region")
    specs_light_chl = [
        {
            "name": "Central",
            "x": "PAR",                 # alias -> light_parEIR
            "y": "chl_total",           # sum of P1/P2/P4 chlorophyll
            "filters": {"months": MONTHS_JJA, "years": YEARS_2018, "where": "PAR_pos"},
            "depth": "surface",
            "scope": {"region": REGIONS[0]},
            "bin": {"x_bins": 40, "agg": "median", "min_count": 20, "iqr": True},
            "style": {"color": "C0"},
            "x_label": "PAR (light_parEIR)",
            "y_label": "Total chlorophyll (P1+P2+P4)",
        },
        {
            "name": "East",
            "x": "PAR",
            "y": "chl_total",
            "filters": {"months": MONTHS_JJA, "years": YEARS_2018, "where": "PAR_pos"},
            "depth": "surface",
            "scope": {"region": REGIONS[1]},
            "bin": {"x_bins": 40, "agg": "median", "min_count": 20, "iqr": True},
            "style": {"color": "C3"},
        },
    ]
    out1 = plot_curves(
        specs=specs_light_chl, ds=ds, groups=GROUPS,
        show_legend=True, legend_outside=True, legend_fontsize=8,
        base_dir=BASE_DIR, figures_root=FIG_DIR,
        dpi=150,
        # Optional: stem="Light_vs_Chl__surface__JJA2018__Central_vs_East",
    )
    kv("Saved", out1)

    # ========================================================================= #
    # 2) Temperature vs Total Phytoplankton C - depth_avg - Apr-Oct 2018 - Domain
    # ========================================================================= #
    info(" Example 2: Temperature vs Total Phytoplankton C - depth_avg - Apr-Oct 2018 - Domain")
    bullet("Binned median with IQR shading; domain-wide depth-average")
    specs_temp_phytoC = [{
        "name": "Domain",
        "x": "temp",
        "y": "phyto_c_total",   # sum of P1/P2/P4 carbon
        "filters": {"months": MONTHS_APR_OCT, "years": YEARS_2018},
        "depth": "depth_avg",
        "scope": {},            # domain
        "bin": {"x_bins": 32, "agg": "median", "min_count": 20, "iqr": True},
        "style": {"color": "C2"},
        "x_label": "Temperature (°C)",
        "y_label": "Total phytoplankton carbon (P1+P2+P4)",
    }]
    out2 = plot_curves(
        specs=specs_temp_phytoC, ds=ds, groups=GROUPS, show_legend=False,
        base_dir=BASE_DIR, figures_root=FIG_DIR,
        dpi=150,
    )
    kv("Saved", out2)

    # ========================================================================= #
    # 3) Temperature vs P5_Cfix (rate) - depth_avg - Apr-Oct 2018 - Domain
    # ========================================================================= #
    info(" Example 3: Temperature vs P5_Cfix (rate) - depth_avg - Apr-Oct 2018 - Domain")
    bullet("Binned median with IQR shading; group-specific productivity rate (P5 only)")
    specs_temp_p5rate = [{
        "name": "Domain",
        "x": "temp",
        "y": "P5_Cfix",         # direct variable available in files
        "filters": {"months": MONTHS_APR_OCT, "years": YEARS_2018},
        "depth": "depth_avg",
        "scope": {},
        "bin": {"x_bins": 32, "agg": "median", "min_count": 20, "iqr": True},
        "style": {"color": "C4"},
        "x_label": "Temperature (°C)",
        "y_label": "P5 carbon fixation rate (P5_Cfix)",
    }]
    out3 = plot_curves(
        specs=specs_temp_p5rate, ds=ds, groups=GROUPS, show_legend=False,
        base_dir=BASE_DIR, figures_root=FIG_DIR,
        dpi=150,
    )
    kv("Saved", out3)
    
    # ============================================================================
    # 4) Temperature vs P5_Cfix (rate) - depth_avg - Apr-Oct 2018 - Domain
    #    Binned backbone + scatter context (same x/y, same filters/scope/depth)
    # ============================================================================
    info(" Example 4: Temperature vs P5_Cfix (rate) - depth_avg - Apr-Oct 2018 - Domain")
    bullet("Scatter cloud for context + binned median with IQR backbone (P5 only)")
    
    specs_temp_p5rate_backbone = [
        # 1) Scatter cloud for context (drawn first → under the line)
        {
            "name": "All points",
            "x": "temp",
            "y": "P5_Cfix",
            "filters": {"months": MONTHS_APR_OCT, "years": YEARS_2018},
            "depth": "depth_avg",
            "scope": {},
            "scatter": {"s": 6, "alpha": 0.08},
            "style": {"marker": ".", "linewidths": 0, "color": "C4"},
            "x_label": "Temperature (°C)",
            "y_label": "P5 carbon fixation rate (P5_Cfix)",
        },
        # 2) Binned median + IQR "backbone" (drawn second → on top)
        {
            "name": "Median (IQR)",
            "x": "temp",
            "y": "P5_Cfix",
            "filters": {"months": MONTHS_APR_OCT, "years": YEARS_2018},
            "depth": "depth_avg",
            "scope": {},
            "bin": {"x_bins": 32, "agg": "median", "min_count": 20, "iqr": True},
            "style": {"color": "C4", "lw": 2},
        },
    ]
    
    # Plot
    out4 = plot_curves(
        specs=specs_temp_p5rate_backbone,
        ds=ds,
        groups=GROUPS,
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        dpi=150,
        legend_outside=True,
    )
    
    kv("Saved", out4)
    
    print(hr("=")); print("Done"); print(hr("="))


if __name__ == "__main__":
    main()
