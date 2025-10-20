#!/usr/bin/env python3
# examples/plot_composition.py
"""
Examples: Phyto/Zoo composition plots (surface, bottom, depth-avg, selected depth)
-------------------------------------------------------------------------------

This runner demonstrates the composition plotting API:

  1) composition_2x2       -> Surface/Bottom • Phyto and Zoo (stacked fractions)
  2) composition_depth_average
  3) composition_at_depth  -> Selected absolute depth (z < 0 m)

Scopes covered:
  • Domain (no region)
  • Region (shapefile polygon)
  • Station (nearest node to lat/lon)

Edit BASE_DIR / FILE_PATTERN / FIG_DIR (and optional shapefiles) below before running.

Run:
    pip install -e .
    python examples/plot_composition.py
"""

from __future__ import annotations
import os
import sys
import warnings
import matplotlib
matplotlib.use("Agg", force=True)  # headless backend for batch runs

from fvcomersemviz.io import load_from_base
from fvcomersemviz.plot import (
    hr, info, kv, bullet,
    list_files, summarize_files, print_dataset_summary, plot_call,
    sample_output_listing,
)
from fvcomersemviz.utils import out_dir, file_prefix
from fvcomersemviz.plots.composition import (
    composition_surface_bottom,
    composition_depth_average_single,
    composition_at_depth_single,

)

# -----------------------------------------------------------------------------
# Project paths (EDIT THESE)
# -----------------------------------------------------------------------------
BASE_DIR     = "/data/proteus1/scratch/yli/project/lake_erie/output_updated_river_var"
FILE_PATTERN = "erie_00??.nc"
FIG_DIR = "/data/proteus1/scratch/moja/projects/Lake_Erie/fvcomersem-viz/examples/plots/"

# -----------------------------------------------------------------------------
# Groups are defined HERE (not inside the module)
# -----------------------------------------------------------------------------
PHYTO_VARS = ["P1_c", "P2_c", "P4_c"]
ZOO_VARS   = ["Z4_c", "Z5_c", "Z6_c"]

# -----------------------------------------------------------------------------
# Optional region & station examples
# -----------------------------------------------------------------------------
REGION = ("Central", {"shapefile": "data/shapefiles/central_basin_single.shp"})
STATION = ("WE12", 41.90, -83.10)  # (name, lat, lon)

# -----------------------------------------------------------------------------
# Time windows
# -----------------------------------------------------------------------------
MONTHS_JJA     = [6, 7, 8]              # Jun–Aug
MONTHS_APR_OCT = [4, 5, 6, 7, 8, 9, 10] # Apr–Oct
YEARS_2018     = [2018]

def main():
    if not os.environ.get("PYTHONWARNINGS"):
        warnings.filterwarnings("default")

    print(hr("=")); print("Phyto/Zoo Composition Examples"); print(hr("="))

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

    # Output location info
    out_root = out_dir(BASE_DIR, FIG_DIR)
    out_folder = os.path.join(out_root, "composition")
    os.makedirs(out_folder, exist_ok=True)
    prefix = file_prefix(BASE_DIR)
    kv("Figure folder", out_folder)
    kv("Filename prefix", prefix)

    # =========================================================================
    # 1) DOMAIN • JJA 2018 • Surface/Bottom composition
    # =========================================================================
    info(" Example 1: DOMAIN • JJA 2018 • Surface/Bottom composition")
    bullet("Produces stacked bars: [Surface Phyto | Surface Zoo] ; [Bottom Phyto | Bottom Zoo]")
    plot_call(
        composition_surface_bottom,
        ds=ds,
        months=MONTHS_JJA, years=YEARS_2018,
        region=None, station=None,
        base_dir=BASE_DIR, figures_root=FIG_DIR,
        phyto_vars=PHYTO_VARS, zoo_vars=ZOO_VARS,
    )

    # =========================================================================
    # 2) REGION (Central) • Apr–Oct 2018 • Surface/Bottom composition
    # =========================================================================
    info(" Example 2: REGION(Central) • Apr–Oct 2018 • Surface/Bottom composition")
    plot_call(
        composition_surface_bottom,
        ds=ds,
        months=MONTHS_APR_OCT, years=YEARS_2018,
        region=REGION, station=None,
        base_dir=BASE_DIR, figures_root=FIG_DIR,
        phyto_vars=PHYTO_VARS, zoo_vars=ZOO_VARS,
    )

    # =========================================================================
    # 3) DOMAIN • JJA 2018 • Depth-averaged composition
    # =========================================================================
    info(" Example 3: DOMAIN • JJA 2018 • Depth-averaged composition")
    bullet("Depth-average computed across ALL sigma layers (weighted by 'layer_thickness' if available; else simple mean)")
    plot_call(
        composition_depth_average_single,
        ds=ds,
        months=MONTHS_JJA, years=YEARS_2018,
        region=None, station=None,
        base_dir=BASE_DIR, figures_root=FIG_DIR,
        phyto_vars=PHYTO_VARS, zoo_vars=ZOO_VARS,
    )

    # =========================================================================
    # 4) STATION (WE12) • JJA 2018 • Composition at selected depth
    # =========================================================================
    info(" Example 4: STATION(WE12) • JJA 2018 • Composition at z=-10 m")
    plot_call(
        composition_at_depth_single,
        ds=ds,
        z_level=-10.0, tol=0.75,
        months=MONTHS_JJA, years=YEARS_2018,
        region=None, station=STATION,
        base_dir=BASE_DIR, figures_root=FIG_DIR,
        phyto_vars=PHYTO_VARS, zoo_vars=ZOO_VARS,
    )

    print(hr("=")); print("Done"); print(hr("="))


if __name__ == "__main__":
    main()

