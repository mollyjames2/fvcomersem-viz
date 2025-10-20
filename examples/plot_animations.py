#!/usr/bin/env python3
# examples/plots/animate.py

from __future__ import annotations

import warnings
import xarray as xr

from fvcomersemviz.io import load_from_base
from fvcomersemviz.plots.animate import animate_timeseries, animate_maps

# -----------------------------------------------------------------------------
# User inputs (EDIT FOR YOUR PROJECT)
# -----------------------------------------------------------------------------
BASE_DIR     = "/data/proteus1/scratch/yli/project/lake_erie/output_updated_river_var"
FILE_PATTERN = "erie_00??.nc"
FIG_DIR      = "/data/proteus1/scratch/moja/projects/Lake_Erie/fvcomersem-viz/examples/plots/"

# -----------------------------
# Variable groups / composites
# -----------------------------
GROUPS = {
    "DOC":   "R1_c + R2_c + R3_c + T1_30d_c + T2_30d_c",  # dissolved organic carbon (sum of pools)
    "phyto": ["P1_c", "P2_c", "P4_c", "P5_c"],            # total phytoplankton carbon (sum)
    "zoo":   ["Z4_c", "Z5_c", "Z6_c"],                    # total zooplankton carbon (sum)
    "chl":   "P1_Chl + P2_Chl + P4_Chl + P5_Chl",         # total chlorophyll (sum)
}

# Optional styles (line colors applied per variable when present)
PLOT_STYLES = {
    "temp":  {"line_color": "lightblue"},
    "DOC":   {"line_color": "blue"},
    "chl":   {"line_color": "lightgreen"},
    "phyto": {"line_color": "darkgreen"},
    "zoo":   {"line_color": "purple"},
}

# -----------------------------
# Stations (nearest-node)
# -----------------------------
# List of (name, lat, lon)
STATIONS = [
    ("WE12", 41.90, -83.10),
    ("WE13", 41.80, -83.20),
]

# -----------------------------
# Regions (polygon masks)
# -----------------------------
# List of (region_name, spec_dict)
REGIONS = [
    ("Central", {"shapefile": "../data/shapefiles/central_basin_single.shp"}),
    ("East",    {"shapefile": "../data/shapefiles/east_basin_single.shp"}),
    ("West",    {"shapefile": "../data/shapefiles/west_basin_single.shp"}),
]

# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
def main():
    if not warnings.filters:
        warnings.filterwarnings("default")
    xr.set_options(use_new_combine_kwarg_defaults=True)

    print("[animate] loading dataset ")
    ds = load_from_base(BASE_DIR, FILE_PATTERN)

    # # ------------------------------------------------------------------
    # # 1) DOMAIN - combine_by='var': one GIF, multiple lines = variables
    # # ------------------------------------------------------------------
    # print("[animate] domain animations (one animation, lines = vars)")
    # animate_timeseries(
    #     ds,
    #     vars=[ "phyto", "zoo"],
    #     groups=GROUPS,
    #     scope="domain",
    #     years=2018,
    #     depth="surface",
    #     base_dir=BASE_DIR, figures_root=FIG_DIR,
    #     combine_by="var",            # one animation for the domain; lines are variables
    #     styles=PLOT_STYLES,
    #     verbose=True,
    # )

    # # ------------------------------------------------------------------
    # # 1b) DOMAIN -  no combining: one GIF per variable (classic behaviour)
    # # ------------------------------------------------------------------
    # print("[animate] domain animations (separate per variable)")
    # animate_timeseries(
    #     ds,
    #     vars=["phyto", "zoo"],
    #     groups=GROUPS,
    #     scope="domain",
    #     years=2018,
    #     depth="surface",
    #     base_dir=BASE_DIR, figures_root=FIG_DIR,
    #     combine_by=None,             # one animation per variable
    #     styles=PLOT_STYLES,
    #     verbose=False,
    # )

    # # ------------------------------------------------------------------
    # # 2) REGIONS -  combine_by='var': one GIF per region, lines = variables
    # # ------------------------------------------------------------------
    # print("[animate] region animations (per region, lines = vars)")
    # animate_timeseries(
    #     ds,
    #     vars=["chl", "phyto", "zoo"],
    #     groups=GROUPS,
    #     scope="region",
    #     regions=REGIONS,             # list of (name, spec) -  passed unchanged
    #     months=[6, 7, 8], years=2018,
    #     depth={"z_m": -10},          # 10 m below surface
    #     base_dir=BASE_DIR, figures_root=FIG_DIR,
    #     combine_by="var",            # one animation per region; lines are variables
    #     styles=PLOT_STYLES,
    #     verbose=False,
    # )

    # # ------------------------------------------------------------------
    # # 2b) REGIONS - combine_by='region': one GIF per variable, lines = regions
    # # ------------------------------------------------------------------
    # print("[animate] region comparison (per var, lines = regions)")
    # animate_timeseries(
    #     ds,
    #     vars=["chl", "phyto"],
    #     groups=GROUPS,
    #     scope="region",
    #     regions=REGIONS,
    #     years=2018,
    #     depth="surface",
    #     base_dir=BASE_DIR, figures_root=FIG_DIR,
    #     combine_by="region",         # one animation per variable; lines are regions
    #     styles=PLOT_STYLES,
    #     verbose=False,
    # )

    # # ------------------------------------------------------------------
    # # 3) STATIONS - combine_by=None: one GIF per (station * variable)
    # # ------------------------------------------------------------------
    # print("[animate] station animations (separate per station * variable)")
    # animate_timeseries(
    #     ds,
    #     vars=["chl", "phyto"],
    #     groups=GROUPS,
    #     scope="station",
    #     stations=STATIONS,           # list of (name, lat, lon)  passed unchanged
    #     start_date="2018-04-01", end_date="2018-10-31",
    #     depth="depth_avg",
    #     base_dir=BASE_DIR, figures_root=FIG_DIR,
    #     combine_by=None,             # one per variable per station
    #     styles=PLOT_STYLES,
    #     verbose=False,
    # )

    # # ------------------------------------------------------------------
    # # 3b) STATIONS  combine_by='station': one GIF per variable, lines = stations
    # # ------------------------------------------------------------------
    # print("[animate] station comparison (per var, lines = stations)")
    # animate_timeseries(
    #     ds,
    #     vars=["chl", "phyto"],
    #     groups=GROUPS,
    #     scope="station",
    #     stations=STATIONS,
    #     start_date="2018-04-01", end_date="2018-10-31",
    #     depth="surface",
    #     base_dir=BASE_DIR, figures_root=FIG_DIR,
    #     combine_by="station",        # one animation per variable; lines are stations
    #     styles=PLOT_STYLES,
    #     verbose=False,
    # )
    
    # ==========================================================================
    # === ANIMATION EXAMPLES =========================================
    # ==========================================================================

    # 4) DOMAIN MAPS - daily frames for June 2018 at the surface
    #    Uses robust color limits per time window unless overridden by MAP_STYLES or clim/norm.
    print("[animate] domain map animation (hourly, surface, June 2018)")
    animate_maps(
        ds,
        variables=["temp", "chl"],   # native variables or GROUPS keys both work
        scope="domain",
        months=6, years=2018,
        depth="surface",
        groups=GROUPS,
        base_dir=BASE_DIR, figures_root=FIG_DIR,
        frequency="daily",          #  hourly | daily | monthly
        grid_on=True,                # draw mesh overlay
        styles=PLOT_STYLES,           # optional per-var map styling
        verbose=True,
    )

    # 4b) REGION MAPS  all avaiablle frames across JJA 2018 at 10 m below surface
    print("[animate] region map animation (daily, z=10 m below surface, JJA 2018)")
    animate_maps(
        ds,
        variables=["chl"],
        scope="region",
        regions=REGIONS,
        months=[6, 7, 8], years=2018,
        depth={"z_m": -10},          # absolute metres below surface (negative down)
        groups=GROUPS,
        base_dir=BASE_DIR, figures_root=FIG_DIR,
        grid_on=False,
        styles=PLOT_STYLES,
        verbose=True,
    )

    # 4c) DOMAIN MAPS - monthly frames for 2018 bottom layer
    print("[animate] domain map animation (monthly, bottom, 2018)")
    animate_maps(
        ds,
        variables=["chl"],
        scope="domain",
        years=2018,
        depth="bottom",
        groups=GROUPS,
        base_dir=BASE_DIR, figures_root=FIG_DIR,
        frequency="monthly",
        styles=PLOT_STYLES,
        verbose=True,
    )

    # 4d) DOMAIN MAPS - explicit instants
    print("[animate] domain map animation (explicit instants)")
    animate_maps(
        ds,
        variables=["temp"],
        scope="domain",
        depth="depth_avg",
        groups=GROUPS,
        base_dir=BASE_DIR, figures_root=FIG_DIR,
        at_times=["2018-06-01 00:00", "2018-06-10 12:00", "2018-06-20 00:00"],
        grid_on=True,
        styles=PLOT_STYLES,
        verbose=True,
    )

    print("[animate] done.")


if __name__ == "__main__":
    main()
