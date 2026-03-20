#!/usr/bin/env python3
# examples/plot_fvcom.py
"""
FVCOM physical output — complete plotting showcase
---------------------------------------------------
Demonstrates every applicable fvcomersemviz plotting function for a
physical-only FVCOM run (no BGC).  One station (L4)

Variables used: temp, salinity, u, v, ua, va, zeta, km, kh, q2

Edit the config block below, then run:
    python examples/plot_fvcom.py
"""

from __future__ import annotations
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg", force=True)  # headless backend for batch runs

import numpy as np
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
from fvcomersemviz.plots.hovmoller import station_hovmoller
from fvcomersemviz.plots.maps import domain_map
from fvcomersemviz.plots.timeseries import (
    domain_mean_timeseries,
    domain_three_panel,
    station_timeseries,
    station_three_panel,
)
from fvcomersemviz.plots.bars_box import plot_bars, plot_box
from fvcomersemviz.plots.curves import plot_curves
from fvcomersemviz.plots.animate import animate_maps, animate_timeseries

xr.set_options(use_new_combine_kwarg_defaults=True)

# -----------------------------------------------------------------------------
# Config (EDIT THESE)
# -----------------------------------------------------------------------------
BASE_DIR = "/data/sthenno1/scratch/modop/Model/FVCOM_tamar/output/archive/"
FILE_PATTERN = "tamar_v2_2021*.nc"
FIG_DIR = "./figures"

# -----------------------------------------------------------------------------
# Stations (name, lat, lon)
# -----------------------------------------------------------------------------
STATIONS = [
    ("L4", 50.25, -4.27),
]

# -----------------------------------------------------------------------------
# Time window  (lightweight: 2 months ~ 1 500 timesteps vs 9 100 for full year)
# -----------------------------------------------------------------------------
MONTHS      = [6, 7]        # Jun–Jul
YEARS       = [2021]
ANIM_MONTHS = [6]           # animations: Jun only (~30 daily frames)

# -----------------------------------------------------------------------------
# Seasons (only those covered by MONTHS above)
# -----------------------------------------------------------------------------
SEASONS = {
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
}

# -----------------------------------------------------------------------------
# Groups — empty for physical-only output; add composites here if needed
# e.g. "speed": "sqrt(u**2 + v**2)"
# -----------------------------------------------------------------------------
GROUPS: dict = {}

# -----------------------------------------------------------------------------
# Per-variable plot styles
# -----------------------------------------------------------------------------
PLOT_STYLES = {
    "temp":     {"cmap": "cmo.thermal"},
    "salinity": {"cmap": "cmo.haline"},
    "u":        {"cmap": "cmo.balance", "vmin": -1.0, "vmax": 1.0},
    "v":        {"cmap": "cmo.balance", "vmin": -1.0, "vmax": 1.0},
    "ua":       {"cmap": "cmo.balance", "vmin": -1.0, "vmax": 1.0},
    "va":       {"cmap": "cmo.balance", "vmin": -1.0, "vmax": 1.0},
    "zeta":     {"cmap": "cmo.balance"},
    "km":       {"cmap": "cmo.matter"},
    "kh":       {"cmap": "cmo.matter"},
    "q2":       {"cmap": "cmo.amp"},
}

# Absolute depth grid for z-mode Hovmöller (L4 ~54 m)
Z_LEVELS = np.linspace(-50.0, 0.0, 25)


def main():
    if not os.environ.get("PYTHONWARNINGS"):
        warnings.filterwarnings("default")

    print(hr("="))
    print("fvcomersemviz: FVCOM physical output showcase")
    print(hr("="))

    # -------------------------------------------------------------------------
    # Discover & load
    # -------------------------------------------------------------------------
    info(" Discovering files")
    files = list_files(BASE_DIR, FILE_PATTERN)
    summarize_files(files)
    if not files:
        print("No files found — check BASE_DIR and FILE_PATTERN.")
        sys.exit(2)

    info(" Loading dataset")
    ds = load_from_base(BASE_DIR, FILE_PATTERN)
    print_dataset_summary(ds)

    out_folder = out_dir(BASE_DIR, FIG_DIR)
    prefix     = file_prefix(BASE_DIR)
    kv("Figure folder",    out_folder)
    kv("Filename prefix",  prefix)

    # =========================================================================
    # 1) DOMAIN MAPS
    # =========================================================================
    print(hr("="))
    print("1) Domain maps")
    print(hr("="))

    bullet("1a: Time-mean surface temperature and salinity (Jun–Jul 2021)")
    plot_call(
        domain_map,
        ds=ds,
        variables=["temp", "salinity"],
        depth="surface",
        months=MONTHS,
        years=YEARS,
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
        figsize=(8, 6),
        dpi=100,
    )

    bullet("1b: Time-mean bottom temperature and salinity (Jun–Jul 2021)")
    plot_call(
        domain_map,
        ds=ds,
        variables=["temp", "salinity"],
        depth="bottom",
        months=MONTHS,
        years=YEARS,
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
        figsize=(8, 6),
        dpi=100,
    )

    bullet("1c: Sea surface elevation (zeta) — 2D variable, time mean")
    plot_call(
        domain_map,
        ds=ds,
        variables=["zeta"],
        months=MONTHS,
        years=YEARS,
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
        figsize=(8, 6),
        dpi=100,
    )

    bullet("1d: Depth-averaged velocity components (ua, va) — 2D, time mean")
    plot_call(
        domain_map,
        ds=ds,
        variables=["ua", "va"],
        months=MONTHS,
        years=YEARS,
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
        figsize=(8, 6),
        dpi=100,
    )

    # =========================================================================
    # 2) DOMAIN TIME SERIES
    # =========================================================================
    print(hr("="))
    print("2) Domain time series")
    print(hr("="))

    bullet("2a: Domain-mean surface temperature and salinity with ±1σ — daily means (Jun–Jul 2021)")
    plot_call(
        domain_mean_timeseries,
        ds=ds,
        variables=["temp", "salinity"],
        depth="surface",
        months=MONTHS,
        years=YEARS,
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
        show_std=True,
        std_alpha=0.25,
        average_by="day",
        figsize=(10, 4),
        dpi=100,
    )

    bullet("2b: Domain-mean sea surface elevation (2D variable, no depth selector)")
    plot_call(
        domain_mean_timeseries,
        ds=ds,
        variables=["zeta"],
        months=MONTHS,
        years=YEARS,
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
        show_std=True,
        std_alpha=0.25,
        figsize=(10, 4),
        dpi=100,
    )

    bullet("2c: Domain three-panel — surface + bottom time series + vertical profile")
    plot_call(
        domain_three_panel,
        ds=ds,
        variables=["temp", "salinity"],
        months=MONTHS,
        years=YEARS,
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
        figsize=(11, 9),
        dpi=100,
    )

    # =========================================================================
    # 3) STATION TIME SERIES — L4
    # =========================================================================
    print(hr("="))
    print("3) Station time series — L4")
    print(hr("="))

    bullet("3a: L4 surface temperature and salinity — daily means")
    plot_call(
        station_timeseries,
        ds=ds,
        variables=["temp", "salinity"],
        stations=STATIONS,
        depth="surface",
        months=MONTHS,
        years=YEARS,
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
        average_by="day",
        figsize=(10, 4),
        dpi=100,
    )

    bullet("3b: L4 sea surface elevation (2D variable)")
    plot_call(
        station_timeseries,
        ds=ds,
        variables=["zeta"],
        stations=STATIONS,
        months=MONTHS,
        years=YEARS,
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
        figsize=(10, 4),
        dpi=100,
    )

    bullet("3c: L4 temp + salinity on one figure (combine_by='var')")
    plot_call(
        station_timeseries,
        ds=ds,
        variables=["temp", "salinity"],
        stations=STATIONS,
        depth="surface",
        months=MONTHS,
        years=YEARS,
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
        combine_by="var",
        figsize=(10, 4),
        dpi=100,
    )

    bullet("3d: L4 three-panel — surface + bottom time series + vertical profile")
    plot_call(
        station_three_panel,
        ds=ds,
        variables=["temp", "salinity"],
        stations=STATIONS,
        months=MONTHS,
        years=YEARS,
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
        figsize=(11, 9),
        dpi=100,
    )

    # =========================================================================
    # 4) HOVMÖLLER — L4
    # =========================================================================
    print(hr("="))
    print("4) Hovmöller diagrams — L4")
    print(hr("="))

    bullet("4a: Temp + salinity on native sigma layers — daily means (axis='sigma', Jun–Jul 2021)")
    plot_call(
        station_hovmoller,
        ds=ds,
        variables=["temp", "salinity"],
        stations=STATIONS,
        axis="sigma",
        months=MONTHS,
        years=YEARS,
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
        average_by="day",
        figsize=(9, 5),
        dpi=100,
    )

    bullet("4b: Temperature interpolated to absolute depth grid — daily means (axis='z')")
    plot_call(
        station_hovmoller,
        ds=ds,
        variables=["temp"],
        stations=STATIONS,
        axis="z",
        z_levels=Z_LEVELS,
        months=MONTHS,
        years=YEARS,
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
        average_by="day",
        figsize=(9, 5),
        dpi=100,
    )

    bullet("4c: Turbulent eddy viscosity km + kh on sigma levels — daily means")
    plot_call(
        station_hovmoller,
        ds=ds,
        variables=["km", "kh"],
        stations=STATIONS,
        axis="sigma",
        months=MONTHS,
        years=YEARS,
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
        average_by="day",
        figsize=(9, 5),
        dpi=100,
    )

    # =========================================================================
    # 5) BAR PLOTS — L4 seasonal × depth
    # =========================================================================
    print(hr("="))
    print("5) Bar plots — L4 seasonal × depth")
    print(hr("="))

    bullet("5a: Seasonal mean temp and salinity — surface vs bottom — error=sd")
    plot_call(
        plot_bars,
        ds=ds,
        variables=["temp", "salinity"],
        stations=STATIONS,
        depth=["surface", "bottom"],
        seasons=SEASONS,
        months=MONTHS,
        years=YEARS,
        groups=GROUPS,
        x_by="season",
        hue_by="depth",
        facet_by="variable",
        error="sd",
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        figsize=(12, 6),
        ncols=2,
        title="L4: seasonal mean by depth (±1 SD, Jun–Jul 2021)",
        dpi=100,
    )

    bullet("5b: Same with CI95 error bars")
    plot_call(
        plot_bars,
        ds=ds,
        variables=["temp", "salinity"],
        stations=STATIONS,
        depth=["surface", "bottom"],
        seasons=SEASONS,
        months=MONTHS,
        years=YEARS,
        groups=GROUPS,
        x_by="season",
        hue_by="depth",
        facet_by="variable",
        error="ci95",
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        figsize=(12, 6),
        ncols=2,
        title="L4: seasonal mean by depth (95% CI, Jun–Jul 2021)",
        dpi=100,
    )

    # =========================================================================
    # 6) BOX PLOTS — L4 seasonal × depth
    # =========================================================================
    print(hr("="))
    print("6) Box plots — L4 seasonal × depth")
    print(hr("="))

    bullet("6a: Seasonal distributions of temp and salinity — surface vs bottom")
    plot_call(
        plot_box,
        ds=ds,
        variables=["temp", "salinity"],
        stations=STATIONS,
        depth=["surface", "bottom"],
        seasons=SEASONS,
        months=MONTHS,
        years=YEARS,
        groups=GROUPS,
        x_by="season",
        hue_by="depth",
        facet_by="variable",
        ylabels={
            "temp":     "Temperature (°C)",
            "salinity": "Salinity (PSU)",
        },
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        figsize=(12, 6),
        ncols=2,
        title="L4: seasonal distributions by depth (Jun–Jul 2021)",
        max_samples_per_bin=20000,
        random_seed=0,
        dpi=100,
    )

    # =========================================================================
    # 7) CURVES — L4 diagnostics
    # =========================================================================
    print(hr("="))
    print("7) Curves — L4 diagnostics")
    print(hr("="))

    bullet("7a: T–S diagram — scatter cloud + binned median (depth-averaged)")
    ts_specs = [
        {
            "name": "All points",
            "x": "temp",
            "y": "salinity",
            "filters": {"months": MONTHS, "years": YEARS},
            "depth": "depth_avg",
            "scope": {"station": STATIONS[0]},
            "scatter": {"s": 4, "alpha": 0.06},
            "style": {"color": "steelblue", "marker": ".", "linewidths": 0},
            "x_label": "Temperature (°C)",
            "y_label": "Salinity (PSU)",
        },
        {
            "name": "Binned median (IQR)",
            "x": "temp",
            "y": "salinity",
            "filters": {"months": MONTHS, "years": YEARS},
            "depth": "depth_avg",
            "scope": {"station": STATIONS[0]},
            "bin": {"x_bins": 30, "agg": "median", "min_count": 10, "iqr": True},
            "style": {"color": "steelblue", "lw": 2},
        },
    ]
    out_ts = plot_curves(
        specs=ts_specs,
        ds=ds,
        groups=GROUPS,
        xlabel="Temperature (°C)",
        ylabel="Salinity (PSU)",
        show_legend=True,
        legend_outside=True,
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        stem="L4__TS_diagram__depth_avg",
        dpi=100,
    )
    kv("Saved", out_ts)

    bullet("7b: Temperature vs turbulent eddy viscosity km (depth-averaged)")
    km_specs = [
        {
            "name": "All points",
            "x": "temp",
            "y": "km",
            "filters": {"months": MONTHS, "years": YEARS},
            "depth": "depth_avg",
            "scope": {"station": STATIONS[0]},
            "scatter": {"s": 4, "alpha": 0.06},
            "style": {"color": "C1", "marker": ".", "linewidths": 0},
            "x_label": "Temperature (°C)",
            "y_label": "km (m² s⁻¹)",
        },
        {
            "name": "Binned median (IQR)",
            "x": "temp",
            "y": "km",
            "filters": {"months": MONTHS, "years": YEARS},
            "depth": "depth_avg",
            "scope": {"station": STATIONS[0]},
            "bin": {"x_bins": 30, "agg": "median", "min_count": 10, "iqr": True},
            "style": {"color": "C1", "lw": 2},
        },
    ]
    out_km = plot_curves(
        specs=km_specs,
        ds=ds,
        groups=GROUPS,
        xlabel="Temperature (°C)",
        ylabel="km (m² s⁻¹)",
        show_legend=True,
        legend_outside=True,
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        stem="L4__temp_vs_km__depth_avg",
        dpi=100,
    )
    kv("Saved", out_km)

    # =========================================================================
    # 8) ANIMATED MAPS
    # =========================================================================
    print(hr("="))
    print("8) Animated domain maps")
    print(hr("="))

    bullet("8a: Surface temp + salinity — daily frames (Jun 2021)")
    animate_maps(
        ds,
        variables=["temp", "salinity"],
        scope="domain",
        depth="surface",
        months=ANIM_MONTHS,
        years=YEARS,
        frequency="daily",
        groups=GROUPS,
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        styles=PLOT_STYLES,
        figsize=(8, 6),
        dpi=100,
        verbose=True,
    )

    # =========================================================================
    # 9) ANIMATED TIME SERIES — L4
    # =========================================================================
    print(hr("="))
    print("9) Animated time series — L4")
    print(hr("="))

    bullet("9a: Growing-line animation of temp + salinity at L4 (one GIF per var)")
    animate_timeseries(
        ds,
        vars=["temp", "salinity"],
        groups=GROUPS,
        scope="station",
        stations=STATIONS,
        depth="surface",
        months=ANIM_MONTHS,
        years=YEARS,
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        combine_by=None,
        styles=PLOT_STYLES,
        figsize=(10, 4),
        dpi=100,
        verbose=True,
    )

    bullet("9b: Same variables combined onto one animation (combine_by='var')")
    animate_timeseries(
        ds,
        vars=["temp", "salinity"],
        groups=GROUPS,
        scope="station",
        stations=STATIONS,
        depth="surface",
        months=ANIM_MONTHS,
        years=YEARS,
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        combine_by="var",
        styles=PLOT_STYLES,
        figsize=(10, 4),
        dpi=100,
        verbose=True,
    )

    # -------------------------------------------------------------------------
    # Done
    # -------------------------------------------------------------------------
    info(" Output recap")
    bullet(f"All figures written under: {out_folder}")
    sample_output_listing(out_folder, prefix)

    print(hr("="))
    print("Done")
    print(hr("="))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
