#!/usr/bin/env python3
"""
Save a variables report (name, long_name, units, dims, shape) from a NetCDF file to a txt file.
"""

from pathlib import Path
from netCDF4 import Dataset

# =========================
# Edit these two lines
NC_FILE = "/data/proteus1/scratch/yli/project/lake_erie/output_updated_river_var/erie_0001.nc"
OUT_TXT = "/data/proteus1/scratch/moja/projects/Lake_Erie/fviz-plots/vars.txt"  # or set e.g. "/path/to/report.txt"
# =========================

def write_nc_var_report(ncfile: str | Path, out_txt: str | Path | None = None) -> Path:
    nc_path = Path(ncfile)
    out_path = Path(out_txt) if out_txt else nc_path.with_suffix(nc_path.suffix + ".vars.txt")

    with Dataset(nc_path, mode="r") as ds, out_path.open("w", encoding="utf-8") as f:
        # Header
        f.write(f"# NetCDF variables report\n")
        f.write(f"# File: {nc_path}\n")
        f.write(f"# Dimensions:\n")
        for dname, dim in ds.dimensions.items():
            size = len(dim) if not dim.isunlimited() else f"{len(dim)} (unlimited)"
            f.write(f"#   {dname}: {size}\n")
        f.write("#\n# Variables:\n")

        # Variables
        for vname, var in ds.variables.items():
            dims = tuple(var.dimensions)
            shape = tuple(var.shape)
            # Prefer long_name; fall back to standard_name; else blank
            long_name = getattr(var, "long_name", None) or getattr(var, "standard_name", "") or ""
            units = getattr(var, "units", "") or ""

            # Normalize whitespace
            long_name = " ".join(str(long_name).split())
            units = " ".join(str(units).split())

            f.write(
                f"{vname}\n"
                f"  long_name : {long_name if long_name else '(none)'}\n"
                f"  units     : {units if units else '(none)'}\n"
                f"  dims      : {dims if dims else '(scalar)'}\n"
                f"  shape     : {shape if shape else '(scalar)'}\n"
            )
        f.write("\n# End of report\n")

    print(f"Wrote variable report to: {out_path}")
    return out_path

if __name__ == "__main__":
    write_nc_var_report(NC_FILE, OUT_TXT)

