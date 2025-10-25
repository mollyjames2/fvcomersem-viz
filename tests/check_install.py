#!/usr/bin/env python
"""
check_install.py - User-facing installation check for fvcomersemviz.

Checks:
  - Python >= 3.11
  - Package import works
  - Distribution and version found
  - Declared runtime dependencies (from pyproject) are present
  - Recommended environment (conda) packages are present (warn-only)
  - Basic smoke test (dir())

Does NOT check:
  - Dev tools (pytest, ruff, etc.)
"""

from __future__ import annotations
import sys
import re
import importlib
import traceback
from dataclasses import dataclass
from typing import List, Optional, Tuple, Sequence
from pathlib import Path

try:
    from importlib import metadata as importlib_metadata  # Python 3.8+
except Exception:  # pragma: no cover
    import importlib_metadata  # type: ignore


# --------------------------- Configuration --------------------------- #

MIN_PYTHON = (3, 11)
DIST_CANDIDATES = ["fvcomersemviz", "fvcomersem-viz"]
MODULE_CANDIDATES = ["fvcomersemviz"]

# Recommended environment (conda) stack to report to users
# Each item: (distribution-name, import-name, [alt-dist-names...])
RECOMMENDED_ENV: Sequence[Tuple[str, str, Sequence[str]]] = (
    ("numpy", "numpy", ()),
    ("scipy", "scipy", ()),
    ("pandas", "pandas", ()),
    ("xarray", "xarray", ()),
    ("dask", "dask", ()),
    ("netCDF4", "netCDF4", ("netcdf4",)),  # metadata dist is "netCDF4"
    ("matplotlib", "matplotlib", ()),
    ("cartopy", "cartopy", ()),
    ("geopandas", "geopandas", ()),
    ("Shapely", "shapely", ("shapely",)),  # metadata dist may be "Shapely"
    ("pyproj", "pyproj", ()),
    ("Rtree", "rtree", ("rtree",)),        # metadata dist may be "Rtree"
)


# --------------------------- Data Model ------------------------------ #

@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    warn: bool = False


# --------------------------- Helper Functions ------------------------ #

def _print(msg: str, *, verbose: bool = True):
    if verbose:
        print(msg)


def check_python(min_version: Tuple[int, int]) -> CheckResult:
    ok = sys.version_info >= (*min_version, 0)
    return CheckResult(
        name=f"Python {min_version[0]}.{min_version[1]}+",
        ok=ok,
        detail=f"Detected Python {sys.version.split()[0]}",
    )


def import_module_any(candidates: List[str]):
    tried: List[str] = []
    for name in candidates:
        try:
            mod = importlib.import_module(name)
            return mod, name, tried
        except Exception as e:
            tried.append(f"{name} ({e.__class__.__name__}: {e})")
    return None, "", tried


def _dist_path(d: importlib_metadata.Distribution) -> Path:
    return Path(getattr(d, "_path", d.locate_file("")))


def _is_site_packages(p: Path) -> bool:
    s = str(p)
    return "site-packages" in s or "dist-packages" in s


def prefer_installed_distribution(
    dist_name: str,
) -> Optional[importlib_metadata.Distribution]:
    matches = []
    for d in importlib_metadata.distributions():
        try:
            if d.metadata.get("Name") == dist_name:
                matches.append(d)
        except Exception:
            continue
    if not matches:
        try:
            return importlib_metadata.distribution(dist_name)
        except Exception:
            return None
    for d in matches:
        if _is_site_packages(_dist_path(d)):
            return d
    return matches[0]


def distribution_any(candidates: List[str]):
    tried: List[str] = []
    for dist_name in candidates:
        d = prefer_installed_distribution(dist_name)
        if d is not None:
            return d, dist_name, tried
        tried.append(f"{dist_name} (not found)")
    return None, "", tried


_REQ_NAME = re.compile(r"^\s*([A-Za-z0-9_.\-]+)")
_EXTRA_MARKER = re.compile(r";\s*extra\s*==", re.IGNORECASE)

def parse_requirements(requires_dist: Optional[List[str]]) -> List[str]:
    """
    Return only runtime requirements (exclude optional extras like dev/test).
    We skip any Requires-Dist entry that has a marker like '; extra == "dev"'.
    """
    if not requires_dist:
        return []
    out: List[str] = []
    for raw in requires_dist:
        s = (raw or "").strip()
        if not s or _EXTRA_MARKER.search(s):
            continue  # ignore extras
        m = _REQ_NAME.match(s)
        if m:
            out.append(m.group(1))
    return out


def version_of(dist_name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(dist_name)
    except Exception:
        return None


def version_of_any(names: Sequence[str]) -> Optional[str]:
    for n in names:
        v = version_of(n)
        if v is not None:
            return v
    return None


def summarize_and_exit(results: List[CheckResult]) -> None:
    print("\n=== fvcomersemviz Installation Summary ===")
    width = max(len(r.name) for r in results) + 2
    for r in results:
        status = "OK" if r.ok else ("WARN" if r.warn else "FAIL")
        print(f"{status:>4}  {r.name:<{width}} {r.detail}")
    print("==========================================")
    if any((not r.ok) and (not r.warn) for r in results):
        print("One or more checks failed. Please review the FAIL items above.")
    elif any(r.warn for r in results):
        print("Environment looks usable, with warnings.")
    else:
        print("All good! fvcomersemviz and its dependencies look ready to run.")


# --------------------------- Main Logic ------------------------------ #

def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="fvcomersemviz user installation check")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    args = parser.parse_args(argv)
    verbose = args.verbose

    results: List[CheckResult] = []

    # 1) Python version
    r = check_python(MIN_PYTHON)
    results.append(r)
    _print(f"[{'OK' if r.ok else 'FAIL'}] {r.name} — {r.detail}", verbose=verbose)

    # 2) Package import
    mod, used_mod, tried_mods = import_module_any(MODULE_CANDIDATES)
    if mod is None:
        detail = "Tried: " + "; ".join(tried_mods)
        results.append(CheckResult("Import fvcomersemviz", ok=False, detail=detail))
        _print(f"[FAIL] Import fvcomersemviz — {detail}", verbose=True)
        summarize_and_exit(results)
        return 1

    version_attr = getattr(mod, "__version__", None)
    results.append(CheckResult("Import fvcomersemviz", ok=True, detail=f"Module '{used_mod}' loaded"))
    if version_attr:
        _print(f"[OK] fvcomersemviz.__version__ = {version_attr}", verbose=True)
    else:
        _print("[WARN] fvcomersemviz has no __version__ attribute", verbose=True)

    # 3) Distribution metadata (pyproject runtime deps)
    dist, dist_name, tried_dists = distribution_any(DIST_CANDIDATES)
    if dist is None:
        detail = "Could not find installed distribution. Tried: " + "; ".join(tried_dists)
        results.append(CheckResult("Distribution located", ok=False, detail=detail))
        _print(f"[FAIL] {detail}", verbose=True)
        summarize_and_exit(results)
        return 1

    this_version = version_of(dist_name) or dist.metadata.get("Version", "unknown")
    _print(f"[OK] Distribution: {dist.metadata.get('Name', dist_name)} {this_version}", verbose=True)
    results.append(CheckResult("Distribution located", ok=True, detail=f"{dist_name} {this_version}"))

    reqs = parse_requirements(dist.requires or [])
    if reqs:
        _print(f"[INFO] Declared runtime dependencies ({len(reqs)}): {', '.join(reqs)}", verbose=True)
        dep_failures: List[str] = []
        dep_warnings: List[str] = []
        for dep in reqs:
            dep_ver = version_of(dep)
            if dep_ver is None:
                dep_failures.append(f"{dep}: not installed")
                _print(f"[FAIL] {dep}: not installed", verbose=True)
            else:
                _print(f"[OK] {dep}: installed (version {dep_ver})", verbose=verbose)
                try:
                    importlib.import_module(dep.replace("-", "_"))
                except Exception as e:
                    dep_warnings.append(f"{dep}: installed but import failed ({e.__class__.__name__})")
                    _print(f"[WARN] {dep}: installed but could not import top-level module.", verbose=True)

        results.append(CheckResult(
            "Dependencies installed",
            ok=(len(dep_failures) == 0),
            detail="All dependencies present" if not dep_failures else "; ".join(dep_failures),
        ))
        results.append(CheckResult(
            "Dependency imports",
            ok=True,
            warn=(len(dep_warnings) > 0),
            detail="All dependency imports succeeded" if not dep_warnings else "; ".join(dep_warnings),
        ))
    else:
        # No runtime deps declared; treat as managed via environment (conda).
        _print("[INFO] Runtime dependencies declared in package metadata: none (using environment packages check below).", verbose=True)
        results.append(CheckResult("Dependencies installed", ok=True, detail="Managed via environment"))
        results.append(CheckResult("Dependency imports", ok=True, detail="Managed via environment"))

    # 4) Recommended environment packages (conda stack) - warn-only
    missing_env: List[str] = []
    detected_env: List[str] = []
    for dist_name, import_name, alts in RECOMMENDED_ENV:
        # Try to import the module (import_name) for a real runtime check
        imported = True
        try:
            importlib.import_module(import_name)
        except Exception:
            imported = False

        # Try to fetch a version from any of the candidate dist names
        ver = version_of_any((dist_name, *alts))

        if imported and ver:
            _print(f"[OK] {dist_name}: installed (version {ver})", verbose=verbose)
            detected_env.append(f"{dist_name} {ver}")
        elif imported:
            _print(f"[OK] {dist_name}: installed (version unknown)", verbose=verbose)
            detected_env.append(f"{dist_name} (version unknown)")
        else:
            _print(f"[WARN] {dist_name}: not detected", verbose=True)
            missing_env.append(dist_name)

    if detected_env:
        results.append(CheckResult(
            "Environment packages (recommended)",
            ok=True,
            detail="Detected: " + ", ".join(detected_env),
            warn=bool(missing_env),
        ))
    if missing_env:
        results.append(CheckResult(
            "Environment packages missing (recommended)",
            ok=True,  # warn-only for users
            detail=", ".join(missing_env),
            warn=True,
        ))

    # 5) Basic smoke test
    try:
        _ = dir(mod)
        results.append(CheckResult("Basic smoke test", ok=True, detail="dir() succeeded"))
        _print("[OK] Basic smoke test passed.", verbose=True)
    except Exception as e:
        results.append(CheckResult("Basic smoke test", ok=False, detail=str(e)))
        _print(f"[FAIL] Basic smoke test failed: {e}", verbose=True)

    summarize_and_exit(results)
    hard_fail = any((not r.ok) and (not r.warn) for r in results)
    return 1 if hard_fail else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception:
        print("Unexpected error:\n" + traceback.format_exc())
        sys.exit(1)

