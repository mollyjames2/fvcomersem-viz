#!/usr/bin/env python
"""
check_fvcomersemviz_install.py

Comprehensive readiness check for the fvcomersemviz / fvcomersem-viz package
and its runtime dependencies.
"""

from __future__ import annotations

import sys
import re
import importlib
import traceback
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set
from pathlib import Path

try:
    # Python 3.8+
    from importlib import metadata as importlib_metadata
except Exception:  # pragma: no cover
    import importlib_metadata  # type: ignore

# --------------------------- Configuration -------------------------------- #

DIST_CANDIDATES = [
    # distribution names (PyPI-style)
    "fvcomersemviz",
    "fvcomersem-viz",
]

MODULE_CANDIDATES = [
    # importable module names (Python package-style)
    "fvcomersemviz",
]

MIN_PYTHON = (3, 8)

# ----------------------------- Helpers ------------------------------------ #


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    warn: bool = False


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


def import_module_any(candidates: List[str]) -> Tuple[Optional[object], str, List[str]]:
    tried = []
    for name in candidates:
        try:
            mod = importlib.import_module(name)
            return mod, name, tried
        except Exception as e:
            tried.append(f"{name} ({e.__class__.__name__}: {e})")
    return None, "", tried


def distribution_any(
    candidates: List[str],
) -> Tuple[Optional[importlib_metadata.Distribution], str, List[str]]:
    tried = []
    for dist_name in candidates:
        d = prefer_installed_distribution(dist_name)
        if d is not None:
            return d, dist_name, tried
        tried.append(f"{dist_name} (not found)")
    return None, "", tried


def _dist_path(d: importlib_metadata.Distribution) -> Path:
    # importlib.metadata doesn't expose a public path attr; this works across impls
    return Path(getattr(d, "_path", d.locate_file("")))


def _is_site_packages(p: Path) -> bool:
    s = str(p)
    return ("site-packages" in s) or ("dist-packages" in s)


def prefer_installed_distribution(
    dist_name: str,
) -> Optional[importlib_metadata.Distribution]:
    """
    Return the fvcomersemviz distribution that lives in site-packages/dist-packages,
    ignoring in-tree egg-info produced by editable installs.
    """
    matches = []
    for d in importlib_metadata.distributions():
        try:
            if d.metadata.get("Name") == dist_name:
                matches.append(d)
        except Exception:
            continue
    if not matches:
        # fall back to the default resolver (may return the in-tree egg-info)
        try:
            return importlib_metadata.distribution(dist_name)
        except Exception:
            return None
    # Prefer the one installed in site-/dist-packages
    for d in matches:
        if _is_site_packages(_dist_path(d)):
            return d
    return matches[0]


_REQ_PATTERN = re.compile(r"^\s*([A-Za-z0-9_.\-]+)")


def parse_requirements(requires_dist: Optional[List[str]]) -> List[str]:
    """
    Extract distribution names from Requires-Dist entries.
    e.g., 'xarray (>=2023.1.0); python_version >= "3.8"' -> 'xarray'
    """
    if not requires_dist:
        return []
    out: List[str] = []
    for line in requires_dist:
        m = _REQ_PATTERN.match(line or "")
        if m:
            out.append(m.group(1))
    return out


def guess_top_level_modules(dist: importlib_metadata.Distribution) -> Set[str]:
    """
    Guess importable top-level module names from a distribution’s file list.
    Falls back to common hyphen->underscore normalization.
    """
    modules: Set[str] = set()
    try:
        files = dist.files or []
        for f in files:
            parts = getattr(f, "parts", None) or str(f).split("/")
            if not parts:
                continue
            top = parts[0]
            # Skip metadata directories
            if top.endswith(".dist-info") or top.endswith(".egg-info"):
                continue
            # Only consider plausible module/package names
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", top):
                modules.add(top)
    except Exception:
        pass
    if not modules:
        # Heuristic: dist-name -> module-name
        norm = re.sub(r"[^A-Za-z0-9_]", "_", dist.metadata["Name"]).lower()
        modules.add(norm)
    return modules


def try_import_distribution(dist_name: str) -> Tuple[bool, List[str]]:
    """
    Try to import at least one plausible top-level module for the given
    distribution. Prefers the distribution record that lives in site-/dist-packages,
    ignoring in-tree egg-info created by editable installs.
    """
    msgs: List[str] = []

    # Prefer the installed dist-info under site-/dist-packages
    dist = prefer_installed_distribution(dist_name)
    if dist is None:
        msgs.append(f"Could not load distribution metadata for {dist_name}")
        return False, msgs

    # Derive likely importable top-level module names from the dist contents
    candidates = list(guess_top_level_modules(dist))

    # Try each candidate
    for modname in candidates:
        try:
            importlib.import_module(modname)
            msgs.append(f"Imported '{modname}' OK")
            return True, msgs
        except Exception as e:
            msgs.append(f"Import '{modname}' failed: {e.__class__.__name__}: {e}")

    # Final heuristic: hyphen→underscore of the dist name
    fallback = dist.metadata.get("Name", dist_name).replace("-", "_")
    if fallback not in candidates:
        try:
            importlib.import_module(fallback)
            msgs.append(f"Imported fallback '{fallback}' OK")
            return True, msgs
        except Exception as e:
            msgs.append(f"Import fallback '{fallback}' failed: {e.__class__.__name__}: {e}")

    return False, msgs


def version_of(dist_name: str) -> Optional[str]:
    d = prefer_installed_distribution(dist_name)
    if d is not None:
        try:
            return d.metadata.get("Version")
        except Exception:
            pass
    try:
        return importlib_metadata.version(dist_name)
    except Exception:
        return None


# ----------------------------- Main logic --------------------------------- #


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="fvcomersemviz / fvcomersem-viz installation check"
    )
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    args = parser.parse_args(argv)

    verbose = args.verbose

    results: List[CheckResult] = []

    # 1) Python version
    r = check_python(MIN_PYTHON)
    results.append(r)
    _print(f"[{'OK' if r.ok else 'FAIL'}] {r.name} — {r.detail}", verbose=verbose)

    # 2) Package import + __version__
    mod, used_mod, tried_mods = import_module_any(MODULE_CANDIDATES)
    if mod is None:
        detail = "Tried: " + "; ".join(tried_mods)
        results.append(CheckResult("Import fvcomersemviz", ok=False, detail=detail))
        _print(f"[FAIL] Import fvcomersemviz — {detail}", verbose=True)
        summarize_and_exit(results)
        return 1

    version_attr = getattr(mod, "__version__", None)
    results.append(
        CheckResult("Import fvcomersemviz", ok=True, detail=f"Module '{used_mod}' loaded")
    )
    if version_attr:
        _print(f"[OK] fvcomersemviz.__version__ = {version_attr}", verbose=True)
    else:
        _print("[WARN] fvcomersemviz has no __version__ attribute", verbose=True)

    # 3) Distribution metadata (to get dependencies)
    dist, dist_name, tried_dists = distribution_any(DIST_CANDIDATES)
    if dist is None:
        detail = "Could not find installed distribution. Tried: " + "; ".join(tried_dists)
        results.append(CheckResult("Distribution lookup", ok=False, detail=detail))
        _print(f"[FAIL] {detail}", verbose=True)
        summarize_and_exit(results)
        return 1

    this_version = version_of(dist_name) or dist.metadata.get("Version", "unknown")
    _print(
        f"[OK] Distribution: {dist.metadata.get('Name', dist_name)} {this_version}",
        verbose=True,
    )
    results.append(
        CheckResult("Distribution located", ok=True, detail=f"{dist_name} {this_version}")
    )

    # 4) Dependencies: verify installed & importable
    reqs = parse_requirements(dist.requires or [])
    if not reqs:
        _print("[OK] No runtime dependencies listed in package metadata.", verbose=True)
    else:
        _print(
            f"[INFO] Declared dependencies ({len(reqs)}): {', '.join(reqs)}",
            verbose=True,
        )

    dep_failures: List[str] = []
    dep_warnings: List[str] = []

    for dep in reqs:
        dep_ver = version_of(dep)
        if dep_ver is None:
            dep_failures.append(f"{dep}: not installed")
            _print(f"[FAIL] {dep}: not installed", verbose=True)
            continue
        _print(f"[OK] {dep}: installed (version {dep_ver})", verbose=verbose)

        ok_import, msgs = try_import_distribution(dep)
        if ok_import:
            _print("      " + " | ".join(msgs), verbose=verbose)
        else:
            # Not always fatal—some packages provide console scripts or are extras.
            dep_warnings.append(f"{dep}: installed but import check failed ({'; '.join(msgs)})")
            _print(
                f"[WARN] {dep}: installed but could not import a top-level module.",
                verbose=True,
            )

    # 5) Final smoke test (optional tiny usage)
    # If the package exposes a simple function or CLI we could check it here.
    # To stay generic (and safe), we just verify we can access __version__ and dir().
    try:
        _ = dir(mod)
        results.append(CheckResult("Basic smoke test", ok=True, detail="dir() succeeded"))
        _print("[OK] Basic smoke test passed (dir(mod) worked).", verbose=True)
    except Exception as e:
        results.append(CheckResult("Basic smoke test", ok=False, detail=str(e)))
        _print(f"[FAIL] Basic smoke test failed: {e}", verbose=True)

    # 6) Summarize and set exit code
    if dep_failures:
        results.append(
            CheckResult("Dependencies installed", ok=False, detail="; ".join(dep_failures))
        )
    else:
        results.append(
            CheckResult("Dependencies installed", ok=True, detail="All dependencies present")
        )

    if dep_warnings:
        results.append(
            CheckResult("Dependency imports", ok=True, warn=True, detail="; ".join(dep_warnings))
        )
    else:
        results.append(
            CheckResult("Dependency imports", ok=True, detail="All dependency imports succeeded")
        )

    summarize_and_exit(results)
    # Exit non-zero if any hard failure
    hard_fail = any((not r.ok) and (not r.warn) for r in results)
    return 1 if hard_fail else 0


def summarize_and_exit(results: List[CheckResult]) -> None:
    print("\n=== fvcomersemviz / fvcomersem-viz Readiness Summary ===")
    width = max(len(r.name) for r in results) + 2
    for r in results:
        status = "OK"
        if not r.ok and not r.warn:
            status = "FAIL"
        elif r.warn:
            status = "WARN"
        line = f"{status:>4}  {r.name:<{width}} {r.detail}"
        print(line)
    print("========================================================")
    # Suggest next step if failed
    if any((not r.ok) and (not r.warn) for r in results):
        print("One or more checks failed. Please review the FAIL items above.")
    elif any(r.warn for r in results):
        print("Environment looks usable, but there were warnings. Review them above.")
    else:
        print("All good! fvcomersemviz and its dependencies look ready to run.")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception:
        print("Unexpected error:\n" + traceback.format_exc())
        sys.exit(1)
