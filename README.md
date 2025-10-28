[![Build and test](https://github.com/mollyjames2/fvcomersem-viz/actions/workflows/build-test.yml/badge.svg)](https://github.com/mollyjames2/fvcomersem-viz/actions/workflows/build-test.yml)
# fvcomersem-viz

Utilities for turning **FVCOM / FVCOM–ERSEM** model output into clear, reproducible figures and animations: time-series, plan-view maps, Hovmöller sections, x–y “curves” diagnostics, stoichiometry KDEs, and more.

---

## Features

* **Time-series** at domain / region / station scopes (surface, bottom, depth-avg, sigma, fixed-z).
* **Plan-view maps** (full domain or polygon-masked regions).
* **Hovmöller** (time × depth) in native σ or interpolated z.
* **Composition** (time × depth) phyto and zoo - surface/bottom comparisons,
  depth averged or specific depth.
* **Curves (x–y diagnostics)** with binned median±IQR or raw scatter.
* **Stoichiometry KDEs** (N:C & P:C vs variable) in a 2×2 panel.
* **Animations** for time-series and maps.
* Sensible **file naming** & directories for batch runs; Dask-friendly.

---

## Requirements

* **Python** ≥ 3.9 (3.11 recommended)
* Core: `numpy`, `pandas`, `xarray`, `matplotlib`, `netCDF4`, `cftime`,
  `scipy`, `notebook`
* Geospatial (for region masks/overlays): `geopandas`, `shapely`, `pyproj`, `rtree` *(optional but recommended)*
* Optional performance: `dask[array]`
---

## Installation

### Option A — User install (recommended)

Use this if you just want to use the package.

1. Create the environment (from `environment.yml`) and activate it:

```bash
mamba env create -f environment.yml
mamba activate fviz
# (use conda instead of mamba if you prefer)
```

2. Verify:

```bash
python tests/check_install.py
# or a tiny smoke test:
python -c "import fvcomersemviz as m; print('fvcomersemviz', getattr(m,'__version__','n/a'))"
```

> Note: `environment.yml` installs all dependencies from conda-forge and installs this package (non-editable) via pip inside the same environment.

---

### Option B — Developer install (contributors)

Use this if you plan to modify the code, run tests, or use linting tools.

1. Clone the repository and create the dev environment (from `environment-dev.yml`), then activate it:

```bash
git clone https://github.com/mollyjames2/fvcomersem-viz.git
cd fvcomersem-viz
mamba env create -f environment-dev.yml
mamba activate fviz-dev
```

2. Verify dev setup:

```bash
python tests/check_install_dev.py
ruff check .
pytest -q
```

> Notes:
>
> * `environment-dev.yml` installs the package in editable mode with `.[dev]` extras (e.g., pytest, ruff).
> * Edits under `src/fvcomersemviz/` take effect immediately.

---

### Option C — Manual install (no YAML)

If you don’t want to use the YAML files, you can create the environment and install packages explicitly. This keeps all heavy geospatial libs consistent via conda-forge, then installs the package with pip.

Create and activate the env:

```bash
mamba create -n fviz -c conda-forge \
  python=3.11 \
  numpy scipy pandas xarray dask netcdf4 \
  matplotlib notebook \
  geopandas shapely pyproj rtree cartopy pip
mamba activate fviz
```

Install the package (users, non-editable):

```bash
python -m pip install "git+https://github.com/mollyjames2/fvcomersem-viz.git"
# SSH alternative (if your GitHub is set up for SSH):
# python -m pip install "git+ssh://git@github.com/mollyjames2/fvcomersem-viz.git"
```

Or, for developers (editable with dev tools) from the repo root:

```bash
git clone https://github.com/mollyjames2/fvcomersem-viz.git
cd fvcomersem-viz
python -m pip install -e ".[dev]"
```

Verify:

```bash
python tests/check_install.py       # users
# or
python tests/check_install_dev.py   # developers
```
If you see
```
  All good! fvcomersemviz and its dependencies look ready to run.
```

---

### Troubleshooting

* Ensure you **activate the environment** before running Python:

  ```bash
  mamba activate fviz       # or: mamba activate fviz-dev
  which python              # should point inside your env
  which pip                 # should point inside your env
  ```
* Keep geospatial libs (GDAL/PROJ/GEOS family via geopandas/shapely/pyproj/rtree/cartopy) from **conda-forge**. Avoid mixing pip wheels for those packages in the same env.
* If you must reinstall cleanly:

  ```bash
  mamba env remove -n fviz -y
  mamba env remove -n fviz-dev -y
  ```

---

## Examples

Browse and run the ready-to-use scripts in the **[examples/](https://github.com/mollyjames2/fvcomersem-viz/tree/main/examples)** folder or run the notebooks in the **[notebooks/](https://github.com/mollyjames2/fvcomersem-viz/tree/main/notebooks)** folder in jupyter notebook

---

## Outputs & naming

Figures/animations are written under:

```
FIG_DIR/<basename(BASE_DIR)>/<module>/
```

Filenames encode **scope**, **variable(s)**, **depth tag**, and **time label** for easy batch comparison.

---

## Project structure

```
fvcomersem-viz/
├─ src/fvcomersemviz/
│  ├─ __init__.py
│  ├─ io.py
│  ├─ utils.py
│  ├─ regions.py
│  ├─ plot.py
│  └─ plots/
│     ├─ timeseries.py
│     ├─ maps.py
│     ├─ hovmoller.py
│     ├─ curves.py
│     ├─ composition.py
│     ├─ kde_stoichiometry.py
│     └─ animate.py
├─ examples/
│  ├─ tutorial.py
│  ├─ plot_timeseries.py
│  ├─ plot_maps.py
│  ├─ plot_kde.py
│  ├─ plot_hovmoller.py
│  ├─ plot_curves.py
│  ├─ plot_composition.py
│  └─ plot_animations.py
├─ notebooks/
│  ├─ quickstart_tutorial.ipynb
│  ├─ plot_timeseries.ipynb
│  ├─ plot_maps.ipynb
│  ├─ plot_kde.ipynb
│  ├─ plot_hovmoller.ipynb
│  ├─ plot_curves.ipynb
│  ├─ plot_composition.ipynb
│  └─ plot_animations.ipynb
├─ tests/
│  └─ check_install.py
├─ LICENSE
├─ README.md
├─ pyproject.toml / setup.cfg (one or both)
└─ .gitignore
```

*(Some filenames may evolve; see the repo for the authoritative layout.)*

---

## How to cite


**APA (template)**

> James, MK. (2025). *fvcomersem-viz (Version 0.0.1)* [Computer software]. GitHub. [https://github.com/mollyjames2/fvcomersem-viz](https://github.com/mollyjames2/fvcomersem-viz)

**BibTeX (template)**

```bibtex
@software{fvcomersem_viz,
  title        = {fvcomersem-viz},
  author       = {Molly James},
  year         = {2025},
  version      = {0.0.1},
  url          = {https://github.com/mollyjames2/fvcomersem-viz},
  note         = {FVCOM / FVCOM–ERSEM visualization utilities}
}
```


---

## Tips & troubleshooting

* **Large files:** avoid committing files >100 MB (GitHub blocks them). Use `.gitignore` for generated figures/data; consider **Git LFS** for large artifacts.
* **Notebooks:** clear outputs before committing:

  ```bash
  jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
  ```
* **Performance:** open datasets with Dask chunks for large runs.

---

## Contributing

Contributions of all kinds are welcome: bug fixes, new features, documentation improvements, and ideas.

### Quick start

1. Clone the repository

    git clone https://github.com/mollyjames2/fvcomersem-viz.git
    cd fvcomersem-viz

2. Create your development environment

    Option A: conda (recommended)

        conda create -n fviz python=3.11 geopandas shapely pyproj rtree notebook -c conda-forge
        conda activate fviz
        pip install -e .[dev]

    Option B: virtual environment

        python -m venv .venv
        . .venv/bin/activate
        pip install -e .[dev]

3. Create a branch for your change

        git checkout dev
        git pull
        git checkout -b feat/my-change

4. Make and test your edits

        python tests/check_install.py
        git add -A
        git commit -m "feat: brief summary of your change"
        git push -u origin feat/my-change

5. Open a Pull Request

    - Base branch: dev
    - Add a clear title and short description
    - Continuous Integration (CI) will run automatically

After review and merge, your branch will be deleted automatically.

### More information

See the full Contributing Guide (CONTRIBUTING.md) and Releasing Guide (RELEASING.md)
for details on branching, review, and release workflows.


### Maintainer note

The fvcomersem-viz package follows a simple branching and release workflow:

- Development work happens on the dev branch.
- Stable releases live on the main branch.
- Version numbers come automatically from git tags (using setuptools-scm).
- When a tag starting with v is pushed to main (for example v1.0.0),
  a GitHub Action builds the package and publishes it to PyPI.

This means that regular development commits on dev are not published until they are
merged into main and tagged as a release. Each tag on main creates a new version on
PyPI that users can install with:

    pip install fvcomersem-viz

Small bug fixes may be applied directly to main as a hotfix and tagged as a patch
release (for example v1.0.1). Larger changes and new features are developed and tested
on dev before being merged and released.

For more details on releases and versioning, see RELEASING.md.

Issues and pull requests are welcome. For development, use `pip install -e .`, run `python tests/check_install.py`, and follow standard Git practices (feature branches + PRs).

---

## License

See the **LICENSE** file in this repository.


# fvcomersem-viz
