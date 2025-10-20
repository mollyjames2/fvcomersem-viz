# fvcomersem-viz

Utilities for turning **FVCOM / FVCOM–ERSEM** model output into clear, reproducible figures and animations: time-series, plan-view maps, Hovmöller sections, x–y “curves” diagnostics, stoichiometry KDEs, and more.

---

## Features

* **Time-series** at domain / region / station scopes (surface, bottom, depth-avg, sigma, fixed-z).
* **Plan-view maps** (full domain or polygon-masked regions).
* **Hovmöller** (time × depth) in native σ or interpolated z.
* **Curves (x–y diagnostics)** with binned median±IQR or raw scatter.
* **Stoichiometry KDEs** (N:C & P:C vs variable) in a 2×2 panel.
* **Animations** for time-series and maps.
* Sensible **file naming** & directories for batch runs; Dask-friendly.

---

## Requirements

* **Python** ≥ 3.9 (3.11 recommended)
* Core: `numpy`, `pandas`, `xarray`, `matplotlib`, `netCDF4`, `cftime`, `scipy`
* Geospatial (for region masks/overlays): `geopandas`, `shapely`, `pyproj`, `rtree` *(optional but recommended)*
* Optional performance: `dask[array]`

---

## Installation

```bash
# create a clean env with geospatial stack
conda create -n fviz python=3.11 geopandas shapely pyproj rtree notebook -c conda-forge
conda activate fviz

# install the package
pip install "git+https://github.com/mollyjames2/fvcomersem-viz.git"
```

**Development (editable) install**

```bash
git clone https://github.com/mollyjames2/fvcomersem-viz.git
cd fvcomersem-viz
pip install -e .
```

**Check your installation**

```bash
python tests/check_install.py
```

---

## Examples

Browse and run the ready-to-use scripts in the **[examples/](https://github.com/mollyjames2/fvcomersem-viz/tree/main/examples)** folder.

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
├─ notebooks/
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

Issues and pull requests are welcome. For development, use `pip install -e .`, run `python tests/check_install.py`, and follow standard Git practices (feature branches + PRs).

---

## License

See the **LICENSE** file in this repository.


# fvcomersem-viz
