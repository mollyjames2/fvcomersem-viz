Contributing to fvcomersem-viz
==============================

Thank you for your interest in contributing to fvcomersem-viz.

This document explains how to set up your environment, make changes, and submit contributions.

-------------------------------------------------------------------------------
1. Overview
-------------------------------------------------------------------------------

The repository uses two main branches:

- main : stable, released versions (published to PyPI)
- dev  : active development branch

You should always branch from dev when making changes.

Releases are automated from git tags pushed to main.

-------------------------------------------------------------------------------
2. Getting started
-------------------------------------------------------------------------------

Clone the repository:

    git clone https://github.com/mollyjames2/fvcomersem-viz.git
    cd fvcomersem-viz

Create a development environment.

Option A: Using conda (recommended)

    conda create -n fviz python=3.11 geopandas shapely pyproj rtree notebook -c conda-forge
    conda activate fviz
    pip install -e .[dev]

Option B: Using venv

    python -m venv .venv
    . .venv/bin/activate
    pip install -e .[dev]

Confirm the package imports:

    python -c "import fvcomersemviz; print('fvcomersemviz OK')"

-------------------------------------------------------------------------------
3. Working on a change
-------------------------------------------------------------------------------

Always start from the dev branch:

    git checkout dev
    git pull

Create a feature branch:

    git checkout -b feat/short-description

Make your edits and commit them:

    git add -A
    git commit -m "feat: short description of change"

Push your branch to GitHub:

    git push -u origin feat/short-description

-------------------------------------------------------------------------------
4. Submitting a pull request
-------------------------------------------------------------------------------

1. Go to the repository on GitHub.
2. Click "Compare & pull request".
3. Set the base branch to dev (not main).
4. Add a short, clear title and description.
5. Submit the pull request (PR).

Continuous Integration (CI) will automatically run build and install checks.

-------------------------------------------------------------------------------
5. Code style and testing
-------------------------------------------------------------------------------

Follow the existing code style in the repository.

General guidelines:
- Use 4 spaces for indentation.
- Write clear, descriptive commit messages.
- Avoid committing large generated files or data outputs.
- Include comments where necessary.

Run any provided tests:

    python tests/check_install.py

If new functionality is added, please include minimal test coverage if possible.

-------------------------------------------------------------------------------
6. Reviewing and merging
-------------------------------------------------------------------------------

All PRs into dev require at least one review (by @mollyjames2 or another maintainer).

When approved, the PR will be merged (squash-merge or merge commit).

After merging:
- Your branch will be automatically deleted (if settings allow).
- You can safely delete your local branch:

      git checkout dev
      git pull
      git branch -d feat/short-description

-------------------------------------------------------------------------------
7. Release process (for maintainers)
-------------------------------------------------------------------------------

Releases are controlled by the maintainer.

Process summary:
1. Merge dev into main.
2. Tag a new version (vX.Y.Z).
3. Push the tag to GitHub.
4. GitHub Actions automatically build and publish to PyPI.

Example:

    git checkout main && git pull
    git merge --no-ff dev -m "merge dev for v1.1.0" && git push
    git tag -a v1.1.0 -m "fvcomersem-viz 1.1.0" && git push origin v1.1.0

-------------------------------------------------------------------------------
8. Licensing
-------------------------------------------------------------------------------

By contributing, you agree that your code will be distributed under the same
license as the project (MIT License, see LICENSE file).

-------------------------------------------------------------------------------
9. Contact
-------------------------------------------------------------------------------

For questions or guidance:
- Maintainer: @mollyjames2
- Issues: https://github.com/mollyjames2/fvcomersem-viz/issues

-------------------------------------------------------------------------------
End of document
-------------------------------------------------------------------------------

