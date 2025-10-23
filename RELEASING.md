fvcomersem-viz: Releasing and Development Guide
===============================================

This document explains how to:
- Release a new version to PyPI
- Fix bugs and publish hotfixes
- Add new features safely
- Manage branches and collaborators

Assumptions:
- main = stable release branch
- dev  = development branch
- Versions come automatically from git tags via setuptools-scm
- PyPI publishing is handled by a GitHub Action (publish.yml)


0. One-time setup checklist
---------------------------

[ ] Repository structure:
    fvcomersem-viz/
      ├─ pyproject.toml
      ├─ src/fvcomersemviz/
      │   ├─ __init__.py
      │   └─ other code...
      └─ .github/workflows/publish.yml

[ ] pyproject.toml includes:
    - dynamic = ["version"]
    - [tool.setuptools_scm] write_to = "src/fvcomersemviz/_version.py"
    - dependencies = [...]  (numpy, pandas, xarray, matplotlib, geopandas, etc.)

[ ] GitHub -> Settings -> Secrets -> Actions -> add
    PYPI_API_TOKEN = <your token from https://pypi.org/manage/account/>

[ ] Create the dev branch:
    git checkout main && git pull
    git branch dev
    git push -u origin dev

[ ] (Optional) Protect main branch:
    - Require Pull Requests and passing checks
    - Restrict push access to the code owner

[ ] Enable "Auto-delete branches after merge" in
    Settings -> General -> Pull Requests


1. Releasing a new version
--------------------------

Goal: publish version vX.Y.Z to PyPI (automated by GitHub Actions).

1. Ensure dev has the latest work:
       git checkout dev
       git pull

2. Merge dev into main:
       git checkout main
       git pull
       git merge --no-ff dev -m "merge dev for vX.Y.Z"
       git push

3. Tag and push (triggers PyPI publish):
       git tag -a vX.Y.Z -m "fvcomersem-viz X.Y.Z"
       git push origin vX.Y.Z

4. (Optional) Test the release:
       python -m venv .venv && . .venv/bin/activate
       pip install -U pip
       pip install fvcomersem-viz
       python -c "import fvcomersemviz as m; print(m.__version__)"
       deactivate


2. Everyday development (features)
----------------------------------

Work on the dev branch.

       git checkout dev
       # edit code
       git add -A
       git commit -m "feat: add new plotting feature"
       git push

Repeat as needed. When ready, merge dev into main and tag a new release.


3. Hotfix an existing release
-----------------------------

If a bug is found in a released version:

1. Fix it on dev:
       git checkout dev
       # edit, test
       git add -A
       git commit -m "fix: correct mask shape bug"
       git push

2. Copy that fix to main:
       git checkout main
       git pull
       git cherry-pick <commit-sha>
       git push

3. Tag and push the patch release:
       git tag -a vX.Y.(Z+1) -m "hotfix"
       git push origin vX.Y.(Z+1)

4. Merge back into dev:
       git checkout dev
       git merge --no-ff main -m "merge hotfix into dev"
       git push


4. Adding collaborators
-----------------------

Invite collaborators:
    GitHub -> Settings -> Collaborators -> Add people
    Give them Write access (not Admin).

Protect main:
    - Require PRs before merge
    - Require status checks
    - Disallow force pushes and deletions
    - Optionally restrict push access to you only

CODEOWNERS (auto-request reviews):
Create .github/CODEOWNERS with:
    * @mollyjames2
GitHub will automatically request your review on PRs.


5. Collaborator instructions
----------------------------

For contributors:

    git clone https://github.com/mollyjames2/fvcomersem-viz.git
    cd fvcomersem-viz

Option A: conda
    conda create -n fviz python=3.11 geopandas shapely pyproj rtree notebook -c conda-forge
    conda activate fviz
    pip install -e .[dev]

Option B: venv
    python -m venv .venv && . .venv/bin/activate
    pip install -e .[dev]

Create a branch for changes:
    git checkout dev
    git pull
    git checkout -b feat/my-change

Commit and push:
    git add -A
    git commit -m "feat: describe change"
    git push -u origin feat/my-change

Open a Pull Request (base = dev) on GitHub.


6. Version numbering (SemVer)
-----------------------------

  Patch : 1.0.1  -> bug fix only
  Minor : 1.1.0  -> new features, backwards compatible
  Major : 2.0.0  -> breaking changes

Do not edit version numbers manually.
The git tag name defines the version.


7. Maintenance commands
-----------------------

Show recent commits:
    git log --oneline --decorate --graph -n 15

Clean up old branches:
    git fetch --prune

Delete merged branches:
    git branch -d feat/something


8. Quick reference
------------------

New release:
    git checkout main && git pull
    git merge --no-ff dev -m "merge dev for vX.Y.Z" && git push
    git tag -a vX.Y.Z -m "release" && git push origin vX.Y.Z

Hotfix:
    git checkout dev
    # fix, commit, push
    git checkout main && git pull
    git cherry-pick <sha> && git push
    git tag -a vX.Y.(Z+1) -m "hotfix" && git push origin vX.Y.(Z+1)
    git checkout dev && git merge --no-ff main && git push

Feature:
    work on dev, commit, push

Clean up:
    git fetch --prune


End of document
Maintainer: @mollyjames2

