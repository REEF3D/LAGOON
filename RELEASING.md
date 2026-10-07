# Releasing LAGOON

Two packages come out of this repository: **lagoon-viz** (the application, GPLv3,
version in `src/lagoonviz/__init__.py`) and **lagoon-format** (the data format
library, MIT, version in `lagoon-format/src/lagoon_format/__init__.py`). A tag
`v<lagoon-viz version>` releases both: the workflow `.github/workflows/release.yml`
builds them, tests the built wheels on macOS, Linux and Windows, publishes them to
PyPI (a lagoon-format version already on PyPI is skipped) and makes the GitHub
release with the changelog's section as its notes.

## Once

1. **PyPI**: with your PyPI account, add a *pending trusted publisher* for each of
   the two projects (pypi.org → Your projects → Publishing):
   project `lagoon-viz` and project `lagoon-format`, each with owner `REEF3D`,
   repository `LAGOON`, workflow `release.yml`, environment `pypi`. No token is
   stored anywhere; GitHub proves to PyPI that the upload comes from that workflow.
2. **GitHub**: Settings → Environments → New environment `pypi`. Add yourself as a
   required reviewer if every upload should wait for your click.
3. **Documentation**: Settings → Pages → Source: *GitHub Actions*. The workflow
   `docs.yml` then publishes the site at https://reef3d.github.io/LAGOON/ from every
   push to `main`.

## Each release

1. **Versions**: set `__version__` in `src/lagoonviz/__init__.py` (e.g. `1.0.0rc1`,
   `1.0.0`), and in `lagoon-format/src/lagoon_format/__init__.py` if the format
   library changed (then also the `lagoon-format>=…,<…` range in `pyproject.toml`).
   The changelog needs a section `## <version>` at the top; the conda-forge recipes
   in `packaging/conda-forge/` carry the same versions. `tests/test_packaging.py`
   checks that all of this agrees.
2. **Check locally**:

   ```bash
   .venv/bin/pytest -q
   uv build --out-dir dist . && uv build --out-dir dist lagoon-format
   uvx twine check --strict dist/*
   ```

3. **Dry run** (optional): GitHub → Actions → Release → *Run workflow* on `main`.
   It builds and tests the wheels on the three systems and publishes nothing.
4. **Tag and push**:

   ```bash
   git tag -a v1.0.0rc1 -m "LAGOON 1.0.0rc1"
   git push origin main v1.0.0rc1
   ```

   The workflow refuses a tag that does not match the version or has no changelog
   section. Release candidates (`rc`, `a`, `b`) become pre-releases on GitHub, and
   `pip install lagoon-viz` keeps installing the last final version; testers use
   `pip install --pre lagoon-viz`.
5. **Afterwards**: set the next development version (e.g. `1.0.1.dev0`) and open a
   new changelog section.

## conda-forge

conda-forge builds from the PyPI source packages, so it comes after a release on
PyPI; submit with a final version (1.0.0), not a release candidate.

1. The sha256 of each source package: on its PyPI page (*Download files* → *view
   hashes*), or

   ```bash
   curl -sL https://pypi.org/packages/source/l/lagoon-viz/lagoon_viz-1.0.0.tar.gz | shasum -a 256
   curl -sL https://pypi.org/packages/source/l/lagoon-format/lagoon_format-0.3.0.tar.gz | shasum -a 256
   ```

2. Put them into `packaging/conda-forge/*/recipe.yaml`, and your GitHub handle as
   the recipe maintainer.
3. Fork `github.com/conda-forge/staged-recipes`, copy both folders to `recipes/`
   (`recipes/lagoon-format/recipe.yaml`, `recipes/lagoon-viz/recipe.yaml`; one pull
   request may hold both, lagoon-format is built first) and open the pull request.
   After review, conda-forge makes a feedstock per package; its bot opens a pull
   request for every later PyPI release.

Then `conda install -c conda-forge lagoon-viz` and `pixi add lagoon-viz` work.
