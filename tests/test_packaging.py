"""The release fits together: versions, changelog, package metadata, recipes.

Run with the rest; the release workflow runs them against the built wheels too
(there the sources are moved away, so only the installed packages are read).
"""

import importlib.util
import re
import tomllib
from pathlib import Path

import lagoon_format

import lagoonviz

ROOT = Path(__file__).resolve().parents[1]
PEP440 = re.compile(r"^\d+\.\d+\.\d+((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?$")


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_versions_are_valid_and_single_sourced():
    assert PEP440.match(lagoonviz.__version__), lagoonviz.__version__
    assert PEP440.match(lagoon_format.__version__), lagoon_format.__version__
    for path, init in (
        (ROOT / "pyproject.toml", "src/lagoonviz/__init__.py"),
        (ROOT / "lagoon-format" / "pyproject.toml", "src/lagoon_format/__init__.py"),
    ):
        project = _toml(path)
        assert "version" not in project["project"]  # only in __init__.py
        assert project["project"]["dynamic"] == ["version"]
        assert project["tool"]["hatch"]["version"]["path"] == init


def test_the_application_takes_this_format_package():
    dependencies = _toml(ROOT / "pyproject.toml")["project"]["dependencies"]
    pin = next(d for d in dependencies if d.startswith("lagoon-format"))
    low, high = re.match(r"lagoon-format>=([\d.]+),<([\d.]+)", pin).groups()

    def key(v):
        return tuple(int(p) for p in re.match(r"[\d.]+", v).group(0).split("."))

    assert key(low) <= key(lagoon_format.__version__) < key(high)


def test_the_changelog_has_this_release():
    notes = _tool("release_notes")
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert notes.notes(text, lagoonviz.__version__)  # the GitHub release's notes
    assert text.index(f"## {lagoonviz.__version__}") < text.index("## 0.")  # newest first
    assert notes.notes("## 1.0 (a)\nx\n## 0.9\ny", "1.0") == "x"
    assert notes.notes("## 1.0\nx", "2.0") == ""


def test_readmes_work_on_pypi():
    # PyPI shows the README without the repository: no relative images or links
    for path in (ROOT / "README.md", ROOT / "lagoon-format" / "README.md"):
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r'(?:src="|\]\()([^")]+)', text):
            assert target.startswith(("http", "#", "mailto:")), (path.name, target)


def test_package_files():
    package = Path(lagoonviz.__file__).parent
    assert (package / "render" / "blender_cycles.py").exists()  # run by Blender
    assert (package / "render" / "assets" / "sea_clear_sky.hdr").exists()  # Cycles' sky
    assert (package / "gui" / "resources" / "icon.png").exists()
    scripts = _toml(ROOT / "pyproject.toml")["project"]["scripts"]
    assert set(scripts) == {"lagoon", "lagoon-render", "lagoon-convert"}


def test_conda_recipes_follow_the_versions():
    for name, version in (
        ("lagoon-viz", lagoonviz.__version__),
        ("lagoon-format", lagoon_format.__version__),
    ):
        recipe = (ROOT / "packaging" / "conda-forge" / name / "recipe.yaml").read_text()
        found = re.search(r'^\s+version: "([^"]+)"', recipe, re.M).group(1)
        assert found == version, (name, found, version)
