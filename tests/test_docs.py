"""The documentation (docs/, mkdocs.yml) holds together without building it:
every page in the navigation exists, links and anchors lead somewhere, pictures
are there or made by tools/docs_images.py, and the command line pages name every
option the commands have.
"""

import contextlib
import importlib.util
import io
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def _images_made() -> set[str]:
    spec = importlib.util.spec_from_file_location("docs_images", ROOT / "tools" / "docs_images.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {f"images/{name}.png" for name in module.IMAGES}


def _nav() -> list[str]:
    text = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    nav = text[text.index("\nnav:") :]
    return re.findall(r":\s+([\w/.-]+\.md)\s*$", nav, re.M)


def _pages() -> list[Path]:
    return sorted(DOCS.rglob("*.md"))


def _text(page: Path) -> str:
    """A page as mkdocs sees it: snippet includes put in."""
    text = page.read_text(encoding="utf-8")

    def include(match):
        return (ROOT / match.group(1)).read_text(encoding="utf-8")

    return re.sub(r'^--8<-- "([^"]+)"\s*$', include, text, flags=re.M)


def _slug(heading: str) -> str:
    heading = re.sub(r"[`*_]", "", heading.strip().lower())
    heading = re.sub(r"[^\w\- ]", "", heading)
    return re.sub(r"\s+", "-", heading)


def _anchors(page: Path) -> set[str]:
    text = re.sub(r"```.*?```", "", _text(page), flags=re.S)
    return {_slug(h) for h in re.findall(r"^#{1,6}\s+(.+)$", text, re.M)}


def test_every_page_in_the_navigation_exists_and_is_listed():
    nav = _nav()
    assert nav and len(nav) == len(set(nav))
    for name in nav:
        assert (DOCS / name).exists(), name
    listed = {DOCS / name for name in nav}
    for page in _pages():
        assert page in listed, f"{page.relative_to(DOCS)} is not in the navigation"


def test_snippets_point_at_files():
    for page in _pages():
        for target in re.findall(r'^--8<-- "([^"]+)"', page.read_text(encoding="utf-8"), re.M):
            assert (ROOT / target).exists(), (page.name, target)


def test_links_and_pictures_lead_somewhere():
    made = _images_made()
    for page in _pages():
        text = re.sub(r"```.*?```", "", _text(page), flags=re.S)
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            path, _, anchor = target.partition("#")
            linked = (page.parent / path).resolve() if path else page
            relative = linked.relative_to(DOCS.resolve()).as_posix() if path else ""
            if relative in made:
                continue
            assert linked.exists(), f"{page.relative_to(DOCS)}: {target}"
            if anchor and linked.suffix == ".md":
                assert anchor in _anchors(linked), f"{page.relative_to(DOCS)}: {target}"


def _options(main) -> set[str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(SystemExit):
        main(["--help"])
    return set(re.findall(r"(?<![\w-])(--[a-z][\w-]*)", out.getvalue())) - {"--help"}


def test_the_command_line_page_names_every_option():
    from lagoonviz.core.convert import main as convert
    from lagoonviz.render.offscreen import main as render

    page = (DOCS / "reference" / "cli.md").read_text(encoding="utf-8")
    sections = re.split(r"^## ", page, flags=re.M)
    section = {s.split("\n", 1)[0].strip(): s for s in sections[1:]}
    for name, main in (("lagoon-render", render), ("lagoon-convert", convert)):
        missing = [o for o in sorted(_options(main)) if f"`{o}" not in section[name]]
        assert not missing, (name, missing)


def test_the_application_options_are_documented():
    app = pytest.importorskip("lagoonviz.gui.app")
    page = (DOCS / "reference" / "cli.md").read_text(encoding="utf-8")
    section = page.split("## lagoon\n", 1)[1].split("\n## ", 1)[0]
    missing = [o for o in sorted(_options(app.main)) if o not in section]
    assert not missing, missing
