import os

# GUI tests run without a display (CI, clusters).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from reef3d_fixtures import make_run  # noqa: E402
from snapshot_tools import Snapshots  # noqa: E402


def pytest_addoption(parser):
    parser.addoption(
        "--update-snapshots",
        action="store_true",
        default=False,
        help="accept the current window structure and renders as the agreed snapshots",
    )


@pytest.fixture
def reef3d_run(tmp_path):
    """A small NHFLOW case: free surface, bed and volume, 3 frames, 2 MPI ranks."""
    return make_run(tmp_path / "run")


@pytest.fixture
def snapshots(request, tmp_path_factory):
    """Compare with the agreed snapshots in tests/snapshots (see snapshot_tools)."""
    folder = os.environ.get("LAGOON_TEST_OUTPUT")
    output = os.path.join(folder, "snapshots") if folder else tmp_path_factory.mktemp("snapshots")
    from pathlib import Path

    return Snapshots(request.config.getoption("--update-snapshots"), Path(output))
