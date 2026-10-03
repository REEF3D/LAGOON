import os

# GUI tests run without a display (CI, clusters).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from reef3d_fixtures import make_run  # noqa: E402


@pytest.fixture
def reef3d_run(tmp_path):
    """A small NHFLOW case: free surface, bed and volume, 3 frames, 2 MPI ranks."""
    return make_run(tmp_path / "run")
