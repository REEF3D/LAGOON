"""LAGOON data format: the reference reader and writer (``lagoon_format.store``), a
small Zarr v3 implementation (``lagoon_format.zarr3``) and the REEF3D run log
(``lagoon_format.run``).

MIT-licensed, independent of the LAGOON application.
"""

__version__ = "0.3.0"

#: Version of the format specification implemented by this package.
SPEC_VERSION = "0.3"

#: Version of the REEF3D run log specification (RUN.md) read by ``lagoon_format.run``.
RUN_LOG_VERSION = 1

#: File extension used for LAGOON datasets (a Zarr store directory).
EXTENSION = ".lagoon"

__all__ = ["EXTENSION", "RUN_LOG_VERSION", "SPEC_VERSION", "__version__"]
