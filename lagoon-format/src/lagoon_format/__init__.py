"""LAGOON data format: specification constants and (later) the reference reader/writer.

MIT-licensed, independent of the LAGOON application.
"""

__version__ = "0.0.1.dev0"

#: Version of the format specification implemented by this package.
SPEC_VERSION = "0.0"

#: Version of the REEF3D run log specification (RUN.md) read by ``lagoon_format.run``.
RUN_LOG_VERSION = 1

#: File extension used for LAGOON datasets (a Zarr store directory).
EXTENSION = ".lagoon"

__all__ = ["EXTENSION", "RUN_LOG_VERSION", "SPEC_VERSION", "__version__"]
