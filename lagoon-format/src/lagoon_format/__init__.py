"""LAGOON data format: specification constants and (later) the reference reader/writer.

MIT-licensed, independent of the LAGOON application.
"""

__version__ = "0.0.1.dev0"

#: Version of the format specification implemented by this package.
SPEC_VERSION = "0.0"

#: File extension used for LAGOON datasets (a Zarr store directory).
EXTENSION = ".lagoon"

__all__ = ["EXTENSION", "SPEC_VERSION", "__version__"]
