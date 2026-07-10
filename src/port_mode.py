"""
PortMode enum — single source of truth for how a CUDA→HIP port should
be structured.

Two modes:

* ``WHOLE_PROGRAM`` — the source is a complete, self-contained program with its
  own ``main()`` and all host dependencies resolveable. The coder reproduces
  everything faithfully, including ``main()``.

* ``DEVICE_SUBSET`` — the source has a ``main()`` but one or more host
  dependencies cannot be resolved locally (missing local headers, undefined host
  symbols, NVIDIA-specific driver APIs). The coder ports only the
  ``__global__`` / ``__device__`` functions and drops the host driver; the
  verification harness synthesizes a test driver.
"""

from enum import Enum


class PortMode(str, Enum):
    WHOLE_PROGRAM = "WHOLE_PROGRAM"
    DEVICE_SUBSET = "DEVICE_SUBSET"
