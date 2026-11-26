"""TopoMemory: hysteretic self-locking memory for long-context sequence modeling.

The fixed-threshold reference module is exported here. The learnable
(reparameterized) variant lives in ``topomemory.learnable`` and is imported
explicitly, since both classes share the name ``TopoMemoryLayer``.
"""

from .core import TopoMemoryLayer

__all__ = ["TopoMemoryLayer"]
