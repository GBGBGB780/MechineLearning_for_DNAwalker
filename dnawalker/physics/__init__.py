"""Core physical simulation and parameter-refinement kernels."""

from . import refinement, simulator

# Short domain aliases retained as part of the public Python API.
pysim = simulator
refine = refinement

__all__ = ["refinement", "simulator", "pysim", "refine"]
