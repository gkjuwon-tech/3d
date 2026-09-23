"""One array API for both CPU and GPU.

Every hot loop in stage 2 is array arithmetic, gathers, filters and sparse
solves, and CuPy mirrors NumPy/SciPy closely enough for those to be written
once. Set THREED_GPU=1 to run on an NVIDIA GPU through CuPy; without it (or
without CuPy) everything runs on NumPy/SciPy, and the pipeline keeps working
on any four-core machine.

  from xp import xp, ndi, sp, spla, GPU, cpu, gpu
"""
import os

GPU = False
if os.environ.get("THREED_GPU") == "1":
    try:
        import cupy as xp                               # noqa: F401
        import cupyx.scipy.ndimage as ndi               # noqa: F401
        import cupyx.scipy.sparse as sp                 # noqa: F401
        import cupyx.scipy.sparse.linalg as spla        # noqa: F401
        xp.zeros(1)                                     # fails without a device
        GPU = True
    except Exception as e:                              # pragma: no cover
        print(f"[xp] GPU requested but unavailable ({e}); using CPU", flush=True)
if not GPU:
    import numpy as xp                                  # noqa: F401
    import scipy.ndimage as ndi                         # noqa: F401
    import scipy.sparse as sp                           # noqa: F401
    import scipy.sparse.linalg as spla                  # noqa: F401


def cpu(a):
    """To a NumPy array, whichever side it lives on."""
    return a.get() if GPU and hasattr(a, "get") else a


def gpu(a):
    """To the active backend's array type."""
    return xp.asarray(a)
