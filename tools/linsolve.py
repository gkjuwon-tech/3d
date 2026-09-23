"""Symmetric positive definite solves for normal integration, CPU or GPU.

CPU: smoothed-aggregation algebraic multigrid (pyamg) as a CG preconditioner.
It converges in tens of iterations, but its setup is serial and costs seconds.

GPU (THREED_GPU=1): CG with a Jacobi preconditioner on CuPy. It needs
thousands of iterations on a Poisson-like system, but each is one sparse
matrix-vector product across the whole device, so it still wins.

The tolerance is a relative residual. 1e-6 reproduces 1e-10 on Lucy's front
view to 0.000 voxels mean and 0.001 at the 99th percentile, in 46% of the
time, so 1e-6 is the default.
"""
import numpy as np

from xp import GPU, cpu


def spd_solve(A, b, x0=None, tol=1e-6, maxiter_amg=400, maxiter_cg=30000):
    """Solve A x = b for a SciPy CSR matrix A. Returns a NumPy array."""
    if not GPU:
        import pyamg
        ml = pyamg.smoothed_aggregation_solver(A, symmetry="symmetric",
                                               max_coarse=500)
        return ml.solve(b, x0=x0, tol=tol, accel="cg", maxiter=maxiter_amg)
    import cupy as cp
    import cupyx.scipy.sparse as csp
    import cupyx.scipy.sparse.linalg as cla
    Ag = csp.csr_matrix(A.astype(np.float64))
    bg = cp.asarray(b, dtype=cp.float64)
    dinv = 1.0 / Ag.diagonal()
    M = cla.LinearOperator(A.shape, matvec=lambda x: dinv * x, dtype=cp.float64)
    x0g = None if x0 is None else cp.asarray(x0, dtype=cp.float64)
    x, _ = cla.cg(Ag, bg, x0=x0g, tol=tol, maxiter=maxiter_cg, M=M)
    return cpu(x)
