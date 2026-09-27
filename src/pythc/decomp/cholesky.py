import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from unittest import mock

import numpy as np
from pythc.decomp.acccholesky.lra import PSDLowRank
from pythc.decomp.acccholesky.matrix import FunctionMatrix
from pythc.decomp.acccholesky.rpcholesky import rpcholesky

from pythc import lib

logger = logging.getLogger()

class Cholesky(ABC):
    @abstractmethod
    def decompose(self, A: FunctionMatrix, rank: int, err_tol: float = -1.0) -> tuple[np.ndarray, np.ndarray, int]:
        pass

class SymMetricOV(FunctionMatrix):
    def __init__(self, A, B):
        assert A.shape[0] == B.shape[0]

        self.A = A
        self.B = B
        self.n = A.shape[0]
        self._diag_values = None

        super().__init__(self.n)

    def _function(self, i, j):
        p, q  = i[0], j[0]
        return np.sum(self.A[p] * self.A[q], axis=0) * np.sum(self.B[p] * self.B[q], axis=0)

    def _function_vec(self, vec_i, vec_j):
        return (self.A[vec_i] @ self.A[vec_j].T) * (self.B[vec_i] @ self.B[vec_j].T)

    def _get_row(self, i):
        return (self.A[i] @ self.A.T) * (self.B[i] @ self.B.T)

    def _function_mtx(self, I, J):
        sub = (self.A[I] @ self.A[J].T) * (self.B[I] @ self.B[J].T)
        return sub

    def _diag_helper(self, vec=None):
        """Use precomputed diagonal values"""
        if self._diag_values is None:
            diag_A = np.einsum('ij,ij->i', self.A, self.A)
            diag_B = np.einsum('ij,ij->i', self.B, self.B)
            self._diag_values = diag_A * diag_B

        if vec is None:
            # Copy: callers (acccholesky) update the returned array in place.
            return self._diag_values.copy()

        return self._diag_values[vec]

class GramMetric(FunctionMatrix):
    def __init__(self, X):
        self.X = X
        self.n, self.p = X.shape
        self.S = {}
        self._access_order = []  # For LRU cache management
        self._diag_values = None

        super().__init__(self.n)

    def _function(self, i, j):
        p, q = i[0], j[0]
        return self.X[p]@self.X[q]

    def _function_vec(self, vec_i, vec_j):
        return self.X[vec_i] @ self.X[vec_j].T

    def _get_row(self, i):
        return self.X[i] @ self.X.T

    def _function_mtx(self, vec_i, vec_j):
        X_i = self.X[vec_i]
        X_j = self.X[vec_j]

        return X_i @ X_j.T

    def _diag_helper(self, vec=None):
        if self._diag_values is None:
            self._diag_values = np.einsum('ij,ij->i', self.X, self.X)

        if vec is None:
            # Copy: callers (acccholesky) update the returned array in place.
            return self._diag_values.copy()

        return self._diag_values[vec]

class SymMetricPBC(FunctionMatrix):
    """
    Gram metric for k-point periodic boundary conditions.

    **AO mode** (X_v=None): compute the full-basis gram metric

        S_{r,r'} = Σ_k | X[k,r,:]† · X[k,r,:'] |²                 (real, ≥ 0)

    **OV mode** (X_v provided): compute the occupied-virtual gram metric

        S_{r,r'} = Σ_k  (X_o[k,r,:]† · X_o[k,r,':])
                       · conj(X_v[k,r,:]† · X_v[k,r,':])           (complex Hermitian)

    Both formulae are derived from

        S_{r,r'} = Σ_k Σ_{μν} [φ_μ^k(r) φ_ν^{-k}(r)]* [φ_μ^k(r') φ_ν^{-k}(r')]

    using time-reversal symmetry φ^{-k}(r) = conj(φ^k(r)).

    Parameters
    ----------
    X : ndarray, shape (k, n, p_o) — complex Bloch-function values on the grid.
        In OV mode this is X_o (occupied block).
    X_v : ndarray, shape (k, n, p_v) or None
        Virtual block. If provided, OV mode is activated.
    kpts : ndarray, shape (k, 3), optional
        k-point coordinates (stored for downstream use, not needed internally).
    """

    def __init__(self, X, X_v=None, kpts=None):
        assert X.ndim == 3, "X must have shape (k, n, p)"
        if X_v is not None:
            assert X_v.ndim == 3 and X_v.shape[:2] == X.shape[:2], \
                "X_v must have shape (k, n, p_v) matching X on axes 0-1"

        self.X = X           # (k, n, p_o) — occupied (or full AO)
        self.X_v = X_v       # (k, n, p_v) — virtual, or None
        self.kpts = kpts
        self.k, self.n, self.p = X.shape
        self._diag_values = None

        super().__init__(self.n)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _inner_products(self, Xi, Xj, Xi_v=None, Xj_v=None):
        """
        Compute per-k inner products between two sets of grid slices.

        Parameters
        ----------
        Xi, Xj : (k, m, p) — row slices of X for indices I and J.
        Xi_v, Xj_v : (k, m, p_v) or None — corresponding virtual slices.

        Returns
        -------
        AO mode  : (k, |I|, |J|) real array of |d^k_{ij}|²
        OV mode  : (k, |I|, |J|) complex array of d_o^k · conj(d_v^k)
        """
        # NOTE: this must stay on the BLAS matmul path (``@``), NOT
        # ``np.einsum('kip,kjp->kij', ...)``. NumPy's default einsum
        # implementation is a single-threaded C loop that ignores
        # OPENBLAS_NUM_THREADS/OMP_NUM_THREADS, which serialized the
        # whole RPCholesky grid-pruning step on one core. Batched
        # matmul dispatches per-k zgemm/dgemm calls and threads.
        # d_o[k,i,j] = Xi[k,i,:]^H · Xj[k,j,:]
        d_o = Xi.conj() @ Xj.transpose(0, 2, 1)  # (k, |I|, |J|) complex

        if Xi_v is None:                  # AO mode  →  |d_o|²  (real)
            return d_o.real ** 2 + d_o.imag ** 2

        # OV mode  →  d_o · conj(d_v)   (complex)
        d_v = Xi_v.conj() @ Xj_v.transpose(0, 2, 1)
        return d_o * d_v.conj()

    # ------------------------------------------------------------------
    # FunctionMatrix interface
    # ------------------------------------------------------------------

    def _function(self, i, j):
        """Return single element S[i[0], j[0]] — scalar (float or complex)."""
        p, q = i[0], j[0]
        Xi = self.X[:, [p], :]   # (k, 1, p_o)
        Xj = self.X[:, [q], :]
        Xi_v = self.X_v[:, [p], :] if self.X_v is not None else None
        Xj_v = self.X_v[:, [q], :] if self.X_v is not None else None

        prods = self._inner_products(Xi, Xj, Xi_v, Xj_v)  # (k, 1, 1)
        val = prods.sum(axis=0)[0, 0]
        return float(val) if np.isrealobj(val) else complex(val)

    def _function_vec(self, vec_i, vec_j):
        """Return element-wise S[vec_i[r], vec_j[r]] — 1-D array."""
        # Paired rows only: computing the full (len×len) block and then
        # keeping its diagonal would waste O(k·len²) work and memory.
        Xi   = self.X[:, vec_i, :]    # (k, len, p_o)
        Xj   = self.X[:, vec_j, :]
        Xi_v = self.X_v[:, vec_i, :] if self.X_v is not None else None
        Xj_v = self.X_v[:, vec_j, :] if self.X_v is not None else None

        d_o = (Xi.conj() * Xj).sum(axis=-1)  # (k, len) complex
        if Xi_v is None:                     # AO mode → Σ_k |d_o|² (real)
            return (d_o.real ** 2 + d_o.imag ** 2).sum(axis=0)

        # OV mode → Σ_k d_o · conj(d_v) (complex)
        d_v = (Xi_v.conj() * Xj_v).sum(axis=-1)
        return (d_o * d_v.conj()).sum(axis=0)

    def _function_mtx(self, vec_i, vec_j, jblock: int = 2048):
        """Return submatrix S[vec_i, :][:, vec_j] — shape (|I|, |J|)."""
        Xi   = self.X[:, vec_i, :]    # (k, |I|, p_o)
        Xi_v = self.X_v[:, vec_i, :] if self.X_v is not None else None

        # Block over J and accumulate per k-point instead of materializing
        # the (k, |I|, |J|) intermediate: for row blocks (|I| ~ b,
        # |J| = n_grid) that temporary is k× larger than the result
        # (tens of GB here). Each per-k ``@`` is a threaded BLAS gemm
        # (see _inner_products). Blocking also keeps every single gemm
        # output small: one giant multi-GB zgemm segfaulted OpenBLAS
        # 0.3.34 (pthreads) on 96-core Zen4, while smaller gemms run fine.
        n_j = len(vec_j)
        out = np.zeros((len(vec_i), n_j),
                       dtype=np.float64 if Xi_v is None else np.complex128)
        for j0 in range(0, n_j, jblock):
            js = slice(j0, min(j0 + jblock, n_j))
            Xjb = self.X[:, vec_j[js], :]
            Xjb_v = self.X_v[:, vec_j[js], :] if self.X_v is not None else None
            for kk in range(self.k):
                d_o = Xi[kk].conj() @ Xjb[kk].T
                if Xi_v is None:              # AO mode → Σ_k |d_o|² (real)
                    out[:, js] += d_o.real ** 2 + d_o.imag ** 2
                else:                         # OV mode → Σ_k d_o·conj(d_v)
                    d_v = Xi_v[kk].conj() @ Xjb_v[kk].T
                    out[:, js] += d_o * d_v.conj()
        return out

    def _diag_helper(self, vec=None):
        """Return diagonal elements S[r,r] — cached, always real and ≥ 0."""
        if self._diag_values is None:
            if self.X_v is None:
                # AO: S[r,r] = Σ_k ||X[k,r,:]||⁴
                norms_sq = (self.X.real ** 2 + self.X.imag ** 2).sum(axis=-1)  # (k, n)
                self._diag_values = (norms_sq ** 2).sum(axis=0)                 # (n,)
            else:
                # OV: S[r,r] = Σ_k ||X_o[k,r,:]||² · ||X_v[k,r,:]||²
                norms_o = (self.X.real ** 2 + self.X.imag ** 2).sum(axis=-1)    # (k, n)
                norms_v = (self.X_v.real ** 2 + self.X_v.imag ** 2).sum(axis=-1)
                self._diag_values = (norms_o * norms_v).sum(axis=0)              # (n,)

        return self._diag_values if vec is None else self._diag_values[vec]


class SymMetric(FunctionMatrix):
    def __init__(self, X):
        self.X = X
        self.n, self.p = X.shape
        self.S = {}
        self._access_order = []  # For LRU cache management
        self._diag_values = None

        super().__init__(self.n)

    def _function(self, i, j):
        p, q  = i[0], j[0]
        return np.sum(self.X[p] * self.X[q], axis=0) ** 2

    def _function_vec(self, vec_i, vec_j):
        return (self.X[vec_i] @ self.X[vec_j].T) ** 2

    def _get_row(self, i):
        return (self.X[i] @ self.X.T) ** 2

    def _function_mtx(self, vec_i, vec_j):
        X_i = self.X[vec_i]
        X_j = self.X[vec_j]

        products = X_i @ X_j.T

        return products ** 2

    def _diag_helper(self, vec=None):
        if self._diag_values is None:
            diag = np.einsum('ij,ij->i', self.X, self.X)
            self._diag_values = diag ** 2

        if vec is None:
            # Copy: callers (acccholesky) update the returned array in place.
            return self._diag_values.copy()

        return self._diag_values[vec]


class AccelRPCholesky(Cholesky):
    """Accelerated randomly-pivoted Cholesky with optional seeding.

    Upstream ``acccholesky`` draws from two unseeded sources: a hardcoded
    ``np.random.default_rng()`` (proposal batches, ignores
    ``np.random.seed``) and the legacy ``np.random.rand()`` (rejection
    step). Both are fixed for the duration of the call when ``seed`` is
    set; the legacy global state is restored afterwards. ``seed=None``
    (default) keeps the unseeded behavior.

    Note: a seed alone does not give reproducibility, because the
    library's ``b='auto'`` block size adapts to wall-clock timing. Hence a
    fixed block size is used whenever ``seed`` is set (overridable via
    ``block_size``); ``block_size=None`` with ``seed=None`` keeps the
    auto-tuned behavior.
    """

    def __init__(self, seed: int | None = None, block_size: int | None = None):
        self.seed = seed
        self.block_size = block_size

    def __str__(self):
        return "accelerated_rpcholesky"

    @staticmethod
    @contextmanager
    def _seeded_rng(seed: int | None):
        if seed is None:
            yield
            return
        legacy_state = np.random.get_state()
        try:
            np.random.seed(seed)
            with mock.patch.object(np.random, 'default_rng',
                                   return_value=np.random.default_rng(seed)):
                yield
        finally:
            np.random.set_state(legacy_state)

    def decompose(self, A: FunctionMatrix, rank: int, err_tol: float = -1.0) -> tuple[np.ndarray, np.ndarray, int]:
        max_mem_bytes = lib.pyscf_max_memory() * (1024**2)
        curr_mem_bytes = lib.current_memory() * (1024**2)
        avail_mem_bytes = max(100.0 * (1024**2), max_mem_bytes - curr_mem_bytes)

        n = A.shape[0]
        # In rpcholesky (accelerated_rpcholesky), factor matrix G and temporary rows matrix
        # are both allocated up front with shape (k, n) as float64 (8 bytes per entry).
        # Memory required per rank step is 2 * n * 8 = 16 * n bytes.
        bytes_per_rank = 16 * n
        max_rank = max(1, int(avail_mem_bytes / bytes_per_rank)) if bytes_per_rank > 0 else rank

        with self._seeded_rng(self.seed):
            k = min(rank, max_rank)
            b = self.block_size
            if b is None and self.seed is not None:
                # Deterministic default mirroring the library's initial auto
                # value; 'auto' itself is timing-adaptive and not reproducible.
                b = max(10, -(-k // 10))
            low_rank: PSDLowRank = rpcholesky(A, k, b, stoptol=err_tol, verbose=False)
        piv = low_rank.get_indices()
        L = low_rank.get_right_factor()

        return L, piv, low_rank.rank()
