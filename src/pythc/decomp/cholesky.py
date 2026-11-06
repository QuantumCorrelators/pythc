import logging
from abc import ABC, abstractmethod

import numpy as np
from pythc.decomp.acccholesky.lra import PSDLowRank
from pythc.decomp.acccholesky.matrix import FunctionMatrix
from pythc.decomp.acccholesky.rpcholesky import rpcholesky, greedy, block_greedy

from pythc import lib
from pythc._tracking import Configurable

logger = logging.getLogger()

class Cholesky(ABC, Configurable):
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

    The seed is injected directly into the (vendored) ``acccholesky``
    routines, which draw proposals and rejection decisions from one
    local ``np.random.default_rng(seed)`` per call. ``seed=None``
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
    def _max_rank(n: int, rank: int) -> int:
        max_mem_bytes = lib.pyscf_max_memory() * (1024**2)
        curr_mem_bytes = lib.current_memory() * (1024**2)
        avail_mem_bytes = max(100.0 * (1024**2), max_mem_bytes - curr_mem_bytes)

        # In rpcholesky (accelerated_rpcholesky), factor matrix G and temporary rows matrix
        # are both allocated up front with shape (k, n) as float64 (8 bytes per entry).
        # Memory required per rank step is 2 * n * 8 = 16 * n bytes.
        bytes_per_rank = 16 * n
        max_rank = max(1, int(avail_mem_bytes / bytes_per_rank)) if bytes_per_rank > 0 else rank
        return min(rank, max_rank)

    def decompose(self, A: FunctionMatrix, rank: int, err_tol: float = -1.0) -> tuple[np.ndarray, np.ndarray, int]:
        n = A.shape[0]
        k = self._max_rank(n, rank)
        b = self.block_size
        if b is None and self.seed is not None:
            # Deterministic default mirroring the library's initial auto
            # value; 'auto' itself is timing-adaptive and not reproducible.
            b = max(10, -(-k // 10))
        low_rank: PSDLowRank = rpcholesky(A, k, b, seed=self.seed, stoptol=err_tol, verbose=False)
        piv = low_rank.get_indices()
        L = low_rank.get_right_factor()

        return L, piv, low_rank.rank()

class GreedyCholesky(AccelRPCholesky):
    """Deterministic greedy pivoting (argmax residual diagonal).

    Same trace-based ``err_tol`` stop as :class:`AccelRPCholesky`, but
    pivots are chosen by argmax instead of random proposals. Comparing
    this against ``AccelRPCholesky`` with the same threshold isolates
    the randomization component: stopping rule and downstream code are
    shared (blocking intentionally differs -- see ``block_size`` note).
    """

    def __init__(self, seed: int | None = None, block_size: int | None = None):
        # ``seed`` is accepted for sweep compatibility but ignored: pure
        # argmax greedy draws no random numbers.
        # ``block_size`` defaults to single-pivot (b=1): block-greedy with
        # large b is numerically unsafe on fast-decaying grid spectra --
        # the unpivoted block Cholesky in 'regularize' strategy spans
        # noise directions, the inflated ||G||^2 then zeroes the tracked
        # residual (clip to 0) and the trace stop fires prematurely at a
        # block boundary (e.g. b=1024 stops after exactly 1024 pivots
        # with Hartree-scale errors). Explicit small b (e.g. 8) is a
        # faster compromise; large b is not recommended.
        super().__init__(seed=seed, block_size=block_size)

    def __str__(self):
        return "greedy_cholesky"

    def decompose(self, A: FunctionMatrix, rank: int, err_tol: float = -1.0) -> tuple[np.ndarray, np.ndarray, int]:
        n = A.shape[0]
        k = self._max_rank(n, rank)
        b = self.block_size if self.block_size is not None else 1
        low_rank: PSDLowRank = greedy(A, k, b=b, stoptol=err_tol)
        piv = low_rank.get_indices()
        L = low_rank.get_right_factor()

        return L, piv, low_rank.rank()


def build_cholesky_decomp(cholesky_kind: str = "accel",
                          seed: int | None = None,
                          block_size: int | None = None) -> Cholesky:
    """Resolver factory: pre-configured pivot rule for THC grid pruning.

    Register as ``"cholesky_decomp"`` so it is injected into THC classes
    by parameter name. Sweeping ``cholesky_kind`` with a fixed threshold
    varies only the pivot rule (random proposals vs. deterministic
    argmax); stopping threshold, grid, and downstream code stay
    identical. Give greedy an explicit small ``block_size`` (or 1) --
    large blocks are numerically unsafe for it, so blocking is
    intentionally not matched -- or compare ranks/energies at fixed
    ``block_size`` only for small values (<= 8).
    """
    if cholesky_kind in ("accel", "rp", "accel_rp", "rp_cholesky"):
        return AccelRPCholesky(seed=seed, block_size=block_size)
    if cholesky_kind in ("greedy", "greedy_cholesky"):
        return GreedyCholesky(seed=seed, block_size=block_size)
    raise ValueError(f"unknown cholesky_kind: {cholesky_kind!r}")
