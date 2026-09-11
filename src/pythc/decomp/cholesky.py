import logging
from abc import ABC, abstractmethod

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
            return self._diag_values

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
            return self._diag_values

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
            return self._diag_values

        return self._diag_values[vec]


class AccelRPCholesky(Cholesky):
    def __str__(self):
        return "accelerated_rpcholesky"

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

        low_rank: PSDLowRank = rpcholesky(A, min(rank, max_rank), stoptol=err_tol, verbose=False)
        piv = low_rank.get_indices()
        L = low_rank.get_right_factor()

        return L, piv, low_rank.rank()
