from abc import abstractmethod, ABC
from pathlib import Path
from typing import Literal

import h5py
import numpy as np

import pythc.lib as lib
from pythc.configurable import Configurable
from pythc.tracking.experiment_run import ExperimentRun

type Mode = Literal['ao', 'ov', 'oo', 'vv', 'ia']


class ERI(ABC):
    """
    Interface for a generic Electron Repulsion Integral
    """

    @abstractmethod
    def get_full(self):
        """
        Get full 4-dimensional <ab|cd> representation of the integral
        :return: <ab|cd>
        """
        pass

    @abstractmethod
    def get_jk(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get J = <ab|cd> and K = <ad|cb> representations of the integral
        :return: <ab|cd>, <ad|cb>
        """
        pass

    @abstractmethod
    def to_backend(self):
        pass


class ThcEri(ERI):
    def __init__(self, nelectron: int, X: np.ndarray, Z: np.ndarray, D: np.ndarray = None):
        self.nelectron = nelectron
        self.X = X  # (N^2, M)
        self.Z = Z  # (M, M)
        self.D = D # (M, n_aux)

        active = ExperimentRun.get_active()
        if active:
            active.log_metric("n_thc", X.shape[0])


    @classmethod
    def from_file(cls, path: str | Path) -> "ThcEri":
        """
        Construct ThcEri directly from an HDF5 file,
        automatically retrieving nelectron from metadata.
        """
        with h5py.File(path, 'r') as f:
            X = f['X'][:]
            Z = f['Z'][:]
            nelectron = f.attrs.get('nelectron', 0)

        return cls(nelectron=int(nelectron), X=X, Z=Z)


    def save(self, path: str):
        with h5py.File(path, 'w') as f:
            f.create_dataset('X', data=self.X)
            f.create_dataset('Z', data=self.Z)
            f.attrs['nelectron'] = self.nelectron


    def get_X_Z(self):
        return self.X, self.Z

    def get_D(self):
        return self.D

    def get_full(self):
        N = self.X.shape[1]
        Xs = lib.einsum("pn,pm->mnp", self.X, self.X).reshape(N ** 2, -1)
        return Xs @ self.Z @ Xs.T

    def get_jk(self):
        nocc = self.nelectron // 2
        o = slice(None, nocc)
        v = slice(nocc, None)

        X_o = self.X[:, o]
        X_v = self.X[:, v]

        X_ai = lib.einsum("pi,pa->iap", X_o, X_v)

        J = lib.einsum("iap,pq,jbq->ijab", X_ai, self.Z, X_ai)

        return J, J.swapaxes(2, 3)

    def to_backend(self):
        self.X = lib.to_backend(self.X)
        self.Z = lib.to_backend(self.Z)


class ThcEriUnrestricted(ERI):
    def get_X_Z(self):
        return self.X_alpha, self.X_beta, self.Z_aa, self.Z_bb, self.Z_ab

    def get_jk(self) -> tuple[np.ndarray, np.ndarray]:
        pass

    def get_full(self):
        pass

    def __init__(self, nelectron, X_alpha, X_beta, Z_aa, Z_bb, Z_ab):
        self.nelectron = nelectron
        self.X_alpha = X_alpha
        self.X_beta = X_beta
        self.Z_aa = Z_aa
        self.Z_bb = Z_bb
        self.Z_ab = Z_ab

        active = ExperimentRun.get_active()
        if active:
            X_a_len = X_alpha.shape[0]
            X_b_len = X_beta.shape[0]
            active.log_metric("n_thc", max(X_a_len, X_b_len))

    def to_backend(self):
        self.X_alpha = lib.to_backend(self.X_alpha)
        self.X_beta = lib.to_backend(self.X_beta)
        self.Z_aa = lib.to_backend(self.Z_aa)
        self.Z_bb = lib.to_backend(self.Z_bb)
        self.Z_ab = lib.to_backend(self.Z_ab)


class THC(ABC, Configurable):
    def with_mo_coeff(self, mo_coeff):
        self.mo_coeff = mo_coeff

    @abstractmethod
    def build(self) -> ThcEri:
        pass

    @abstractmethod
    def build_unrestricted(self) -> ThcEriUnrestricted:
        pass
