from abc import abstractmethod, ABC
from pathlib import Path

import h5py
import numpy as np

import pythc.lib as lib
from pythc import observe
from pythc.configurable import Configurable
from pythc.thc.thc_base import Mode, ERI


class THC_ERI_kpts(ERI):
    def __init__(self, nelectron: int, X_kpts: np.ndarray, Z_kpts: np.ndarray, D_kpts: np.ndarray = None):
        self.nelectron = nelectron
        self.X_kpts = X_kpts  # (k, N^2, M)
        self.Z_kpts = Z_kpts  # (k, M, M)
        self.D_kpts = D_kpts # (k, M, n_aux)

        observe.log_metric("n_thc", int(X_kpts.shape[0]))


    @classmethod
    def from_file(cls, path: str | Path) -> "THC_ERI_kpts":
        """
        Construct ThcEri directly from an HDF5 file,
        automatically retrieving nelectron from metadata.
        """
        with h5py.File(path, 'r') as f:
            X = f['X'][:]
            Z = f['Z'][:]
            nelectron = f.attrs.get('nelectron', 0)

        return cls(nelectron=int(nelectron), X_kpts=X, Z_kpts=Z)


    def save(self, path: str):
        with h5py.File(path, 'w') as f:
            f.create_dataset('X', data=self.X_kpts)
            f.create_dataset('Z', data=self.Z_kpts)
            f.attrs['nelectron'] = self.nelectron


    def get_X_Z(self):
        return self.X_kpts, self.Z_kpts

    def get_D(self):
        return self.D_kpts

    def get_full(self):
        N = self.X_kpts.shape[1]
        Xs = lib.einsum("pn,pm->mnp", self.X_kpts, self.X_kpts).reshape(N ** 2, -1)
        return Xs @ self.Z_kpts @ Xs.T

    def get_jk(self):
        nocc = self.nelectron // 2
        o = slice(None, nocc)
        v = slice(nocc, None)

        X_o = self.X_kpts[:, o]
        X_v = self.X_kpts[:, v]

        X_ai = lib.einsum("pi,pa->iap", X_o, X_v)

        J = lib.einsum("iap,pq,jbq->ijab", X_ai, self.Z_kpts, X_ai)

        return J, J.swapaxes(2, 3)

    def to_backend(self):
        self.X_kpts = lib.to_backend(self.X_kpts)
        self.Z_kpts = lib.to_backend(self.Z_kpts)


class THC(ABC, Configurable):
    def with_mo_coeff(self, mo_coeff):
        self.mo_coeff = mo_coeff

    @abstractmethod
    def build_kpts(self, mode: Mode = "ao") -> THC_ERI_kpts:
        pass
