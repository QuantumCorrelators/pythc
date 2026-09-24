import logging
from typing import Optional

import numpy as np
from numba import prange, njit
from pyscf.gw.rpa import _get_scaled_legendre_roots
from pyscf.scf.hf import SCF

from pythc.thc.ls_ri_cholesky import LS_RI_Cholesky
from pythc.thc.thc_base import ThcEri, THC

logger = logging.getLogger()

@njit(fastmath=True, parallel=True, cache=True)
def build_pi_optimized(X_o, X_v, e_o, e_v, freq):
    """
    Computes the grid-projected polarizability Pi for a single frequency.
    X_o shape: (n_occ, n_grid)
    X_v shape: (n_vir, n_grid)
    """
    n_occ, n_grid = X_o.shape
    n_vir = X_v.shape[0]

    # 1. Precompute the denominator matrix to avoid redundant flops
    D = np.empty((n_occ, n_vir), dtype=np.float64)
    for i in range(n_occ):
        for a in range(n_vir):
            d = e_v[a] - e_o[i]
            D[i, a] = 4.0 * d / (d**2 + freq**2)

    # 2. Transpose for Cache Locality (Crucial for CPU speed)
    # This aligns the orbital loops to contiguous memory in RAM.
    X_o_T = np.ascontiguousarray(X_o.T)
    X_v_T = np.ascontiguousarray(X_v.T)

    pi = np.zeros((n_grid, n_grid), dtype=np.float64)

    # 3. Parallelize over the grid to guarantee thread safety
    for p in prange(n_grid):
        for q in range(p, n_grid):

            pi_pq = 0.0

            for i in range(n_occ):
                # Form the Occupied Outer Product scalar
                O_pq = X_o_T[p, i] * X_o_T[q, i]

                # Form the Virtual Contraction scalar
                Y_pq = 0.0
                for a in range(n_vir):
                    Y_pq += X_v_T[p, a] * D[i, a] * X_v_T[q, a]

                # Hadamard product accumulation
                pi_pq += O_pq * Y_pq

            # 4. Symmetrize on the fly
            pi[p, q] = pi_pq
            if p != q:
                pi[q, p] = pi_pq

    return pi

def thc_rpa(mf: SCF, eri_thc: ThcEri):
    mol = mf.mol
    X, Z = eri_thc.get_X_Z()
    D = eri_thc.get_D()

    n_occ = mol.nelectron // 2
    n_aux = D.shape[1]

    X_o = np.ascontiguousarray(X[:, :n_occ].T)
    X_v = np.ascontiguousarray(X[:, n_occ:].T)

    e = mf.mo_energy
    e_o = e[:n_occ]
    e_v = e[n_occ:]

    E = 0.0
    # Optimal scale is the HOMO-LUMO gap
    homo_lumo_gap = e_v[0] - e_o[-1]
    freqs, weights = _get_scaled_legendre_roots(40, homo_lumo_gap)

    for freq, w in zip(freqs, weights):
        pi = build_pi_optimized(X_o, X_v, e_o, e_v, freq)
        M = D.T @ pi @ D # dims (n_aux, n_aux)
        sign, logabsdet = np.linalg.slogdet(np.eye(n_aux) + M)
        trace = np.trace(M)
        E += w*(logabsdet - trace)

    E_corr = E / (2.0 * np.pi)
    return E_corr

class RPA:
    def __init__(self, mf: SCF, thc: Optional[THC]):
        self.mol = mf.mol
        self.mf = mf
        self.thc = thc if thc else LS_RI_Cholesky(self.mol, mo_coeff=self.mf.mo_coeff, cholesky_threshold=1e-5)

    def kernel(self):
        eri = self.thc.build(mode='ov')
        return thc_rpa(self.mf, eri)

