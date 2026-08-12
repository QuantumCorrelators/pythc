import logging

import numpy as np
from numba import prange, njit
from pyscf import gto
from pyscf.gw.rpa import _get_scaled_legendre_roots
from scipy.optimize import least_squares

from pythc.methods.runner import SCF
from pythc.thc.thc_base import ThcEri

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


def thc_rpa(mol: gto.Mole, mf: SCF, eri_thc: ThcEri):
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


def build_M_tau(X_o, X_v, e_o, e_v, D, tau):
    """
    Computes the auxiliary matrix M(tau) in O(N^3) steps using pure BLAS.
    """
    # 1. Scale rows by exponential factors
    # np.newaxis ensures the arrays broadcast correctly across the grid
    X_o_tau = X_o * np.exp(e_o * tau)[:, np.newaxis]
    X_v_tau = X_v * np.exp(-e_v * tau)[:, np.newaxis]

    # 2. Form Occupied and Virtual intermediates (O(N_grid^2 * N_occ/vir))
    # Resulting shapes are (n_grid, n_grid)
    O = np.dot(X_o_tau.T, X_o)
    V = np.dot(X_v_tau.T, X_v)

    # 3. Hadamard product for Pi(tau)
    Pi_tau = 4.0 * (O * V)

    # 4. Project into auxiliary basis to form M(tau) (O(N_aux^2 * N_grid))
    return D.T @ Pi_tau @ D

class RPA():
    def __init__(self, mol, mf, eri_thc):
        self.mol = mol
        self.mf = mf
        self.eri_thc = eri_thc

    def kernel(self):
        e = self.mf.mo_energy
        n_occ = self.mol.nelectron // 2

        e_o = e[:n_occ]
        e_v = e[n_occ:]

        # For spacetime RPA, the Laplace variables exponentiate the single particle-hole
        # gap (e_a - e_i), as seen in build_M_tau. Therefore, the grid must be
        # optimized for the single gap spectrum, not the doubled MP2 gap.
        ymin = np.min(e_v) - np.max(e_o)
        ymax = np.max(e_v) - np.min(e_o)

        import laplace_minimax as lm
        grid = lm.get_laplace_grid(ymin=ymin, ymax=ymax, tolerr=1e-6)

        return thc_rpa_spacetime(self.mol, self.mf, self.eri_thc, grid.exponents, grid.weights)



def thc_rpa_spacetime(mol, mf, eri_thc, taus, tau_weights):
    X, Z = eri_thc.get_X_Z()
    D = eri_thc.get_D()

    n_occ = mol.nelectron // 2
    n_aux = D.shape[1]

    X_o = np.ascontiguousarray(X[:, :n_occ].T)
    X_v = np.ascontiguousarray(X[:, n_occ:].T)

    e = mf.mo_energy
    e_o = e[:n_occ]
    e_v = e[n_occ:]

    # --- 1. IMAGINARY TIME LOOP (The Heavy Lifting) ---
    # We do the O(N^3) operations strictly on the short tau grid (e.g. ~15 points)
    M_taus = []
    for tau in taus:
        M_taus.append(build_M_tau(X_o, X_v, e_o, e_v, D, tau))

    # --- 2. FREQUENCY LOOP (The Cheap Part) ---
    # We loop over the 40 frequency points, but only do O(N_aux^3) and O(N_aux^2) math
    E = 0.0
    homo_lumo_gap = e_v[0] - e_o[-1]
    freqs, omega_weights = _get_scaled_legendre_roots(40, homo_lumo_gap)

    for freq, w_omega in zip(freqs, omega_weights):
        # Fourier transform M(tau) -> M(i omega)
        M_omega = np.zeros((n_aux, n_aux))
        for tau, w_tau, M_t in zip(taus, tau_weights, M_taus):
            M_omega += M_t * w_tau * np.cos(freq * tau)

        # Standard RPA energy evaluation
        sign, logabsdet = np.linalg.slogdet(np.eye(n_aux) + M_omega)
        trace = np.trace(M_omega)
        E += w_omega * (logabsdet - trace)

    E_corr = E / (2.0 * np.pi)
    return E_corr


