import logging

import numpy as np
from pyscf.pbc import gto
from pyscf.pbc import tools as pbctools

from pythc.decomp.cholesky import SymMetric, SymMetricOV, AccelRPCholesky, SymMetricPBC
from pythc.grid import UniformGrid
from pythc.pbc.lib import get_supercell_phase, contract_fft_k, lstsq_svd
from pythc.thc.thc_base import THC, Mode, ThcEri, ThcEriUnrestricted

logger = logging.getLogger()



class PBC_LS_RI_Cholesky(THC):
    """
    Periodic Boundary Condition implementation of LS-RI-Cholesky.
    Adapts the FFTISDF pruning strategy for Gamma-point and k-points.
    """

    def build_unrestricted(self) -> ThcEriUnrestricted:
        pass

    def __init__(self, cell: gto.Cell, mo_coeff=None, e_cut: float = 70, cholesky_threshold: float = 1e-10):
        super().__init__()
        self.cell = cell
        self.mo_coeff = mo_coeff if mo_coeff is not None else np.eye(cell.nao_nr())
        self.e_cut = e_cut
        self.cholesky_threshold = cholesky_threshold
        self.grid_builder = UniformGrid(cell, e_cut)

    def build(self, mode: Mode = "ao") -> ThcEri:
        grid = self.grid_builder.build()
        X = self.cell.pbc_eval_gto('GTOval_sph', coords=grid)

        if mode == 'ov':
            X = X @ self.mo_coeff

        n_occ = self.cell.nelectron // 2

        if mode == 'ao':
            A = SymMetric(X)
        elif mode == 'ov':
            A = SymMetricOV(X[:, :n_occ], X[:, n_occ:])
        else:
            raise NotImplementedError

        n_grid = X.shape[0]
        U, piv, num_rank = AccelRPCholesky().decompose(A, n_grid, self.cholesky_threshold)
        piv = piv[:num_rank]
        U = U[:, piv]

        X_pruned = X[piv, :]

        S_Lg = A._function_mtx(piv, np.arange(n_grid))

        mesh = self.grid_builder.mesh  # [Nx, Ny, Nz]
        omega = self.cell.vol  # Unit cell volume

        Gv = self.cell.get_Gv(mesh)
        G2 = np.einsum('gx,gx->g', Gv, Gv)

        v_G = np.zeros(n_grid, dtype=np.float64)
        non_zero = G2 > 1e-12
        v_G[non_zero] = 4.0 * np.pi / G2[non_zero]

        S_Lg_3d = S_Lg.reshape(num_rank, mesh[0], mesh[1], mesh[2])

        dV = omega / n_grid
        S_Lg_G = np.fft.fftn(S_Lg_3d, axes=(1, 2, 3)).reshape(num_rank, n_grid) * dV

        V = (1.0 / omega) * (S_Lg_G.conj() * v_G) @ S_Lg_G.T
        V = np.real(V)

        import scipy.linalg as sla
        A = sla.solve_triangular(U.T, V, lower=True)
        B = sla.solve_triangular(U, A, lower=False)

        C_T = sla.solve_triangular(U.T, B.T, lower=True)
        Z_T = sla.solve_triangular(U, C_T, lower=False)
        Z = Z_T.T

        return ThcEri(self.cell.nelectron, X_pruned, Z, None)

    def build_kpts(self, mode: Mode = "ao", kpts=None) -> ThcEri:
        mesh = self.cell.mesh
        grid = self.cell.gen_uniform_grids(mesh)
        n_kpt = len(kpts)
        logger.info(f"building k-points THC: n_grid={len(grid)}; n_kpts={n_kpt}; n_ao={self.cell.nao_nr()}; n_electron={self.cell.nelectron}")

        # 1. Evaluate Bloch AOs on grid: shape (nkpt, ngrid, nao)
        X_kpt = np.ascontiguousarray(
            np.asarray(self.cell.pbc_eval_gto('GTOval', coords=grid, kpts=kpts), dtype=np.complex128)
        )

        n_occ = self.cell.nelectron // 2

        if mode == 'ov':
            for k in range(len(kpts)):
                X_kpt[k] = X_kpt[k] @ self.mo_coeff[k]
            # Split into occupied / virtual blocks across all k-points
            X_o = X_kpt[:, :, :n_occ]  # (n_kpts, n_grid, n_occ)
            X_v = X_kpt[:, :, n_occ:]  # (n_kpts, n_grid, n_virt)
            A = SymMetricPBC(X_o, X_v=X_v, kpts=kpts)
        elif mode == 'ao':
            A = SymMetricPBC(X_kpt, kpts=kpts)
        else:
            raise NotImplementedError(f"mode={mode!r} not supported for build_kpts")

        n_grid = X_kpt.shape[1]
        _, piv, num_rank = AccelRPCholesky().decompose(A, n_grid, self.cholesky_threshold)
        piv = piv[:num_rank]
        X_kpt_pruned = np.ascontiguousarray(X_kpt[:, piv, :])
        n_grid_pruned = len(piv)

        # 3. Compute eta_kpt (pair density at grid) and Pi (metric at pivots)
        phase = get_supercell_phase(self.cell, kpts)
        eta_kpt = contract_fft_k(X_kpt_pruned, X_kpt, phase)         # (nkpt, nip, ngrid)
        Pi = contract_fft_k(X_kpt_pruned, X_kpt_pruned, phase)       # (nkpt, nip, nip)

        # 4. Compute Coulomb kernel W_q (coul_kpt)
        v0 = self.cell.get_Gv(mesh)
        Z_kpt = np.zeros((n_kpt, n_grid_pruned, n_grid_pruned), dtype=np.complex128)

        for q in range(n_kpt):
            logger.info("\tk-point %d/%d: building Coulomb kernel", q + 1, n_kpt)
            fqs = np.exp(-1j * grid @ kpts[q])
            dV = (self.cell.vol / n_grid)
            vq = pbctools.get_coulG(self.cell, k=kpts[q], exx=False, Gv=v0, mesh=mesh) * dV
            eta_q = eta_kpt[q] * fqs
            eta_q_fft = pbctools.fft(eta_q, mesh)
            v_eta_q = pbctools.ifft(eta_q_fft * vq, mesh).conj()
            W = (eta_q @ v_eta_q.T) / np.sqrt(n_grid) # We keep the normalization out of the lstsq solve

            Z_q, _ = lstsq_svd(Pi[q], W, tol=1e-8) # solve Z = Pi^(-1) W Pi^(-1)
            Z_q = (Z_q + Z_q.conj().T) / 2.0 # Ensure Z is Hermitian
            Z_kpt[q] = Z_q * np.sqrt(n_grid)
            logger.info("\tk-point %d/%d: Z=%s", q + 1, n_kpt, Z_kpt[q].shape)

        return ThcEri(self.cell.nelectron, X_kpt_pruned, Z_kpt, None)

    def build_unrestricted(self, mode: Mode = "ov") -> ThcEriUnrestricted:
        """
        Builds a spin-unrestricted THC representation.
        Currently only supported in 'ov' mode.
        """
        if mode != 'ov':
            raise NotImplementedError("Unrestricted fitting is only supported in 'ov' mode")

        # ---------------------------------------------------------
        # 1. SETUP
        # ---------------------------------------------------------
        # TODO: Get grid, weights, and evaluate basis X

        # TODO: Transform X for alpha and beta spins separately using mo_coeff[0] and mo_coeff[1]

        # ---------------------------------------------------------
        # 2. PRUNING
        # ---------------------------------------------------------
        # TODO: Perform pruning separately for alpha and beta to get pivots_alpha, pivots_beta

        # ---------------------------------------------------------
        # 3. METRIC INVERSION
        # ---------------------------------------------------------
        # TODO: Form S_pruned and S_cross for alpha and beta

        # TODO: Solve for \Xi_alpha and \Xi_beta

        # ---------------------------------------------------------
        # 4. COULOMB KERNEL
        # ---------------------------------------------------------
        # TODO: Evaluate 3-center integrals

        # TODO: Form Z_aa, Z_bb, and the asymmetric Z_ab

        pass
