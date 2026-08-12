import logging

import numpy as np
import scipy as sp
from pyscf import gto

from pythc.decomp.cholesky import SymMetricOV, SymMetric, AccelRPCholesky
from pythc.grid import BeckeGrid
from pythc.thc.checkpoints import GRID_BUILD, BASIS_FUNCTION_EVAL, GRID_PRUNING, FITTING_MATRIX, METRIC_INVERSION
from pythc.thc.ls_thc_funcs import build_auxmol, build_aux_coulomb_inv, eval_basefuncs
from pythc.thc.thc_base import ThcEri, Mode, ThcEriUnrestricted, THC
from pythc.tracking.experiment_run import ExperimentRun

logger = logging.getLogger(__name__)

class LS_snRI_Cholesky(THC):
    """
    Implements Block Tensor Decomposition (BTD) THC kernel evaluation 
    combined with Cholesky decomposition for grid pruning based on
    Zhang et al. (2026 - DOI: 10.1063/5.0289370).
    
    This avoids the O(N^4) 3c2e integral contractions by using 
    Resolution of the Identity (RI) on the real space Coulomb interaction.
    """

    def __init__(self,
                 mol: gto.Mole,
                 grid=None,
                 mo_coeff: np.ndarray = None,
                 auxbasis: str = None,
                 cholesky_decomp=AccelRPCholesky,
                 cholesky_threshold: float = 1e-5,
                 epsilon: float = 1e-11):

        super().__init__()

        self.mol = mol
        self.N = mol.nao_nr()
        self.auxbasis = auxbasis if auxbasis is not None else f'{mol.basis}-ri'
        self.cholesky_decomp = cholesky_decomp()
        self.cholesky_threshold = cholesky_threshold
        self.grid = grid if grid is not None else BeckeGrid(mol, level=0)
        self.mo_coeff = mo_coeff if mo_coeff is not None and len(mo_coeff) > 0 else np.eye(mol.nao_nr())
        self.epsilon = epsilon


    @classmethod
    def __str__(cls):
        return "btd_thc"

    def build(self, mode: Mode = "ao") -> ThcEri:
        active = ExperimentRun.get_active()

        grid_dense, weights_dense = BeckeGrid(self.mol, level=1).build()
        grid_prune, weights_prune = self.grid.build()
        logger.info(f"built dense grid of size: {len(grid_dense)}, pruned grid of size: {len(grid_prune)}")
        if active: active.checkpoint(GRID_BUILD)

        R_dense = eval_basefuncs(self.mol, grid_dense)
        X_dense = (np.sqrt(np.sqrt(weights_dense))[:, np.newaxis] * R_dense)
        
        R_prune = eval_basefuncs(self.mol, grid_prune)
        logger.info(f"got {R_dense.shape} basis function on dense grid matrix")
        X_prune = (np.sqrt(np.sqrt(weights_prune))[:, np.newaxis] * R_prune)
        
        if mode != 'ao':
            X_dense = X_dense @ self.mo_coeff
            X_prune = X_prune @ self.mo_coeff
        if active: active.checkpoint(BASIS_FUNCTION_EVAL)
        logger.info(f"using cholesky threshold: {self.cholesky_threshold}")

        n_occ = self.mol.nelectron // 2 if self.mol.nelectron > 1 else 1

        n_grid_prune = X_prune.shape[0]

        L, X_pruned, piv, _ = self.prune_grid(X_prune, n_grid_prune, n_occ, is_ao=(mode == 'ao'))
        
        if mode == 'ao':
            S_Lg = (X_pruned @ X_dense.T) ** 2
        else:
            X_o_pruned = X_pruned[:, :n_occ]
            X_v_pruned = X_pruned[:, n_occ:]
            X_o_dense = X_dense[:, :n_occ]
            X_v_dense = X_dense[:, n_occ:]
            S_Lg = (X_o_pruned @ X_o_dense.T) * (X_v_pruned @ X_v_dense.T)

        if active: active.checkpoint(GRID_PRUNING)
        logger.info("building BTD fitting matrices")

        auxmol = build_auxmol(self.mol, self.auxbasis)
        j2c_inv = build_aux_coulomb_inv(auxmol)

        # 1. Compute B_gM on FULL grid
        logger.info("building fakemol and 2c1e grid-auxiliary integrals")
        fakemol = gto.fakemol_for_charges(grid_dense)
        int2c2e_gM = gto.intor_cross('int2c2e', fakemol, auxmol)
        B_gM = int2c2e_gM @ j2c_inv  # Shape: (n_grid, n_aux)

        # Multiply by w_g^(1/2) to correct the integration weight since X uses w_g^(1/4)
        B_gM = B_gM * np.sqrt(weights_dense)[:, np.newaxis]

        # 3. Compute A_LM
        A_LM = S_Lg @ B_gM  # Shape: (num_rank, n_aux)
        if active: active.checkpoint(FITTING_MATRIX)

        D = self.build_coulomb_matrix_sep(L, A_LM.T)
        if active: active.checkpoint(METRIC_INVERSION)

        logger.info("building coulomb kernel matrix")
        Z = D @ D.T

        return ThcEri(self.mol.nelectron, X_pruned, Z, D)

    def build_unrestricted(self, mode: Mode = "ao") -> ThcEriUnrestricted:
        if mode != 'ov':
            raise NotImplementedError("fitting an unrestricted THC in AO mode is currently not supported")

        S = self.mol.spin
        nocc_alpha = (self.mol.nelectron + S) // 2
        nocc_beta = (self.mol.nelectron - S) // 2

        active = ExperimentRun.get_active()
        mo_coeff_alpha = self.mo_coeff[0]
        mo_coeff_beta = self.mo_coeff[1]

        grid_dense, weights_dense = BeckeGrid(self.mol, level=1).build()
        grid_prune, weights_prune = self.grid.build()
        if active: active.checkpoint(GRID_BUILD)

        logger.info("evaluating basis functions on grid")
        R_dense = eval_basefuncs(self.mol, coords=grid_dense)
        R_prune = eval_basefuncs(self.mol, coords=grid_prune)

        logger.info("building metric matrix")
        X_dense = np.sqrt(np.sqrt(weights_dense))[:, np.newaxis] * R_dense
        X_prune = np.sqrt(np.sqrt(weights_prune))[:, np.newaxis] * R_prune

        X_alpha_dense = X_dense @ mo_coeff_alpha
        X_beta_dense = X_dense @ mo_coeff_beta

        X_alpha_prune = X_prune @ mo_coeff_alpha
        X_beta_prune = X_prune @ mo_coeff_beta
        if active: active.checkpoint(BASIS_FUNCTION_EVAL)

        auxmol = build_auxmol(self.mol, self.auxbasis)
        j2c_inv = build_aux_coulomb_inv(auxmol)

        logger.info("building fakemol and 2c1e grid-auxiliary integrals")
        fakemol = gto.fakemol_for_charges(grid_dense)
        int2c2e_gM = gto.intor_cross('int2c2e', fakemol, auxmol)
        B_gM = int2c2e_gM @ j2c_inv  # Shape: (n_grid, n_aux)
        B_gM = B_gM * np.sqrt(weights_dense)[:, np.newaxis]

        n_grid_prune = X_prune.shape[0]
        L_aa, X_alpha_pruned, piv_alpha, _ = self.prune_grid(X_alpha_prune, n_grid_prune, nocc_alpha)
        L_bb, X_beta_pruned, piv_beta, _ = self.prune_grid(X_beta_prune, n_grid_prune, nocc_beta)

        X_o_alpha_pruned = X_alpha_pruned[:, :nocc_alpha]
        X_v_alpha_pruned = X_alpha_pruned[:, nocc_alpha:]
        X_o_alpha_dense = X_alpha_dense[:, :nocc_alpha]
        X_v_alpha_dense = X_alpha_dense[:, nocc_alpha:]
        S_Lg_alpha = (X_o_alpha_pruned @ X_o_alpha_dense.T) * (X_v_alpha_pruned @ X_v_alpha_dense.T)

        X_o_beta_pruned = X_beta_pruned[:, :nocc_beta]
        X_v_beta_pruned = X_beta_pruned[:, nocc_beta:]
        X_o_beta_dense = X_beta_dense[:, :nocc_beta]
        X_v_beta_dense = X_beta_dense[:, nocc_beta:]
        S_Lg_beta = (X_o_beta_pruned @ X_o_beta_dense.T) * (X_v_beta_pruned @ X_v_beta_dense.T)

        Y_alpha = S_Lg_alpha @ B_gM
        Y_beta = S_Lg_beta @ B_gM

        n_grid_aa = piv_alpha.shape[0]
        n_grid_bb = piv_beta.shape[0]
        Z_aa = np.zeros((n_grid_aa, n_grid_aa))
        Z_bb = np.zeros((n_grid_bb, n_grid_bb))
        if L_aa.shape[0] > 0:
            Z_aa = self.build_coulomb_matrix_sym(L_aa, Y_alpha.T)

        if L_bb.shape[0] > 0:
            Z_bb = self.build_coulomb_matrix_sym(L_bb, Y_beta.T)

        Z_ab = self.build_coulomb_matrix_asym(L_aa, Y_alpha.T, L_bb, Y_beta.T)
        if active: active.checkpoint(METRIC_INVERSION)

        return ThcEriUnrestricted(self.mol.nelectron, X_alpha_pruned, X_beta_pruned, Z_aa, Z_bb, Z_ab)

    def prune_grid(self, X: np.ndarray, n_grid: int, n_occ: int, is_ao: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if n_occ > 0:
            if is_ao:
                A = SymMetric(X)
            else:
                X_o = X[:, :n_occ]
                X_v = X[:, n_occ:]
                A = SymMetricOV(X_o, X_v)
            L, piv, num_rank_a = self.cholesky_decomp.decompose(A, n_grid, self.cholesky_threshold)
            piv = piv[:num_rank_a]
            L = L[:, piv]
            X_pruned = X[piv, :]
            S_Lg = A._function_mtx(piv, np.arange(n_grid))
        else:
            L = np.zeros((0, n_grid))
            piv = np.empty((0, n_grid))
            X_pruned = X
            S_Lg = np.empty((0, n_grid))

        return L, X_pruned, piv, S_Lg

    def build_coulomb_matrix_asym(self, L, Y_alpha, R, Y_beta):
        orb_a, _ = L.shape
        orb_b, _ = R.shape

        if orb_a == 0 or orb_b == 0:
            return np.empty((0, 0))

        B_alpha = self.build_coulomb_matrix_sep(L, Y_alpha)
        B_beta = self.build_coulomb_matrix_sep(R, Y_beta)
        return B_alpha @ B_beta.T

    def build_coulomb_matrix_sym(self, L, Y):
        B = self.build_coulomb_matrix_sep(L, Y)
        return B @ B.T

    def build_coulomb_matrix_sep(self, L, Y):
        A = sp.linalg.solve_triangular(L.T, Y.T, lower=True)
        B = sp.linalg.solve_triangular(L, A, lower=False)
        return B


    def prune_grid_ao(self, X: np.ndarray, n_grid: int, n_occ: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.prune_grid(X, n_grid, n_occ, is_ao=True)
