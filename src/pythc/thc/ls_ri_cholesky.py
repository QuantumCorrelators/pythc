import logging
from typing import Type

import numpy as np
import scipy as sp
from pyscf import gto

from pythc.decomp.cholesky import AccelRPCholesky
from pythc.decomp.cholesky import Cholesky
from pythc.decomp.cholesky import SymMetric, SymMetricOV
from pythc.grid import GridProvider, BeckeGrid
from pythc.thc.checkpoints import GRID_BUILD, BASIS_FUNCTION_EVAL, GRID_PRUNING, FITTING_MATRIX, METRIC_INVERSION
from pythc.thc.ls_thc_funcs import build_auxmol, build_aux_coulomb_inv, eval_basefuncs, contract_codensity_df_eri
from pythc.thc.thc_base import ThcEri, Mode, THC
from pythc.thc.thc_base import ThcEriUnrestricted
from pythc.tracking.experiment_run import ExperimentRun

logger = logging.getLogger()


class LS_RI_Cholesky(THC):
    """
    Implements Least-Squares Fitted Tensor Hypercontraction (THC) with Cholesky Decomposition
    following Matthews (2020 - DOI: 10.1021/acs.jctc.9b01205).

    This class extends THC applying a pivoted Cholesky decomposition
    to the metric matrix. This identifies a linear dependency threshold and prunes
    the grid, effectively reducing the number of grid points and the dimensionality
    of the fitting matrices to improve computational efficiency and numerical stability.
    """

    def __init__(self,
                 mol: gto.Mole,
                 grid: GridProvider = None,
                 mo_coeff: np.ndarray = None,
                 auxbasis: str = None,
                 cholesky_decomp: Type[Cholesky] = AccelRPCholesky,
                 cholesky_threshold: float = 1e-5,
                 symmetry: str = 's2'):

        super().__init__()
        self.mol = mol
        self.N = mol.nao_nr()
        self.auxbasis = auxbasis if auxbasis is not None else f'{self.mol.basis}-ri'
        self.cholesky_decomp = cholesky_decomp()
        self.cholesky_threshold = cholesky_threshold
        self.grid = grid if grid is not None else BeckeGrid(mol)
        self.mol = mol
        self.symmetry = symmetry
        self.mo_coeff = mo_coeff if mo_coeff is not None and len(mo_coeff) > 0 else np.eye(mol.nao_nr())

    @classmethod
    def __str__(cls):
        return "cd_ls_thc"

    def build(self, mode: Mode = "ao") -> ThcEri:
        """
        Builds a spin-restricted Tensor Hypercontraction (THC) representation with grid pruning.

        Workflow summary:
        1. Generate grid points and integration weights.
        2. Evaluate basis functions to construct the collocation matrix (X).
        3. Form an implicit representation of the metric matrix (A) and perform
           pivoted Cholesky decomposition to obtain the lower triangular factor (L)
           and pivot indices.
        4. Prune the location matrix (X) using the selected pivots.
        5. Construct the fitting matrix (E) using the pruned collocation matrix.
        6. Solve for the Coulomb kernel matrix (Z) efficiently using triangular solvers.

        Returns:
            ThcEri: Dataclass containing the number of electrons, pruned X, and Z.

        Raises:
            NotImplementedError: If the fitting mode is not 'ao' or 'ov'.
        """
        active = ExperimentRun.get_active()

        grid, weigths = self.grid.build()
        if active: active.checkpoint(GRID_BUILD)

        logger.info(f"built grid of size: {len(grid)}")

        R = eval_basefuncs(self.mol, grid)  # (grid, N)
        logger.info(f"got {R.shape} basis function on grid matrix")

        X = (np.sqrt(np.sqrt(weigths))[:, np.newaxis] * R)
        if mode != 'ao':
            X = X @ self.mo_coeff

        if active: active.checkpoint(BASIS_FUNCTION_EVAL)
        logger.info(f"using cholesky threshold: {self.cholesky_threshold}")

        n_occ = self.mol.nelectron // 2
        n_virt = self.N - n_occ

        if mode == 'ao':
            A = SymMetric(X)
        elif mode == 'ov':
            A = SymMetricOV(X[:, :n_occ], X[:, n_occ:])
        else:
            raise NotImplementedError

        n_grid = X.shape[0]
        L, piv, num_rank = self.cholesky_decomp.decompose(A, n_grid, self.cholesky_threshold)
        piv = piv[:num_rank]
        L = L[:, piv]
        self.pruned_grid = grid[piv]

        X_pruned = X[piv, :]

        if active: active.checkpoint(GRID_PRUNING)
        logger.info(f"building fitting matrix")

        auxmol = build_auxmol(self.mol, self.auxbasis)
        j2c_inv = build_aux_coulomb_inv(auxmol)
        logger.info("building coulomb kernel matrix")

        Y = contract_codensity_df_eri(mode, X_pruned, self.mo_coeff, self.mol, auxmol, j2c_inv, n_occ, n_virt)
        if active: active.checkpoint(FITTING_MATRIX)

        D = self.build_coulomb_matrix_sep(L, Y)
        if active: active.checkpoint(METRIC_INVERSION)

        Z = D @ D.T

        return ThcEri(self.mol.nelectron, X_pruned, Z, D)

    def build_unrestricted(self, mode: Mode = "ao") -> ThcEriUnrestricted:
        """
        Builds a spin-unrestricted Tensor Hypercontraction (THC) representation with grid pruning.

        Note: Currently, this method is only supported in 'ov' (occupied-virtual) mode.

        Workflow summary:
        1. Extract alpha/beta molecular orbital coefficients and occupations.
        2. Evaluate basis functions to construct separate alpha and beta collocation matrices (X_alpha, X_beta).
        3. Perform matrix-free pivoted Cholesky decompositions on the alpha and beta
           metric matrices to obtain respective factors (L_aa, L_bb) and pivot indices.
        4. Prune the alpha and beta location matrices based on their specific pivots.
        5. Construct symmetric (E_aa, E_bb) and asymmetric (E_ab) fitting matrices using pruned grids.
        6. Compute the unrestricted Coulomb kernel matrices (Z_aa, Z_bb, Z_ab) via triangular solvers.

        Returns:
            ThcEriUnrestricted: Dataclass containing the number of electrons, pruned collocation
                                matrices, and Coulomb kernels.

        Raises:
            NotImplementedError: If the fitting mode is not set to 'ov'.
        """
        if mode != 'ov':
            raise NotImplementedError("fitting an unrestricted THC in AO mode is currently not supported")

        S = self.mol.spin
        N = self.mol.nao_nr()
        nocc_alpha = (self.mol.nelectron + S) // 2
        nocc_beta = (self.mol.nelectron - S) // 2
        nvir_alpha = N - nocc_alpha
        nvir_beta = N - nocc_beta

        active = ExperimentRun.get_active()
        mo_coeff_alpha = self.mo_coeff[0]
        mo_coeff_beta = self.mo_coeff[1]

        grid, weights = self.grid.build()
        if active: active.checkpoint(GRID_BUILD)

        logger.info("evaluating basis functions on grid")
        R = eval_basefuncs(self.mol, coords=grid)

        logger.info("building metric matrix")
        X = np.sqrt(np.sqrt(weights))[:, np.newaxis] * R

        X_alpha = X @ mo_coeff_alpha
        X_beta = X @ mo_coeff_beta
        if active: active.checkpoint(BASIS_FUNCTION_EVAL)

        n_grid = X.shape[0]

        logger.info(f"using cholesky threshold: {self.cholesky_threshold}")
        L_aa, X_alpha_pruned_aa = self.prune_grid(X_alpha, n_grid, nocc_alpha)
        L_bb, X_beta_pruned_bb = self.prune_grid(X_beta, n_grid, nocc_beta)

        if active: active.checkpoint(GRID_PRUNING)
        logger.info(f"building fitting matrices")

        auxmol = build_auxmol(self.mol, self.auxbasis)
        j2c_inv = build_aux_coulomb_inv(auxmol)

        logger.info("building coulomb kernel matrices")

        Y_alpha = contract_codensity_df_eri(mode, X_alpha_pruned_aa, mo_coeff_alpha, self.mol, auxmol, j2c_inv, nocc_alpha, nvir_alpha)
        Y_beta = contract_codensity_df_eri(mode, X_beta_pruned_bb, mo_coeff_beta, self.mol, auxmol, j2c_inv, nocc_beta, nvir_beta)
        if active: active.checkpoint(FITTING_MATRIX)

        Z_aa = Z_bb = np.zeros((n_grid, n_grid))
        if L_aa.shape[0] > 0:
            Z_aa = self.build_coulomb_matrix_sym(L_aa, Y_alpha)

        if L_bb.shape[0] > 0:
            Z_bb = self.build_coulomb_matrix_sym(L_bb, Y_beta)

        Z_ab = self.build_coulomb_matrix_asym(L_aa, Y_alpha, L_bb, Y_beta)
        if active: active.checkpoint(METRIC_INVERSION)

        return ThcEriUnrestricted(self.mol.nelectron, X_alpha_pruned_aa, X_beta_pruned_bb, Z_aa, Z_bb, Z_ab)

    def prune_grid(self, X: np.ndarray, n_grid: int, n_occ: int) -> tuple[np.ndarray, np.ndarray]:
        if n_occ > 0:
            X_o = X[:, :n_occ]
            X_v = X[:, n_occ:]
            A = SymMetricOV(X_o, X_v)
            L, piv, num_rank_a = self.cholesky_decomp.decompose(A, n_grid, self.cholesky_threshold)
            piv = piv[:num_rank_a]
            L = L[:, piv]
            X_pruned = X[piv, :]
        else:
            L = np.zeros((0, n_grid))
            X_pruned = X

        return L, X_pruned

    def build_coulomb_matrix_sep(self, L, Y):
        A = sp.linalg.solve_triangular(L.T, Y.T, lower=True)
        D = sp.linalg.solve_triangular(L, A, lower=False)
        return D

    def build_coulomb_matrix_sym(self, L, Y):
        B = self.build_coulomb_matrix_sep(L, Y)
        return B @ B.T

    def build_coulomb_matrix_asym(self, L, Y_alpha, R, Y_beta):
        orb_a, _ = L.shape
        orb_b, _ = R.shape

        if orb_a == 0 or orb_b == 0:
            return np.empty((0, 0))

        B_alpha = self.build_coulomb_matrix_sep(L, Y_alpha)
        B_beta = self.build_coulomb_matrix_sep(R, Y_beta)
        return B_alpha @ B_beta.T

