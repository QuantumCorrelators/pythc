import logging
from typing import Any, Type

import numpy as np
import scipy
from pyscf import df, gto
from scipy.linalg.lapack import dpstrf

from pythc.decomp.cholesky import Cholesky, AccelRPCholesky, GramMetric
from pythc.grid import GridProvider, BeckeGrid
from pythc.thc.checkpoints import GRID_BUILD, BASIS_FUNCTION_EVAL, GRID_PRUNING, FITTING_MATRIX
from pythc.thc.ls_thc_funcs import eval_basefuncs
from pythc.thc.thc_base import THC, ThcEri, ERI, Mode
from pythc.thc.thc_base import ThcEriUnrestricted
from pythc.tracking.experiment_run import ExperimentRun

logger = logging.getLogger()

class LS_Aux_Becke(THC):
    """
    Implements Density-Fitted Least-Squares Tensor Hypercontraction (LS-Aux-Becke) following
    Hillers-Bendtsen and Martinez (2025 - DOI: 10.48550/arXiv.2508.19212)

    This class leverages an auxiliary basis set to approximate the electron
    repulsion integrals. The Coulomb kernel (Z) is constructed by fitting
    the auxiliary 2-center integrals to the auxiliary basis evaluated on the
    integration grid, circumventing the need for 4-index or 3-index integral
    contractions directly with the grid.
    """
    def __init__(self,
                 mol: gto.Mole,
                 grid: GridProvider = None,
                 mo_coeff: np.ndarray = None,
                 cholesky_decomp: Type[Cholesky] = AccelRPCholesky,
                 cholesky_threshold: float = 1e-8,
                 fit_auxbasis: str = None,
                 auxbasis: str = None,
                 regularization: float = 1e-6,
                 level=0,
                 ):
        self.N = mol.nao_nr()
        self.mol = mol
        self.grid = grid if grid else BeckeGrid(mol, level=level)
        self.cholesky_decomp = cholesky_decomp()
        self.cholesky_threshold = cholesky_threshold
        self.regularization = regularization

        HEAVY_ELEMENTS = ['Sb', 'Te', 'I', 'Bi']
        if isinstance(fit_auxbasis, str) and "etb" in fit_auxbasis:
            beta = float(fit_auxbasis.split("-")[1])
            self.fit_auxbasis = df.aug_etb(mol, beta=beta)
        elif fit_auxbasis:
            m = gto.Mole()
            m.atom = mol.atom
            m.charge = mol.charge
            m.spin = mol.spin

            basis = {
                'default': fit_auxbasis
            }
            for el in HEAVY_ELEMENTS:
                basis[el] = fit_auxbasis + '-pp'
            logger.info(f'using fit auxbasis: {basis}')

            m.basis = basis
            m.ecp = mol.ecp
            m.build()

            self.fit_auxbasis = df.make_auxbasis(m, mp2fit=True)
        else:
            self.fit_auxbasis = df.make_auxbasis(mol, mp2fit=True)

        logger.info(f"build fit auxbasis: {str(self.fit_auxbasis)[:100]}")
        self.mo_coeff = mo_coeff if mo_coeff is not None and len(mo_coeff) > 0 else np.eye(mol.nao_nr())


    def build_auxmol(self) -> gto.Mole:
        return df.addons.make_auxmol(self.mol, auxbasis=self.fit_auxbasis)

    @classmethod
    def __str__(cls):
        return "df_ls_thc"

    def build(self, mode: Mode = "ao") -> ERI:
        """
        Builds a spin-restricted LS-Aux-Becke representation.

        Workflow summary:
        1. Obtain the evaluated (and potentially pruned) collocation matrices
           for the auxiliary and primary bases via `get_pruned_grid`.
        2. Construct the intermediate fitting matrix (Y) by projecting the
           auxiliary basis integrals onto the grid.
        3. Compute the Coulomb kernel matrix directly as Z = Y.T @ Y.

        Returns:
            ThcEri: Dataclass containing the number of electrons, collocation matrix (X),
                    and the Coulomb kernel (Z).
        """
        active: ExperimentRun = ExperimentRun.get_active()
        auxmol = self.build_auxmol()
        if active: active.log_metric("fit_auxbasis_size", auxmol.nao_nr())


        logger.info("evaluating basis functions on grid")
        Xaux_pruned, X_pruned = self.get_pruned_grid(mode, auxmol)

        D = self.build_D(Xaux_pruned, auxmol)
        Z = D.T @ D
        if active: active.checkpoint(FITTING_MATRIX)

        logger.info(f"built THC with Z: {Z.shape}, X: {X_pruned.shape}")
        return ThcEri(self.mol.nelectron, X_pruned, Z, D)

    def build_unrestricted(self, mode: Mode = "ao") -> ThcEriUnrestricted:
        """
        Builds a spin-unrestricted LS-Aux-Becke representation.

        Note: This method is only supported in 'ov' (occupied-virtual) mode.

        Workflow summary:
        1. Obtain the evaluated auxiliary collocation matrix and spin-separated primary
           collocation matrices (X_alpha, X_beta) via `get_pruned_grid`.
        2. Construct the intermediate fitting matrix (Y) using the auxiliary grid data.
        3. Compute the common Coulomb kernel matrix as Z = Y.T @ Y.
           (In this approximation, the kernel is spin-independent, so Z_aa = Z_bb = Z_ab = Z).

        Returns:
            ThcEriUnrestricted: Dataclass containing the number of electrons, spin-specific
                                collocation matrices, and the shared Coulomb kernels.

        Raises:
            NotImplementedError: If the fitting mode is not set to 'ov'.
        """
        if mode != 'ov':
            raise NotImplementedError("fitting an unrestricted THC in AO mode is currently not supported")

        active = ExperimentRun.get_active()
        auxmol = self.build_auxmol()
        if active: active.log_metric("fit_auxbasis_size", auxmol.nao_nr())

        logger.info("evaluating basis functions on grid")
        Xaux_pruned, X_pruned_alpha, X_pruned_beta = self.get_pruned_grid(mode, auxmol)

        Y = self.build_D(Xaux_pruned, auxmol)
        Z = Y.T @ Y
        if active: active.checkpoint(FITTING_MATRIX)


        return ThcEriUnrestricted(self.mol.nelectron, X_pruned_alpha, X_pruned_beta, Z, Z, Z)

    prune = False
    def get_pruned_grid(self, mode, auxmol):
        active = ExperimentRun.get_active()

        coords, weights = self.grid.build()
        if active: active.checkpoint(GRID_BUILD)

        logger.info("evaluating basis functions on grid")
        Raux = eval_basefuncs(auxmol, coords)
        Xaux = np.sqrt(weights)[:, np.newaxis] * Raux

        R = eval_basefuncs(self.mol, coords)
        X = (np.sqrt(np.sqrt(weights))[:, np.newaxis] * R)
        if active: active.checkpoint(BASIS_FUNCTION_EVAL)

        if self.prune:
            logger.info("pruning active: reducing grid size")

            A = GramMetric(Xaux)
            L, piv, num_rank = self.cholesky_decomp.decompose(A, Xaux.shape[0], self.cholesky_threshold)
            piv = piv[:num_rank]

            X_aux_pruned = Xaux[piv, :]
            X_pruned = X[piv, :]

            logger.info(f"selected {len(X_pruned)} pruned points from {len(coords)} parent")
            if active: active.checkpoint(GRID_PRUNING)
        else:
            X_aux_pruned = Xaux
            X_pruned = X

        if len(self.mo_coeff.shape) == 3:  # unrestricted (must be mode ov)
            mo_coeff_alpha = self.mo_coeff[0]
            mo_coeff_beta = self.mo_coeff[1]

            X_alpha = X_pruned @ mo_coeff_alpha
            X_beta = X_pruned @ mo_coeff_beta

            return X_aux_pruned, X_alpha, X_beta
        else:
            if mode != 'ao':
                X_pruned = X_pruned @ self.mo_coeff

            return X_aux_pruned, X_pruned

    def build_D(self, Xaux, auxmol) -> Any:
        logger.info("computing auxiliary integrals")
        ints_2c2e = auxmol.intor('int2c2e')

        logger.info("performing low-rank decomposition of (K|L) integrals")

        L_out, piv, rank, info = dpstrf(ints_2c2e)
        logger.info(f"(K|L) has numerical rank {rank}/{ints_2c2e.shape[0]}")

        piv = piv - 1
        U = np.triu(L_out[:rank, :])

        L_permuted = U.T

        inv_piv = np.argsort(piv)
        L_left = L_permuted[inv_piv]

        L_full = np.zeros_like(ints_2c2e)
        L_full[:, :rank] = L_left

        # This is almost what the original publication does.
        # SciPy does not provide LQ factorization, so we replace it with
        # standard QR.
        #
        # Q, R = scipy.linalg.qr(Xaux, mode='economic')
        # C = scipy.linalg.solve_triangular(R.T, L_full, lower=True)
        # Y_T = Q @ C

        S_aux = Xaux.T @ Xaux
        S_aux_reg = S_aux + self.regularization * np.eye(S_aux.shape[0])
        M = scipy.linalg.solve(S_aux_reg, L_full, assume_a='pos')
        D_T = Xaux @ M

        return D_T.T


class LSTHC_AuxCoulomb_Cholesky(LS_Aux_Becke):
    @classmethod
    def __str__(cls):
        return "cd_df_ls_thc"

    prune = True


