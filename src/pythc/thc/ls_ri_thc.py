import logging
from abc import abstractmethod

import numpy as np
from pyscf import gto

from pythc.grid import GridProvider, BeckeGrid
from pythc.thc.checkpoints import GRID_PRUNING, FITTING_MATRIX
from pythc.thc.ls_thc_funcs import build_auxmol, build_coulomb_matrix, \
    build_coulomb_matrix_asym, eval_basefuncs
from pythc.thc.thc_base import THC, ThcEri, Mode, ERI, ThcEriUnrestricted
from pythc.tracking.experiment_run import ExperimentRun

logger = logging.getLogger()


class LS_RI_THC(THC):
    """

    This class implements Least-Squares Fitted THC following Hohenstein, Parrish, Martinez
    (DOI: 10.1063/1.4768233) sometimes also referred to as Interpolative Separable Density Fitting (ISDF)
    (see Lee, Lin, Head-Gordon (2020 - DOI: 10.1021/acs.jctc.9b00820)).

    This abstract base class establishes the framework for LS-THC. It relies on density
    fitting (resolution of the identity) to factorize the electron repulsion integrals.
    The Coulomb kernel (Z) is constructed via an intermediate tensor (D), such that
    Z = D.T @ D. Subclasses must implement the `build_pruned_X` method to define the
    grid selection strategy.
    """
    def __init__(self,
                 mol: gto.Mole,
                 auxbasis: str,
                 grid: GridProvider = None,
                 mo_coeff: np.ndarray = None,
                 ):
        self.N = mol.nao_nr()
        self.mol = mol
        self.grid = grid if grid is not None else BeckeGrid(mol)
        self.auxbasis = auxbasis if auxbasis else f'{mol.basis}-ri'
        self.mo_coeff = mo_coeff if mo_coeff is not None and len(mo_coeff) > 0 else np.eye(mol.nao_nr())

    @abstractmethod
    def build_pruned_X(self, mode: Mode, mo_coeff: np.ndarray, auxmol: gto.Mole) -> np.ndarray:
        """
        Constructs the pruned collocation matrix (X).

        This is an abstract method that must be implemented by subclasses to define
        the specific grid pruning or selection strategy used in the ISDF procedure.

        Args:
            mo_coeff (np.ndarray): The molecular orbital coefficients.

        Returns:
            np.ndarray: The pruned collocation matrix.
            :param auxmol:
            :param mo_coeff:
            :param mode:
        """
        pass

    def build(self, mode: Mode = "ao") -> ERI:
        """
        Builds a spin-restricted LS-THC representation.

        Workflow summary:
        1. Grid Pruning (`build_pruned_X`): Selects a representative subset of grid points
           to form the collocation matrix (X) based on the subclass's ISDF strategy.
        2. 2-Center Integrals: Evaluates the auxiliary basis 2-center integrals (K|L)
           and computes their inverse square root.
        3. Intermediate Matrix Construction:
           - In 'ao' mode (`build_D_ao`): Computes the metric inversion directly,
             contracts 3-center integrals with the AO collocation pair products using
             spatial symmetry, and multiplies by the 2-center inverse.
           - In 'ov' mode (`build_D_ov`): Projects X into the MO basis, isolates the
             occupied and virtual blocks to compute the metric inversion, contracts
             with 3-center integrals, and multiplies by the 2-center inverse.
        4. Coulomb Kernel: Computes the final kernel matrix as Z = D.T @ D.

        Returns:
            ThcEri: Dataclass containing the number of electrons, pruned X, and Z.

        Raises:
            NotImplementedError: If the fitting mode is not 'ao' or 'ov'.
        """
        active = ExperimentRun.get_active()

        auxmol = build_auxmol(self.mol, self.auxbasis)

        X = self.build_pruned_X(mode, self.mo_coeff, auxmol)
        if mode == 'ov':
            X = X @ self.mo_coeff

        if active: active.checkpoint(GRID_PRUNING)

        D = build_coulomb_matrix(mode, self.mol, auxmol, X, self.mo_coeff)
        Z = D.T @ D
        if active: active.checkpoint(FITTING_MATRIX)

        return ThcEri(self.mol.nelectron, X, Z, D.T)

    def build_unrestricted(self, mode: Mode = "ao") -> ThcEriUnrestricted:
        """
        Builds a spin-unrestricted LS-THC representation.

        Workflow summary:
        1. 2-Center Integrals: Evaluates the auxiliary basis 2-center integrals and
           computes their inverse square root.
        2. Spin Separation: Extracts alpha and beta molecular orbital coefficients
           and calculates their respective orbital occupations.
        3. Grid Pruning: Independently selects grid points for
           alpha and beta spins (if occupied) to form spin-specific collocation
           matrices (X_alpha, X_beta).
        4. Intermediate Matrix Construction:
           - Projects X_alpha and X_beta into their respective MO bases.
           - Computes independent intermediate matrices (D_aa, D_bb) by performing
             metric inversion on the occupied-virtual blocks and contracting with
             the 3-center integrals.
        5. Coulomb Kernels: Computes the unrestricted kernel matrices using the
           intermediate matrices:
           - Z_aa = D_aa.T @ D_aa
           - Z_bb = D_bb.T @ D_bb
           - Z_ab = D_aa.T @ D_bb

        Returns:
            ThcEriUnrestricted: Dataclass containing the number of electrons, spin-specific
                                collocation matrices, and Coulomb kernels.

        Raises:
            NotImplementedError: If the fitting mode is set to 'ao'.
        """
        active = ExperimentRun.get_active()

        if mode == 'ao':
            raise NotImplementedError("fitting an unrestricted THC in AO mode is currently not supported")

        auxmol = build_auxmol(self.mol, self.auxbasis)

        S = self.mol.spin
        nocc_alpha = (self.mol.nelectron + S) // 2
        nocc_beta = (self.mol.nelectron - S) // 2
        mo_coeff_alpha = self.mo_coeff[0]
        mo_coeff_beta = self.mo_coeff[1]

        if nocc_alpha == 0 or nocc_beta == 0:
            grid, weights = self.grid.build()
            R = eval_basefuncs(self.mol, coords=grid)
            X = np.sqrt(np.sqrt(weights))[:, np.newaxis] * R

        if nocc_alpha > 0:
            X_alpha = self.build_pruned_X(mode, mo_coeff_alpha, auxmol)
        else:
            X_alpha = X

        if nocc_beta > 0:
            X_beta = self.build_pruned_X(mode, mo_coeff_beta, auxmol)
        else:
            X_beta = X

        X_alpha = X_alpha @ mo_coeff_alpha
        X_beta = X_beta @ mo_coeff_beta

        if active: active.checkpoint(GRID_PRUNING)

        Z_aa, Z_bb, Z_ab = build_coulomb_matrix_asym(mode, self.mol, auxmol, X_alpha, X_beta, self.mo_coeff)
        if active: active.checkpoint(FITTING_MATRIX)

        return ThcEriUnrestricted(self.mol.nelectron, X_alpha, X_beta, Z_aa, Z_bb, Z_ab)