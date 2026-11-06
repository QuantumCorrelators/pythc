import logging

import numba as nb
import numpy as np
import scipy as sp
from pyscf import gto

import pythc.lib as lib
from pythc.grid import GridProvider, BeckeGrid
from pythc.thc.ls_thc_funcs import eval_basefuncs
from pythc.thc.thc_base import Mode
from pythc.thc.checkpoints import GRID_BUILD, BASIS_FUNCTION_EVAL
from pythc.thc.ls_ri_thc import LS_RI_THC
from tracking import observe

logger = logging.getLogger()

@nb.njit(parallel=True, fastmath=True)
def _compute_codensity(R, etas, rho_mat):
    g_size, N = R.shape

    for gi in nb.prange(g_size):

        pair_idx = 0

        for i in range(N):
            R_gi_i = R[gi, i]

            for j in range(i, N):
                rho_mat[pair_idx, gi] = R_gi_i * R[gi, j] * etas[pair_idx]
                pair_idx += 1


@nb.njit(parallel=True, fastmath=True)
def _compute_codensity_ov(R_occ, R_vir, etas, rho_mat):
    b_size, n_occ = R_occ.shape
    _, n_vir = R_vir.shape

    for gi in nb.prange(b_size):
        pair_idx = 0

        for i in range(n_occ):
            R_gi_i = R_occ[gi, i]

            for a in range(n_vir):
                rho_mat[pair_idx, gi] = R_gi_i * R_vir[gi, a] * etas[pair_idx]
                pair_idx += 1


class LS_RI_QRCP(LS_RI_THC):
    """
    Implements a QR Decomposition with Column Pivoting (QRCP) strategy for LS-THC grid pruning
    following Lu and Ying (2015 - DOI: 10.1016/j.jcp.2015.09.014).

    This class selects interpolating points by finding the most linearly independent
    columns of the orbital co-density matrix. To handle the O(N^4) scaling of the full
    co-density matrix decomposition, it utilizes a randomized sketching technique (similar to a
    Subsampled Randomized Hadamard Transform, but using FFT) to compress the pair
    dimension before performing the QRCP.
    """
    def __init__(self,
                 mol: gto.Mole,
                 auxbasis: str,
                 grid: GridProvider = None,
                 mo_coeff: np.ndarray = None,
                 tolerance: float = 1e-6,
                 seed: int | None = None):
        super().__init__(mol=mol, auxbasis=auxbasis, grid=grid, mo_coeff=mo_coeff)
        self.mol = mol
        self.pruning_threshold = tolerance
        self.seed = seed

        # beta = 10/3 to matches the literature default of keep_rows=20 at 1e-6
        self.keep_rows = max(1, int(np.ceil(-(10 / 3) * np.log10(tolerance))))

        self.grid = grid if grid is not None else BeckeGrid(mol)

    def build_pruned_X(self, mode: Mode, mo_coeff: np.ndarray, auxmol,
                       n_occ: int | None = None) -> np.ndarray:
        """
        Constructs the pruned collocation matrix (X) using sketched QRCP.

        Workflow summary:
        1. Base Grid Generation: Generates the integration grid and evaluates the
           weighted AO basis functions (Rs).
        2. Matrix Sketching (Dimensionality Reduction):
           - If the number of orbital pairs is large, generates a sketched pair density matrix.
           - Computes AO co-densities weighted by random complex phase factors.
           - Applies a Fast Fourier Transform (FFT) across the pair dimension to mix the data.
           - Shuffles and truncates the rows to a requested subsample size (r_requested).
           - If the system is small, exact AO co-densities are computed without sketching.
        3. QR with Column Pivoting: Performs a pivoted QR decomposition (zgeqp3 for
           complex sketched data, dgeqp3 for real exact data) on the density matrix.
        4. Truncation: Evaluates the absolute values of the R-matrix diagonal. Retains
           pivot indices where the diagonal value is greater than the maximum diagonal
           value multiplied by the `pruning_threshold`.
        5. Matrix Assembly: Slices the evaluated AO basis matrix (Rs) using the selected
           pivots to form the final pruned collocation matrix.

        Args:
            mo_coeff (np.ndarray): The molecular orbital coefficients.

        Returns:
            np.ndarray: The pruned collocation matrix containing only the selected
                        interpolation points.

        Raises:
            ValueError: If the threshold is too high and all auxiliary functions are pruned.
        """
        grid, weigths = self.grid.build()
        observe.checkpoint(GRID_BUILD)

        logger.info("building QRCP grid: grid=%d, nao=%d, keep_rows=%d", len(grid), self.N, self.keep_rows)

        Rs = eval_basefuncs(self.mol, grid)  # Renamed for clarity
        Rs = (np.sqrt(np.sqrt(weigths))[:, np.newaxis] * Rs)

        observe.checkpoint(BASIS_FUNCTION_EVAL)
        logger.info("evaluated basis functions on grid: R=%s", Rs.shape)

        if n_occ is None:
            n_occ = self.mol.nelectron // 2
        use_ov = (mode == 'ov' and n_occ > 0 and n_occ < self.N)

        if use_ov:
            X = Rs @ mo_coeff
            X_occ = np.ascontiguousarray(X[:, :n_occ])
            X_vir = np.ascontiguousarray(X[:, n_occ:])
            n_pairs = n_occ * (self.N - n_occ)
            logger.info(f"computing product densities with {n_occ}x{self.N - n_occ} occupied-virtual block")
        else:
            rows, cols = np.triu_indices(self.N, k=0)
            n_pairs = len(rows)
            logger.info(f"computing product densities with {self.N}x{self.N} basis functions")
        n_grid = len(grid)
        logger.info("computing product densities: pairs=%d", n_pairs)
        r_requested = self.N * self.keep_rows if self.keep_rows < self.N else n_pairs

        if r_requested < n_pairs:
            logger.info("sketching codensity: pairs=%d -> rows=%d", n_pairs, r_requested)
            
            # Local generator: reproducible per call for a given seed, and no
            # mutation of the global lib.rng state.
            rng = np.random.default_rng(self.seed)
            etas = np.exp(1j * rng.uniform(0, 2 * np.pi, size=n_pairs)).astype(np.complex128)

            # Pre-select random rows to keep after FFT
            chosen_rows = rng.permutation(n_pairs)[:r_requested]
            
            # Pre-allocate the final sketched matrix
            M_sketch = np.empty((r_requested, n_grid), dtype=np.complex128, order='F')
            
            block_size = 2048
            n_blocks = (n_grid + block_size - 1) // block_size
            
            logger.info("sketching grid blocks: n_blocks=%d, block_size=%d", n_blocks, block_size)
            
            for b in range(n_blocks):
                logger.info("\tsketch block %d/%d", b + 1, n_blocks)
                start = b * block_size
                end = min((b + 1) * block_size, n_grid)
                b_size = end - start
                
                # Allocate block as Fortran-contiguous for optimal column-wise memory access
                rho_block = np.empty((n_pairs, b_size), dtype=np.complex128, order='F')
                
                # Compute codensity for this grid block
                if use_ov:
                    X_occ_block = np.ascontiguousarray(X_occ[start:end, :])
                    X_vir_block = np.ascontiguousarray(X_vir[start:end, :])
                    _compute_codensity_ov(X_occ_block, X_vir_block, etas, rho_block)
                else:
                    Rs_block = np.ascontiguousarray(Rs[start:end, :])
                    _compute_codensity(Rs_block, etas, rho_block)
                
                # Apply FFT in place
                sp.fft.fft(rho_block, axis=0, workers=lib.cpu_count(), overwrite_x=True)
                
                # Extract chosen rows
                M_sketch[:, start:end] = rho_block[chosen_rows, :]
                
            logger.info("QRCP on sketched matrix: threshold=%.1e", self.pruning_threshold)

            R, piv, _, _, _ = sp.linalg.lapack.zgeqp3(M_sketch, overwrite_a=1)
        else:
            # these cases are so small we can copy safely
            logger.info("QRCP on exact codensity: threshold=%.1e", self.pruning_threshold)
            if use_ov:
                rho_initial = (X_occ[:, :, None] * X_vir[:, None, :]).reshape(n_grid, -1)
            else:
                rho_initial = Rs[:, rows] * Rs[:, cols]
            rho = np.asfortranarray(rho_initial.swapaxes(0, 1))
            R, piv, _, _, _ = sp.linalg.lapack.dgeqp3(rho, overwrite_a=1)

        piv = piv - 1

        R_diag = np.abs(np.diag(R))
        if R_diag.size > 0:
            max_val = R_diag[0]
            # Avoid division by zero if matrix is empty or zero
            if max_val > 1e-15:
                mask = R_diag > (max_val * self.pruning_threshold)
                N_aux = np.sum(mask)
            else:
                N_aux = 0
        else:
            N_aux = 0

        logger.info("pruned grid: %d points (R_max=%.2e)", N_aux, R_diag[0])

        if N_aux == 0:
            raise ValueError("ISDF pruning removed all auxiliary functions. Check threshold or basis.")

        x_mu = piv[:N_aux]

        # Select AO values at interpolating points and then transform to MO basis
        logger.info("built pruned collocation matrix: X=%s", (N_aux, Rs.shape[1]))
        X = Rs[x_mu, :]
        return X
