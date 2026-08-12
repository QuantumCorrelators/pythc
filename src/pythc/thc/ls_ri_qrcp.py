import logging

import numba as nb
import numpy as np
import scipy as sp
from pyscf import gto

import pythc.lib as lib
from pythc.grid import GridProvider, BeckeGrid
from pythc.thc.ls_thc_funcs import eval_basefuncs
from pythc.thc.thc_base import Mode
from pythc.thc.checkpoints import GRID_BUILD, GRID_PRUNING, BASIS_FUNCTION_EVAL
from pythc.thc.ls_ri_thc import LS_RI_THC
from pythc.tracking.experiment_run import ExperimentRun

logger = logging.getLogger()

def _random_unit() -> complex:
    theta = lib.rng.uniform(0, 2 * np.pi)
    return complex(np.cos(theta), np.sin(theta))

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
                 tolerance: float = 1e-6):
        super().__init__(mol=mol, auxbasis=auxbasis, grid=grid, mo_coeff=mo_coeff)
        self.mol = mol
        self.pruning_threshold = tolerance

        # beta = 10/3 to matches the literature default of keep_rows=20 at 1e-6
        self.keep_rows = max(1, int(np.ceil(-(10 / 3) * np.log10(tolerance))))

        self.grid = grid if grid is not None else BeckeGrid(mol)

    @classmethod
    def __str__(cls):
        return "isdf_qrcp"


    def build_pruned_X(self, mode: Mode, mo_coeff: np.ndarray, auxmol) -> np.ndarray:
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
        active = ExperimentRun.get_active()
        grid, weigths = self.grid.build()
        if active: active.checkpoint(GRID_BUILD)

        logger.info(f"built grid of size: {len(grid)}")

        Rs = eval_basefuncs(self.mol, grid)  # Renamed for clarity
        Rs = (np.sqrt(np.sqrt(weigths))[:, np.newaxis] * Rs)
        logger.info(f"got {Rs.shape} basis function on grid matrix")
        if active: active.checkpoint(BASIS_FUNCTION_EVAL)

        rows, cols = np.triu_indices(self.N, k=0)

        logger.info(f"computing product densities with {self.N}x{self.N} basis functions")

        n_pairs = len(rows)
        n_grid = len(grid)
        r_requested = self.N * self.keep_rows if self.keep_rows < self.N else n_pairs

        if r_requested < n_pairs:
            logger.info(f"Sketching active: compressing {n_pairs} pairs to {r_requested} auxiliary rows")
            
            # The product densities should be in the AO basis for the ISDF selection process
            etas = np.array([_random_unit() for _ in range(n_pairs)])
            
            # Pre-select random rows to keep after FFT
            shuffled_indices = np.arange(n_pairs)
            lib.rng.shuffle(shuffled_indices)
            chosen_rows = shuffled_indices[:r_requested]
            
            # Pre-allocate the final sketched matrix
            M_sketch = np.empty((r_requested, n_grid), dtype=np.complex128, order='F')
            
            block_size = 2048
            n_blocks = (n_grid + block_size - 1) // block_size
            
            logger.info(f"processing in {n_blocks} blocks of size {block_size}")
            
            for b in range(n_blocks):
                start = b * block_size
                end = min((b + 1) * block_size, n_grid)
                b_size = end - start
                
                # Allocate block as Fortran-contiguous for optimal column-wise memory access
                rho_block = np.empty((n_pairs, b_size), dtype=np.complex128, order='F')
                
                # Compute codensity for this grid block
                Rs_block = np.ascontiguousarray(Rs[start:end, :])
                _compute_codensity(Rs_block, etas, rho_block)
                
                # Apply FFT in place
                sp.fft.fft(rho_block, axis=0, workers=lib.cpu_count(), overwrite_x=True)
                
                # Extract chosen rows
                M_sketch[:, start:end] = rho_block[chosen_rows, :]
                
            logger.info(f"performing QR decomposition with pivoting, pruning threshold: {self.pruning_threshold}")

            R, piv, _, _, _ = sp.linalg.lapack.zgeqp3(M_sketch, overwrite_a=1)
        else:
            # these cases are so small we can copy safely
            logger.info(f"performing QR decomposition with pivoting, pruning threshold: {self.pruning_threshold}")
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

        logger.info(f"Pruned to {N_aux} auxiliary functions (Max val: {R_diag[0]:.2e})")

        if N_aux == 0:
            raise ValueError("ISDF pruning removed all auxiliary functions. Check threshold or basis.")

        x_mu = piv[:N_aux]
        self.pruned_coords = grid[x_mu]

        # Select AO values at interpolating points and then transform to MO basis
        logger.info(f"transforming to MO basis at {N_aux} interpolating points")
        X = Rs[x_mu, :]
        return X
