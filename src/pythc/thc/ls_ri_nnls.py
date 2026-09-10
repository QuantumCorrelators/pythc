import logging

import numpy as np
import scipy as sp

import pythc.lib as lib
from pythc.decomp.cholesky import AccelRPCholesky, SymMetric, SymMetricOV
from pythc.thc.ls_thc_funcs import eval_basefuncs
from pythc.thc.thc_base import Mode
from pythc.thc.ls_ri_thc import LS_RI_THC

logger = logging.getLogger()

class LS_RI_NNLS(LS_RI_THC):
    def build_pruned_X(self, mode: Mode, mo_coeff: np.ndarray, auxmol) -> np.ndarray:
        logger.info("building NNLS grid: mode=%s", mode)
        grid, weights = self.grid.build()
        logger.info("built grid: grid=%d", len(grid))
        R = eval_basefuncs(self.mol, grid)
        logger.info("evaluated basis functions on grid: R=%s", R.shape)
        
        # 1. Point Selection using Pivoted Cholesky
        X_tmp = np.sqrt(np.sqrt(weights))[:, np.newaxis] * R
        if mode != 'ao':
            X_tmp = X_tmp @ mo_coeff
            
        n_occ = self.mol.nelectron // 2
        
        if mode == 'ao':
            metric = SymMetric(X_tmp)
        elif mode == 'ov':
            metric = SymMetricOV(X_tmp[:, :n_occ], X_tmp[:, n_occ:])
        else:
            raise NotImplementedError
            
        decomp = AccelRPCholesky(seed=getattr(self, 'seed', None))
        n_grid = X_tmp.shape[0]
        # Defaulting to 1e-5 threshold similar to the Cholesky THC implementation
        cholesky_threshold = getattr(self, 'cholesky_threshold', 1e-5)
        logger.info("selecting points: threshold=%.1e", cholesky_threshold)
        _, piv, num_rank = decomp.decompose(metric, n_grid, cholesky_threshold)
        piv = piv[:num_rank]
        logger.info("selected points: %d -> %d candidates", n_grid, num_rank)
        
        # 2. Weight Optimization using NNLS on the pruned grid
        R_pruned = R[piv]
        if mode != 'ao':
            R_pruned_mo = R_pruned @ mo_coeff
        else:
            R_pruned_mo = R_pruned

        N = R_pruned_mo.shape[1]
        S = self.mol.intor('int1e_ovlp_sph')
        A = lib.einsum('pn,pm->mnp', R_pruned_mo, R_pruned_mo).reshape(N**2, -1)
        y = S.reshape(N**2)

        logger.info("optimizing weights: A=%s", A.shape)
        # A is now heavily overdetermined (e.g. 841 x ~300), making NNLS converge in ms
        w_pruned, _ = sp.optimize.nnls(A, y)

        # Drop any points that NNLS assigned zero weight
        piv_final = np.where(w_pruned > 1e-8)[0]
        w_final = w_pruned[piv_final]
        X_final = R_pruned[piv_final]
        X_final = X_final * np.sqrt(np.sqrt(w_final))[:, np.newaxis]
        logger.info("built pruned collocation matrix: %d -> %d points, X=%s",
                    num_rank, len(piv_final), X_final.shape)

        return X_final