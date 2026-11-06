import logging

import numpy as np

from pythc.thc.ls_thc_funcs import eval_basefuncs
from pythc.thc.thc_base import Mode
from pythc.thc.ls_ri_thc import LS_RI_THC

logger = logging.getLogger()


class LS_RI_Becke(LS_RI_THC):
    """
    This class implements Least-Squares Fitted THC following Hohenstein, Parrish, Martinez
    (DOI: 10.1063/1.4768233).
    """

    def build_pruned_X(self, mode: Mode, mo_coeff: np.ndarray, auxmol, n_occ: int | None = None) -> np.ndarray:
        logger.info("building Becke grid: mode=%s", mode)
        grid, weights = self.grid.build()
        logger.info("built grid: grid=%d", len(grid))
        R = eval_basefuncs(self.mol, coords=grid)
        logger.info("evaluated basis functions on grid: R=%s", R.shape)
        X = np.sqrt(np.sqrt(weights))[:, np.newaxis] * R

        logger.info("built collocation matrix: X=%s", X.shape)

        return X

