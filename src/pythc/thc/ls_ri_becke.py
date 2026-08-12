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

    @classmethod
    def __str__(cls):
        return "ls_thc"

    def build_pruned_X(self, mode: Mode, mo_coeff: np.ndarray, auxmol) -> np.ndarray:
        grid, weights = self.grid.build()
        R = eval_basefuncs(self.mol, coords=grid)
        X = np.sqrt(np.sqrt(weights))[:, np.newaxis] * R

        self.pruned_grid = grid

        return X

