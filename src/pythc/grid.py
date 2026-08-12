from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from pyscf import gto, dft
from pyscf.dft import treutler_prune

from pythc.configurable import Configurable
from pythc.tracking.experiment_run import ExperimentRun


class GridProvider(ABC,Configurable):
    @abstractmethod
    def build(self) -> tuple[np.ndarray, np.ndarray]:
        pass

class BeckeGrid(GridProvider):
    def __repr__(self):
        return f"BeckeGrid(level={self.level})"

    def __str__(self):
        return "becke"

    def __init__(self, mol: gto.Mole, level=0, prune: Callable[..., np.ndarray] = treutler_prune):
        self.mol = mol
        self.level = level
        self.prune = prune

    def build(self):
        grid = dft.gen_grid.Grids(self.mol)
        grid.level = self.level
        grid.prune = self.prune
        grid.build()

        active: ExperimentRun = ExperimentRun.get_active()
        if active:
            active.log_metric("grid_points", len(grid.coords))

        return grid.coords, grid.weights
