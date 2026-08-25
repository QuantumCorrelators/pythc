from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from pyscf import gto, dft, pbc
from pyscf.dft import treutler_prune

from pythc.configurable import Configurable
from pythc import observe


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

        observe.log_metric("grid_points", len(grid.coords))

        return grid.coords, grid.weights


class UniformGrid(GridProvider):
    def __init__(self, cell: pbc.gto.Cell, e_cut: float = 100.0, wrap_around: bool = False):
        self.cell = cell
        self.spacing = np.pi/np.sqrt(2*e_cut)
        self.wrap_around = wrap_around
        self.mesh = None

    def build(self) -> tuple[np.ndarray, np.ndarray]:
        N1 = int(np.ceil(np.linalg.norm(self.cell.a[0]) / self.spacing))
        N2 = int(np.ceil(np.linalg.norm(self.cell.a[1]) / self.spacing))
        N3 = int(np.ceil(np.linalg.norm(self.cell.a[2]) / self.spacing))

        self.mesh = [N1, N2, N3]

        return self.cell.get_uniform_grids(mesh=self.mesh, wrap_around=self.wrap_around)



