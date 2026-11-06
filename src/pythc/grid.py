from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from pyscf import gto, dft
from pyscf.dft import treutler_prune

from pythc._tracking import Configurable, observe


class GridProvider(ABC,Configurable):
    @abstractmethod
    def build(self) -> tuple[np.ndarray, np.ndarray]:
        pass

class BeckeGrid(GridProvider):
    def __repr__(self):
        return f"BeckeGrid(level={self.level})"

    def __str__(self):
        return "becke"

    def __init__(self, mol: gto.Mole, level=0, prune: Callable[..., np.ndarray] = treutler_prune,
                 atom_grid: dict | tuple | None = None):
        self.mol = mol
        self.level = level
        self.prune = prune
        # Parent (radial, angular) sizes, passed through to PySCF, e.g.
        # ``{'At': (130, 590)}`` or ``(120, 590)`` for all atoms. This is the
        # radial direction of a grid-convergence series; the angular direction
        # is controlled by ``prune`` (see :func:`zone_prune`).
        self.atom_grid = atom_grid

    def build(self):
        grid = dft.gen_grid.Grids(self.mol)
        grid.level = self.level
        grid.prune = self.prune
        if self.atom_grid is not None:
            grid.atom_grid = self.atom_grid
        grid.build()

        observe.log_metric("grid_points", len(grid.coords))

        return grid.coords, grid.weights

def zone_prune(inner: int | None = None, mid: int | None = None, outer: int | None = None,
               base: Callable[..., np.ndarray] = treutler_prune) -> Callable[..., np.ndarray]:
    """Adjustable angular pruning with per-zone Lebedev point counts.

    Splits the radial shells into the Treutler zones (inner third / middle
    sixth / outer half) and overrides the angular point count of each zone.
    ``None`` keeps the ``base`` scheme's value for that zone. This is the
    angular direction of a grid-convergence series; pair it with the parent
    grid size (``BeckeGrid(level=...)`` / ``atom_grid=...``) for the radial
    direction, e.g.::

        for n_rad in (95, 115, 130):
            for inner in (14, 50, 110, 194):
                grid = BeckeGrid(mol, level=4,  # parent radial/angular sizes
                                 prune=zone_prune(inner=inner))

    Args:
        inner: Angular points per shell for the inner (core) third.
        mid: Angular points per shell for the middle sixth.
        outer: Angular points per shell for the outer half (also acts as a
            cap when set below the parent ``n_ang``).
        base: Underlying prune scheme defining the zone split and the
            fallback values; same ``(nuc, rads, n_ang)`` signature as
            :func:`pyscf.dft.gen_grid.treutler_prune`.

    Each value is either a Lebedev point count (1, 6, 14, 26, 50, 74, 86,
    110, 146, 170, 194, ...) or a Lebedev order, which is mapped to its
    point count (e.g. order 11 -> 50 points, order 17 -> 110 points).
    Anything else snaps up to the next valid Lebedev size.
    """
    inner = _snap_angular(inner) if inner is not None else None
    mid = _snap_angular(mid) if mid is not None else None
    outer = _snap_angular(outer) if outer is not None else None

    def prune(nuc, rads, n_ang):
        counts = np.asarray(base(nuc, rads, n_ang)).copy()
        nr = len(rads)
        if inner is not None:
            counts[:nr // 3] = inner
        if mid is not None:
            counts[nr // 3:nr // 2] = mid
        if outer is not None:
            counts[nr // 2:] = outer
        return counts

    return prune


def _snap_angular(value: int) -> int:
    """Map a requested angular size onto a valid Lebedev point count."""
    from pyscf.dft.LebedevGrid import LEBEDEV_NGRID, LEBEDEV_ORDER

    counts = [int(n) for n in LEBEDEV_NGRID]
    if value in counts:
        return value
    if value in LEBEDEV_ORDER:
        return int(LEBEDEV_ORDER[value])
    if value < 1:
        raise ValueError(f"angular size must be positive, got {value}")
    larger = [n for n in counts if n >= value]
    if not larger:
        raise ValueError(f"angular size {value} exceeds largest Lebedev grid ({counts[-1]})")
    return larger[0]
