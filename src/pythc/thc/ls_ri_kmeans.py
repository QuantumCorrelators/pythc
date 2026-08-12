import logging

import numpy as np
from pyscf import gto
from pyscf.dft import gen_grid, treutler_prune
from scipy.spatial import KDTree
from sklearn.cluster import KMeans

import pythc.lib as lib
from pythc.grid import GridProvider
from pythc.thc.ls_ri_thc import LS_RI_THC
from pythc.thc.ls_thc_funcs import eval_basefuncs
from pythc.thc.thc_base import Mode
from pythc.tracking.experiment_run import ExperimentRun

logger = logging.getLogger()

class LS_RI_KMeans(LS_RI_THC):
    """
    Implements a K-Means clustering strategy for LSTHC grid pruning following
    Lee, Lin, Head-Gordon (2020 - DOI: 10.1021/acs.jctc.9b00820) and Dong, Hu,
    Lin (2017 - DOI: 10.48550/arXiv.1711.01531).


    This class selects a minimal set of interpolation points (grid points) by
    clustering the standard DFT integration grid. The clustering is weighted by
    the local orbital co-density to ensure points are concentrated in physically
    important regions of the molecule. The target number of points is determined
    by a multiple of the auxiliary basis set size (`ips_per_naux`).
    """
    def __init__(self,
                 mol: gto.Mole,
                 auxbasis: str,
                 mo_coeff: np.ndarray = None,
                 grid: GridProvider = None,
                 ips_per_naux = None):
        super().__init__(mol=mol, auxbasis=auxbasis, grid=grid, mo_coeff=mo_coeff)
        self.ips_per_naux = ips_per_naux if ips_per_naux else 2.5

    @classmethod
    def __str__(cls):
        return "isdf_kmeans"

    def build_pruned_X(self, mode: Mode, mo_coeff: np.ndarray, auxmol) -> np.ndarray:
        """
        Constructs the pruned collocation matrix (X) using density-weighted K-Means.

        Workflow summary:
        1. Base Grid Generation: Generates standard atom-partitioned DFT grids
           (with Treutler pruning) and computes standard integration and partition weights.
        2. Target Point Allocation: Calculates the total target number of interpolation
           points (N_aux * ips_per_naux) and distributes them across atoms proportionally
           based on their initial grid sizes.
        3. Density Weighting: For each atom, calculates a local density metric to
           guide the clustering:
           - 'ov' mode: Uses the product of occupied and virtual co-densities.
           - 'ao' mode: Uses the sum of squared AO collocation values.
        4. K-Means Clustering: Runs a K-Means algorithm on the physical grid coordinates
           for each atom, weighting the points by the product of their spatial partition
           weights and the computed density.
        5. Nearest-Neighbor Mapping: Uses a KDTree to snap the continuous K-Means
           cluster centers back to the nearest exact, discrete grid points.
        6. Matrix Assembly: Slices the collocation matrix using the selected indices
           and stacks them to form the final pruned matrix.

        Args:
            mo_coeff (np.ndarray): The molecular orbital coefficients.

        Returns:
            np.ndarray: The pruned collocation matrix containing only the selected
                        interpolation points.
        """
        active: ExperimentRun = ExperimentRun.get_active()
        n_aux = auxmol.nao_nr()

        g = gen_grid.Grids(self.mol)
        agt = g.gen_atomic_grids(self.mol, prune=treutler_prune, level=2)
        coords, weights = g.gen_partition(self.mol, agt, concat=False)

        for symb in agt:
            c, vol = agt[symb]
            agt[symb] = (c, np.ones_like(vol))
        _, partition_weights = g.gen_partition(self.mol, agt, concat=False)

        X = []

        total_grid_size = 0.0
        for ia, atom_coords in enumerate(coords):
            total_grid_size += len(atom_coords)

        if active: active.log_metric("grid_points", total_grid_size)
        requested_grid_size = (n_aux*self.ips_per_naux)
        if requested_grid_size>total_grid_size:
            logger.warning('Attempting to build a bigger grid than the parent. Falling back to full parent grid size')
            requested_grid_size=total_grid_size

        for aidx, (atom_coords, atom_weights, atom_p_weights) in enumerate(zip(coords, weights, partition_weights)):
            kdtree = KDTree(atom_coords)

            w = np.sqrt(np.sqrt(np.abs(atom_weights)))
            R_atm = eval_basefuncs(self.mol, atom_coords)
            X_atm = w[:, np.newaxis] * R_atm

            n_clusters = int((len(atom_coords)/total_grid_size)*requested_grid_size)

            logger.info(f"atom grid: {len(atom_coords)} -> nclusters={n_clusters}")
            kmeans = KMeans(n_clusters=n_clusters, max_iter=1_000, random_state=aidx)

            n_occ = self.mol.nelectron // 2
            if mode == 'ov' and n_occ > 0:
                X_atm_mo = X_atm @ mo_coeff
                X_occ = X_atm_mo[:, :n_occ]
                X_vir = X_atm_mo[:, n_occ:]
                density  = np.sum(X_occ ** 2, axis=1) + np.sum(X_vir ** 2, axis=1)
                #density = np.sum((X_atm @ mo_coeff) ** 2, axis=1)
            else:
                density = np.sum(X_atm ** 2, axis=1)

            cluster_weights = density

            kmeans.fit(atom_coords, sample_weight=cluster_weights)
            centers = kmeans.cluster_centers_

            dist, idxs = kdtree.query(centers, k=1, workers=lib.cpu_count())
            X.append(X_atm[idxs])

            if not hasattr(self, 'pruned_coords'):
                self.pruned_coords = []
                self.pruned_weights = []

            self.pruned_coords.append(atom_coords[idxs])
            self.pruned_weights.append(density[idxs])  # <--- ADD [idxs] HERE

        X_pruned = np.vstack(X)

        return X_pruned

