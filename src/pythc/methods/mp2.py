import gc
import logging
import math
from typing import Optional

from numba import njit, prange
from pyscf import gto
from pyscf.gto import Mole
from pyscf.gw.gw_ac import _get_scaled_legendre_roots
from pyscf.scf.hf import SCF

import pythc.lib as lib
from pythc.thc.ls_ri_cholesky import LS_RI_Cholesky
from pythc.thc.ls_snri_cholesky import LS_snRI_Cholesky
from pythc.thc.thc_base import ERI, ThcEri, THC
from pythc.thc.thc_base import ThcEriUnrestricted
from pythc import observe

if lib.has_cuda_gpu():
    import cupy as xp
else:
    import numpy as xp

logger = logging.getLogger()


def calculate_optimal_tile_size(a, b, cache_fraction: float = 0.85) -> int:
    """Calculates the optimal grid tile size for the MP2 J term using the quadratic footprint model."""
    max_memory = lib.pyscf_max_memory() * 1024 * 1024

    # Target memory limit
    L = max_memory * cache_fraction

    # Quadratic coefficients: a*T^2 + b*T + c = 0
    c = -L

    # Solve for T
    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        return 64  # Failsafe

    T = (-b + math.sqrt(discriminant)) / (2 * a)

    # Round down to the nearest multiple of 64 for optimal cache line alignment
    T_aligned = int(T // 64) * 64

    # Keep within reasonable bounds (min 64, max 2048)
    return max(64, T_aligned)



def _calculate_ump2_J_mixed(
        tau_o_a, tau_v_a, X_o_a, X_v_a,
        tau_o_b, tau_v_b, X_o_b, X_v_b,
        Z
) -> float:
    n_lapl = tau_o_a.shape[0]
    n_grid_a = Z.shape[0]
    n_grid_b = Z.shape[1]

    mp2_J = 0.0

    E_a = xp.empty((n_grid_a, n_grid_a))
    E_b = xp.empty((n_grid_b, n_grid_b))
    W_a = xp.empty((n_grid_a, n_grid_b))
    W_b = xp.empty((n_grid_b, n_grid_a))

    for v in range(n_lapl):
        X_o_tau_a = X_o_a * tau_o_a[v][None, :]
        X_v_tau_a = X_v_a * tau_v_a[v][None, :]
        X_o_tau_b = X_o_b * tau_o_b[v][None, :]
        X_v_tau_b = X_v_b * tau_v_b[v][None, :]

        A_a = xp.dot(X_o_tau_a, X_o_a.T)
        B_a = xp.dot(X_v_tau_a, X_v_a.T)
        xp.multiply(A_a, B_a, out=E_a)

        A_b = xp.dot(X_o_tau_b, X_o_b.T)
        B_b = xp.dot(X_v_tau_b, X_v_b.T)
        xp.multiply(A_b, B_b, out=E_b)

        xp.dot(E_a, Z, out=W_a)
        xp.dot(E_b, Z.T, out=W_b)

        laplace_J = xp.sum(W_a * W_b.T)
        mp2_J += laplace_J

    return mp2_J


def _contract_B_chunked(Z, X_o, X_v, B, max_chunk_bytes=2 * 1024**3):
    """Compute B[P,j,b] = Σ_Q Z[P,Q] * X_o[Q,j] * X_v[Q,b] with bounded intermediates.

    Chunks over the occupied index to avoid allocating the full (N_grid, N_occ, N_virt)
    intermediate.  Peak temporary memory: O(j_chunk × N_grid × N_virt).
    """
    n_grid = Z.shape[0]
    n_occ = X_o.shape[1]
    n_virt = X_v.shape[1]
    itemsize = X_o.dtype.itemsize

    # bytes for one occupied orbital's intermediate: (N_grid × N_virt) × itemsize
    bytes_per_j = n_grid * n_virt * itemsize
    j_chunk = max(1, min(n_occ, int(max_chunk_bytes / bytes_per_j)))

    for j0 in range(0, n_occ, j_chunk):
        j1 = min(j0 + j_chunk, n_occ)
        # intermediate: (n_grid, chunk_j, n_virt) — bounded by max_chunk_bytes
        X_sub = X_o[:, j0:j1, None] * X_v[:, None, :]
        X_sub_2d = X_sub.reshape(n_grid, -1)
        B[:, j0:j1, :] = (Z @ X_sub_2d).reshape(n_grid, j1 - j0, n_virt)
        del X_sub, X_sub_2d


@njit(parallel=True, fastmath=True, cache=True)
def _numba_accumulate_tile(tau_slice, E_tile, S_tile):
    dp, dr, dj = E_tile.shape
    acc = 0.0
    BLK = 32

    n_blocks_p = (dp + BLK - 1) // BLK
    n_blocks_r = (dr + BLK - 1) // BLK
    total_blocks = n_blocks_p * n_blocks_r

    for b in prange(total_blocks):
        bp = b // n_blocks_r
        br = b % n_blocks_r

        p0 = bp * BLK
        p1 = min(p0 + BLK, dp)
        r0 = br * BLK
        r1 = min(r0 + BLK, dr)

        for p in range(p0, p1):
            for r in range(r0, r1):
                h_val = 0.0
                for j in range(dj):
                    h_val += E_tile[p, r, j] * E_tile[r, p, j] * tau_slice[j]

                acc += h_val * S_tile[p, r]

    return acc


# Optimized to handle the Cross-Term (E_pr and E_rp)
@njit(parallel=True, fastmath=True, cache=True)
def _numba_accumulate_cross_tiles(tau_slice, E_pr, E_rp, S_pr):
    dp, dr, dj = E_pr.shape
    acc = 0.0
    BLK = 32

    n_blocks_p = (dp + BLK - 1) // BLK
    n_blocks_r = (dr + BLK - 1) // BLK
    total_blocks = n_blocks_p * n_blocks_r

    for b in prange(total_blocks):
        bp = b // n_blocks_r
        br = b % n_blocks_r

        p0 = bp * BLK
        p1 = min(p0 + BLK, dp)
        r0 = br * BLK
        r1 = min(r0 + BLK, dr)

        for p in range(p0, p1):
            for r in range(r0, r1):
                h_val = 0.0
                for j in range(dj):
                    h_val += E_pr[p, r, j] * E_rp[r, p, j] * tau_slice[j]

                acc += h_val * S_pr[p, r]

    return acc


def _build_B(Z, X_o, X_v):
    n_grid = Z.shape[0]
    n_occ = X_o.shape[1]
    n_virt = X_v.shape[1]

    B = xp.empty((n_grid, n_occ, n_virt), dtype=X_o.dtype)
    _contract_B_chunked(Z, X_o, X_v, B)

    return B


# ============================================================================
# Optimized MP2 J/K: memory-safe, symmetry-exploiting, BLAS-optimized
# ============================================================================
def _tile_size_from_budget(budget_bytes, n_j, itemsize):
    """Compute the largest tile T such that tile-pair memory fits within budget.

    Per tile pair, peak concurrent memory is:
        E_pr_buf + E_rp_buf + E_flat + E_rp_flat + S_tile
        = T^2 * (4*n_j + 1) * itemsize
    """
    a_coeff = (4 * n_j + 1) * itemsize
    if a_coeff <= 0 or budget_bytes <= 0:
        return 64
    T = math.sqrt(budget_bytes / a_coeff)
    T_aligned = int(T // 64) * 64
    return max(64, min(T_aligned, 16384))


def _calculate_mp2_J_optimized(tau_o, tau_v, X_o, X_v, Z):
    """Performance-optimized MP2 J-term with pre-allocated buffers and symmetry.

    Algorithm (per Laplace point v):
        E[P,R] = (X_o_tau @ X_o^T) * (X_v_tau @ X_v^T)  -- tiled, upper-triangle only
        W = E @ Z                                         -- BLAS GEMM (dominant cost)
        J(v) = sum W[P,R] * W[R,P] = trace(W^2)          -- einsum, no temporaries

    Symmetry: E[P,R] = E[R,P]^T -- only upper triangle computed, lower mirrored.
    Memory: 2 x (N_grid^2) persistent buffers + O(tile^2) temporaries.
    """
    n_grid, n_occ = X_o.shape
    n_virt = X_v.shape[1]
    n_lapl = tau_o.shape[0]

    GRID_TILE = calculate_optimal_tile_size(24.0, 16.0 * n_virt * n_occ)
    logger.info("J-optimized: grid=%d, occ=%d, virt=%d, tile=%d",
                n_grid, n_occ, n_virt, GRID_TILE)

    E = xp.empty((n_grid, n_grid))
    W = xp.empty((n_grid, n_grid))

    mp2_J = 0.0

    for v in range(n_lapl):
        logger.info("\tJ Laplace point %d/%d", v + 1, n_lapl)

        X_o_tau = X_o * tau_o[v][None, :]
        X_v_tau = X_v * tau_v[v][None, :]

        # Build E -- upper triangle only, mirror lower
        for p0 in range(0, n_grid, GRID_TILE):
            p1 = min(p0 + GRID_TILE, n_grid)
            for r0 in range(p0, n_grid, GRID_TILE):
                r1 = min(r0 + GRID_TILE, n_grid)

                A_tile = xp.dot(X_o_tau[p0:p1], X_o[r0:r1].T)
                B_tile = xp.dot(X_v_tau[p0:p1], X_v[r0:r1].T)
                xp.multiply(A_tile, B_tile, out=A_tile)
                E[p0:p1, r0:r1] = A_tile

                if p0 != r0:
                    E[r0:r1, p0:p1] = A_tile.T

        # W = E @ Z  -- dominant O(N^3) BLAS GEMM
        xp.dot(E, Z, out=W)

        # trace(W^2) without N^2-sized temporary
        laplace_J = float(xp.einsum('ij,ji->', W, W))
        mp2_J += laplace_J

    # Explicitly release N_grid^2 buffers before K starts
    del E, W
    gc.collect()

    return mp2_J


def _build_B_block(Z, X_o_block, X_v):
    """Pre-compute B for a block of occupied orbitals (weight-independent).

    B_block[P,j,b] = sum_Q Z[P,Q] * X_o[Q,j] * X_v[Q,b]

    Uses chunked construction to avoid (N_grid × block_j × N_virt) intermediate.
    """
    n_grid = Z.shape[0]
    block_j = X_o_block.shape[1]
    n_virt = X_v.shape[1]

    B_block = xp.empty((n_grid, block_j, n_virt), dtype=X_o_block.dtype)
    _contract_B_chunked(Z, X_o_block, X_v, B_block)

    return B_block


def _mp2_K_tile_loop(B_data, tau_o_slice, X_o_tau, X_o, X_v_tau_T,
                      n_grid, GRID_TILE, E_pr_buf, E_rp_buf):
    """Accumulate K from B data across all grid tile pairs.

    Upper-triangle symmetry: diagonal tiles once, off-diagonal x 2.
    """
    n_j = B_data.shape[1]
    n_virt = B_data.shape[2]

    k_acc = 0.0

    for p0 in range(0, n_grid, GRID_TILE):
        p1 = min(p0 + GRID_TILE, n_grid)
        dp = p1 - p0

        for r0 in range(p0, n_grid, GRID_TILE):
            r1 = min(r0 + GRID_TILE, n_grid)
            dr = r1 - r0

            S_tile = xp.dot(X_o_tau[p0:p1], X_o[r0:r1].T)

            B_flat = B_data[p0:p1].reshape(dp * n_j, n_virt)
            E_flat = xp.dot(B_flat, X_v_tau_T[:, r0:r1])
            E_pr_view = E_flat.reshape(dp, n_j, dr).transpose(0, 2, 1)
            E_pr_tile = E_pr_buf[:dp, :dr, :n_j]
            E_pr_tile[:] = E_pr_view

            if p0 == r0:
                term = _numba_accumulate_tile(tau_o_slice, E_pr_tile, S_tile)
                k_acc += term
            else:
                B_r_flat = B_data[r0:r1].reshape(dr * n_j, n_virt)
                E_rp_flat = xp.dot(B_r_flat, X_v_tau_T[:, p0:p1])
                E_rp_view = E_rp_flat.reshape(dr, n_j, dp).transpose(0, 2, 1)
                E_rp_tile = E_rp_buf[:dr, :dp, :n_j]
                E_rp_tile[:] = E_rp_view

                term = _numba_accumulate_cross_tiles(
                    tau_o_slice, E_pr_tile, E_rp_tile, S_tile)
                k_acc += 2 * term

    return k_acc


def _calculate_mp2_K_optimized(tau_o, tau_v, X_o, X_v, Z):
    """Memory-safe, performance-optimized MP2 K-term with adaptive B strategy.

    Pre-computes B[P,j,b] = sum_Q Z[P,Q]*X_o[Q,j]*X_v[Q,b] (weight-independent).
    Full B if memory allows (including tile buffers), blocked over occupied index
    otherwise.

    Memory accounting:
        - B storage:       N_grid × n_j × N_virt × itemsize
        - B construction:  +O(N_grid × N_virt) temporary (chunked)
        - Tile buffers:    T^2 × (4*n_j + 1) × itemsize  (E_pr, E_rp, E_flat, S)
        - Per-Laplace:     N_grid × (N_occ + N_virt) × itemsize
    """
    n_grid, n_occ = X_o.shape
    n_virt = X_v.shape[1]
    itemsize = X_o.dtype.itemsize

    available = lib.available_memory_bytes()
    B_full_bytes = n_grid * n_occ * n_virt * itemsize

    # Per-Laplace overhead: X_o_tau + X_v_tau_T (always needed)
    per_laplace = n_grid * (n_occ + n_virt) * itemsize
    # Construction intermediate (chunked — bounded)
    construct_overhead = min(2 * 1024**3, n_grid * n_virt * itemsize)

    # Budget available for B + tile buffers (leave 15% safety margin)
    budget = available * 0.85 - per_laplace - construct_overhead
    if budget < 0:
        budget = available * 0.5  # Extreme fallback

    logger.info("K-optimized: available=%.2f GB, budget=%.2f GB, "
                "full_B=%.2f GB, grid=%d, occ=%d, virt=%d",
                available / 1e9, budget / 1e9, B_full_bytes / 1e9,
                n_grid, n_occ, n_virt)

    # Check if full B + reasonable tile buffers fit
    # After B is allocated, remaining budget for tiles:
    remaining_after_B = budget - B_full_bytes
    if remaining_after_B > 0:
        tile_full = _tile_size_from_budget(remaining_after_B, n_occ, itemsize)
    else:
        tile_full = 0

    if tile_full >= 64:
        logger.info("K-optimized: full B path. B=%.2f GB, tile=%d",
                     B_full_bytes / 1e9, tile_full)
        return _mp2_K_full_B(tau_o, tau_v, X_o, X_v, Z, tile_full)
    else:
        # Blocked path: jointly solve for j_block and tile
        # Split budget: 55% for B, 45% for tile buffers
        b_budget = budget * 0.55
        t_budget = budget * 0.45

        bytes_per_j_storage = n_grid * n_virt * itemsize
        j_block_size = max(1, min(n_occ, int(b_budget / bytes_per_j_storage)))

        tile_blocked = _tile_size_from_budget(t_budget, j_block_size, itemsize)

        # If tile is still too small, reduce j_block further
        while tile_blocked < 64 and j_block_size > 1:
            j_block_size = max(1, j_block_size // 2)
            b_actual = j_block_size * bytes_per_j_storage
            t_budget_adj = budget - b_actual
            tile_blocked = _tile_size_from_budget(t_budget_adj, j_block_size, itemsize)

        n_blocks = (n_occ + j_block_size - 1) // j_block_size
        logger.info("K-optimized: blocked path. B_block=%.2f GB, "
                     "j_block=%d, %d blocks, tile=%d",
                     j_block_size * bytes_per_j_storage / 1e9,
                     j_block_size, n_blocks, tile_blocked)
        return _mp2_K_blocked(tau_o, tau_v, X_o, X_v, Z,
                              j_block_size, tile_blocked)


def _mp2_K_full_B(tau_o, tau_v, X_o, X_v, Z, GRID_TILE):
    """K-term with full B pre-computation. Used when B fits in memory."""
    n_grid, n_occ = X_o.shape
    n_virt = X_v.shape[1]
    n_lapl = tau_o.shape[0]

    logger.info("K (full B): grid=%d, occ=%d, virt=%d, tile=%d",
                n_grid, n_occ, n_virt, GRID_TILE)

    B = _build_B(Z, X_o, X_v)

    E_pr_buf = xp.empty((GRID_TILE, GRID_TILE, n_occ), dtype=B.dtype)
    E_rp_buf = xp.empty((GRID_TILE, GRID_TILE, n_occ), dtype=B.dtype)

    mp2_K = 0.0
    for v in range(n_lapl):
        logger.info("\tK Laplace point %d/%d", v + 1, n_lapl)
        X_o_tau = X_o * tau_o[v][None, :]
        X_v_tau_T = xp.ascontiguousarray((X_v * tau_v[v][None, :]).T)

        mp2_K += _mp2_K_tile_loop(
            B, tau_o[v], X_o_tau, X_o, X_v_tau_T,
            n_grid, GRID_TILE, E_pr_buf, E_rp_buf)

    return mp2_K


def _mp2_K_blocked(tau_o, tau_v, X_o, X_v, Z, j_block_size, GRID_TILE):
    """K-term with blocked B. Each block computed once, reused for all Laplace."""
    n_grid, n_occ = X_o.shape
    n_virt = X_v.shape[1]
    n_lapl = tau_o.shape[0]

    n_blocks = (n_occ + j_block_size - 1) // j_block_size
    logger.info("K (blocked): grid=%d, occ=%d, virt=%d, tile=%d, j_block=%d",
                n_grid, n_occ, n_virt, GRID_TILE, j_block_size)

    mp2_K = 0.0

    for j0 in range(0, n_occ, j_block_size):
        j1 = min(j0 + j_block_size, n_occ)
        block_j = j1 - j0
        block_idx = j0 // j_block_size + 1

        logger.info("K: building B block [%d:%d] (%d/%d)",
                     j0, j1, block_idx, n_blocks)
        B_block = _build_B_block(Z, X_o[:, j0:j1], X_v)

        E_pr_buf = xp.empty((GRID_TILE, GRID_TILE, block_j), dtype=X_o.dtype)
        E_rp_buf = xp.empty((GRID_TILE, GRID_TILE, block_j), dtype=X_o.dtype)

        for v in range(n_lapl):
            logger.info("\tK Laplace %d/%d (block %d/%d)",
                        v + 1, n_lapl, block_idx, n_blocks)
            X_o_tau = X_o * tau_o[v][None, :]
            X_v_tau_T = xp.ascontiguousarray((X_v * tau_v[v][None, :]).T)

            mp2_K += _mp2_K_tile_loop(
                B_block, tau_o[v, j0:j1], X_o_tau, X_o, X_v_tau_T,
                n_grid, GRID_TILE, E_pr_buf, E_rp_buf)

        del B_block, E_pr_buf, E_rp_buf
        gc.collect()

    return mp2_K



def get_ump2_inputs(mf: SCF, mol: Mole, n_laplace: int, thc: ThcEriUnrestricted):
    S = mol.spin
    nocc_alpha = (mol.nelectron + S) // 2
    nocc_beta = (mol.nelectron - S) // 2
    nvir_alpha = mol.nao_nr() - nocc_alpha
    nvir_beta = mol.nao_nr() - nocc_beta

    e_alpha = lib.to_backend(mf.mo_energy[0])
    e_beta = lib.to_backend(mf.mo_energy[1])

    e_alpha_o = e_alpha[:nocc_alpha]
    e_alpha_v = e_alpha[nocc_alpha:]
    e_beta_o = e_beta[:nocc_beta]
    e_beta_v = e_beta[nocc_beta:]

    tau_o_alpha = xp.zeros((n_laplace, nocc_alpha))
    tau_v_alpha = xp.zeros((n_laplace, nvir_alpha))
    tau_o_beta = xp.zeros((n_laplace, nocc_beta))
    tau_v_beta = xp.zeros((n_laplace, nvir_beta))

    t, w = lib.to_backend(_get_scaled_legendre_roots(n_laplace))
    for v in range(n_laplace):
        weight = xp.sqrt(xp.sqrt(w[v]))
        tau_o_alpha[v, :] = weight * xp.exp(+t[v] * e_alpha_o)
        tau_v_alpha[v, :] = weight * xp.exp(-t[v] * e_alpha_v)
        tau_o_beta[v, :] = weight * xp.exp(+t[v] * e_beta_o)
        tau_v_beta[v, :] = weight * xp.exp(-t[v] * e_beta_v)

    thc.to_backend()
    X_alpha, X_beta, Z_aa, Z_bb, Z_ab = thc.get_X_Z()
    X_o_alpha = X_alpha[:, :nocc_alpha]
    X_v_alpha = X_alpha[:, nocc_alpha:]
    X_o_beta = X_beta[:, :nocc_beta]
    X_v_beta = X_beta[:, nocc_beta:]

    return X_o_alpha, X_o_beta, X_v_alpha, X_v_beta, Z_aa, Z_bb, Z_ab, tau_o_alpha, tau_o_beta, tau_v_alpha, tau_v_beta, nocc_alpha, nocc_beta


def ump2_energy_laplace(mol: gto.Mole, mf: SCF, thc: ThcEriUnrestricted, n_laplace: int = 10,
                        aac = 0.5, bbc = 0.5, abc = 1.0,
                        J_calc=_calculate_mp2_J_optimized, K_calc=_calculate_mp2_K_optimized, J_calc_mixed=_calculate_ump2_J_mixed)\
        -> tuple[float,...]:

    X_o_alpha, X_o_beta, X_v_alpha, X_v_beta, Z_aa, Z_bb, Z_ab, tau_o_alpha, tau_o_beta, tau_v_alpha, tau_v_beta, nocc_alpha, nocc_beta \
        = get_ump2_inputs(mf, mol, n_laplace, thc)

    mp2_J_aa, mp2_J_ab, mp2_J_bb = 0.0, 0.0, 0.0
    mp2_K_aa, mp2_K_bb = 0.0, 0.0

    if nocc_alpha > 0 and aac != 0:
        mp2_J_aa = J_calc(tau_o_alpha, tau_v_alpha, X_o_alpha, X_v_alpha, Z_aa)
        logger.debug(f"J-alpha = {mp2_J_aa}")

        mp2_K_aa = K_calc(tau_o_alpha, tau_v_alpha, X_o_alpha, X_v_alpha, Z_aa)
        logger.debug(f"K-alpha = {mp2_K_aa}")
    else:
        mp2_J_aa = mp2_K_aa = 0.0

    if nocc_beta > 0 and bbc != 0:
        mp2_J_bb = J_calc(tau_o_beta, tau_v_beta, X_o_beta, X_v_beta, Z_bb)
        logger.debug(f"J-beta = {mp2_J_bb}")

        mp2_K_bb = K_calc(tau_o_beta, tau_v_beta, X_o_beta, X_v_beta, Z_bb)
        logger.debug(f"K-beta = {mp2_K_bb}")
    else:
        mp2_J_bb = mp2_K_bb = 0.0

    if nocc_alpha > 0 and nocc_beta > 0 and abc != 0:
        mp2_J_ab = J_calc_mixed(
            tau_o_alpha, tau_v_alpha, X_o_alpha, X_v_alpha,
            tau_o_beta, tau_v_beta, X_o_beta, X_v_beta,
            Z_ab
        )
        logger.debug(f"J-alpha/beta = {mp2_J_ab}")
    else:
        mp2_J_ab = 0.0

    E_aa = aac * (-mp2_J_aa + mp2_K_aa)
    E_bb = bbc * (-mp2_J_bb + mp2_K_bb)
    E_ab = -abc * mp2_J_ab

    logger.info(f"E_aa: {E_aa:3f} + E_bb: {E_bb:3f} + E_ab: {E_ab:3f} = {E_aa + E_bb + E_ab}")
    return E_aa + E_bb + E_ab, mp2_J_aa, mp2_J_ab, mp2_J_bb, mp2_K_aa, mp2_K_bb


def mp2_energy_laplace(mol: gto.Mole, mf: SCF, thc: ThcEri, n_laplace: int = 10, jc = 2.0, kc = 1.0, J_calc=_calculate_mp2_J_optimized,
                       K_calc=_calculate_mp2_K_optimized) -> tuple[float,...]:
    thc.to_backend() # copy to GPU is needed

    nocc = mol.nelectron // 2

    e = lib.to_backend(mf.mo_energy)
    e_o = e[:nocc]
    e_v = e[nocc:]

    t, w = lib.to_backend(_get_scaled_legendre_roots(n_laplace))

    tau_v = xp.pow(w[:, None], 0.25) * xp.exp(-t[:, None] * e_v[None, :])
    tau_o = xp.pow(w[:, None], 0.25) * xp.exp(+t[:, None] * e_o[None, :])

    X, Z = thc.get_X_Z()
    X_o = X[:, :nocc]
    X_v = X[:, nocc:]
    observe.checkpoint("mp2_laplace_setup")

    mp2_J, mp2_K = 0.0, 0.0

    if jc != 0:
        logger.info("calculating J")
        mp2_J = J_calc(tau_o, tau_v, X_o, X_v, Z)
        observe.checkpoint("mp2_laplace_J_build")
        gc.collect()

    if kc != 0:
        logger.info("calculating K")
        mp2_K = K_calc(tau_o, tau_v, X_o, X_v, Z)
        observe.checkpoint("mp2_laplace_K_build")

    observe.append_metric("mp2_e_thc_J", float(mp2_J))
    observe.append_metric("mp2_e_thc_K", float(mp2_K))

    return (-jc * mp2_J) + (kc * mp2_K), mp2_J, mp2_K


def _build_delta(e_o, e_v):
    delta = (
            e_o[:, None, None, None]
            + e_o[None, :, None, None]
            - e_v[None, None, :, None]
            - e_v[None, None, None, :]
    )

    return delta


def mp2_energy(mol: gto.Mole, mf: SCF, eri_thc: ERI) -> float:
    nocc = mol.nelectron // 2
    e = mf.mo_energy

    J, K = eri_thc.get_jk()

    e_o = e[:nocc]
    e_v = e[nocc:]

    delta = _build_delta(e_o, e_v)

    mp2_e = xp.float64(lib.einsum('ijab,ijab,ijab->',
                                  2 * J - K,
                                  J,
                                  1.0 / delta))

    return mp2_e

class MP2:
    def __init__(self, mf: SCF, thc: Optional[THC] = None):
        self.mf = mf
        self.mol = mf.mol
        self.thc = thc if thc else LS_snRI_Cholesky(self.mol, cholesky_threshold=1e-9)

    def kernel(self) -> tuple[float,...]:
        eri = self.thc.build(mode='ov')
        return (mp2_energy(self.mol, self.mf, eri),)

class LaplaceMP2:
    def __init__(self, mf: SCF, thc: Optional[THC] = None, n_laplace : int = 10, jc: Optional[float] = None, kc: Optional[float] = None):
        self.mf = mf
        self.mol = mf.mol
        self.thc = thc if thc else LS_snRI_Cholesky(self.mol, cholesky_threshold=1e-5)
        self.n_laplace = n_laplace
        self.jc, self.kc = jc, kc

    def kernel(self) -> tuple[float,...]:
        eri = self.thc.build(mode='ov')
        return mp2_energy_laplace(self.mol, self.mf, eri, self.n_laplace)

class LaplaceSOSMP2:
    def __init__(self, mf: SCF, thc: Optional[THC] = None, n_laplace : int = 10, jc: Optional[float] = None, kc: Optional[float] = None):
        self.mf = mf
        self.mol = mf.mol
        self.thc = thc if thc else LS_snRI_Cholesky(self.mol, cholesky_threshold=1e-5)
        self.n_laplace = n_laplace
        self.jc, self.kc = jc, kc

    def kernel(self) -> tuple[float,...]:
        eri = self.thc.build(mode='ov')
        return mp2_energy_laplace(self.mol, self.mf, eri, self.n_laplace, jc=1.3, kc=0.0)

class LaplaceUMP2:
    def __init__(self, mf: SCF, thc: Optional[THC] = None, n_laplace : int = 10):
        self.mf = mf
        self.mol = mf.mol
        self.thc = thc if thc else LS_RI_Cholesky(self.mol, mo_coeff=mf.mo_coeff, cholesky_threshold=1e-5)
        self.n_laplace = n_laplace

    def kernel(self) -> tuple[float,...]:
        eri = self.thc.build_unrestricted(mode='ov')
        return ump2_energy_laplace(self.mol, self.mf, eri, self.n_laplace)

class LaplaceSOSUMP2:
    def __init__(self, mf: SCF, thc: Optional[THC] = None, n_laplace : int = 10):
        self.mf = mf
        self.mol = mf.mol
        self.thc = thc if thc else LS_RI_Cholesky(self.mol, mo_coeff=mf.mo_coeff, cholesky_threshold=1e-5)
        self.n_laplace = n_laplace

    def kernel(self) -> tuple[float,...]:
        eri = self.thc.build_unrestricted(mode='ov')
        return ump2_energy_laplace(self.mol, self.mf, eri, self.n_laplace, aac=0.0, bbc=0.0, abc=1.3)
