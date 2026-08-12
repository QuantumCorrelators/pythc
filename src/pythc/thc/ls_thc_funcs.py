import logging
from typing import Optional

import numba as nb
import numpy as np
import pyscf.lib as pyscflib
from pyscf import df, gto, ao2mo

from pythc import lib
from pythc.thc.checkpoints import METRIC_INVERSION
from pythc.thc.thc_base import Mode
from pythc.tracking.experiment_run import ExperimentRun

logger = logging.getLogger()

def eval_basefuncs(mol: gto.Mole, coords: np.ndarray):
    return mol.eval_gto('GTOval_sph', coords=coords)

def get_reference_eri(mol: gto.Mole, nelec: int, mo_coeff: Optional[np.ndarray]) -> np.ndarray:
    if len(mo_coeff) > 0:
        eri = ao2mo.kernel(mol, mo_coeff, compact=False)
    else:
        eri = ao2mo.kernel(mol, np.eye(nelec), compact=False)

    return eri

@nb.njit(parallel=True, fastmath=True, cache=True)
def build_codensity_kernel_tril(X_block, n, out):
    n_grid = X_block.shape[0]

    for g in nb.prange(n_grid):
        k = 0
        for i in range(n):
            val_i = X_block[g, i]
            for j in range(i + 1):
                out[g, k] = val_i * X_block[g, j]
                k += 1


@nb.njit(parallel=True, fastmath=True, cache=True)
def build_codensity_kernel_rect(X_block, n_occ, n_vir, out):
    n_grid = X_block.shape[0]
    n_total = n_occ + n_vir

    for g in nb.prange(n_grid):
        k = 0
        for i in range(n_occ):
            val_i = X_block[g, i]
            for j in range(n_occ, n_total):
                out[g, k] = val_i * X_block[g, j]
                k += 1


@nb.jit(parallel=True, fastmath=True, cache=True)
def build_S_ov_block(X, n_occ, S):
    n_grid = X.shape[0]
    n_basis = X.shape[1]

    for ig in nb.prange(n_grid):
        dot_occ_diag = 0.0
        dot_vir_diag = 0.0

        for i in range(n_occ):
            dot_occ_diag += X[ig, i] * X[ig, i]

        for a in range(n_occ, n_basis):
            dot_vir_diag += X[ig, a] * X[ig, a]

        S[ig, ig] = dot_occ_diag * dot_vir_diag

        for jg in range(ig + 1, n_grid):
            dot_occ = 0.0
            dot_vir = 0.0

            for i in range(n_occ):
                dot_occ += X[ig, i] * X[jg, i]

            for a in range(n_occ, n_basis):
                dot_vir += X[ig, a] * X[jg, a]

            val = dot_occ * dot_vir

            S[ig, jg] = val
            S[jg, ig] = val


@nb.njit(parallel=True, fastmath=True, cache=True)
def build_S_full(X, S):
    n_grid = X.shape[0]
    n_basis = X.shape[1]

    for ig in nb.prange(n_grid):
        dot_prod_diag = 0.0
        for p in range(n_basis):
            dot_prod_diag += X[ig, p] * X[ig, p]
        S[ig, ig] = dot_prod_diag * dot_prod_diag

        for jg in range(ig + 1, n_grid):
            dot_prod = 0.0
            for q in range(n_basis):
                dot_prod += X[ig, q] * X[jg, q]

            val = dot_prod * dot_prod

            S[ig, jg] = val
            S[jg, ig] = val


def build_S(mode: Mode, X, n_occ):
    n_grid = X.shape[0]
    S = np.empty((n_grid, n_grid))
    if mode == 'ov':
        build_S_ov_block(X, n_occ, S)
    else:
        build_S_full(X, S)
    return S


def contract_codensity_full_eri(self, X, mo_coeff):
    logger.info("using full 4-index integrals for fitting matrix")
    Xs = lib.einsum("pn,pm->mnp", X, X).reshape(self.N ** 2, -1)
    eri = get_reference_eri(self.mol, self.N, mo_coeff)
    E = Xs.T @ eri @ Xs
    return E


def build_aux_coulomb_inv(auxmol):
    logger.info("Computing 2-center Coulomb metric and J^{-1/2}...")
    j2c = auxmol.intor('int2c2e', aosym='s1')
    j2c_cholesky = lib.pseudo_inv_sqrt(j2c)

    return j2c_cholesky

def build_auxmol(mol, auxbasis) -> gto.Mole:
    return df.addons.make_auxmol(mol, auxbasis=auxbasis)


def compute_ao_slices(mol, auxmol):
    n_ao = int(mol.nao_nr())
    n_aux = int(auxmol.nao_nr())

    max_mem_mb = lib.pyscf_max_memory()
    curr_mem_mb = lib.current_memory()
    live_avail_mb = max(100.0, max_mem_mb - curr_mem_mb)

    # 30% budget for V_chunk buffer
    v_budget_bytes = live_avail_mb * 0.30 * 1024.0 * 1024.0

    ao_loc = mol.ao_loc_nr()
    blocks = []
    start_shl = 0

    for shl in range(1, mol.nbas + 1):
        ao1 = ao_loc[shl]
        ao0 = ao_loc[start_shl]
        ni = int(ao1 - ao0)

        chunk_bytes = ni * n_ao * n_aux * 8

        if chunk_bytes >= v_budget_bytes:
            if shl - 1 == start_shl:
                blocks.append((start_shl, shl, ao_loc[start_shl], ao_loc[shl]))
                start_shl = shl
            else:
                blocks.append((start_shl, shl - 1, ao_loc[start_shl], ao_loc[shl - 1]))
                start_shl = shl - 1

    if start_shl < mol.nbas:
        blocks.append((start_shl, mol.nbas, ao_loc[start_shl], ao_loc[mol.nbas]))

    logger.info(f"Dynamic Blocking: V_chunk budget = {v_budget_bytes / 1e6:.0f} MB. Processing in {len(blocks)} chunks.")

    return blocks

def build_coulomb_matrix(mode: Mode, mol: gto.Mole, auxmol: gto.Mole, X, mo_coeff):
    active = ExperimentRun.get_active()
    n_occ = mol.nelectron // 2
    n_vir = mol.nao_nr() - n_occ

    S = build_S(mode, X, n_occ)
    S_inv = lib.pinv(S)
    if active: active.checkpoint(METRIC_INVERSION)

    j2c_inv = build_aux_coulomb_inv(auxmol)
    E = contract_codensity_df_eri(mode, X, mo_coeff, mol, auxmol, j2c_inv, n_occ, n_vir)

    D = E @ S_inv

    return D

def build_coulomb_matrix_asym(mode: Mode, mol: gto.Mole, auxmol: gto.Mole, X_alpha, X_beta, mo_coeff):
    active = ExperimentRun.get_active()

    S = mol.spin
    N = mol.nao_nr()
    nocc_alpha = (mol.nelectron + S) // 2
    nocc_beta = (mol.nelectron - S) // 2
    nvir_alpha = N - nocc_alpha
    nvir_beta = N - nocc_beta
    mo_coeff_alpha = mo_coeff[0]
    mo_coeff_beta = mo_coeff[1]


    S_aa = build_S(mode, X_alpha, nocc_alpha)
    S_aa_inv = lib.pinv(S_aa)

    S_bb = build_S(mode, X_beta, nocc_beta)
    S_bb_inv = lib.pinv(S_bb)
    if active: active.checkpoint(METRIC_INVERSION)

    j2c_inv = build_aux_coulomb_inv(auxmol)
    E_aa = contract_codensity_df_eri(mode, X_alpha, mo_coeff_alpha,
                                     mol, auxmol, j2c_inv,
                                     nocc_alpha, nvir_alpha)

    E_bb = contract_codensity_df_eri(mode, X_beta, mo_coeff_beta,
                                     mol, auxmol, j2c_inv,
                                     nocc_beta, nvir_beta)

    D_aa = E_aa @ S_aa_inv
    D_bb = E_bb @ S_bb_inv

    Z_aa = D_aa.T @ D_aa
    Z_bb = D_bb.T @ D_bb
    Z_ab = D_aa.T @ D_bb

    return Z_aa, Z_bb, Z_ab

def contract_codensity_sri_eri(mode: Mode, X: np.ndarray, mo_coeff: np.ndarray, auxmol, S_Lg, j2c_inv, grid, weights,):
    logger.info("building fakemol and 2c1e grid-auxiliary integrals")
    fakemol = gto.fakemol_for_charges(grid)
    int2c2e_gM = gto.intor_cross('int2c2e', fakemol, auxmol)
    B_gM = int2c2e_gM @ j2c_inv

    B_gM = B_gM * np.sqrt(weights)[:, np.newaxis]

    Y = S_Lg @ B_gM  # Shape: (num_rank, n_aux)

    return Y


def contract_codensity_df_eri(mode, X, mo_coeff, mol, auxmol, j2c_inv, n_occ, n_vir):
    logger.info("Using DIRECT density fitting with 2D Chunking and pure BLAS contraction")
    p, n = X.shape

    n_aux = auxmol.nao_nr()
    n_ao = mol.nao_nr()

    if mode == 'ao':
        X_left = X
        X_right = X
    else:
        C_occ = mo_coeff[:, :n_occ]
        C_vir = mo_coeff[:, n_occ:n_occ + n_vir]
        X_left = pyscflib.dot(X[:, :n_occ], C_occ.T)
        X_right = pyscflib.dot(X[:, n_occ:], C_vir.T)

    W = np.zeros((n_aux, p))

    for shl0, shl1, ao0, ao1 in compute_ao_slices(mol, auxmol):
        logger.info(f"Direct DF: Processing AO shells {shl0}-{shl1} / {mol.nbas} (AOs {ao0}-{ao1})")

        shls_slice = (shl0, shl1, 0, mol.nbas, 0, auxmol.nbas)

        V_chunk = df.incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s1', shls_slice=shls_slice)

        ni = ao1 - ao0
        nj = n_ao
        V_chunk = V_chunk.reshape(ni, nj, n_aux)

        W += lib.einsum('ijq,gi,gj->qg', V_chunk, X_left[:, ao0:ao1], X_right)

        del V_chunk

    logger.info("Applying inverse metric to final grid...")
    Y = pyscflib.dot(j2c_inv, W)

    return Y


