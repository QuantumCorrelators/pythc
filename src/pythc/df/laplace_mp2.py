"""RI Laplace MP2 for validation of THC-Laplace MP2.

Uses PySCF density-fitting (RI) integrals and the exact same Laplace
quadrature as :mod:`pythc.methods.mp2` (minimax ``_get_scaled_legendre_roots``).

Closed-shell energy (chemist notation, ``J=(ia|jb)``, ``K=(ib|ja)``)::

    E = -jc * J + kc * K
    J = sum_v sum_ijab tau_i tau_j tau_a tau_b (ia|jb)^2
    K = sum_v sum_ijab tau_i tau_j tau_a tau_b (ia|jb)(ib|ja)

with ``tau_o[v,i] = w[v]**1/4 * exp(+t[v]*e_o[i])`` and
``tau_v[v,a] = w[v]**1/4 * exp(-t[v]*e_v[a])``.

Memory strategy (must hold the biggest Diet-GMTKN-150 species in cc-pVQZ,
81 atoms / nao=3205 / nocc=119 / naux=6842, in 700 GB):
  * 3-center RI integrals come from ``pyscf.mp.dfmp2`` (incore or H5-outcore
    selected by ``max_memory``), accessed strictly through
    ``eris.get_occ_blk`` so at most two occupied blocks are resident.
  * ``J``-only (SOS, ``kc == 0``) uses the O(ov*Naux^2) ``M = B^T diag(s) B``
    formulation with a single streaming pass over occupied blocks.
  * Full ``J+K`` forms ``(ia|jb) = B_i @ B_j.T`` per occupied-block pair
    (upper triangle, doubled off-diagonal) and Laplace-contracts on the fly,
    so the only large resident is one ``(bi*v) x (bj*v)`` block.

Not hyper-optimized; validation only.
"""

import gc
import logging
import math
from typing import Optional

import numpy as np
from pyscf import df
from pyscf.gw.gw_ac import _get_scaled_legendre_roots
from pyscf.mp import dfmp2, dfump2
from pyscf.scf.hf import SCF

from pythc._tracking import Configurable, observe
import pythc.lib as lib

logger = logging.getLogger(__name__)

_FULL_BLOCK_CAP = 8
_J_ONLY_BLOCK_CAP = 32


def _laplace_taus(e_o: np.ndarray, e_v: np.ndarray, n_laplace: int):
    """Exact same grid/factors as methods/mp2.py."""
    t, w = _get_scaled_legendre_roots(n_laplace)
    t = np.asarray(t, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    wq = w[:, None] ** 0.25
    tau_o = wq * np.exp(t[:, None] * e_o[None, :])
    tau_v = wq * np.exp(-t[:, None] * e_v[None, :])
    return tau_o, tau_v


def _budget_bytes(fraction: float = 0.5) -> float:
    try:
        avail = lib.available_memory_bytes()
    except Exception:  # pragma: no cover
        import psutil
        avail = psutil.virtual_memory().available
    return max(1 << 30, avail * fraction)


def _choose_full_blksize(nocc: int, nvirt: int, naux: int,
                         budget: Optional[float] = None) -> int:
    if budget is None:
        budget = _budget_bytes(0.5)
    a = float(nvirt * nvirt * 8)
    b = float(2 * nvirt * naux * 8)
    disc = b * b + 4.0 * a * budget
    t = int((-b + math.sqrt(disc)) / (2.0 * a)) if a > 0 else nocc
    return max(1, min(nocc, t, _FULL_BLOCK_CAP))


def _make_rmp2(mf: SCF, auxbasis: Optional[str] = None):
    mp = dfmp2.DFRMP2(mf)
    if auxbasis is not None:
        mp.with_df = df.DF(mf.mol, auxbasis=auxbasis)
    try:
        avail_mb = _budget_bytes(0.9) / 1e6
        if avail_mb > mp.max_memory:
            mp.max_memory = float(avail_mb)
    except Exception:
        pass
    return mp


def _ri_j_only(eris, tau_o: np.ndarray, tau_v: np.ndarray,
               occ_blksize: Optional[int] = None) -> float:
    nocc = eris.nocc
    nvirt = eris.nvir
    naux = eris.naux
    nlap = tau_o.shape[0]
    if occ_blksize is None:
        budget = _budget_bytes(0.5)
        # M block (L,Naux,Naux) + one B block
        m_bytes = nlap * naux * naux * 8
        per_o = nvirt * naux * 8
        occ_blksize = max(1, min(nocc, _J_ONLY_BLOCK_CAP,
                                int((budget - m_bytes) // per_o) if per_o else nocc))
        occ_blksize = max(1, occ_blksize)
    logger.info("RI-Laplace J-only: occ=%d virt=%d aux=%d lap=%d blk=%d",
                nocc, nvirt, naux, nlap, occ_blksize)
    M = np.zeros((nlap, naux, naux), dtype=np.float64)
    for i0 in range(0, nocc, occ_blksize):
        i1 = min(i0 + occ_blksize, nocc)
        Bb = np.asarray(eris.get_occ_blk(i0, i1), order="C").reshape((i1 - i0) * nvirt, naux)
        for v in range(nlap):
            s = (tau_o[v, i0:i1, None] * tau_v[v, None, :]).reshape(-1)
            Bs = Bb * s[:, None]
            M[v] += Bs.T @ Bb
        del Bb
        gc.collect()
    j = 0.0
    for v in range(nlap):
        j += float(np.dot(M[v].ravel(), M[v].ravel()))
    del M
    gc.collect()
    return j


def _ri_jk_full(eris, tau_o: np.ndarray, tau_v: np.ndarray,
                occ_blksize: Optional[int] = None) -> tuple[float, float]:
    nocc = eris.nocc
    nvirt = eris.nvir
    naux = eris.naux
    nlap = tau_o.shape[0]
    if occ_blksize is None:
        occ_blksize = _choose_full_blksize(nocc, nvirt, naux)
    logger.info("RI-Laplace full: occ=%d virt=%d aux=%d lap=%d blk=%d",
                nocc, nvirt, naux, nlap, occ_blksize)
    # Virtual outer products per Laplace point: (L,V,V), 760 MB for V=3086/L=10.
    Tv = np.empty((nlap, nvirt, nvirt), dtype=np.float64)
    for v in range(nlap):
        tv = tau_v[v]
        Tv[v] = tv[:, None] * tv[None, :]
    blocks = [(i, min(i + occ_blksize, nocc)) for i in range(0, nocc, occ_blksize)]
    J = 0.0
    K = 0.0
    for bi, (i0, i1) in enumerate(blocks):
        Bi = np.asarray(eris.get_occ_blk(i0, i1), order="C").reshape((i1 - i0) * nvirt, naux)
        nbi = i1 - i0
        for bj, (j0, j1) in enumerate(blocks):
            if bj < bi:
                continue
            nbj = j1 - j0
            if bj == bi:
                Bj = Bi
            else:
                Bj = np.asarray(eris.get_occ_blk(j0, j1), order="C").reshape(nbj * nvirt, naux)
            M = Bi @ Bj.T
            M4 = M.reshape(nbi, nvirt, nbj, nvirt)
            factor = 1.0 if bj == bi else 2.0
            for ii in range(nbi):
                toi = tau_o[:, i0 + ii]
                for jj in range(nbj):
                    Mij = np.ascontiguousarray(M4[ii, :, jj, :])
                    toj = tau_o[:, j0 + jj]
                    j_loc, k_loc = _ri_jk_pair_contrib(
                        Mij, toi, toj, Tv, do_k=True)
                    J += factor * j_loc
                    K += factor * k_loc
                    del Mij
            del M, M4
            if bj != bi:
                del Bj
            gc.collect()
        del Bi
        gc.collect()
    del Tv
    gc.collect()
    return J, K


def _ri_jk_pair_contrib(Mij: np.ndarray, toi_v: np.ndarray, toj_v: np.ndarray,
                       Tv: np.ndarray, do_k: bool = True) -> tuple[float, float]:
    Mij2 = Mij * Mij
    Kmat = Mij * Mij.T if do_k else None
    j = k = 0.0
    for v in range(Tv.shape[0]):
        w = float(toi_v[v] * toj_v[v])
        j += w * float(np.dot(Mij2.ravel(), Tv[v].ravel()))
        if do_k:
            k += w * float(np.dot(Kmat.ravel(), Tv[v].ravel()))
    return j, k


def mp2_energy_ri_laplace(mol, mf: SCF, n_laplace: int = 10,
                          jc: float = 2.0, kc: float = 1.0,
                          auxbasis: Optional[str] = None,
                          occ_blksize: Optional[int] = None) -> tuple[float, ...]:
    """Closed-shell RI Laplace MP2. Returns (e_corr, J, K)."""
    assert mol.spin == 0, "closed-shell driver requires singlet"
    nocc = mol.nelectron // 2
    e = np.asarray(mf.mo_energy, dtype=np.float64)
    e_o, e_v = e[:nocc], e[nocc:]
    tau_o, tau_v = _laplace_taus(e_o, e_v, n_laplace)
    observe.checkpoint("ri_laplace_setup")

    mp = _make_rmp2(mf, auxbasis)
    eris = mp.ao2mo()
    assert eris.nocc == nocc, f"eris.nocc={eris.nocc} != {nocc}"
    observe.checkpoint("ri_laplace_eris")

    if kc == 0.0:
        J = _ri_j_only(eris, tau_o, tau_v, occ_blksize)
        K = 0.0
    else:
        J, K = _ri_jk_full(eris, tau_o, tau_v, occ_blksize)
    observe.checkpoint("ri_laplace_contract")

    gc.collect()
    return (-jc * J) + (kc * K), J, K


def ump2_energy_ri_laplace(mol, mf: SCF, n_laplace: int = 10,
                           aac: float = 0.5, bbc: float = 0.5, abc: float = 1.0,
                           auxbasis: Optional[str] = None) -> tuple[float, ...]:
    """Unrestricted RI Laplace MP2. Small-system incore path (no big blocking)."""
    mo_e = np.asarray(mf.mo_energy, dtype=np.float64)
    assert mo_e.ndim == 2 and mo_e.shape[0] == 2
    nocc_a, nocc_b = int(mol.nelec[0]), int(mol.nelec[1])
    eoa, eva = mo_e[0, :nocc_a], mo_e[0, nocc_a:]
    eob, evb = mo_e[1, :nocc_b], mo_e[1, nocc_b:]
    tao_o, tav_a = _laplace_taus(eoa, eva, n_laplace)
    tbo_o, tbv_b = _laplace_taus(eob, evb, n_laplace)

    mp = dfump2.DFUMP2(mf)
    if auxbasis is not None:
        mp.with_df = df.DF(mol, auxbasis=auxbasis)
    eris = mp.ao2mo()

    def _get_blk(s, i0, i1):
        try:
            return np.asarray(eris.get_occ_blk(s, i0, i1), order="C")
        except TypeError:
            return np.asarray(eris.ovL[s][i0 * (eris.nvir[s]):i1 * (eris.nvir[s])],
                              order="C")

    nva = eris.nvir[0] if nocc_a else 0
    nvb = eris.nvir[1] if nocc_b else 0
    Ba = _get_blk(0, 0, nocc_a).reshape(nocc_a, nva, eris.naux) \
        if nocc_a else np.zeros((0, max(nva, 0), eris.naux))
    Bb = _get_blk(1, 0, nocc_b).reshape(nocc_b, nvb, eris.naux) \
        if nocc_b else np.zeros((0, max(nvb, 0), eris.naux))

    def pair_jk(Bi, Bj, toi_v, toj_v, tv_i, tv_j, do_k: bool):
        M = Bi @ Bj.T
        M2 = M * M
        j = k = 0.0
        for v in range(n_laplace):
            outer = tv_i[v][:, None] * tv_j[v][None, :]
            w = toi_v[v] * toj_v[v]
            j += w * float(np.dot(M2.ravel(), outer.ravel()))
            if do_k:
                k += w * float(np.dot((M * M.T).ravel(), outer.ravel()))
        return j, k

    Jaa = Kaa = Jbb = Kbb = Jab = 0.0
    if nocc_a and aac != 0:
        for i in range(nocc_a):
            for j in range(nocc_a):
                jj, kk = pair_jk(Ba[i], Ba[j], tao_o[:, i], tao_o[:, j],
                                 tav_a, tav_a, True)
                Jaa += jj
                Kaa += kk
    if nocc_b and bbc != 0:
        for i in range(nocc_b):
            for j in range(nocc_b):
                jj, kk = pair_jk(Bb[i], Bb[j], tbo_o[:, i], tbo_o[:, j],
                                 tbv_b, tbv_b, True)
                Jbb += jj
                Kbb += kk
    if nocc_a and nocc_b and abc != 0:
        for i in range(nocc_a):
            for j in range(nocc_b):
                jj, _ = pair_jk(Ba[i], Bb[j], tao_o[:, i], tbo_o[:, j],
                                tav_a, tbv_b, False)
                Jab += jj
    Eaa = aac * (-Jaa + Kaa)
    Ebb = bbc * (-Jbb + Kbb)
    Eab = -abc * Jab
    logger.info(f"E_aa: {Eaa} + E_bb: {Ebb} + E_ab: {Eab} = {Eaa + Ebb + Eab}")
    return Eaa + Ebb + Eab, Jaa, Jab, Jbb, Kaa, Kbb


class LaplaceDFMP2(Configurable):
    """Closed-shell RI Laplace MP2, interface mirroring methods/mp2.LaplaceMP2."""

    def __init__(self, mf: SCF, n_laplace: int = 10, auxbasis: Optional[str] = None,
                 jc: float = 2.0, kc: float = 1.0,
                 occ_blksize: Optional[int] = None):
        self.mf = mf
        self.mol = mf.mol
        self.n_laplace = n_laplace
        self.auxbasis = auxbasis
        self.jc, self.kc = jc, kc
        self.occ_blksize = occ_blksize

    def kernel(self) -> tuple[float, ...]:
        return mp2_energy_ri_laplace(self.mol, self.mf, self.n_laplace,
                                     jc=self.jc, kc=self.kc,
                                     auxbasis=self.auxbasis,
                                     occ_blksize=self.occ_blksize)


class LaplaceDFSOSMP2(LaplaceDFMP2):
    def __init__(self, mf: SCF, n_laplace: int = 10, auxbasis: Optional[str] = None,
                 occ_blksize: Optional[int] = None):
        super().__init__(mf, n_laplace, auxbasis, jc=1.3, kc=0.0,
                         occ_blksize=occ_blksize)


class LaplaceDFUMP2(Configurable):
    def __init__(self, mf: SCF, n_laplace: int = 10, auxbasis: Optional[str] = None):
        self.mf = mf
        self.mol = mf.mol
        self.n_laplace = n_laplace
        self.auxbasis = auxbasis

    def kernel(self) -> tuple[float, ...]:
        return ump2_energy_ri_laplace(self.mol, self.mf, self.n_laplace,
                                      auxbasis=self.auxbasis)


class LaplaceDFSOSUMP2(Configurable):
    def __init__(self, mf: SCF, n_laplace: int = 10, auxbasis: Optional[str] = None):
        self.mf = mf
        self.mol = mf.mol
        self.n_laplace = n_laplace
        self.auxbasis = auxbasis

    def kernel(self) -> tuple[float, ...]:
        return ump2_energy_ri_laplace(self.mol, self.mf, self.n_laplace,
                                      aac=0.0, bbc=0.0, abc=1.3,
                                      auxbasis=self.auxbasis)
