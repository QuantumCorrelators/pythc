import numpy as np
from pyscf import __config__, pbc
from pyscf.pbc import dft
from pythc.pbc.scf.df import get_k_kpts, Yang_FFTISDF
from pythc.pbc.thc.pbc_ls_ri_cholesky import PBC_LS_RI_Cholesky


def _normalize_single_kpt(kpt):
    """Normalize a single k-point to shape (3,)."""
    if kpt is None:
        return np.zeros(3)
    kpt = np.asarray(kpt, dtype=float)
    return kpt.reshape(3)


def _normalize_kpts(kpts):
    """Normalize sampled k-points to shape (nkpts, 3)."""
    if kpts is None:
        return np.zeros((1, 3))
    kpts = np.asarray(kpts, dtype=float)
    if kpts.ndim == 1:
        # Single k-point given as (3,) -> (1, 3).
        # NOTE: a Monkhorst-Pack mesh (e.g. [2, 2, 2]) must be expanded
        # explicitly via cell.make_kpts(mesh) before passing here.
        kpts = kpts.reshape(1, 3)
    return kpts.reshape(-1, 3)


def _is_band_calc(kpts_scf, kpts_band):
    """True if kpts_band requests band k-points distinct from the SCF k-points."""
    if kpts_band is None:
        return False
    band = np.asarray(kpts_band, dtype=float).reshape(-1, 3)
    scf = np.asarray(kpts_scf, dtype=float).reshape(-1, 3)
    if len(band) != len(scf):
        return True
    return not np.allclose(band, scf)


def _hermitianize(vk):
    """Symmetrize (..., nao, nao) array as Hermitian, preserving shape."""
    return 0.5 * (vk + vk.conj().swapaxes(-1, -2))


def _build_thc_dfo(cell, kpts):
    thc = PBC_LS_RI_Cholesky(cell=cell, cholesky_threshold=1e-8)
    thc_eri = thc.build_kpts(mode="ao", kpts=kpts)
    return Yang_FFTISDF(cell=cell, kpts=kpts, thc_eri=thc_eri)


class THCHybridRKS(dft.rks.RKS):
    """THC hybrid at a single k-point (Gamma-point RKS).

    Args:
        cell: PySCF PBC Cell.
        kpt: single k-point in absolute coordinates, shape (3,).
            NOTE: this is a k-point coordinate, NOT a Monkhorst-Pack mesh.
            For k-point-sampled calculations use :class:`THCHybridKRKS`
            with ``kpts=cell.make_kpts(mesh)``.
    """
    def __init__(self, cell: pbc.gto.Cell, kpt=None, xc='LDA,VWN',
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald'), dfo=None,):
        super().__init__(cell, kpt, xc, exxdiv)
        if kpt is None:
            kpt = self.kpt
        kpt = _normalize_single_kpt(kpt)
        if not np.allclose(kpt, 0):
            raise ValueError(
                f"THCHybridRKS supports the Gamma point only, got kpt={kpt}. "
                "Note kpt is a k-point coordinate in absolute units, not a "
                "Monkhorst-Pack mesh. For k-point-sampled calculations use "
                "THCHybridKRKS with kpts=cell.make_kpts(mesh), e.g. "
                "THCHybridKRKS(cell, kpts=cell.make_kpts([2, 2, 2]))."
            )
        # Keep parent's kpt in sync when explicitly given.
        self.kpt = kpt
        kpts = kpt.reshape(1, 3)
        if dfo is None:
            dfo = _build_thc_dfo(cell, kpts)

        self.dfo = dfo
        self.kpts_full = kpts
        self._keys.update(['dfo', 'kpts_full'])

    def get_jk(self, cell=None, dm=None, hermi=1, kpt=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, **kwargs):
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if kpt is None: kpt = self.kpt

        # 1. Let PySCF build J (forward k-point args so bands are correct)
        if with_j:
            vj, _ = super().get_jk(cell=cell, dm=dm, hermi=hermi, kpt=kpt,
                                   kpts_band=kpts_band, with_j=True,
                                   with_k=False, omega=omega)
        else:
            vj = None

        # 2. Compute K using THC implementation for SCF k-points,
        #    exact exchange for band k-points (THC tensors only cover SCF k-points).
        if with_k:
            kpts_scf = np.asarray(kpt, dtype=float).reshape(1, 3)
            if _is_band_calc(kpts_scf, kpts_band):
                _, vk = super().get_jk(cell=cell, dm=dm, hermi=hermi, kpt=kpt,
                                       kpts_band=kpts_band, with_j=False,
                                       with_k=True, omega=omega)
            else:
                vk = get_k_kpts(self.dfo, dm, hermi, kpts_scf, None, self.exxdiv)
                if hermi == 1:
                    vk = _hermitianize(vk)
        else:
            vk = None

        return vj, vk


class THCHybridKRKS(dft.krks.KRKS):
    """THC hybrid with k-point sampling (KRKS).

    Args:
        cell: PySCF PBC Cell.
        kpts: sampled k-points in absolute coordinates, shape (nkpts, 3).
            Build with ``cell.make_kpts(mesh)``.
    """
    def __init__(self, cell: pbc.gto.Cell, kpts=None, xc='LDA,VWN',
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald'), dfo=None,):
        super().__init__(cell, kpts, xc, exxdiv)
        if kpts is None:
            kpts = self.kpts
        kpts = _normalize_kpts(kpts)
        self.kpts = kpts
        if dfo is None:
            dfo = _build_thc_dfo(cell, kpts)

        self.dfo = dfo
        self.kpts_full = kpts
        self._keys.update(['dfo', 'kpts_full'])

    def get_jk(self, cell=None, dm_kpts=None, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, **kwargs):
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        kpts = _normalize_kpts(kpts)

        # 1. J via PySCF (forward k-point args so bands are correct)
        if with_j:
            vj, _ = super().get_jk(cell=cell, dm_kpts=dm_kpts, hermi=hermi,
                                   kpts=kpts, kpts_band=kpts_band,
                                   with_j=True, with_k=False,
                                   omega=omega, **kwargs)
        else:
            vj = None

        # 2. K via THC for SCF k-points, exact for band k-points.
        if with_k:
            if _is_band_calc(kpts, kpts_band):
                _, vk = super().get_jk(cell=cell, dm_kpts=dm_kpts, hermi=hermi,
                                       kpts=kpts, kpts_band=kpts_band,
                                       with_j=False, with_k=True,
                                       omega=omega, **kwargs)
            else:
                vk = get_k_kpts(self.dfo, dm_kpts, hermi, kpts, None, self.exxdiv)
                if hermi == 1:
                    vk = _hermitianize(vk)
        else:
            vk = None

        return vj, vk
