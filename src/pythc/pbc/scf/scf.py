from typing import Optional

import numpy as np
from pyscf import lib as pyscflib
from pyscf.pbc import gto
from pyscf.pbc.df.aft import _check_kpts
from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0, _format_dms, _format_jks, _format_kpts_band
from pyscf.pbc.df.fft import FFTDF
from pyscf.pbc.df.fft_jk import get_j_kpts
from pyscf.pbc.lib.kpts_helper import is_zero

from pythc.pbc.lib import get_supercell_phase, kpt_to_spc, spc_to_kpt
from pythc.pbc.thc.thc_base import THC_ERI_kpts, THC

class PBC_THC_DF(FFTDF):
    """
    Periodic Boundary Condition Density Fitting driver using THC/ISDF representations.
    """

    def __init__(self, cell: gto.Cell, thc: THC, kpts = None, k_mesh = None):
        super().__init__(cell, kpts=kpts)
        self.thc = thc
        self.k_mesh = k_mesh if k_mesh else [1,1,1]
        self.kpts = kpts if kpts is not None else cell.make_kpts(k_mesh)
        self.eri: Optional[THC_ERI_kpts] = None

    def build(self):
        if self.eri is None:
            self.eri = self.thc.build_kpts(mode="ao", kpts=self.kpts)

        return self

def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        assert omega is None
        kpts, is_single_kpt = _check_kpts(self, kpts)

        vj = vk = None
        if with_k:
            vk = get_k_kpts(self, dm, hermi, kpts, kpts_band, exxdiv)

        if with_j:
            vj = get_j_kpts(self, dm, hermi, kpts, kpts_band)

        return vj, vk

# Alias for backward compatibility
Yang_FFTISDF = PBC_THC_DF

def get_k_kpts(dfo: PBC_THC_DF, dm_kpts, hermi=1, kpts=np.zeros((1, 3)), kpts_band=None, exxdiv=None):
    """
    Compute PBC Hartree-Fock exchange matrix V_K for k-points using THC/ISDF tensors.

    Args:
        dfo: Density fitting object containing inpv_kpt and coul_kpt.
        dm_kpts: Density matrix array for k-points.
        hermi: Matrix hermiticity (1 for Hermitian).
        kpts: k-point coordinates.
        kpts_band: Band k-points (optional).
        exxdiv: Exchange divergence treatment ('ewald' or None).

    Returns:
        vk_kpts: Exchange potential matrix in AO basis.
    """
    if dfo.eri is None:
        dfo.build()

    X_kpt, Z_kpt = dfo.eri.get_X_Z()

    cell = dfo.cell
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension == 3

    # Normalize sampled k-points to (nkpts, 3). A single k-point passed as
    # (3,) is reshaped to (1, 3) so downstream phase/DM handling is consistent.
    kpts = np.asarray(kpts, dtype=float).reshape(-1, 3)
    phase = get_supercell_phase(cell, kpts)
    nspc, n_kpt = phase.shape

    dm_kpts = pyscflib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    n_set, n_kpt, n_ao = dms.shape[:3]

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    if nband != n_kpt or not np.allclose(np.asarray(kpts_band).reshape(-1, 3), kpts):
        raise NotImplementedError(
            "THC get_k_kpts only supports SCF k-points (kpts_band=None or "
            f"kpts_band==kpts). Got nband={nband} with nkpts={n_kpt}. "
            "Band-structure K matrices must use exact exchange (see "
            "THCHybridRKS/THCHybridKRKS.get_jk fallback)."
        )

    n_grid = X_kpt.shape[1]

    # Supercell Coulomb kernel
    coul_spc = kpt_to_spc(Z_kpt, phase) * np.sqrt(n_kpt)
    coul_spc = coul_spc.reshape(nspc, n_grid, n_grid)

    vk_kpts = []
    for dm_kpt in dms:
        # Projected density in k-space: rho_{IJ}^k = (1/Nk) [X_k P_k X_k^H]_{IJ}
        rho_kpt = (X_kpt @ dm_kpt @ X_kpt.conj().transpose(0, 2, 1)) / n_kpt

        # Transform to real-space supercell and transpose (I, J)
        rho_spc = kpt_to_spc(rho_kpt, phase).transpose(0, 2, 1)

        # Element-wise product in supercell space
        v_spc = coul_spc * rho_spc

        # Transform back to k-space
        v_kpt = spc_to_kpt(v_spc, phase).reshape(n_kpt, n_grid, n_grid)

        # Back-project to AO basis: V_K^k = (X_k^T v_k X_k^*)^*
        vk_kpt = X_kpt.transpose(0, 2, 1) @ v_kpt @ X_kpt.conj()
        vk_kpt = vk_kpt.conj().reshape(n_kpt, n_ao, n_ao)
        vk_kpts.append(vk_kpt)

    vk_kpts = np.asarray(vk_kpts).reshape(n_set, n_kpt, n_ao, n_ao)
    if is_zero(kpts_band):
        vk_kpts = vk_kpts.real

    if exxdiv is not None:
        assert exxdiv.lower() == "ewald"
        _ewald_exxdiv_for_G0(cell, kpts, dms, vk_kpts, kpts_band=kpts_band)

    return _format_jks(vk_kpts, dm_kpts, input_band, kpts)


