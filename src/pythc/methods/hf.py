import logging

import numpy as np
import pyscf.lib as pyscflib
from pyscf import gto, df
from pyscf.scf import ghf
from pyscf.scf.ghf import GHF
from pyscf.scf.hf import RHF
from pyscf.scf.uhf import UHF

from pythc.methods.thc_df import THCDF
from pythc.thc.thc_base import ThcEri

logger = logging.getLogger()

# J/K kernels live in THCDF (pythc.methods.thc_df), exposed as
# ``self.with_df``. The SCF classes below are a thin density-difference
# wrapper: exact DF baseline on the first cycle, THC on ``dm - ref_dm``
# afterwards, plus spin combining and the RHF/UHF/GHF Fock factors.


class THCMixin:
    """
    Thin density-difference wrapper around the THC J/K engine.

    The first SCF cycle builds an exact DF baseline; every later cycle
    evaluates THC on the density difference only and adds it to the frozen
    baseline. Pure THC (no baseline) is served by stock PySCF SCF with
    ``mf.with_df = THCDF(...)``.
    """
    def init_thc(self, mol: gto.Mole, eri_thc: ThcEri, auxbasis: str = None,
                 verbose: int = 0,
                 with_j_thc: bool = True, with_k_thc: bool = True):
        self.verbose = verbose
        self.max_cycle = 200
        self.conv_tol = 1e-9
        self.direct_scf = True
        self.auxbasis = auxbasis
        self.cycles = 0
        self.diis = pyscflib.diis.DIIS

        self.ref_dm = None
        self.ref_vhf = None

        self.df_obj = df.DF(mol)
        if self.auxbasis:
            self.df_obj.auxbasis = self.auxbasis

        # Single DF-style J/K engine. It shares the exact DF object as its
        # internal baseline, so the DF-ERI is built at most once. Flip
        # with_df.with_j_thc / .with_k_thc for THC/DF routing per term
        # (with_j_thc=False selects "only-K": J via DF/RI, K via THC).
        self.with_df = THCDF(mol, eri_thc, auxbasis,
                             baseline=self.df_obj,
                             with_j_thc=with_j_thc, with_k_thc=with_k_thc)

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()

        self.cycles += 1

        if self.ref_dm is None:
            logger.info(f"SCF Cycle {self.cycles}: exact DF baseline")
            self.ref_dm = np.asarray(dm)
            self.ref_vhf = self._exact_vhf(mol, dm, hermi)
            return self.ref_vhf

        # Density Difference Ansatz: THC sees only dm - ref_dm.
        ddm = np.asarray(dm) - self.ref_dm
        vj_thc, vk_thc = self.get_jk_thc(ddm, hermi=hermi)

        return self._combine_thc_vhf(self.ref_vhf, vj_thc, vk_thc)

    def _exact_vhf(self, mol, dm, hermi):
        raise NotImplementedError

    def _combine_thc_vhf(self, ref_vhf, vj_thc, vk_thc):
        raise NotImplementedError

    def get_jk_thc(self, dm=None, hermi=1, with_j=True, with_k=True):
        raise NotImplementedError


class THC_UHF(THCMixin, UHF):
    def __init__(self, mol: gto.Mole, eri_thc, auxbasis: str = None,
                 verbose: int = 4,
                 with_j_thc: bool = True, with_k_thc: bool = True):
        UHF.__init__(self, mol)
        self.init_thc(mol, eri_thc, auxbasis, verbose=verbose,
                      with_j_thc=with_j_thc, with_k_thc=with_k_thc)

    def _exact_vhf(self, mol, dm, hermi):
        vj, vk = self.df_obj.get_jk(dm, hermi=hermi)
        vj_tot = vj[0] + vj[1]
        return vj_tot - vk

    def _combine_thc_vhf(self, ref_vhf, vj_thc, vk_thc):
        # UHF V_eff is J - K (no 0.5 factor because dm is already split by spin)
        return ref_vhf + vj_thc - vk_thc

    def get_jk_thc(self, dm=None, hermi=1, with_j=True, with_k=True):
        if dm is None: dm = self.make_rdm1()
        dm = np.asarray(dm)
        # No uhf.get_jk module splitter exists (unlike ghf.get_jk), so the
        # one UHF-specific step lives here: J is linear, hence
        # J_tot = J(Da) + J(Db), broadcast to both spins. K is per-spin.
        *_, nspin, N1, N2 = dm.shape
        assert (nspin, N1) == (2, N2), f"UHF dm must have shape (..., 2, N, N), got {dm.shape}"
        N = N2
        vj_s, vk_s = self.with_df.get_jk(dm.reshape(-1, N, N), hermi, with_j, with_k)
        vj = vk = None
        if with_j:
            j_tot = vj_s.reshape(-1, 2, N, N).sum(axis=1)
            vj = np.stack([j_tot, j_tot], axis=1).reshape(dm.shape)
        if with_k:
            vk = vk_s.reshape(dm.shape)
        return vj, vk


class THC_RHF(THCMixin, RHF):
    def __init__(self, mol: gto.Mole, eri_thc: ThcEri, auxbasis: str = None,
                 verbose: int = 0,
                 with_j_thc: bool = True, with_k_thc: bool = True):
        RHF.__init__(self, mol)
        self.init_thc(mol, eri_thc, auxbasis, verbose=verbose,
                      with_j_thc=with_j_thc, with_k_thc=with_k_thc)

    def _exact_vhf(self, mol, dm, hermi):
        vj, vk = self.df_obj.get_jk(dm, hermi=hermi)
        return vj - vk * 0.5

    def _combine_thc_vhf(self, ref_vhf, vj_thc, vk_thc):
        return ref_vhf + vj_thc - vk_thc * 0.5

    def get_jk_thc(self, dm=None, hermi=1, with_j=True, with_k=True):
        if dm is None: dm = self.make_rdm1()
        # with_df handles arbitrary leading batch dims itself.
        return self.with_df.get_jk(np.asarray(dm), hermi, with_j, with_k)

class THC_GHF(THCMixin, GHF):
    def __init__(self, mol: gto.Mole, eri_thc: ThcEri, auxbasis: str = None,
                 verbose: int = 0,
                 with_j_thc: bool = True, with_k_thc: bool = True):
        GHF.__init__(self, mol)
        self.init_thc(mol, eri_thc, auxbasis, verbose=verbose,
                      with_j_thc=with_j_thc, with_k_thc=with_k_thc)
        # Exact baseline must understand spin-orbital (2N, 2N) densities.
        # (with_df keeps the plain DF baseline for spatial densities.)
        self.df_obj = GHF(mol).density_fit(self.auxbasis)

    def _exact_vhf(self, mol, dm, hermi):
        vj, vk = self.df_obj.get_jk(mol, dm, hermi=hermi)
        # GHF Fock in spin-orbitals, like UHF: no 0.5 (cf. RHF vj - vk*0.5).
        return vj - vk

    def get_jk_thc(self, dm=None, hermi=1, with_j=True, with_k=True):
        if dm is None: dm = self.make_rdm1()

        def get_jk_func(mol, dm, hermi, with_j, with_k, omega=None):
            # NOTE: signature must match the 6-arg jkbuild convention used by
            # ghf.get_jk (no vhfopt), like GHF.get_jk's inner jkbuild.
            # with_df flattens the (nblocks, n_dm, N, N) spin-block stacks
            # and handles complex densities itself.
            return self.with_df.get_jk(dm, hermi, with_j, with_k)

        return ghf.get_jk(self, dm, hermi, with_j, with_k, jkbuild=get_jk_func)

    def _combine_thc_vhf(self, ref_vhf, vj_thc, vk_thc):
        return ref_vhf + vj_thc - vk_thc

