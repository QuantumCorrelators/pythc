import logging

import cotengra as ctg
import numpy as np
from pyscf import gto, df
from pyscf.scf.hf import RHF
from pyscf.scf.uhf import UHF
import pyscf.lib as pyscflib

import pythc.lib as lib
from pythc.thc.thc_base import ThcEri

logger = logging.getLogger()

class THCMixin:
    """
    Base mixin class implementing the common density difference ansatz,
    DIIS reset logic, and energy threshold tracking for THC-SCF calculations.
    """
    def init_thc(self, mol: gto.Mole, eri_thc: ThcEri, auxbasis: str = None,
                 min_exact_cycles: int = None, thc_threshold: float = None,
                 verbose: int = 0, dtype = np.float64,
                 thc_only: bool = False, use_thc_only: bool = None):
        eri_thc.to_backend()
        X, Z = eri_thc.get_X_Z()
        self.X, self.Z = X, Z

        self.verbose = verbose
        self.max_cycle = 200
        self.conv_tol = 1e-9
        self.direct_scf = True
        self.auxbasis = auxbasis
        self.cycles = 0
        self.dtype = dtype
        self.diis = pyscflib.diis.DIIS

        self.thc_only = bool(thc_only or (use_thc_only if use_thc_only is not None else False))
        self.min_exact_cycles = min_exact_cycles if min_exact_cycles is not None else 1
        self.thc_threshold = thc_threshold if thc_threshold is not None else 0.1

        if self.thc_only:
            logger.info("configured SCF runner: Pure THC mode (no exact DF cycles, no RI ERI built)")
        else:
            logger.info(f"configured SCF runner: at least {self.min_exact_cycles} exact cycles or converge to <={self.thc_threshold}")

        self.optimizer = ctg.ReusableHyperOptimizer(
            minimize='combo',
            slicing_opts={'target_size': lib.cotengra_target_size(bytes_per_float=np.dtype(dtype).itemsize)},
            progbar=False,
            max_time=10.0
        )

        N = mol.nao_nr()
        self.contract_j_tree = ctg.einsum_tree(
            'mn,Pm,Pn,PQ,Ql,Qs->ls',
            (N, N), X.shape, X.shape, Z.shape, X.shape, X.shape,
            optimize=self.optimizer
        )
        self.contract_k_tree = ctg.einsum_tree(
            'mn,Pm,Pl,PQ,Qn,Qs->ls',
            (N, N), X.shape, X.shape, Z.shape, X.shape, X.shape,
            optimize=self.optimizer
        )

        self.current_e = None
        self.last_e = None
        self.e_diff = float('inf')
        self.thc_active = self.thc_only
        self.ref_dm = None
        self.ref_vhf = None

        if not self.thc_only:
            self.df_obj = df.DF(mol)
            if self.auxbasis:
                self.df_obj.auxbasis = self.auxbasis
        else:
            self.df_obj = None

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        """Override to intercept and track the total energy."""
        e = super().energy_tot(dm, h1e, vhf)

        self.last_e = self.current_e
        self.current_e = e
        if self.last_e is not None:
            self.e_diff = abs(self.current_e - self.last_e)

        return e

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()

        self.cycles += 1

        if self.thc_only:
            vj_thc, vk_thc = self.get_jk_thc(dm, hermi=hermi)
            return self._combine_thc_vhf(0, vj_thc, vk_thc)

        # Check the exact cycle minimum and the convergence threshold
        if not self.thc_active and (self.cycles <= self.min_exact_cycles or (self.e_diff >= self.thc_threshold)):
            logger.info(f"SCF Cycle {self.cycles}: Computing DF V_HF baseline (e_diff = {self.e_diff:.4f} Eh)")

            self.ref_dm = np.asarray(dm)
            self.ref_vhf = self._exact_vhf(mol, dm, hermi)

            return self.ref_vhf

        # Switch to THC once below the threshold
        if not self.thc_active:
            logger.info(f"SCF Cycle {self.cycles}: Energy diff {self.e_diff:.4f} < {self.thc_threshold}. Switching to THC.")
            self.thc_active = True

            # Reset DIIS history due to the numerical discontinuity of the V_HF evaluation switch
            if hasattr(self, 'diis') and self.diis:
                self.diis.space = 0

        # Perform the Density Difference Ansatz
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
                 min_exact_cycles=0, thc_threshold=100,
                 verbose=4, dtype=np.float64,
                 thc_only: bool = False, use_thc_only: bool = None):
        UHF.__init__(self, mol)
        self.init_thc(mol, eri_thc, auxbasis, min_exact_cycles, thc_threshold, verbose, dtype, thc_only, use_thc_only)

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

        # UHF dm has shape (2, N, N) for a single batch
        is_batch = (dm.ndim == 4)
        if not is_batch:
            dms = dm[:, np.newaxis, ...]
        else:
            dms = dm

        vj = np.zeros_like(dms) if with_j else None
        vk = np.zeros_like(dms) if with_k else None
        X, Z = self.X, self.Z

        for i in range(dms.shape[1]):
            p_mat_a = lib.to_backend(dms[0, i])
            p_mat_b = lib.to_backend(dms[1, i])

            if with_j:
                p_mat_tot = p_mat_a + p_mat_b
                j_tot = self.contract_j_tree.contract([p_mat_tot, X, X, Z, X, X])
                if lib.has_cuda_gpu(): j_tot = j_tot.get()

                if hermi == 1:
                    j_tot = 0.5 * (j_tot + j_tot.T)

                vj[0, i] = j_tot
                vj[1, i] = j_tot

            if with_k:
                k_a = self.contract_k_tree.contract([p_mat_a, X, X, Z, X, X])
                if lib.has_cuda_gpu(): k_a = k_a.get()

                k_b = self.contract_k_tree.contract([p_mat_b, X, X, Z, X, X])
                if lib.has_cuda_gpu(): k_b = k_b.get()

                if hermi == 1:
                    k_a = 0.5 * (k_a + k_a.T)
                    k_b = 0.5 * (k_b + k_b.T)

                vk[0, i] = k_a
                vk[1, i] = k_b

        if not is_batch:
            if with_j: vj = vj[:, 0, ...]
            if with_k: vk = vk[:, 0, ...]

        return vj, vk


class THC_RHF(THCMixin, RHF):
    def __init__(self, mol: gto.Mole, eri_thc: ThcEri, auxbasis: str = None,
                 min_exact_cycles=None, thc_threshold=None,
                 verbose=0, dtype=np.float64,
                 thc_only: bool = False, use_thc_only: bool = None):
        RHF.__init__(self, mol)
        self.init_thc(mol, eri_thc, auxbasis, min_exact_cycles, thc_threshold, verbose, dtype, thc_only, use_thc_only)

    def _exact_vhf(self, mol, dm, hermi):
        vj, vk = self.df_obj.get_jk(dm, hermi=hermi)
        return vj - vk * 0.5

    def _combine_thc_vhf(self, ref_vhf, vj_thc, vk_thc):
        return ref_vhf + vj_thc - vk_thc * 0.5

    def get_jk_thc(self, dm=None, hermi=1, with_j=True, with_k=True):
        if dm is None: dm = self.make_rdm1()
        dm = np.asarray(dm)

        is_2d = (dm.ndim == 2)
        if is_2d:
            dms = dm[np.newaxis, ...]
        else:
            dms = dm

        vj = np.zeros_like(dms) if with_j else None
        vk = np.zeros_like(dms) if with_k else None
        X, Z = self.X, self.Z

        for i, p_mat in enumerate(dms):
            p_mat = lib.to_backend(p_mat)
            if with_j:
                vji = self.contract_j_tree.contract([p_mat, X, X, Z, X, X])
                if lib.has_cuda_gpu(): vji = vji.get()
                vj[i] = vji

            if with_k:
                vki = self.contract_k_tree.contract([p_mat, X, X, Z, X, X])
                if lib.has_cuda_gpu(): vki = vki.get()
                vk[i] = vki

            if with_j: vj[i] = 0.5 * (vj[i] + vj[i].T)
            if with_k: vk[i] = 0.5 * (vk[i] + vk[i].T)

        if is_2d:
            if with_j: vj = vj[0]
            if with_k: vk = vk[0]

        return vj, vk
