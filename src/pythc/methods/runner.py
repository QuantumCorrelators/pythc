import logging
import pathlib
from typing import Type, Callable

from pyscf import gto, scf
from pyscf.mp import RMP2, UMP2
from pyscf.mp.dfmp2 import DFRMP2
from pyscf.mp.dfump2 import DFUMP2

import pythc.thc.checkpoints as ch
from pythc import lib, grid
from pythc.methods.hf import THC_RHF, THC_UHF
from pythc.methods.mp2 import mp2_energy_laplace, ump2_energy_laplace, mp2_energy_sos, ump2_energy_sos
from pythc.methods.runnable import Runnable
from pythc.thc.ls_snri_cholesky import LS_snRI_Cholesky
from pythc.thc.thc_base import THC
from pythc.thc.ls_ri_cholesky import LS_RI_Cholesky
from pythc.tracking.experiment_run import ExperimentRun

logger = logging.getLogger()

class NOOP(Runnable):
    def __init__(self):
        return
    def kernel(self) -> tuple[float,...]:
        return (0,)

def build_mf(mol, auxbasis = '', chkfile = '', verbose=4):
    active: ExperimentRun = ExperimentRun.get_active()
    if mol.spin > 0:
        mf = scf.UHF(mol)
    else:
        mf = scf.RHF(mol)

    if chkfile != "":
        mf.chkfile = chkfile

    if pathlib.Path(chkfile).is_file():
        logger.info(f"taking initial guess from {chkfile}")
        data = scf.chkfile.load(chkfile, 'scf')
        mf.mo_coeff = data['mo_coeff']
        mf.mo_energy = data['mo_energy']
        mf.mo_occ = data['mo_occ']
        mf.e_tot = data['e_tot']
        mf.max_cycle = 0
        mf.converged = True
    else:
        if auxbasis != '':
            logger.info("building mean field with DF")
            mf = mf.density_fit(auxbasis=auxbasis)
            mf.with_df = lib.get_dfo(mol, auxbasis)

        mf.get_init_guess()
        mf.verbose = verbose
        mf.max_cycle = 200
        mf.max_memory = lib.pyscf_max_memory()
        mf.kernel()
        if active:
            active.log_metric('mf_converged', mf.converged)
            active.log_metric("mf_cycles", mf.cycles)


    return mf


class SCF(Runnable):
    def __init__(self, mol: gto.Mole, chkfile: str = ''):
        self.mol = mol
        self.chkfile = chkfile

    def __str__(self):
        return "scf"

    def kernel(self) -> tuple[float,...]:
        mf = build_mf(self.mol, self.chkfile)
        return (float(mf.e_tot),)

class DFSCF(Runnable):
    def __init__(self, mol: gto.Mole, auxbasis, chkfile: str = ''):
        self.mol = mol
        self.auxbasis = auxbasis
        self.chkfile = chkfile

    def __str__(self):
        return "scf_ri"

    def kernel(self) -> tuple[float,...]:
        mf = build_mf(self.mol, self.auxbasis, self.chkfile)
        return (float(mf.e_tot),)


class THCSCF(Runnable):
    @classmethod
    def with_thc(cls, thc_cls: Type[THC]) -> Callable[..., Runnable]:
        def build(cfg):
            thc = thc_cls.from_config(cfg)
            return cls.from_config(cfg | {'thc': thc})

        return build

    def __str__(self):
        return f"scf_thc_{self.thc.__str__()}"

    def __init__(self, mol, thc, auxbasis='', thc_threshold=None, min_exact_cycles=None, thc_only: bool = False, use_thc_only: bool = None, chkfile=''):
        self.mol = mol
        self.thc = thc
        self.auxbasis = auxbasis
        self.thc_threshold = thc_threshold
        self.min_exact_cycles = min_exact_cycles
        self.thc_only = bool(thc_only or (use_thc_only if use_thc_only is not None else False))
        self.chkfile = chkfile

    def kernel(self) -> tuple[float,...]:
        active = ExperimentRun.get_active()
        eri = self.thc.build(mode="ao")

        if self.mol.spin > 0:
            mf = THC_UHF(self.mol,
                         eri,
                         self.auxbasis,
                         verbose=4,
                         thc_threshold=self.thc_threshold,
                         min_exact_cycles=self.min_exact_cycles,
                         thc_only=self.thc_only)
        else:
            mf = THC_RHF(self.mol,
                         eri,
                         self.auxbasis,
                         verbose=4,
                         thc_threshold=self.thc_threshold,
                         min_exact_cycles=self.min_exact_cycles,
                         thc_only=self.thc_only)

        if self.chkfile:
            mf.chkfile = self.chkfile
            if pathlib.Path(self.chkfile).is_file():
                logger.info(f"taking initial guess from {self.chkfile}")
                data = scf.chkfile.load(self.chkfile, 'scf')
                mf.mo_coeff = data['mo_coeff']
                mf.mo_energy = data['mo_energy']
                mf.mo_occ = data['mo_occ']
                if 'e_tot' in data:
                    mf.e_tot = data['e_tot']

        mf.kernel()

        if active:
            active.log_metric('mf_converged', mf.converged)
            active.log_metric('mf_cycles', mf.cycles)

        e_scf = mf.e_tot

        if active: active.checkpoint(ch.MF_BUILD)

        return (float(e_scf),)

class MP2(Runnable):
    def __init__(self, mol, auxbasis, chkfile = ''):
        self.mol = mol
        self.auxbasis = auxbasis
        self.chkfile = chkfile

    def __str__(self):
        return "mp2_pyscf_dfmp2"

    def kernel(self) -> tuple[float,...]:
        active = ExperimentRun.get_active()
        mf = build_mf(self.mol, "", self.chkfile)

        if active: active.checkpoint(ch.MF_BUILD)

        if self.mol.spin > 0:
            e_corr = UMP2(mf).kernel(with_t2=False)
        else:
            e_corr = RMP2(mf).kernel(with_t2=False)

        if active: active.checkpoint(ch.MP2_ENERGY)

        return (float(e_corr[0]),)

class DFMP2(Runnable):
    def __init__(self, mol, auxbasis, chkfile = "", verbose=4):
        self.mol = mol
        self.auxbasis = auxbasis
        self.mf = None
        self.chkfile = chkfile
        self.verbose = verbose

    def __str__(self):
        return "mp2_pyscf_dfmp2"

    def with_mf(self, mf):
        self.mf = mf

    def kernel(self) -> tuple[float,...]:
        active = ExperimentRun.get_active()
        mf = self.mf
        if mf is None:
            logger.info("building mean field ...")
            mf = build_mf(self.mol, self.auxbasis, self.chkfile, self.verbose)
            self.mf = mf
            if active: active.checkpoint(ch.MF_BUILD)

        e_scf = mf.e_tot

        if not mf.converged:
            logger.info("MF not converged. Can not compute MP2 energy ...")
            return 0, e_scf, e_scf

        logger.info("calculating RI-MP2 ...")
        if self.mol.spin > 0:
            e_corr = DFUMP2(mf).kernel(with_t2=False)
        else:
            e_corr = DFRMP2(mf).kernel(with_t2=False)

        if active: active.checkpoint(ch.MP2_ENERGY)
        e_corr = float(e_corr[0])

        return e_corr, e_scf, e_scf + e_corr


class THCMP2_DFSCF(Runnable):
    @classmethod
    def with_thc(cls, thc_cls: Type[THC]) -> Callable[..., Runnable]:
        def build(cfg):
            thc = thc_cls.from_config(cfg)
            return cls.from_config(cfg | {'thc': thc})

        return build

    def __str__(self):
        return f"{self.thc.__str__()}"

    def __init__(self, mol, thc, auxbasis, chkfile = "", verbose=4):
        self.mol = mol
        self.thc = thc
        self.auxbasis = auxbasis
        self.mf = None
        self.chkfile = chkfile
        self.verbose = verbose

    def with_mf(self, mf):
        self.mf = mf

    def kernel(self) -> tuple[float,...]:
        active: ExperimentRun = ExperimentRun.get_active()
        logger.info("building mean field ...")
        mf = self.mf
        if mf is None:
            mf = build_mf(self.mol, self.auxbasis, self.chkfile, self.verbose)
            self.mf = mf

        e_scf = mf.e_tot

        if active: active.checkpoint(ch.MF_BUILD)

        self.thc.with_mo_coeff(mf.mo_coeff)
        if self.mol.spin > 0:
            logger.info("constructing THC ...")
            eri = self.thc.build_unrestricted(mode="ov")
            if active: active.checkpoint(ch.THC_BUILD)
            logger.info("calculating THC-RI-MP2 ...")
            e_corr = ump2_energy_laplace(self.mol, mf, eri)
        else:
            logger.info("constructing THC ...")
            eri = self.thc.build(mode="ov")
            if active: active.checkpoint(ch.THC_BUILD)
            logger.info("calculating THC-RI-MP2 ...")

            e_corr = mp2_energy_laplace(self.mol, mf, eri)

        logger.info(f"got MP2 correction: {e_corr}")

        return float(e_corr), float(e_scf), float(e_scf) + float(e_corr)


class THCMP2_SOS_THCSCF(Runnable):
    @classmethod
    def with_thc(cls, thc_cls: Type[THC]) -> Callable[..., Runnable]:
        def build(cfg):
            thc = thc_cls.from_config(cfg)
            return cls.from_config(cfg | {'thc': thc})

        return build

    def __str__(self):
        return f"{self.thc.__str__()}"

    def __init__(self, mol, thc, auxbasis='', thc_only: bool = False, use_thc_only: bool = None):
        self.mol = mol
        self.thc = thc
        self.auxbasis = auxbasis
        self.thc_only = bool(thc_only or (use_thc_only if use_thc_only is not None else False))

    def kernel(self) -> tuple[float,...]:
        scfthc = LS_snRI_Cholesky(mol=self.mol, auxbasis=self.auxbasis, cholesky_threshold=1e-8, grid=grid.BeckeGrid(self.mol, level=1))
        scf_eri = scfthc.build(mode="ao")

        active: ExperimentRun = ExperimentRun.get_active()

        if self.mol.spin > 0:
            mf = THC_UHF(self.mol, scf_eri, self.auxbasis, thc_only=self.thc_only)
            e_scf = mf.kernel()
            if not mf.converged:
                raise RuntimeError(f"SCF failed to converge after {mf.max_cycle} cycles!")

            self.thc.with_mo_coeff(mf.mo_coeff)

            eri = self.thc.build_unrestricted(mode="ov")
            if active: active.checkpoint(ch.THC_BUILD)
            e_corr = ump2_energy_laplace(self.mol, mf, eri)

        else:
            mf = THC_RHF(self.mol, scf_eri, self.auxbasis, thc_only=self.thc_only)
            e_scf = mf.kernel()
            if not mf.converged:
                raise RuntimeError(f"SCF failed to converge after {mf.max_cycle} cycles!")

            self.thc.with_mo_coeff(mf.mo_coeff)

            eri = self.thc.build(mode="ov")
            e_corr = mp2_energy_sos(self.mol, mf, eri)

        self.mf = mf
        return float(e_corr), float(e_scf), float(e_scf) + float(e_corr)

class THCMP2_THCSCF(Runnable):
    @classmethod
    def with_thc(cls, thc_cls: Type[THC]) -> Callable[..., Runnable]:
        def build(cfg):
            thc = thc_cls.from_config(cfg)
            return cls.from_config(cfg | {'thc': thc})

        return build

    def __str__(self):
        return f"{self.thc.__str__()}"

    def __init__(self, mol, thc, auxbasis='', thc_only: bool = False, use_thc_only: bool = None):
        self.mol = mol
        self.thc = thc
        self.auxbasis = auxbasis
        self.thc_only = bool(thc_only or (use_thc_only if use_thc_only is not None else False))

    def kernel(self) -> tuple[float,...]:
        scfthc = LS_RI_Cholesky(mol=self.mol, auxbasis=self.auxbasis, cholesky_threshold=1e-8, grid=grid.BeckeGrid(self.mol))
        scf_eri = scfthc.build(mode="ao")

        active: ExperimentRun = ExperimentRun.get_active()

        if self.mol.spin > 0:
            mf = THC_UHF(self.mol, scf_eri, self.auxbasis, thc_only=self.thc_only)
            e_scf = mf.kernel()
            if not mf.converged:
                raise RuntimeError(f"SCF failed to converge after {mf.max_cycle} cycles!")

            self.thc.with_mo_coeff(mf.mo_coeff)

            eri = self.thc.build_unrestricted(mode="ov")
            if active: active.checkpoint(ch.THC_BUILD)
            e_corr = ump2_energy_laplace(self.mol, mf, eri)

        else:
            mf = THC_RHF(self.mol, scf_eri, self.auxbasis, thc_only=self.thc_only)
            e_scf = mf.kernel()
            if not mf.converged:
                raise RuntimeError(f"SCF failed to converge after {mf.max_cycle} cycles!")

            self.thc.with_mo_coeff(mf.mo_coeff)

            eri = self.thc.build(mode="ov")
            e_corr = mp2_energy_laplace(self.mol, mf, eri)

        self.mf = mf


        return float(e_corr), float(e_scf), float(e_scf) + float(e_corr)


class THCMP2_SOS(Runnable):
    @classmethod
    def with_thc(cls, thc_cls: Type[THC]) -> Callable[..., Runnable]:
        def build(cfg):
            thc = thc_cls.from_config(cfg)
            return cls.from_config(cfg | {'thc': thc})

        return build

    def __str__(self):
        return f"ump2_thc_sos_{self.thc.__str__()}_"

    def __init__(self, mol, thc, auxbasis, chkfile = ''):
        self.mol = mol
        self.thc = thc
        self.auxbasis = auxbasis
        self.chkfile = chkfile

    def kernel(self) -> tuple[float,...]:
        active: ExperimentRun = ExperimentRun.get_active()

        mf = build_mf(self.mol, self.auxbasis, self.chkfile)
        e_scf = mf.e_tot

        self.mf = mf

        if active: active.checkpoint(ch.MF_BUILD)

        self.thc.with_mo_coeff(mf.mo_coeff)
        if self.mol.spin > 0:
            eri = self.thc.build_unrestricted(mode="ov")
            if active: active.checkpoint(ch.THC_BUILD)
            e_corr = ump2_energy_sos(self.mol, mf, eri)
        else:
            eri = self.thc.build(mode="ov")
            if active: active.checkpoint(ch.THC_BUILD)
            e_corr = mp2_energy_sos(self.mol, mf, eri)

        return float(e_corr), float(e_scf), float(e_scf) + float(e_corr)
