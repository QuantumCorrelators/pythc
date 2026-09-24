"""Shared builders for the THC end-to-end tests.

Mirrors the HF/MP2 call pattern used in ``src/pythc/examples``:

* HF: ``pythc.methods.scfdd.RHF`` / ``UHF`` vs. stock PySCF DF-SCF.
* MP2: ``pythc.methods.mp2.LaplaceMP2`` / ``LaplaceUMP2`` /
  ``LaplaceSOSMP2`` / ``LaplaceSOSUMP2`` vs. ``DFRMP2`` / ``DFUMP2``.
"""

import glob
import os

from pyscf import df, gto, scf
from pyscf.dft import treutler_prune
from pyscf.lib.exceptions import BasisNotFoundError
from pyscf.mp.dfmp2 import DFRMP2
from pyscf.mp.dfump2 import DFUMP2

from pythc.grid import BeckeGrid
from pythc.methods.scfdd import RHF as THCRHF
from pythc.methods.scfdd import UHF as THCUHF
from pythc.methods.mp2 import (
    LaplaceMP2,
    LaplaceSOSMP2,
    LaplaceSOSUMP2,
    LaplaceUMP2,
)
from pythc.thc.ls_aux_becke import LS_Aux_Becke
from pythc.thc.ls_ri_becke import LS_RI_Becke
from pythc.thc.ls_ri_cholesky import LS_RI_Cholesky
from pythc.thc.ls_ri_kmeans import LS_RI_KMeans
from pythc.thc.ls_ri_qrcp import LS_RI_QRCP
from pythc.thc.ls_snri_cholesky import LS_snRI_Cholesky

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYSCF_MAX_MEMORY"] = "4000"
os.environ["PYTHC_USE_CUDA"] = "False"
os.environ["PYTHC_CUDA_PRECISION"] = "float64"
os.environ["PYSCF_SCF_UHF_INIT_GUESS_BREAKSYM"] = "2"

BASIS = "def2svp"
AUXBASIS = "def2universal-jkfit"
MOL_FILES = sorted(glob.glob("tests/mols/*.xyz"))

# Per-variant accuracy thresholds, carried over from the old test_e2e.py.
# MO variants are used for (U)MP2, AO variants for HF.
MO_THRESHOLDS = {
    "LS-RI-Becke": 0.01,
    "LS-RI-QRCP": 1,
    "LS-RI-KMeans": 1,
    "LS-RI-Cholesky": 1,
    "LS-snRI-Cholesky": 1,
    "LS-Aux-Becke": 5,
}

AO_THRESHOLDS = {
    "LS-RI-Becke": 1,
    "LS-RI-QRCP": 1,
    "LS-RI-KMeans": 1,
    "LS-RI-Cholesky": 1,
    "LS-snRI-Cholesky": 1,
    "LS-Aux-Becke": 5,
}

MO_NAMES = list(MO_THRESHOLDS)
AO_NAMES = list(AO_THRESHOLDS)

# Per-variant RPA correlation-energy thresholds (Eh) on the water/cc-pvdz
# probe: Becke/QRCP/KMeans ~1e-9, Cholesky/snRI ~8e-6, Aux-Becke ~1.1e-4.
RPA_THRESHOLDS = {
    "LS-RI-Becke": 1,
    "LS-RI-QRCP": 1,
    "LS-RI-KMeans": 1,
    "LS-RI-Cholesky": 1,
    "LS-snRI-Cholesky": 1,
    "LS-Aux-Becke": 3,
}


def build_auxbasis(mol: gto.Mole, auxbasis: str):
    """Resolve ``auxbasis``, falling back to an even-tempered basis."""
    if auxbasis:
        try:
            df.addons.make_auxmol(mol, auxbasis=auxbasis)
        except BasisNotFoundError:
            return df.make_auxbasis(mol, mp2fit=True)
    return auxbasis


def build_mol(mol_file: str, basis: str = BASIS) -> gto.Mole:
    return gto.M(atom=mol_file, basis=basis)


def df_scf(mol: gto.Mole, auxbasis: str, verbose: int = 0):
    """Stock PySCF DF-SCF, used as the reference and the MP2 starting point."""
    mf = (scf.UHF if mol.spin > 0 else scf.RHF)(mol).density_fit(auxbasis=auxbasis)
    mf.verbose = verbose
    mf.max_cycle = 200
    mf.kernel()
    return mf


def df_mp2_corr(mf) -> float:
    """DF-MP2 correlation energy, dispatching on restricted/unrestricted."""
    if mf.mol.spin > 0:
        return float(DFUMP2(mf).kernel()[0])
    return float(DFRMP2(mf).kernel()[0])


def make_thc_ao(thc_name: str, mol: gto.Mole, auxbasis: str, grid=None):
    """Build a THC factorization for HF (AO mode)."""
    grid = grid if grid is not None else BeckeGrid(mol)
    if thc_name == "LS-RI-Becke":
        return LS_RI_Becke(mol=mol, auxbasis=auxbasis, grid=grid)
    if thc_name == "LS-RI-QRCP":
        return LS_RI_QRCP(mol=mol, auxbasis=auxbasis, grid=grid, tolerance=1e-6)
    if thc_name == "LS-RI-KMeans":
        return LS_RI_KMeans(mol=mol, auxbasis=auxbasis, grid=grid, ips_per_naux=5.5)
    if thc_name == "LS-RI-Cholesky":
        return LS_RI_Cholesky(mol=mol, auxbasis=auxbasis, grid=grid, cholesky_threshold=1e-8)
    if thc_name == "LS-snRI-Cholesky":
        return LS_snRI_Cholesky(mol=mol, auxbasis=auxbasis, grid=grid, cholesky_threshold=1e-8)
    if thc_name == "LS-Aux-Becke":
        return LS_Aux_Becke(mol=mol, grid=grid, fit_auxbasis="etb-1.1")
    raise ValueError(f"unknown THC variant: {thc_name}")


def make_thc_mo(thc_name: str, mol: gto.Mole, mf, auxbasis: str, grid=None):
    """Build a THC factorization for (U)MP2 (OV mode)."""
    grid = grid if grid is not None else BeckeGrid(mol)
    mo_coeff = mf.mo_coeff
    if thc_name == "LS-RI-Becke":
        return LS_RI_Becke(mol=mol, auxbasis=auxbasis, grid=grid, mo_coeff=mo_coeff)
    if thc_name == "LS-RI-QRCP":
        return LS_RI_QRCP(mol=mol, auxbasis=auxbasis, grid=grid, mo_coeff=mo_coeff, tolerance=1e-4)
    if thc_name == "LS-RI-KMeans":
        return LS_RI_KMeans(mol=mol, auxbasis=auxbasis, grid=grid, mo_coeff=mo_coeff, ips_per_naux=6)
    if thc_name == "LS-RI-Cholesky":
        return LS_RI_Cholesky(
            mol=mol, auxbasis=auxbasis, grid=grid, mo_coeff=mo_coeff, cholesky_threshold=1e-5
        )
    if thc_name == "LS-snRI-Cholesky":
        return LS_snRI_Cholesky(
            mol=mol, auxbasis=auxbasis, grid=grid, mo_coeff=mo_coeff, cholesky_threshold=1e-5
        )
    if thc_name == "LS-Aux-Becke":
        return LS_Aux_Becke(mol=mol, grid=grid, mo_coeff=mo_coeff, fit_auxbasis="cc-pv5z")
    raise ValueError(f"unknown THC variant: {thc_name}")


def thc_hf_energy(mol: gto.Mole, thc_name: str, auxbasis: str, grid=None) -> tuple[float, float]:
    """Return ``(e_thc, e_ref)`` HF total energies, as in the SCF examples."""
    thc = make_thc_ao(thc_name, mol, auxbasis, grid)
    cls = THCUHF if mol.spin > 0 else THCRHF
    e_thc = float(cls(mol, thc, auxbasis, verbose=0).kernel())
    e_ref = float(df_scf(mol, auxbasis).e_tot)
    return e_thc, e_ref


def thc_mp2_corr(mol: gto.Mole, mf, thc_name: str, auxbasis: str, grid=None) -> tuple[float, float]:
    """Return ``(corr_thc, corr_ref)`` MP2 correlation energies, as in the MP2 examples."""
    thc = make_thc_mo(thc_name, mol, mf, auxbasis, grid)
    cls = LaplaceUMP2 if mol.spin > 0 else LaplaceMP2
    corr_thc = float(cls(mf, thc).kernel()[0])
    corr_ref = df_mp2_corr(mf)
    return corr_thc, corr_ref


def thc_sos_mp2_corr(mol: gto.Mole, mf, thc_name: str, auxbasis: str, grid=None) -> tuple[float, float]:
    """Return ``(sos_corr_thc, corr_ref)`` SOS-MP2 vs. full DF-MP2 correlation energies."""
    thc = make_thc_mo(thc_name, mol, mf, auxbasis, grid)
    cls = LaplaceSOSUMP2 if mol.spin > 0 else LaplaceSOSMP2
    corr_thc = float(cls(mf, thc).kernel()[0])
    corr_ref = df_mp2_corr(mf)
    return corr_thc, corr_ref


def default_mp2_grid(mol: gto.Mole) -> BeckeGrid:
    return BeckeGrid(mol=mol, level=0, prune=treutler_prune)
