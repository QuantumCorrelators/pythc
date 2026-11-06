import logging

from pythc.const import KCALPERMOL_PER_HARTREE
import pytest
from pyscf import gto

from tests.thc_helpers import (
    AUXBASIS,
    MO_NAMES,
    MO_THRESHOLDS,
    df_scf,
    thc_mp2_corr, BASIS,
)
from pythc.grid import BeckeGrid

logger = logging.getLogger(__name__)

PROTON_GEOM = "H 0 0 0"

METHYL_GEOM = """
C 0.000000 0.000000 0.000000
H 0.000000 1.079000 0.000000
H 0.934449 -0.539500 0.000000
H -0.934449 -0.539500 0.000000
"""

def _unrestricted_mol(geom: str) -> gto.Mole:
    mol = gto.Mole()
    mol.atom = geom
    mol.basis = BASIS
    mol.spin = 1
    mol.build()
    return mol

@pytest.mark.parametrize("thc_name", MO_NAMES)
def test_proton_ump2(thc_name):
    mol = _unrestricted_mol(PROTON_GEOM)
    mf = df_scf(mol, AUXBASIS)
    corr_thc, corr_ref = thc_mp2_corr(mol, mf, thc_name, AUXBASIS, BeckeGrid(mol))

    diff = (corr_ref - corr_thc) * KCALPERMOL_PER_HARTREE
    logger.info("Proton UMP2 %s: DIFF=%s", thc_name, diff)
    assert abs(diff) <= MO_THRESHOLDS[thc_name]


@pytest.mark.parametrize("thc_name", MO_NAMES)
def test_ump2_radical(thc_name):
    mol = _unrestricted_mol(METHYL_GEOM)
    mf = df_scf(mol, AUXBASIS)
    corr_thc, corr_ref = thc_mp2_corr(mol, mf, thc_name, AUXBASIS, BeckeGrid(mol))

    diff = (corr_ref - corr_thc) * KCALPERMOL_PER_HARTREE
    logger.info("Methyl UMP2 %s: DIFF=%s", thc_name, diff)
    assert abs(diff) <= MO_THRESHOLDS[thc_name]
