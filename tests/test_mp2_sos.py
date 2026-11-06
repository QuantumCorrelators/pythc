import logging

import pytest

from pythc.const import KCALPERMOL_PER_HARTREE
from tests.thc_helpers import (
    AUXBASIS,
    MOL_FILES,
    MO_NAMES,
    build_mol,
    df_scf,
    thc_sos_mp2_corr,
)
from pythc.grid import BeckeGrid

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("thc_name", MO_NAMES)
@pytest.mark.parametrize("mol_file", MOL_FILES)
def test_mp2_sos(mol_file, thc_name):
    mol = build_mol(mol_file)
    mf = df_scf(mol, AUXBASIS)
    corr_thc, corr_ref = thc_sos_mp2_corr(mol, mf, thc_name, AUXBASIS, BeckeGrid(mol))

    logger.info("MP2-SOS %s on %s: THC=%s DF=%s", thc_name, mol_file, corr_thc, corr_ref)
    assert abs(corr_ref - corr_thc) * KCALPERMOL_PER_HARTREE < 5
