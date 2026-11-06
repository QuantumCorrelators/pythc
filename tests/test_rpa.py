import logging
import time

import pytest
from pyscf import gto
from pyscf.gw.rpa import RPA as PyscfRPA

from pythc.const import KCALPERMOL_PER_HARTREE
from pythc.methods.rpa import RPA
from tests.thc_helpers import MO_NAMES, RPA_THRESHOLDS, df_scf, make_thc_mo

logger = logging.getLogger(__name__)

WATER_CLUSTER = """
H        0.087529        0.023820        0.930805
O        0.657172        0.599414        0.406256
H        0.792448        1.344387        1.004310
"""

BASIS = "cc-pvdz"
AUXBASIS = "cc-pvdz-ri"


# ----------------------------------------------------------------------
# Test Function (Discovered by Pytest / IDE Test Runners)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("thc_name", MO_NAMES)
def test_thc_rpa_vs_pyscf(thc_name):
    mol = gto.Mole()
    mol.basis = BASIS
    mol.atom = WATER_CLUSTER
    mol.build()

    print("\n--- Running SCF ---")
    mf = df_scf(mol, AUXBASIS)

    # 1. Run Standard PySCF RPA
    print("--- Running Standard PySCF RPA ---")
    rpa_ref = PyscfRPA(mf)
    rpa_ref.verbose = 0

    t0 = time.perf_counter()
    e_corr_ref = rpa_ref.kernel(nw=40, x0=0.5)
    t1 = time.perf_counter()
    time_ref = t1 - t0

    # 2. Run Custom THC-RPA
    print(f"--- Running THC-RPA ({thc_name}) ---")
    thc = make_thc_mo(thc_name, mol, mf, AUXBASIS)
    t0 = time.perf_counter()
    e_corr_thc = RPA(mf, thc).kernel()
    t1 = time.perf_counter()
    time_thc = t1 - t0

    # 3. Report
    print("\n" + "=" * 40)
    print(f"           RPA RESULTS ({thc_name})")
    print("=" * 40)
    print(f"PySCF E_corr : {e_corr_ref: 15.8f} Eh  ({time_ref:.3f} s)")
    print(f"THC   E_corr : {e_corr_thc: 15.8f} Eh  ({time_thc:.3f} s)")
    print("-" * 40)

    error = abs(e_corr_ref - e_corr_thc) * KCALPERMOL_PER_HARTREE
    print(f"Absolute Diff: {error: 15.8e} kcal/mol")
    logger.info("RPA %s: ref=%s thc=%s err=%s", thc_name, e_corr_ref, e_corr_thc, error)

    assert error < RPA_THRESHOLDS[thc_name], f"THC error ({error}) exceeds expected bounds."
    print("TEST PASSED: THC correlation energy matches RI-RPA within truncation threshold.")


if __name__ == "__main__":
    for _thc_name in MO_NAMES:
        test_thc_rpa_vs_pyscf(_thc_name)
