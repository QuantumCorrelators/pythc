import pathlib
import time

from pyscf import gto, scf

from pythc.decomp.cholesky import AccelRPCholesky
from pythc.methods.rpa import thc_rpa
from pythc.thc.ls_ri_cholesky import LS_RI_Cholesky
from pythc.thc.thc_base import ThcEri

# ----------------------------------------------------------------------
# 3. Test Function (Discovered by Pytest / IDE Test Runners)
# ----------------------------------------------------------------------
def test_thc_rpa_vs_pyscf():
    water_cluster = """
    H        0.087529        0.023820        0.930805
    O        0.657172        0.599414        0.406256
    H        0.792448        1.344387        1.004310
    """

    mol = gto.Mole()
    mol.basis = 'cc-pvdz'
    mol.atom = water_cluster
    mol.build()

    auxbasis = 'cc-pvdz-ri'

    print("\n--- Running SCF ---")
    mf = scf.RHF(mol)
    mf = mf.density_fit(auxbasis=auxbasis)
    mf.verbose = 0
    mf.kernel()

    print("--- Generating THC ERIs ---")
    thc = LS_RI_Cholesky(mol=mol, auxbasis=auxbasis, mo_coeff=mf.mo_coeff,
                         cholesky_threshold=1e-5,
                         cholesky_decomp=AccelRPCholesky)
    eri_thc = thc.build(mode='ov')

    # 1. Run Standard PySCF RPA
    print("--- Running Standard PySCF RPA ---")
    try:
        from pyscf.gw.rpa import RPA
    except ImportError:
        raise ImportError("PySCF RPA module not found. Ensure PySCF is installed.")

    rpa_ref = RPA(mf)
    rpa_ref.verbose = 0

    t0 = time.perf_counter()
    e_corr_ref = rpa_ref.kernel(nw=40, x0=0.5)
    t1 = time.perf_counter()
    time_ref = t1 - t0

    # 2. Run Custom THC-RPA
    print("--- Running THC-RPA ---")
    t0 = time.perf_counter()
    e_corr_thc = thc_rpa(mol, mf, eri_thc)
    t1 = time.perf_counter()
    time_thc = t1 - t0

    # 3. Report
    print("\n" + "=" * 40)
    print("           RPA RESULTS")
    print("=" * 40)
    print(f"PySCF E_corr : {e_corr_ref: 15.8f} Eh  ({time_ref:.3f} s)")
    print(f"THC   E_corr : {e_corr_thc: 15.8f} Eh  ({time_thc:.3f} s)")
    print("-" * 40)

    error = abs(e_corr_ref - e_corr_thc)
    print(f"Absolute Diff: {error: 15.8e} Eh")

    assert error < 1e-4, f"THC error ({error}) exceeds expected bounds."
    print("TEST PASSED: THC correlation energy matches RI-RPA within truncation threshold.")


if __name__ == "__main__":
    test_thc_rpa_vs_pyscf()
