import logging
import os
import sys
from contextlib import contextmanager
from time import perf_counter

import numpy as np
from pyscf import lib
# 1. Import PBC variants from PySCF
from pyscf.pbc import gto, scf

# 2. Import your existing MP2 logic and the NEW PBC THC class
from pythc.pbc.scf.scf import Yang_FFTISDF
from pythc.pbc.thc.pbc_ls_ri_cholesky import PBC_LS_RI_Cholesky

KCALPERMOL_PER_HARTREE = 627.509_474

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


@contextmanager
def timed_step(name: str):
    start = perf_counter()
    try:
        yield
    finally:
        elapsed = perf_counter() - start
        logger.info("[TIMING] %s completed in %.3f seconds (%.2f minutes)", name, elapsed, elapsed / 60.0)


def main():
    total_start = perf_counter()

    os.environ["OMP_NUM_THREADS"] = "18"
    os.environ["PYSCF_MAX_MEMORY"] = "16000"
    os.environ["PYTHC_USE_CUDA"] = "False"
    os.environ["PYTHC_CUDA_PRECISION"] = "float64"
    os.environ["PYSCF_SCF_UHF_INIT_GUESS_BREAKSYM"] = "2"
    lib.parameters.MAX_MEMORY = 160000

    # 1. Define and build unit cell
    with timed_step("Cell Construction & Build"):
        cell = gto.Cell()
        a = 5.431
        cell.unit = "A"
        cell.a = np.array([
            [0.0, a / 2, a / 2],
            [a / 2, 0.0, a / 2],
            [a / 2, a / 2, 0.0]
        ])
        cell.atom = [
            ["Si", [0.0, 0.0, 0.0]],
            ["Si", [a / 4, a / 4, a / 4]]
        ]
        cell.basis = "gth-dzvp"
        cell.pseudo = "gth-pade"
        cell.ke_cutoff = 20
        cell.build()

    auxbasis = "weigend"
    k_mesh = [3, 3, 3]  # 27 k-points
    kpts = cell.make_kpts(k_mesh)

    # 2. Reference KRHF with Standard Density Fitting
    with timed_step("Reference KRHF (with standard DF)"):
        mf = scf.KRHF(cell, kpts=kpts)
        mf.max_memory = 16000
        mf = mf.density_fit(auxbasis=auxbasis)
        mf.verbose = 4
        mf.kernel()

    print(f"MF energy: {mf.e_tot}")

    # 3. THC ERI Tensor Construction
    with timed_step("THC Initialization & ERI Build"):
        thc = PBC_LS_RI_Cholesky(cell=cell, cholesky_threshold=1e-8)
        thc_eri = thc.build_kpts(mode="ao", kpts=kpts)

    # 4. KRHF with THC Density Fitting
    with timed_step("MF-THC KRHF Kernel"):
        mf_thc = scf.KRHF(cell, kpts)
        mf_thc.verbose = 4
        mf_thc.max_memory = 16000
        mf_thc.with_df = Yang_FFTISDF(cell, kpts, k_mesh, thc_eri)
        mf_thc.kernel()

    print(f"MF-THC energy: {mf_thc.e_tot*KCALPERMOL_PER_HARTREE}")
    print(f"THC Z shape: {thc_eri.Z_kpts.shape}")
    print(f"Error Me vs. RI: {(mf.e_tot - mf_thc.e_tot)*KCALPERMOL_PER_HARTREE}")

    total_elapsed = perf_counter() - total_start
    logger.info("[TIMING] Entire workflow completed in %.3f seconds (%.2f minutes)", total_elapsed, total_elapsed / 60.0)


if __name__ == "__main__":
    main()
