"""Benzene on Au(111): PBC THC vs. PySCF reference with k-points.

System
------
- Benzene (C6H6) physisorbed flat ~3.3 A above a 2-layer Au(111) slab
  (2x2 in-plane repetition, i.e. 8 Au atoms + 12 benzene atoms).
- Slab + vacuum supercell along z (``cell.dimension == 3``). The surface is
  therefore modelled as a 3D slab, not a true 2D cell, because the THC
  k-point driver (``PBC_THC_DF``) requires ``dimension == 3``. The
  "2D surface" sampling is the ``3 x 3 x 1`` k-mesh (single k-point along z).

Basis / pseudopotential (double-zeta GTH)
-----------------------------------------
- ``basis = 'gth-dzvp-molopt-sr'``: double-zeta MOLOPT short-range GTH basis.
  NOTE: plain ``gth-dzvp`` / ``gth-dzv`` shipped with PySCF contain no Au
  entry, so they cannot build this system. The ``-molopt-sr`` DZ variant
  covers Au (q11), C and H and is the DZ GTH basis used here.
- ``pseudo = 'gth-pade'`` (covers Au-q11, C-q4, H-q1).

What it does
------------
1. Builds the slab + benzene cell with ASE.
2. Runs a reference periodic HF calculation (``KRHF`` + default FFTDF,
   ``exxdiv='ewald'``) on the ``3 x 3 x 1`` k-mesh and times it.
3. Builds the periodic THC/ISDF ERI (``PBC_LS_RI_Cholesky.build_kpts``,
   ``mode='ao'``) and times it.
4. Runs ``KRHF`` with ``PBC_THC_DF`` (FFT Coulomb + THC exchange) and times it.
5. Prints an energy/timing comparison table.

Cost control
------------
The FFT (uniform) grid grows with ``ke_cutoff`` and the vacuum thickness.
The defaults (``KE_CUTOFF = 20`` Ry, 2 Au layers, 10 A vacuum) match the
resolution of the existing ``scf_with_pbc_kpoints_ls_ri_cholesky.py`` Si
example and keep the demo tractable. Production numbers need a converged
``ke_cutoff`` (typically >> 100 Ry for Au MOLOPT), more Au layers, and
k-mesh/vacuum convergence tests. Tune via the module-level constants or
the ``PYTHC_BENZENE_AU_*`` environment variables.

Reference: PyTHC PBC implementation (``pythc.pbc.thc.pbc_ls_ri_cholesky``,
``pythc.pbc.scf.df.PBC_THC_DF``) vs. stock PySCF PBC SCF.
"""

import logging
import os
import sys
from contextlib import contextmanager
from time import perf_counter

import numpy as np

from pyscf.pbc import gto, scf
from pyscf.pbc.tools import pyscf_ase

from pythc.pbc.scf.df import PBC_THC_DF
from pythc.pbc.thc.pbc_ls_ri_cholesky import PBC_LS_RI_Cholesky

# ---------------------------------------------------------------------------
# Config (overridable via environment for quick tests)
# ---------------------------------------------------------------------------
K_MESH = [3, 3, 1]  # required: 3 x 3 x 1 surface sampling
BASIS = "gth-dzvp-molopt-sr"  # double-zeta GTH (MOLOPT-SR; supports Au/C/H)
PSEUDO = "gth-pade"
KE_CUTOFF = float(os.environ.get("PYTHC_BENZENE_AU_KE_CUTOFF", "20"))  # Ry
N_LAYERS = int(os.environ.get("PYTHC_BENZENE_AU_LAYERS", "2"))  # Au(111) layers
INPLANE = int(os.environ.get("PYTHC_BENZENE_AU_INPLANE", "2"))  # NxN repetition
VACUUM = float(os.environ.get("PYTHC_BENZENE_AU_VACUUM", "10.0"))  # Angstrom
AU_LATTICE_A = 4.0782  # Angstrom, bulk Au lattice constant
BENZENE_HEIGHT = 3.3  # Angstrom above the top Au layer
CHOLESKY_THRESHOLD = 1e-6
SMEARING_SIGMA = 0.01  # Ha, Fermi smearing for the metallic slab

HARTREE_TO_MEV = 27_211.386_245_988
HARTREE_TO_KCALMOL = 627.509_474

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@contextmanager
def timed_step(name: str, store: dict | None = None, key: str | None = None):
    start = perf_counter()
    try:
        yield
    finally:
        elapsed = perf_counter() - start
        logger.info("[TIMING] %s completed in %.3f s (%.2f min)", name, elapsed, elapsed / 60.0)
        if store is not None and key is not None:
            store[key] = elapsed


def build_benzene_on_gold_cell() -> gto.Cell:
    """Build the Au(111) slab + flat benzene cell with ASE."""
    from ase.build import fcc111, molecule  # local import: only needed here

    slab = fcc111(
        "Au",
        size=(INPLANE, INPLANE, N_LAYERS),
        a=AU_LATTICE_A,
        vacuum=VACUUM,
    )
    benz = molecule("C6H6")  # planar in xy, centred near origin
    top_z = slab.get_positions()[:, 2].max()
    shift_xy = slab.get_positions()[:, :2].mean(axis=0) - benz.get_positions()[:, :2].mean(axis=0)
    benz.translate([shift_xy[0], shift_xy[1], 0.0])
    benz.translate([0.0, 0.0, top_z + BENZENE_HEIGHT - benz.get_positions()[:, 2].mean()])
    combined = slab + benz

    cell = gto.Cell()
    cell.atom = pyscf_ase.ase_atoms_to_pyscf(combined)
    cell.a = np.array(combined.cell)
    cell.unit = "Angstrom"
    cell.basis = BASIS
    cell.pseudo = PSEUDO
    cell.ke_cutoff = KE_CUTOFF
    # Slab with vacuum along z; THC k-point DF requires dimension == 3.
    cell.dimension = 3
    cell.verbose = 4
    cell.build()
    logger.info(
        "cell: natm=%d nelec=%d nao=%d mesh=%s vol=%.1f A^3",
        cell.natm, cell.nelectron, cell.nao_nr(), list(cell.mesh), cell.vol,
    )
    return cell


def run_krhf(cell: gto.Cell, kpts: np.ndarray, with_df=None, label: str = "KRHF") -> tuple:
    """Run periodic RHF on the given k-mesh, optionally with a custom DF object."""
    mf = scf.KRHF(cell, kpts=kpts)
    mf.verbose = 4
    mf.max_cycle = 60
    mf.conv_tol = 1e-7
    mf.level_shift = 0.2
    mf.diis_space = 12
    mf.exxdiv = "ewald"
    mf = mf.smearing(sigma=SMEARING_SIGMA, method="fermi")
    if with_df is not None:
        mf.with_df = with_df
    e_tot = mf.kernel()
    logger.info("%s converged=%s e_tot=%.10f Ha", label, mf.converged, e_tot)
    return mf, e_tot


def main() -> dict:
    timings: dict = {}

    with timed_step("Cell construction (benzene/Au(111))", timings, "cell"):
        cell = build_benzene_on_gold_cell()
    kpts = cell.make_kpts(K_MESH)
    logger.info("k-mesh=%s nkpts=%d", K_MESH, len(kpts))

    # 1. Stock PySCF reference: KRHF with default FFTDF (FFT J + FFT K).
    with timed_step("Reference KRHF (PySCF FFTDF)", timings, "ref_scf"):
        mf_ref, e_ref = run_krhf(cell, kpts, label="Reference KRHF")

    # 2. THC factorization of the periodic ERIs (AO mode for SCF).
    with timed_step("THC build_kpts (PBC LS-RI-Cholesky, mode='ao')", timings, "thc_build"):
        thc = PBC_LS_RI_Cholesky(cell=cell, cholesky_threshold=CHOLESKY_THRESHOLD)
        thc_eri = thc.build_kpts(mode="ao", kpts=kpts)
    X_kpt, Z_kpt = thc_eri.get_X_Z()
    logger.info("THC shapes: X=%s Z=%s (ngrid=%d)", X_kpt.shape, Z_kpt.shape, int(np.prod(cell.mesh)))

    # 3. THC SCF: FFT Coulomb + THC exchange via PBC_THC_DF.
    with timed_step("THC-KRHF (PBC_THC_DF)", timings, "thc_scf"):
        thc_df = PBC_THC_DF(cell, kpts, K_MESH, thc_eri)
        mf_thc, e_thc = run_krhf(cell, kpts, with_df=thc_df, label="THC-KRHF")

    # 4. Comparison.
    d_e = e_thc - e_ref
    total_thc = timings["thc_build"] + timings["thc_scf"]
    print("\n================ benzene/Au(111) 3x3x1 comparison ================")
    print(f"basis={BASIS}  pseudo={PSEUDO}  ke_cutoff={KE_CUTOFF} Ry")
    print(f"slab={INPLANE}x{INPLANE}x{N_LAYERS} Au + C6H6  nao={cell.nao_nr()}  nkpts={len(kpts)}")
    print(f"reference KRHF energy : {e_ref:.10f} Ha  ({timings['ref_scf']:.1f} s)")
    print(f"THC-KRHF energy       : {e_thc:.10f} Ha  (scf {timings['thc_scf']:.1f} s + build {timings['thc_build']:.1f} s)")
    print(f"ΔE (THC - ref)        : {d_e:+.6e} Ha = {d_e * HARTREE_TO_MEV:+.3f} meV = {d_e * HARTREE_TO_KCALMOL:+.4f} kcal/mol")
    print(f"wall time THC/SCF-ref : {total_thc:.1f} s / {timings['ref_scf']:.1f} s  (ratio {total_thc / max(timings['ref_scf'], 1e-9):.2f}x)")
    print(f"converged: ref={mf_ref.converged} thc={mf_thc.converged}")
    print("=================================================================\n")

    return {"e_ref": e_ref, "e_thc": e_thc, "timings": timings}


if __name__ == "__main__":
    main()
