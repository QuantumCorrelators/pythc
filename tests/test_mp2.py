import logging

import pytest
from pyscf import gto

from pythc.const import KCALPERMOL_PER_HARTREE
from tests.parse_diet_gmtkn import parse_yaml_to_reactions
from tests.thc_helpers import (
    AUXBASIS,
    MOL_FILES,
    MO_NAMES,
    MO_THRESHOLDS,
    build_auxbasis,
    build_mol,
    default_mp2_grid,
    df_mp2_corr,
    df_scf,
    make_thc_mo,
    thc_mp2_corr, BASIS, df_mp2_corr_laplace,
)
from pythc.grid import BeckeGrid
from pythc.methods.mp2 import LaplaceMP2, LaplaceUMP2

logger = logging.getLogger(__name__)

DIET_GMTKN_55_150_MOLS = [
    r for r in parse_yaml_to_reactions("tests/DietGMTKN55/GoodSamples/AllElements_150.yaml")
    if r.dataset not in ["C60ISO", "UPU23", "ISOL24"]
]
DIET_GMTKN_55_30_MOLS = [
    r for r in parse_yaml_to_reactions("tests/DietGMTKN55/GoodSamples/AllElements_030.yaml")
    if r.dataset not in ["C60ISO", "UPU23", "ISOL24"]
]

@pytest.mark.parametrize("thc_name", MO_NAMES)
@pytest.mark.parametrize("mol_file", MOL_FILES)
def test_mp2(mol_file, thc_name):
    mol = build_mol(mol_file)
    mf = df_scf(mol, AUXBASIS)
    corr_thc, corr_ref = thc_mp2_corr(mol, mf, thc_name, AUXBASIS, default_mp2_grid(mol))

    logger.info("MP2 %s on %s: THC=%s DF=%s", thc_name, mol_file, corr_thc, corr_ref)
    assert abs(corr_ref - corr_thc) * KCALPERMOL_PER_HARTREE <= MO_THRESHOLDS[thc_name]



@pytest.mark.parametrize("seed", [42, 45])
def test_reaction_mp2_qrcp_bf_outlier(seed):
    """W4-11 #57 (BF -> B + F, cc-pVQZ): unrestricted OV-pruning regression.

    ``LS_RI_THC.build_unrestricted`` used to prune the alpha grid with
    ``n_occ = nelectron // 2``, dropping the singly-occupied alpha
    orbital(s) from the pair space (B: 2p never entered QRCP selection).
    The B-atom alpha blocks then plateaued (~+1 kcal/mol at tol 1e-3..1e-5
    while Cholesky/KMeans/snRI all reached ~0), and loose thresholds kept
    knife-edge grids whose pivots vary run-to-run under threaded LAPACK
    (benchmark outlier: -8..-14 kcal/mol reaction error at tol=0.1).
    With spin-correct occupations the reaction converges (tol 1e-3:
    ~-0.2 kcal/mol); this test pins that.
    """
    reaction = [r for r in DIET_GMTKN_55_30_MOLS if r.dataset == "W4-11" and r.reaction == 57][0]
    basis, auxbasis_seed = "cc-pvqz", "cc-pvqz-ri"

    def calculate_mp2(m) -> tuple[float, float]:
        mol = gto.Mole()
        mol.spin = m.spin
        mol.charge = m.charge
        mol.atom = m.geom
        mol.basis = basis
        mol.build()

        auxbasis = build_auxbasis(mol, auxbasis_seed)
        mf = df_scf(mol, auxbasis)
        e_scf = float(mf.e_tot)
        corr_ref = df_mp2_corr_laplace(mf)

        from pythc.thc.ls_ri_qrcp import LS_RI_QRCP
        grid = BeckeGrid(mol=mol, level=0)
        thc = LS_RI_QRCP(mol=mol, auxbasis=auxbasis, grid=grid,
                         mo_coeff=mf.mo_coeff, tolerance=1e-4, seed=seed)
        cls = LaplaceUMP2 if mol.spin > 0 else LaplaceMP2
        corr_thc = float(cls(mf, thc, n_laplace=10).kernel()[0])

        # Repeat with the same seed: must agree to solver noise (tests run
        # single-threaded; guards the threaded-LAPACK pivot nondeterminism
        # seen at loose tolerance, which moves kcal, not ulps).
        thc2 = LS_RI_QRCP(mol=mol, auxbasis=auxbasis, grid=grid,
                          mo_coeff=mf.mo_coeff, tolerance=1e-4, seed=seed)
        corr_thc2 = float(cls(mf, thc2, n_laplace=10).kernel()[0])
        assert corr_thc == pytest.approx(corr_thc2, rel=1e-9)

        e_total_thc = abs(e_scf + corr_thc)
        e_total_ref = abs(e_scf + corr_ref)
        logger.info("DIFF: %s THC: %s; RI: %s", m.filename, e_total_thc, e_total_ref)
        return e_total_thc, e_total_ref

    energy_in_thc = energy_in_ref = 0.0
    for educt in reaction.educts:
        e_thc, e_ref = calculate_mp2(educt.mol)
        weight = abs(educt.weight)
        energy_in_thc += weight * e_thc
        energy_in_ref += weight * e_ref

    energy_out_thc = energy_out_ref = 0.0
    for product in reaction.products:
        e_thc, e_ref = calculate_mp2(product.mol)
        weight = abs(product.weight)
        energy_out_thc += weight * e_thc
        energy_out_ref += weight * e_ref

    energy_diff_thc = (energy_out_thc - energy_in_thc) * KCALPERMOL_PER_HARTREE
    energy_diff_ref = (energy_out_ref - energy_in_ref) * KCALPERMOL_PER_HARTREE
    energy_diff_diff = energy_diff_ref - energy_diff_thc

    logger.info("DELTA THC-DF: %s", energy_diff_diff)
    assert abs(energy_diff_diff) < 2.0


@pytest.mark.parametrize("thc_name", MO_NAMES)
def test_reaction_mp2(thc_name):
    reaction = [r for r in DIET_GMTKN_55_30_MOLS if r.dataset == "W4-11" and r.reaction == 132][0]

    def calculate_mp2(m) -> tuple[float, float]:
        mol = gto.Mole()
        mol.spin = m.spin
        mol.charge = m.charge
        mol.atom = m.geom
        mol.basis = {"default": BASIS}
        mol.build()

        auxbasis = build_auxbasis(mol, AUXBASIS)
        mf = df_scf(mol, auxbasis)
        e_scf = float(mf.e_tot)
        corr_ref = df_mp2_corr_laplace(mf)

        grid = BeckeGrid(mol=mol, level=0)
        thc = make_thc_mo(thc_name, mol, mf, auxbasis, grid)
        cls = LaplaceUMP2 if mol.spin > 0 else LaplaceMP2
        corr_thc = float(cls(mf, thc, n_laplace=10).kernel()[0])

        e_total_thc = abs(e_scf + corr_thc)
        e_total_ref = abs(e_scf + corr_ref)
        logger.info("DIFF: %s THC: %s; RI: %s", m.filename, e_total_thc, e_total_ref)
        return e_total_thc, e_total_ref

    energy_in_thc = energy_in_ref = 0.0
    for educt in reaction.educts:
        e_thc, e_ref = calculate_mp2(educt.mol)
        weight = abs(educt.weight)
        energy_in_thc += weight * e_thc
        energy_in_ref += weight * e_ref

    energy_out_thc = energy_out_ref = 0.0
    for product in reaction.products:
        e_thc, e_ref = calculate_mp2(product.mol)
        weight = abs(product.weight)
        energy_out_thc += weight * e_thc
        energy_out_ref += weight * e_ref

    energy_diff_thc = (energy_out_thc - energy_in_thc) * KCALPERMOL_PER_HARTREE
    energy_diff_ref = (energy_out_ref - energy_in_ref) * KCALPERMOL_PER_HARTREE
    energy_diff_diff = energy_diff_ref - energy_diff_thc

    logger.info("DELTA THC-DF: %s", energy_diff_diff)
    assert abs(energy_diff_diff) < MO_THRESHOLDS[thc_name]