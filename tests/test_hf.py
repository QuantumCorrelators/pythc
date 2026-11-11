import logging

import numpy as np
import pytest
from pyscf import gto, df, scf

from pythc.const import KCALPERMOL_PER_HARTREE
from pythc.decomp.cholesky import build_cholesky_decomp
from pythc.grid import BeckeGrid, zone_prune
from pythc.methods import scf as scfthc
from pythc.thc.ls_ri_cholesky import LS_RI_Cholesky
from tests.parse_diet_gmtkn import parse_yaml_to_reactions
from tests.thc_helpers import (
    AO_NAMES,
    AO_THRESHOLDS,
    AUXBASIS,
    MOL_FILES,
    build_auxbasis,
    build_mol,
    thc_hf_energy_density_difference, BASIS, thc_hf_energy, )

logger = logging.getLogger(__name__)

DIET_GMTKN_55_150_MOLS = [
    r for r in parse_yaml_to_reactions("tests/DietGMTKN55/GoodSamples/AllElements_150.yaml")
    if r.dataset not in ["C60ISO", "UPU23", "ISOL24"]
]

@pytest.mark.parametrize("thc_name", AO_NAMES)
@pytest.mark.parametrize("mol_file", MOL_FILES)
def test_hf_density_difference(mol_file, thc_name):
    mol = build_mol(mol_file)
    e_thc, e_ref = thc_hf_energy_density_difference(mol, thc_name, AUXBASIS, BeckeGrid(mol))

    diff = e_ref - e_thc
    logger.info("HF %s on %s: THC=%s DF=%s DIFF=%s", thc_name, mol_file, e_thc, e_ref, diff)
    assert abs(diff) <= AO_THRESHOLDS[thc_name]


halogens = {
    'F2': ''''F   0.0000  0.0000  0.0000; F   0.0000  0.0000  1.412''',
    'Cl2': ''''Cl  0.0000  0.0000  0.0000; Cl  0.0000  0.0000  1.988''',
    'Br2': ''''Br  0.0000  0.0000  0.0000; Br  0.0000  0.0000  2.281''',
    'I2': ''''I   0.0000  0.0000  0.0000; I   0.0000  0.0000  2.665''',
    'At2': ''''At  0.0000  0.0000  0.0000; At  0.0000  0.0000  3.000'''
}

@pytest.mark.parametrize("thc_name", ("LS-RI-Cholesky",))
def test_hf_halogens(thc_name):
    mol = gto.M(atom=halogens['I2'], basis="x2c-svpall")
    e_thc, e_ref = thc_hf_energy(mol,
                                 thc_name,
                                 df.aug_etb(mol, 2.0),
                                 BeckeGrid(mol, level=4, prune=zone_prune(inner=110)),
                                 with_k_thc=True,
                                 with_j_thc=True)

    diff = e_ref - e_thc
    logger.info("HF %s on %s: THC=%s DF=%s DIFF=%s", thc_name, "", e_thc, e_ref, diff)
    assert abs(diff) <= 100

@pytest.mark.parametrize("mol_file", MOL_FILES)
def test_cholesky_random_vs_greedy(mol_file):
    mol = build_mol(mol_file)
    # Same threshold, same grid; greedy runs single-pivot (b=1 default)
    # while accel uses blocked proposals -- the pivot rule (random vs.
    # argmax) is what differs. b=8 is a faster greedy compromise;
    # large greedy blocks are numerically unsafe, don't use them.
    decomp_greedy = build_cholesky_decomp("greedy", seed=42, block_size=8)
    decomp_accel = build_cholesky_decomp("accel", seed=42, block_size=8)

    thc_greedy = LS_RI_Cholesky(mol, grid=BeckeGrid(mol, level=0), cholesky_threshold=1e-8, cholesky_decomp=decomp_greedy)
    thc_accel = LS_RI_Cholesky(mol, grid=BeckeGrid(mol, level=0), cholesky_threshold=1e-8, cholesky_decomp=decomp_accel)

    mf_greedy = scf.UHF(mol) if mol.spin > 0 else scf.RHF(mol)
    mf_greedy = mf_greedy.density_fit(with_df=scfthc.THCDF(mol, thc_greedy, AUXBASIS))
    mf_greedy.kernel()

    mf_accell = scf.UHF(mol) if mol.spin > 0 else scf.RHF(mol)
    mf_accell = mf_accell.density_fit(with_df=scfthc.THCDF(mol, thc_accel, AUXBASIS))
    mf_accell.kernel()

    e_greedy = mf_greedy.e_tot * KCALPERMOL_PER_HARTREE
    e_accell = mf_accell.e_tot * KCALPERMOL_PER_HARTREE

    diff = e_greedy - e_accell
    logger.info("HF on %s: THC (Greedy) =%s THC (Accell) =%s DIFF=%s (kcal/mol)", mol_file, e_greedy, e_accell, diff)
    # this should always yield the same result and break if the seeding is ignored somewhere
    assert np.isclose(diff, -0.0767, atol=0.001)

@pytest.mark.parametrize("thc_name", AO_NAMES)
def test_reaction_hf(thc_name):
    reaction = [r for r in DIET_GMTKN_55_150_MOLS if r.dataset == "W4-11" and r.reaction == 38][0]

    def calculate_scf(m) -> tuple[float, float]:
        mol = gto.Mole()
        mol.spin = m.spin
        mol.charge = m.charge
        mol.atom = m.geom
        mol.basis = {"default": BASIS}
        mol.build()

        auxbasis = build_auxbasis(mol, AUXBASIS)
        e_thc, e_ref = thc_hf_energy_density_difference(mol, thc_name, auxbasis, df_min_cycles=3)
        logger.info("DIFF: %s THC: %s; RI: %s", m.filename, e_thc, e_ref)
        return e_thc, e_ref

    energy_in_thc = energy_in_ref = 0.0
    for educt in reaction.educts:
        e_thc, e_ref = calculate_scf(educt.mol)
        weight = abs(educt.weight)
        energy_in_thc += weight * e_thc
        energy_in_ref += weight * e_ref

    energy_out_thc = energy_out_ref = 0.0
    for product in reaction.products:
        e_thc, e_ref = calculate_scf(product.mol)
        weight = abs(product.weight)
        energy_out_thc += weight * e_thc
        energy_out_ref += weight * e_ref

    energy_diff_thc = (energy_out_thc - energy_in_thc) * KCALPERMOL_PER_HARTREE
    energy_diff_ref = (energy_out_ref - energy_in_ref) * KCALPERMOL_PER_HARTREE
    energy_diff_diff = energy_diff_ref - energy_diff_thc

    logger.info("DELTA THC-DF: %s", energy_diff_diff)
    assert abs(energy_diff_diff) < AO_THRESHOLDS[thc_name]