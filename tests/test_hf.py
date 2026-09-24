import logging

import pytest
from pyscf import gto

from tests.parse_diet_gmtkn import parse_yaml_to_reactions
from pythc.const import KCALPERMOL_PER_HARTREE
from tests.thc_helpers import (
    AO_NAMES,
    AO_THRESHOLDS,
    AUXBASIS,
    MOL_FILES,
    build_auxbasis,
    build_mol,
    thc_hf_energy, BASIS,
)
from pythc.grid import BeckeGrid

logger = logging.getLogger(__name__)

DIET_GMTKN_55_150_MOLS = [
    r for r in parse_yaml_to_reactions("tests/DietGMTKN55/GoodSamples/AllElements_150.yaml")
    if r.dataset not in ["C60ISO", "UPU23", "ISOL24"]
]


@pytest.mark.parametrize("thc_name", AO_NAMES)
@pytest.mark.parametrize("mol_file", MOL_FILES)
def test_hf(mol_file, thc_name):
    mol = build_mol(mol_file)
    e_thc, e_ref = thc_hf_energy(mol, thc_name, AUXBASIS, BeckeGrid(mol))

    diff = e_ref - e_thc
    logger.info("HF %s on %s: THC=%s DF=%s DIFF=%s", thc_name, mol_file, e_thc, e_ref, diff)
    assert abs(diff) <= AO_THRESHOLDS[thc_name]


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
        e_thc, e_ref = thc_hf_energy(mol, thc_name, auxbasis)
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
    assert abs(energy_diff_diff) <= 1 or abs(
        reaction.energy_diff_kcalmol - energy_diff_ref
    ) > abs(reaction.energy_diff_kcalmol - energy_diff_thc)
