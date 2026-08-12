import glob
import os
import sys
import unittest

from pyscf import df
from pyscf.dft import treutler_prune
from pyscf.lib.exceptions import BasisNotFoundError

from tests.parse_diet_gmtkn import parse_yaml_to_reactions
from tests.reaction import MolInfo
from pythc.grid import BeckeGrid
from pythc.methods.runner import *
from pythc.thc.ls_aux_becke import LS_Aux_Becke
from pythc.thc.ls_snri_cholesky import LS_snRI_Cholesky
from pythc.thc.ls_ri_cholesky import LS_RI_Cholesky
from pythc.thc.ls_ri_becke import LS_RI_Becke
from pythc.thc.ls_ri_kmeans import LS_RI_KMeans
from pythc.thc.ls_ri_qrcp import LS_RI_QRCP

DIET_GMTKN_55_150_MOLS = [r for r in parse_yaml_to_reactions('tests/DietGMTKN55/GoodSamples/AllElements_150.yaml')
                          if r.dataset not in ['C60ISO', 'UPU23', 'ISOL24']]

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def build_auxbasis(mol, auxbasis):
    resolved_auxbasis = auxbasis
    if auxbasis:
        try:
            # Test if it works
            df.addons.make_auxmol(mol, auxbasis=auxbasis)
        except BasisNotFoundError:
            logger.warning(
                f"Molecule lacks elements in {auxbasis}. Generating even-tempered basis centrally.")
            # Generate a dictionary mapping of the basis and pass THIS dictionary to all runners
            resolved_auxbasis = df.make_auxbasis(mol, mp2fit=True)

    return resolved_auxbasis


logger = logging.getLogger()

mols = glob.glob("tests/mols/*.xyz")

KCALPERMOL_PER_HARTREE = 627.509_474

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYSCF_MAX_MEMORY"] = "4000"
os.environ["PYTHC_USE_CUDA"] = "False"
os.environ["PYTHC_CUDA_PRECISION"] = "float64"
os.environ["PYSCF_SCF_UHF_INIT_GUESS_BREAKSYM"] = "2"


class TestEriApprox(unittest.TestCase):

    def get_mol(self, file: str):
        mol = gto.M(atom=file, basis='def2-svp')

        return mol

    thcs_mo = {
        'LS-RI-Becke': (LS_RI_Becke, {}, 0.01),
        'LS-RI-QRCP': (LS_RI_QRCP, {'tolerance': 1e-4}, 1),
        'LS-RI-KMeans': (LS_RI_KMeans, {'ips_per_naux': 6}, 1),
        'LS-RI-Cholesky': (LS_RI_Cholesky, {'cholesky_threshold': 1e-5}, 1),
        'LS-snRI-Cholesky': (LS_snRI_Cholesky, {'cholesky_threshold': 1e-5}, 1),
        'LS-Aux-Becke': (LS_Aux_Becke, {'fit_auxbasis': 'cc-pv5z'}, 5),
    }

    thcs_ao = {
        'LS-RI-Becke': (LS_RI_Becke, {}, 1),
        'LS-RI-QRCP': (LS_RI_QRCP, {'tolerance': 1e-6}, 1),
        'LS-RI-KMeans': (LS_RI_KMeans, {'ips_per_naux': 5.5}, 1),
        'LS-RI-Cholesky': (LS_RI_Cholesky, {'cholesky_threshold': 1e-8}, 1),
        'LS-snRI-Cholesky': (LS_snRI_Cholesky, {'cholesky_threshold': 1e-8}, 1),
        'LS-Aux-Becke': (LS_Aux_Becke, {'fit_auxbasis': 'etb-1.1'}, 5),
    }

    def test_proton(self):
        for thc_name, (thc_class, thc_kwargs, threshold) in self.thcs_mo.items():
            logger.info(f"\n\n====== UMP2 Testing {thc_name} on radical ======")
            mol = gto.Mole()
            # Standard geometry for Methyl Radical
            mol.atom = '''
            H -0.000000 -0.000000 0.000000
            '''
            mol.basis = 'def2-svp'
            auxbasis = f'def2universal-jkfit'
            mol.spin = 1  # 1 unpaired electron
            mol.build()

            params = {
                'auxbasis': auxbasis,
                'mol': mol,
                'grid': BeckeGrid(mol),
                'verbose': 0
            }
            params.update(thc_kwargs)

            runner = THCMP2_DFSCF.with_thc(thc_class)(params)
            ump2_thc = runner.kernel()[0] * KCALPERMOL_PER_HARTREE

            ump2_ref = DFMP2.from_config(params)
            ref = ump2_ref.kernel()[0] * KCALPERMOL_PER_HARTREE

            diff = ref - ump2_thc

            logger.info(f"=== Error / electron (DF) ===\nDIFF = {diff}")
            assert abs(diff) <= threshold


    def test_ump2_radical(self):
        for thc_name, (thc_class, thc_kwargs, threshold) in self.thcs_mo.items():
            logger.info(f"\n\n====== UMP2 Testing {thc_name} on radical ======")
            mol = gto.Mole()
            # Standard geometry for Methyl Radical
            mol.atom = '''
            C 0.000000 0.000000 0.000000
            H 0.000000 1.079000 0.000000
            H 0.934449 -0.539500 0.000000
            H -0.934449 -0.539500 0.000000
            '''
            mol.basis = 'def2-svp'
            auxbasis = f'def2universal-jkfit'
            mol.spin = 1  # 1 unpaired electron
            mol.build()

            params = {
                'auxbasis': auxbasis,
                'mol': mol,
                'grid': BeckeGrid(mol),
                'verbose': 0
            }
            params.update(thc_kwargs)

            runner = THCMP2_DFSCF.with_thc(thc_class)(params)
            ump2_thc = runner.kernel()[0] * KCALPERMOL_PER_HARTREE

            ump2_ref = DFMP2.from_config(params)
            ref = ump2_ref.kernel()[0] * KCALPERMOL_PER_HARTREE

            diff = ref - ump2_thc

            logger.info(f"=== Error / electron (DF) ===\nDIFF = {diff}")
            assert abs(diff) <= threshold

    def test_reaction_mp2(self):
        for thc_name, (thc_class, thc_kwargs, threshold) in self.thcs_mo.items():
            reaction = [r for r in DIET_GMTKN_55_150_MOLS if r.dataset== "W4-11" and r.reaction==38][0]
            logger.info(f"==== {reaction.reaction} ====")

            def calculate_mp2(m: MolInfo) -> tuple[float, ...]:
                basis = 'def2-svp'
                auxbasis = f'def2universal-jkfit'
                basis_dict = {'default': basis}

                mol = gto.Mole()
                mol.spin = m.spin
                mol.charge = m.charge
                mol.atom = m.geom
                mol.basis = basis_dict
                mol.build()

                params = {
                    'auxbasis': build_auxbasis(mol, auxbasis),
                    'mol': mol,
                    'grid': BeckeGrid(mol=mol, level=0),
                }

                runner_pyscf = DFMP2.from_config(params)
                e_total_hartree_pyscf = abs(runner_pyscf.kernel()[-1])

                params.update(thc_kwargs)
                runner = THCMP2_DFSCF.with_thc(thc_class)(params)
                e_total_hartree_thc = abs(runner.kernel()[-1])

                logger.info(
                    f"DIFF: {m.filename} THC: {e_total_hartree_thc}; RI: {e_total_hartree_pyscf}; DIFF: {e_total_hartree_thc - e_total_hartree_pyscf}")

                return e_total_hartree_thc, e_total_hartree_pyscf

            energy_in_thc = 0.0
            energy_in_pyscf = 0.0
            for educt in reaction.educts:
                e_thc, e_pyscf = calculate_mp2(educt.mol)
                weight = abs(educt.weight)
                energy_in_thc += weight * e_thc
                energy_in_pyscf += weight * e_pyscf

            energy_out_thc = 0.0
            energy_out_pyscf = 0.0
            for product in reaction.products:
                e_thc, e_pyscf = calculate_mp2(product.mol)
                weight = abs(product.weight)
                energy_out_thc += weight * e_thc
                energy_out_pyscf += weight * e_pyscf

            energy_diff_thc = (energy_out_thc - energy_in_thc) * KCALPERMOL_PER_HARTREE
            energy_diff_pyscf = (energy_out_pyscf - energy_in_pyscf) * KCALPERMOL_PER_HARTREE
            energy_diff_diff = energy_diff_pyscf - energy_diff_thc

            logger.info(f"""
            DELTA THC:\t({energy_out_thc} - {energy_in_thc}) * {KCALPERMOL_PER_HARTREE} = {energy_diff_thc}
            DELTA DF:\t({energy_out_pyscf} - {energy_in_pyscf}) * {KCALPERMOL_PER_HARTREE} = {energy_diff_pyscf}
            DELTA CCSD:\t {reaction.energy_diff_kcalmol}

            DELTA THC-DF: {energy_diff_diff}
            DELTA DF-CCSD: {reaction.energy_diff_kcalmol - energy_diff_pyscf}
            DELTA THC-CCSD: {reaction.energy_diff_kcalmol - energy_diff_thc}
            """)

            assert (abs(energy_diff_diff) <= 1 or abs(reaction.energy_diff_kcalmol - energy_diff_pyscf) > abs(
                reaction.energy_diff_kcalmol - energy_diff_thc))

            if (energy_diff_pyscf - reaction.energy_diff_kcalmol) >= (reaction.energy_diff_kcalmol / 5):
                logger.warning(f"MP2 seems to be failing here")

    def test_reaction_hf(self):
        for thc_name, (thc_class, thc_kwargs, threshold) in self.thcs_ao.items():
            reaction = [r for r in DIET_GMTKN_55_150_MOLS if r.dataset== "W4-11" and r.reaction==38][0]
            logger.info(f"==== {reaction.file_name} ====")

            def calculate_scf(m: MolInfo) -> tuple[float, ...]:
                basis = 'def2-svp'
                auxbasis = f'def2universal-jkfit'
                basis_dict = {'default': basis}

                mol = gto.Mole()
                mol.spin = m.spin
                mol.charge = m.charge
                mol.atom = m.geom
                mol.basis = basis_dict
                mol.build()

                params = {
                    'auxbasis': build_auxbasis(mol, auxbasis),
                    'mol': mol,
                    'grid': BeckeGrid(mol=mol, level=0),
                    'thc_threshold': 0.1,
                    'min_exact_cycles': 1,
                }

                runner_pyscf = DFSCF.from_config(params)
                e_total_hartree_pyscf = runner_pyscf.kernel()[-1]

                params.update(thc_kwargs)
                runner = THCSCF.with_thc(thc_class)(params)
                e_total_hartree_thc = runner.kernel()[-1]

                logger.info(
                    f"DIFF: {m.filename} THC: {e_total_hartree_thc}; RI: {e_total_hartree_pyscf}; DIFF: {e_total_hartree_thc - e_total_hartree_pyscf}")

                params.update(thc_kwargs)
                runner = THCSCF.with_thc(thc_class)(params)
                e_total_hartree_thc = runner.kernel()[-1]

                logger.info(
                    f"DIFF: {m.filename} THC: {e_total_hartree_thc}; RI: {e_total_hartree_pyscf}; DIFF: {e_total_hartree_thc - e_total_hartree_pyscf}")

                return e_total_hartree_thc, e_total_hartree_pyscf

            energy_in_thc = 0.0
            energy_in_pyscf = 0.0
            for educt in reaction.educts:
                weight = abs(educt.weight)
                e_thc, e_pyscf = calculate_scf(educt.mol)
                energy_in_thc += weight * e_thc
                energy_in_pyscf += weight * e_pyscf

            energy_out_thc = 0.0
            energy_out_pyscf = 0.0
            for product in reaction.products:
                weight = abs(product.weight)
                e_thc, e_pyscf = calculate_scf(product.mol)
                energy_out_thc += weight * e_thc
                energy_out_pyscf += weight * e_pyscf

            energy_diff_thc = (energy_out_thc - energy_in_thc) * KCALPERMOL_PER_HARTREE
            energy_diff_pyscf = (energy_out_pyscf - energy_in_pyscf) * KCALPERMOL_PER_HARTREE
            energy_diff_diff = energy_diff_pyscf - energy_diff_thc

            logger.info(f"""
            DELTA THC:\t({energy_out_thc} - {energy_in_thc}) * {KCALPERMOL_PER_HARTREE} = {energy_diff_thc}
            DELTA DF:\t({energy_out_pyscf} - {energy_in_pyscf}) * {KCALPERMOL_PER_HARTREE} = {energy_diff_pyscf}
            DELTA CCSD:\t {reaction.energy_diff_kcalmol}
            
            DELTA THC-DF: {energy_diff_diff}
            DELTA DF-CCSD: {reaction.energy_diff_kcalmol - energy_diff_pyscf}
            DELTA THC-CCSD: {reaction.energy_diff_kcalmol - energy_diff_thc}
            """)

            assert (abs(energy_diff_diff) <= 1 or abs(reaction.energy_diff_kcalmol - energy_diff_pyscf) > abs(
                reaction.energy_diff_kcalmol - energy_diff_thc))

            if (energy_diff_pyscf - reaction.energy_diff_kcalmol) >= (reaction.energy_diff_kcalmol / 5):
                logger.warning(f"MP2 seems to be failing here")

    def test_hf(self):
        for thc_name, (thc_class, thc_kwargs, threshold) in self.thcs_ao.items():
            for mol_file in sorted(mols):
                with self.subTest(mol=mol_file, thc=thc_name):
                    logging.getLogger().setLevel(logging.INFO)

                    logger.info(f"\n\n====== HF Testing {thc_name} on {mol_file} ======")
                    mol = self.get_mol(mol_file)
                    auxbasis = f'def2universal-jkfit'

                    params = {
                        'auxbasis': auxbasis,
                        'mol': mol,
                        'grid': BeckeGrid(mol),
                        'thc_only': True,
                        'chkfile': ''
                    }
                    params.update(thc_kwargs)

                    hfthc = THCSCF.with_thc(thc_class)(params)
                    val = hfthc.kernel()[0]

                    hfri = DFSCF.from_config(params)
                    ref = hfri.kernel()[0]

                    diff = ref - val

                    logger.info(f"=== Error / electron (DF) ===\nDIFF = {diff}")
                    assert abs(diff) <= threshold

    def test_mp2(self):
        for thc_name, (thc_class, thc_kwargs, threshold) in self.thcs_mo.items():
            for mol_file in sorted(mols):
                with self.subTest(mol=mol_file, thc=thc_name):
                    logger.info(f"\n\n====== MP2 Testing {thc_name} on {mol_file} ======")

                    mol = self.get_mol(mol_file)
                    auxbasis = f'def2universal-jkfit'

                    params = {
                        'auxbasis': auxbasis,
                        'mol': mol,
                        'grid': BeckeGrid(mol=mol, level=0, prune=treutler_prune),
                    }
                    params.update(thc_kwargs)

                    runner = THCMP2_DFSCF.with_thc(thc_class)(params)
                    val = runner.kernel()[0]

                    mp2 = DFMP2.from_config(params)
                    ref = mp2.kernel()[0]

                    diff = ref - val

                    logger.info(f"=== REFERENCE = {ref}")
                    logger.info(f"=== THC VALUE = {val}")
                    logger.info(f"=== Error / electron (DF) ===\nDIFF = {diff}")
                    assert abs(diff) <= threshold

    def test_mp2_sos(self):
        for thc_name, (thc_class, thc_kwargs, threshold) in self.thcs_mo.items():
            for mol_file in sorted(mols):
                with self.subTest(mol=mol_file, thc=thc_name):
                    logger.info(f"\n\n====== MP2-SOS Testing {thc_name} on {mol_file} ======")
                    mol = self.get_mol(mol_file)
                    auxbasis = f'def2universal-jkfit'

                    params = {
                        'auxbasis': auxbasis,
                        'mol': mol,
                        'grid': BeckeGrid(mol),
                    }
                    params.update(thc_kwargs)

                    runner = THCMP2_SOS.with_thc(thc_class)(params)
                    val = runner.kernel()[0]

                    mp2 = DFMP2.from_config(params)
                    ref = mp2.kernel()[0]

                    diff = ref - val

                    logger.info(f"=== Error / electron (DF) ===\nDIFF = {diff}")
                    assert abs(diff) <= 3e-2
