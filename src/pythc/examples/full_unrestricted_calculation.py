import logging
import sys

from pyscf import gto
from pythc.methods.hf import THC_UHF
from pythc.methods.mp2 import ump2_energy_laplace
from pythc.thc.ls_ri_cholesky import LS_RI_Cholesky
from pythc.thc.ls_aux_becke import LS_Aux_Becke

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def main():
    mol = gto.Mole()
    mol.basis = 'cc-pvdz'
    mol.atom = '''
    C 0.000000 0.000000 0.000000
    H 0.000000 1.079000 0.000000
    H 0.934449 -0.539500 0.000000
    H -0.934449 -0.539500 0.000000
    '''
    mol.spin = 1  # 1 unpaired electron
    mol.build()

    auxbasis = 'cc-pvdz-ri'

    ao_thc = LS_RI_Cholesky(mol, auxbasis=auxbasis, cholesky_threshold=1e-9)
    ao_eri = ao_thc.build(mode='ao')

    mf = THC_UHF(mol=mol,
                 eri_thc=ao_eri,
                 auxbasis=auxbasis,
                 thc_threshold=0.1,
                 min_exact_cycles=1)

    mf.kernel()

    print(f'SCF E: {mf.e_tot} after {mf.cycles} cycles')

    thc = LS_Aux_Becke(mol=mol, fit_auxbasis='cc-pvqz', mo_coeff=mf.mo_coeff)
    eri_mo = thc.build_unrestricted(mode='ov')

    mp2e = ump2_energy_laplace(mol, mf, eri_mo, n_laplace=10)
    print(f'MP2 E corr: {mp2e}')

if __name__ == '__main__':
    main()
