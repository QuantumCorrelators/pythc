import logging
import sys

from pyscf import gto

from pythc.methods import scfdd
from pythc.methods.mp2 import LaplaceUMP2
from pythc.thc.ls_ri_cholesky import LS_RI_Cholesky

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

    thc_ao = LS_RI_Cholesky(mol, auxbasis=auxbasis, cholesky_threshold=1e-9)
    mf = scfdd.UHF(mol=mol, auxbasis=auxbasis, thc=thc_ao)
    mf.kernel()

    print(f'SCF E: {mf.e_tot} after {mf.cycles} cycles')

    thc_mo = LS_RI_Cholesky(mol, mo_coeff=mf.mo_coeff, auxbasis=auxbasis, cholesky_threshold=1e-5)
    mp2e = LaplaceUMP2(mf, thc_mo, n_laplace=10).kernel()
    print(f'MP2 E corr: {mp2e}')

if __name__ == '__main__':
    main()
