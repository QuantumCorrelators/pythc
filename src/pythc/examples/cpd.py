import logging
import pathlib
import sys

from pyscf import gto, scf
from pyscf.mp.dfmp2 import DFMP2

from pythc.methods.mp2 import LaplaceRMP2
from pythc.thc.ls_ri_cholesky import LS_RI_Cholesky
from pythc.thc.thc_base import ThcEri

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def main():
    water_cluster = """
    H            0.087529        0.023820        0.930805
    O            0.657172        0.599414        0.406256
    H            0.792448        1.344387        1.004310
    """

    mol = gto.Mole()
    mol.basis = 'cc-pvdz'
    mol.atom = water_cluster
    mol.build()

    auxbasis = 'cc-pvdz-ri'

    mf = scf.RHF(mol)
    mf = mf.density_fit(auxbasis=auxbasis)
    mf.verbose = 4
    mf.kernel()

    if pathlib.Path("eri.hdf5").exists():
        eri = ThcEri.from_file("eri.hdf5")
    else:
        thc = LS_RI_Cholesky(mol=mol, auxbasis=auxbasis, mo_coeff=mf.mo_coeff,
                             cholesky_threshold=1e-5)
        eri = thc.build(mode='ov')
        eri.save("eri.hdf5")

    mp2_ref = DFMP2(mf).kernel()[0]
    print(f'MP2 RI Reference: {mp2_ref}')

    mp2e = LaplaceRMP2(mol, mf, eri, n_laplace=10)
    print(f'MP2 E corr: {mp2e}')


if __name__ == '__main__':
    main()
