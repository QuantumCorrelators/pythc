import logging
import sys

from pyscf import gto

from pythc.methods.hf import THC_RHF
from pythc.thc.ls_snri_cholesky import LS_snRI_Cholesky

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

    thc = LS_snRI_Cholesky(mol=mol, cholesky_threshold=1e-8)
    eri = thc.build(mode='ao')

    mf = THC_RHF(mol, eri, auxbasis, thc_only=True)
    scf_e = mf.kernel()

    print(f'SCF E: {scf_e}')

if __name__ == '__main__':
    main()