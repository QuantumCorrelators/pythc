import logging
import sys

from pyscf import gto
from pythc.thc.ls_ri_cholesky import LS_RI_Cholesky

from pythc.methods import scfdd

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

    thc = LS_RI_Cholesky(mol=mol, auxbasis=auxbasis)

    mf = scfdd.RHF(mol,
                   thc,
                   auxbasis,
                   with_j_thc=False,
                   with_k_thc=True,
                   # minimum number of RI cycles may be specified instead
                   #min_df_cycles=4,
                   thc_threshold=0.1)
    scf_e = mf.kernel()

    print(f'SCF E: {scf_e}')

if __name__ == '__main__':
    main()