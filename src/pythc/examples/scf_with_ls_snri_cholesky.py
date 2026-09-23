import logging
import sys

from pyscf import gto, scf

from pythc.methods.thc_df import THCDF
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

    # Pure THC-SCF: stock PySCF SCF with the THC integral object.
    # density_fit() re-classes mf so get_jk routes through with_df; passing
    # with_df directly avoids building a throwaway DF object.
    # (For an RI baseline that switches to THC mid-SCF, use THC_RHF from
    # pythc.methods.hf instead.)
    mf = scf.RHF(mol).density_fit(with_df=THCDF(mol, eri, auxbasis=auxbasis))
    scf_e = mf.kernel()

    print(f'SCF E: {scf_e}')

if __name__ == '__main__':
    main()
