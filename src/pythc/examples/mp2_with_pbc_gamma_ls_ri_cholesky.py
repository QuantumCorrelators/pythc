import logging
import sys
import numpy as np

# 1. Import PBC variants from PySCF
from pyscf.pbc import gto, scf
from pyscf.pbc.mp.mp2 import RMP2
from pyscf.pbc.tools import super_cell

# 2. Import your existing MP2 logic and the NEW PBC THC class
from pythc.methods.mp2 import mp2_energy_laplace
from pythc.pbc.thc.pbc_ls_ri_cholesky import PBC_LS_RI_Cholesky

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def main():
    # 3. Define the unit cell instead of a molecule
    cell = gto.Cell()
    cell.atom = '''
    He 0.0 0.0 0.0
    He 1.0 1.0 1.0
    '''
    cell.a = np.eye(3) * 2.0  # Lattice vectors (cubic cell, 2 Bohr length)
    cell.basis = 'gth-dzv'  # PBC typically uses specialized bases (e.g., GTH)
    cell.pseudo = 'gth-pade'  # Pseudopotentials are usually required in PBC
    cell.build()

    # We will use periodic density fitting
    auxbasis = 'cc-pvdz-ri'

    gamma_cell = super_cell(cell, [2, 2, 2])

    # 4. Run Gamma-point SCF
    # (By default, pbc.scf.RHF runs at the Gamma point unless kpts are specified)
    mf = scf.RHF(gamma_cell)
    mf = mf.density_fit(auxbasis=auxbasis)
    mf.verbose = 4
    mf.kernel()

    # 5. Initialize the NEW PBC THC implementation
    # Notice the interface is exactly the same, just passing `cell` instead of `mol`
    thc = PBC_LS_RI_Cholesky(cell=gamma_cell, mo_coeff=mf.mo_coeff, cholesky_threshold=1e-5)

    # 6. Build the THC ERI (X and Z) using Gamma-point real-space uniform grids
    eri = thc.build(mode='ov')

    # Reference PySCF DF-MP2 (for comparison)
    mp2_ref = RMP2(mf).kernel()[0]
    print(f'PBC MP2 RI Reference: {mp2_ref}')

    # 7. Compute MP2 energy using THC
    # (Assuming mp2_energy_laplace is written generally enough to handle the
    #  eri format, which it should be if ThcEri implements the standard interface)
    mp2e = mp2_energy_laplace(gamma_cell, mf, eri, n_laplace=10)
    print(f'PBC MP2 E corr: {mp2e}')


if __name__ == '__main__':
    main()