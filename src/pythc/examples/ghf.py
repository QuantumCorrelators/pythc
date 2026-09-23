from pyscf import gto, scf

from pythc.methods.thc_df import THCDF
from pythc.thc.ls_snri_cholesky import LS_snRI_Cholesky

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

mf = scf.GHF(mol=mol)
mf.density_fit()
mf.verbose = 4
e_ghf = mf.kernel()

thc = LS_snRI_Cholesky(mol, cholesky_threshold=1e-9)
eri = thc.build()

# Pure THC-SCF: stock PySCF GHF with the THC integral object.
# Spin-block splitting and the spin-orbital Fock factors stay in PySCF.
thc_mf = scf.GHF(mol).density_fit(with_df=THCDF(mol, eri, auxbasis=auxbasis))
thc_mf.verbose = 4
e_ghf_thc = thc_mf.kernel()


print(f"E GHF: {e_ghf}; THC: {e_ghf_thc}; ERROR: {e_ghf_thc - e_ghf}")
