from pyscf import gto, scf

from pythc.methods.scf import THCDF
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

mf = scf.GHF(mol=mol).density_fit(auxbasis=auxbasis)
mf.verbose = 4
e_ghf = mf.kernel()
dm_exact = mf.make_rdm1()

thc = LS_snRI_Cholesky(mol, cholesky_threshold=1e-9)
# Pure THC-SCF: stock PySCF GHF with the THC integral object.
# Spin-block splitting and the spin-orbital Fock factors stay in PySCF.
dfo = THCDF(mol, thc, auxbasis=auxbasis).build()

thc_mf = scf.GHF(mol).density_fit(with_df=dfo)
thc_mf.verbose = 4
e_ghf_thc = thc_mf.kernel()
dm_thc = thc_mf.make_rdm1()


print(f"E GHF: {e_ghf} <S^2>={mf.spin_square()[0]:.4f}; "
      f"THC: {e_ghf_thc} <S^2>={thc_mf.spin_square()[0]:.4f}; "
      f"ERROR: {e_ghf_thc - e_ghf}")

# Cross-restart: converge each theory starting from the other's density.
# If both keep their energies, the minima coexist in both theories and any
# gap is basin selection (note <S^2>), not an integral error. If one side
# collapses to the other's minimum, that minimum is likely spurious.
mf_cross = scf.GHF(mol=mol).density_fit(auxbasis=auxbasis)
mf_cross.verbose = 0
e_xrt = mf_cross.kernel(dm0=dm_thc)

thc_mf_cross = scf.GHF(mol).density_fit(with_df=dfo)
thc_mf_cross.verbose = 0
e_trt = thc_mf_cross.kernel(dm0=dm_exact)

print(f"exact-from-THC-dm: {e_xrt} <S^2>={mf_cross.spin_square()[0]:.4f}")
print(f"thc-from-exact-dm: {e_trt} <S^2>={thc_mf_cross.spin_square()[0]:.4f}")
