import logging
import sys

import numpy as np
import pyscf.pbc.gto as pbcgto
from pyscf.pbc.tools import pyscf_ase
from ase.build import bulk
import matplotlib.pyplot as plt


logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

from pythc.pbc.dft.ks_thc import THCHybridKRKS

c = bulk('C', 'diamond', a=3.5668)
print(c.get_volume())

cell = pbcgto.Cell()
cell.atom = pyscf_ase.ase_atoms_to_pyscf(c)
cell.a = np.array(c.cell)
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.verbose = 5
cell.build()

bp = pyscf_ase.bandpath(cell)

kpts = cell.make_kpts([2,2,2])
mf = THCHybridKRKS(cell, kpts=kpts)
mf.xc = 'pbe0'
mf.run()

band_kpts = cell.get_abs_kpts(bp.kpts)
e_kn = mf.get_bands(band_kpts)[0]

kmf = cell.KRKS(kpts=kpts)
kmf.xc = 'pbe0'
kmf.run()

e_kn_2 = kmf.get_bands(band_kpts)[0]

nocc = cell.nelectron // 2
au2ev = 27.21139
e_kn = (np.array(e_kn) - mf.get_fermi()) * au2ev
e_kn_2 = (np.array(e_kn_2) - kmf.get_fermi()) * au2ev

fig, ax = plt.subplots(figsize=(5, 6))
pyscf_ase.plot_band_structure(bp, e_kn, ax, 'black')
pyscf_ase.plot_band_structure(bp, e_kn_2, ax, 'blue')

emin = -1*au2ev
emax = 1*au2ev
ax.set_ylim(emin, emax)

plt.tight_layout()
plt.show()