import numpy as np
from ase.visualize import view
from pyscf.pbc import gto, scf

if __name__ == '__main__':
    a = np.array([
        [0.0000, 1.7834, 1.7834],
        [1.7834, 0.0000, 1.7834],
        [1.7834, 1.7834, 0.0000]
    ])

    atoms_xyz = '''C 0.0000 0.0000 0.0000
                  C 0.8917 0.8917 0.8917'''

    cell = gto.Cell()
    cell.atom = atoms_xyz
    cell.pseudo = 'gth-pade'
    cell.basis = 'gth-szv'
    cell.unit = 'A'
    cell.a = a
    cell.build()
    #%%
    from ase import Atoms

    ase_atoms = Atoms(
        symbols=[atom[0] for atom in cell._atom],
        positions=cell.atom_coords(),
        cell=cell.a,
        pbc=True,

    )
    supercell = ase_atoms.repeat((2,2,2))
    view(supercell, block=False)