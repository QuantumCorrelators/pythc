# PyTHC

![logo](pythc_logo.svg)



Python-based implementations of Tensor Hypercontraction (THC).
Preprint paper available at: https://arxiv.org/abs/2608.17885.

## Description

This project implements Tensor Hypercontraction for use with Electron Repulsion Integrals (ERIs). It is built in pure Python on top of:

* [PySCF](https://github.com/pyscf/pyscf)
* [NumPy](https://numpy.org/)
* [SciPy](https://scipy.org/)
* [Cotengra](https://cotengra.readthedocs.io/en/latest/)

## Installation

The project is not yet published to PyPI, so you will need to install it directly from Git.

### Installing into a virtual environment

On Debian-based distributions, ensure that the following packages are installed and up to date: `python3`, `python3-pip`, `python3-venv`, `gcc`, `gfortran`, `libopenblas-dev`. (Tested on Ubuntu 24.04/26.04).

To create a new virtual environment, run:

```shell
python3 -m venv .venv
```

Activate the environment with:

```shell
source .venv/bin/activate
```

Now install the project using:

```shell
pip install --upgrade pip
pip install git+https://github.com/QuantumCorrelators/pythc.git
```

### Installing with Conda / Mamba

To install the library in a new environment using Conda or Mamba:

```shell
mamba create -n my_env python=3.12
mamba activate my_env
pip install git+ssh://git@github.com/QuantumCorrelators/pythc.git
```

### Installing into an existing uv project 

This is the easiest way to install the project. You can create a new uv project using `uv init my-project-name`.

If you already have a project set up using the [uv](https://docs.astral.sh/uv/) package manager, running:

```shell
uv add git+ssh://git@github.com/QuantumCorrelators/pythc.git
```

will add the package to your `pyproject.toml`.


## Developer Instructions

If you want to contribute to `PyTHC`, you can clone the repository and install the dependencies either via mamba
```shell
mamba create -n pythc python=3.12
mamba activate pythc
git clone git@github.com:QuantumCorrelators/pythc.git
cd pythc
pip install . 
```

or via uv
```shell
git clone git@github.com:QuantumCorrelators/pythc.git
cd pythc
uv sync
```
given uv is installed on your system.


## Basic Usage

Detailed examples can be found in [`src/pythc/examples`](src/pythc/examples).

All THC calculations require a PySCF `Molecule` object as well as a completed mean-field calculation:

```python
from pyscf import gto, scf

mol = gto.M('path/to/your/mol', basis='cc-pvdz')
mf = scf.RHF(mol)
mf.kernel()
```

From those objects, we can build the THC representation by importing the appropriate THC class and grid builder:

```python
from pythc.thc.ls_ri_becke import LS_RI_Becke
from pythc.grid import BeckeGrid

grid_builder = BeckeGrid(mol)
thc_builder = LS_RI_Becke(mol=mol, grid=grid_builder, auxbasis='cc-pvdz-ri')
thc_eri = thc_builder.build(mode='ao')
```
Supported modes are `ao` for the AO-ERI $(\mu \nu|\lambda \sigma)$ and `ov` for the occupied-virtual block if the MO-ERI $(i j|a b)$.
The THC representation **must** be built in `mode='ao'` for the HF-SCF calculation.

Once the THC ERI is built, we can retrieve the $X_\mu^P$ and $Z^{P Q}$ matrices using:

```python
X, Z = thc_eri.get_X_Z()
```

or construct the full 4-dimensional quantity:

$$
(\mu \nu | \lambda \sigma) = \sum_{P Q} X_\mu^P X_\nu^P Z^{P Q} X_\lambda^Q X_\sigma^Q
$$

```python
full_eri = thc_eri.get_full()
```

The matrices can also be saved to HDF5 format:

```python
thc_eri.save('path/to/place/eri.hdf5')
```

Alternatively, you can load an existing ERI from disk:

```python
from pythc.thc.thc_base import ThcEri

thc_eri = ThcEri.from_file("path/to/place/eri.hdf5")
```

With this THC ERI interface, you can implement improved quantum chemistry algorithms. This project currently implements:

1. The HF-SCF algorithm (`THC_RHF` / `THC_UHF`)
2. Møller-Plesset Perturbation Theory of 2nd order (`mp2_energy_laplace`)

The HF-SCF implementations inherit from PySCF's `RHF`/`UHF` classes. We build in `ao` mode so the THC does not need to be rebuilt between SCF iterations:

```python
from pythc.grid import BeckeGrid
from pythc.thc.ls_ri_becke import LS_RI_Becke
from pythc.methods.hf import THC_RHF

auxbasis = 'cc-pvdz-ri'
grid_builder = BeckeGrid(mol)
thc_builder = LS_RI_Becke(mol=mol, grid=grid_builder, auxbasis=auxbasis)
thc_eri = thc_builder.build(mode='ao')

mf = THC_RHF(mol,
             thc_eri,
             auxbasis,
             verbose=4,
             thc_threshold=0.1,
             min_exact_cycles=1)
mf.kernel()
```

The usage of THC during SCF can be configured using `thc_threshold` and `min_exact_cycles`. The `thc_threshold` parameter specifies the energy difference threshold (in Hartree) below which the solver switches from exact RI/DF cycles to THC, provided at least `min_exact_cycles` iterations have passed. If the parameter `thc_only=True` is provided, the SCF will be converged exclusively using the THC-ERI ([details](src/pythc/examples/scf_with_ls_snri_cholesky.py)).

To calculate a Laplace-transformed MP2 energy contribution with 10 integration points, build the THC in `ov` mode:

```python
from pythc.grid import BeckeGrid
from pythc.thc.ls_ri_becke import LS_RI_Becke
from pythc.methods.mp2 import mp2_energy_laplace

grid_builder = BeckeGrid(mol)
thc_builder = LS_RI_Becke(mol=mol, mo_coeff=mf.mo_coeff, grid=grid_builder,
                          auxbasis='cc-pvdz-ri')
thc_eri = thc_builder.build(mode='ov')

mp2_e_thc = mp2_energy_laplace(mol, mf, thc_eri, n_laplace=10)
```

The THC representation **must** be built in `mode='ov'` for the MP2 calculation.

