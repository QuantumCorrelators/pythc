# PyTHC

![logo](pythc_logo.svg)



Python-based implementations of Tensor Hypercontraction (THC).
Preprint paper available at: https://arxiv.org/abs/2608.17885.
The full code can be found at https://github.com/QuantumCorrelators/pythc.

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
pip install pythc
```

### Installing with Conda / Mamba

To install the library in a new environment using Conda or Mamba:

```shell
mamba create -n my_env python=3.12
mamba activate my_env
pip install pythc 
```

### Installing into an existing uv project 

This is the easiest way to install the project. You can create a new uv project using `uv init my-project-name`.

If you already have a project set up using the [uv](https://docs.astral.sh/uv/) package manager, running:

```shell
uv add pythc 
```

will add the package to your `pyproject.toml`.

### Optional experiment tracking

`PyTHC` works without the external [`tracking`](https://github.com/nikplx/tracking) library. When it is not installed, checkpoint and metric logging calls are no-ops. To enable tracking, install it manually:

```shell
pip install git+https://github.com/nikplx/tracking.git
```

It is intentionally not a PyPI dependency while the `tracking` package is not published there.


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

To also install the optional `tracking` integration while developing, run:

```shell
uv sync --group tracking
```


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

thc = LS_RI_Becke(mol=mol, auxbasis='cc-pvdz-ri')
thc_eri = thc.build(mode='ao')
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
from pythc.thc.thc_base import THC_ERI

thc_eri = THC_ERI.from_file("path/to/place/eri.hdf5")
```

With this THC ERI interface, you can implement improved quantum chemistry algorithms. This project currently implements:

1. The HF-SCF algorithm 
2. Møller-Plesset Perturbation Theory of 2nd order
3. Random Phase Approximation (RPA)

The THC J/K engine  (`pythc.methods.scf`), which implements the `pyscf.df.DF` interface and can be dropped into any stock PySCF SCF via `with_df`. Spin-block splitting (UHF/GHF) and the RHF/UHF/GHF Fock factors then stay in PySCF. The THC ERI is automatically built in `ao` mode so the THC does not need to be rebuilt between SCF iterations:

```python
from pyscf import scf
from pythc.thc.ls_ri_becke import LS_RI_Becke
from pythc.methods.scf import THCDF

auxbasis = 'cc-pvdz-ri'
thc = LS_RI_Becke(mol=mol, auxbasis=auxbasis)
dfo = THCDF(mol, thc, auxbasis=auxbasis)
mf = scf.RHF(mol).density_fit(with_df=dfo)
mf.kernel()
```

`density_fit` re-classes the SCF object so that `get_jk` routes through `with_df`; passing `with_df` directly avoids building a throwaway DF object. The same `dfo` can be dropped into `scf.UHF(mol)` or `scf.GHF(mol)` in place of `scf.RHF(mol)`; spin-block splitting and the RHF/UHF/GHF Fock factors stay in PySCF.

Each of J and K can independently be routed to THC or to conventional DF/RI via the `with_j_thc` / `with_k_thc` flags. In particular `with_j_thc=False` selects "only-K" mode (Coulomb through DF/RI, exchange through THC):

```python
dfo = THCDF(mol, thc, auxbasis=auxbasis, with_j_thc=False, with_k_thc=True)
```

For an exact-DF baseline that switches to the THC density-difference ansatz after a certain threshold in the energy difference it reached, use the `RHF`, `UHF` and `GHF`  implementations from `pythc.methods.scfdd`:
```python
from pythc.methods import scfdd

mf = scfdd.RHF(mol, thc, auxbasis, thc_threshold=0.1)
mf.kernel()
```

`scfdd.UHF` and `scfdd.GHF` take the same arguments and switch to THC on the unrestricted / generalized density difference respectively:

```python
mf = scfdd.UHF(mol, thc, auxbasis, thc_threshold=0.1)
mf.kernel()
```

To calculate a Laplace-transformed MP2 energy contribution with 10 integration points use:

```python
from pythc.thc.ls_ri_becke import LS_RI_Becke
from pythc.methods.mp2 import LaplaceMP2

thc = LS_RI_Becke(mol=mol, auxbasis='cc-pvdz-ri')
mp2_e_thc = LaplaceMP2(mf, thc, n_laplace=10).kernel()
```

The THC representation **must** be built in `mode='ov'` for the MP2 calculation.

To calculate the direct-RPA correlation energy use:

```python
from pythc.thc.ls_ri_becke import LS_RI_Becke
from pythc.methods.rpa import RPA

thc = LS_RI_Becke(mol=mol, auxbasis='cc-pvdz-ri')
rpa_e_thc = RPA(mf, thc).kernel()
```

The THC representation **must** be built in `mode='ov'` for the RPA calculation.

