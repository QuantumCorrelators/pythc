import importlib.util
import json
import logging
import os
import subprocess
from typing import Any

import cotengra as ctg
import numpy as np
import opt_einsum as oe
import psutil
import pyscf.lib as pyscflib
import pytblis as pt
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from pyscf import df

logger = logging.getLogger()

def cpu_count() -> int:
    return int(os.getenv("OMP_NUM_THREADS", os.cpu_count()))

def einsum(*args, **kwargs):
    """
    Drop-in replacement for numpy.einsum

    Can be configured via PYTHC_EINSUM environment variable to either to TBLIS ("tblis"), standard numpy ("numpy")
    or opt_einsum ("oe_numpy").
    """
    backend = os.environ.get("PYTHC_EINSUM", "opt_einsum")

    if backend == "tblis":
        pt.set_num_threads(cpu_count())
        return pt.einsum(*args, **kwargs, optimize='optimal')
    elif backend == "numpy":
        return np.einsum(*args, optimize=True, **kwargs)
    elif backend == "opt_einsum":
        return oe.contract(*args, **kwargs)
    elif backend == "cotengra":
        return contract_cotengra(*args)
    else:
        raise ValueError(f"Unknown einsum backend: {backend}")

rng = np.random.default_rng(42)

def pinv(M: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Pseudo invert a hermitian matrix.
    
    Builds a Moore-Penrose pseudo inversion using an eigenvalue decomposition. For safety, all values smaller than
    1e-15 are set to zero.

    :param M: Hermitian matrix
    :param epsilon: Percentage of the biggest eigenvalue to use as a cutoff
    :return: Pseudo inverted matrix
    """
    M[np.abs(M) < 1e-15] = 0.0

    if not M.flags['F_CONTIGUOUS']:
        M = np.asfortranarray(M)

    logger.info("pseudo inverting matrix of shape %s", M.shape)
    eig_vals, eig_vecs = np.linalg.eigh(M)

    max_eig = eig_vals[-1]
    thresh = max(epsilon * max_eig, 1e-12)
    mask = eig_vals > thresh

    inv_eigvals = np.zeros_like(eig_vals)
    inv_eigvals[mask] = 1.0 / eig_vals[mask]
    logger.info(f"selected {len(inv_eigvals[mask])} eigenvalues")

    S_inv = (eig_vecs * inv_eigvals) @ eig_vecs.T
    return S_inv


def pseudo_inv_sqrt(M: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Builds the M^(-1/2)

    Builds square root of the pseudo inversion of a hermitian matrix M using an eigenvalue decomposition.
    :param M:  Hermitian matrix
    :param epsilon:
    :return: Pseudo inverse square root
    """
    M = M + np.eye(M.shape[0]) * 1e-8
    eig_vals, eig_vecs = np.linalg.eigh(M)
    thresh = epsilon * eig_vals[-1]
    mask = eig_vals > thresh

    inv_sqrt_vals = np.zeros_like(eig_vals)
    inv_sqrt_vals[mask] = 1.0 / np.sqrt(eig_vals[mask])

    return (eig_vecs * inv_sqrt_vals) @ eig_vecs.T

def current_memory() -> float:
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss # Resident Set Size
    mem_mb = mem_bytes / 1024 / 1024
    return mem_mb

def latest_tag() -> str:
    import git
    repo = git.Repo(search_parent_directories=True)
    tags = sorted(repo.tags, key=lambda t: t.commit.committed_datetime)
    latest_tag = tags[-1]

    return latest_tag.name

def get_l3_cache_size_bytes() ->int:
    result = subprocess.run(['lscpu', '-J', '--bytes'],
                            capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    for item in data.get('lscpu', []):
        field = item.get('field', '').lower().strip()
        # Accounts for "L3 cache:" or "L3:" depending on the lscpu version
        if 'l3 cache' in field or field == 'l3:':
            return int(item.get('data', 0).split(" ")[0])

    return 64 * 1024**2

def pyscf_max_memory():
    return pyscflib.param.MAX_MEMORY

def get_gpu_memory_mb() -> float:
    import cupy as cp

    device = cp.cuda.Device()
    _, total_bytes = device.mem_info
    total_bytes = total_bytes / (1024 ** 2)

    return total_bytes



def cotengra_target_size(safety=0.85, bytes_per_float=8) -> int:
    if has_cuda_gpu():
        remaining_mem_mb = get_gpu_memory_mb()
        logger.info(f"GPU has {remaining_mem_mb}MB memory left")
    else:
        # 1. Get raw system memory
        available_bytes = psutil.virtual_memory().available

        # 2. Subtract a flat reserve for MKL thread buffers and system overhead.
        # Adjust this depending on your thread count.
        # (e.g., 192 threads * ~15MB per thread = ~2.8GB)
        mkl_reserve_bytes = 4 * 1024 * 1024 * 1024
        available_bytes = max(0, available_bytes - mkl_reserve_bytes)
        remaining_mem_mb = available_bytes / 1024 / 1024
    # 3. Apply safety margin to what's left
    available_memory_bytes = remaining_mem_mb * 1024 * 1024
    usable_bytes = available_memory_bytes * safety

    # 4. Calculate total allowable elements in the footprint
    total_elements = int(usable_bytes / bytes_per_float)

    # 5. Divide by the concurrent tensor factor (safest is 4)
    # This ensures size(A) + size(B) + size(C) + overhead <= total_elements
    target_size = total_elements // 4
    logger.info(
        f"Allowed peak footprint: {usable_bytes / (1024**2):.2f} MB, "
        f"Target size per tensor: {(target_size * bytes_per_float) / (1024**2):.2f} MB"
    )
    return target_size


def get_optimizer(itemsize: int, max_repeats=1024) -> ctg.HyperOptimizer:
    opt = ctg.ReusableHyperOptimizer(
        minimize='flops', # STRICTLY prioritize flops
        slicing_opts={'target_size': cotengra_target_size(bytes_per_float=itemsize)},
        max_repeats=max_repeats, # Give it plenty of time to find the best sliced path
        progbar=False,
        parallel=False,
    )
    return opt

def contract_cotengra(expr: str, *inputs):
    logger.debug("building cotengra contraction")
    if has_cuda_gpu():
        import cupy as cp
        shapes = [cp.shape(arr) for arr in inputs]
    else:
        shapes = [np.shape(arr) for arr in inputs]
    itemsize = inputs[0].dtype.itemsize

    # 1. Build the tree
    tree = ctg.einsum_tree(expr, *shapes, optimize=get_optimizer(itemsize))

    # 2. Reorder the contraction paths to minimize peak concurrent memory
    tree.reorder_for_peak_size()
    # 3. Check the exact peak memory the tree requires
    peak_bytes = tree.peak_size() * itemsize
    logger.info(f"Tree requires a peak concurrent memory of {peak_bytes / (1024**3):.2f} GB")
    logger.info(f"carrying out contraction: {tree.contract_stats()}")
    # 4. Execute the contraction
    return tree.contract(arrays=inputs, progbar=False, implementation='cotengra')

def has_cuda_gpu() -> bool:
    use_cuda = os.getenv("PYTHC_USE_CUDA", "True") == "True"
    if not use_cuda:
        return False

    cupy_available = importlib.util.find_spec("cupy")
    if not cupy_available:
        return False
    import cupy as cp
    return cp.cuda.is_available()

def to_backend(*inputs, dtype=None):
    # Determine if we should return a single array or a tuple
    return_single = False

    # Flatten the inputs if a single tuple/list was passed
    if len(inputs) == 1:
        if isinstance(inputs[0], (tuple, list)):
            inputs = inputs[0]
        else:
            # A single, standalone array was passed
            return_single = True

    if dtype is None:
        type_env = os.getenv("PYTHC_CUDA_PRECISION", "float64").lower()
        if type_env in ["float64", "double", "f64"]:
            dtype = np.float64
        elif type_env in ["float32", "single", "f32"]:
            dtype = np.float32
        elif type_env in ["float16", "f16"]:
            dtype = np.float16
        else:
            dtype = np.float64

    # Process arrays
    if has_cuda_gpu():
        import cupy as cp
        logger.info(f"Copying arrays to GPU in type: {dtype}")
        out_arrays = [cp.asarray(i, dtype=dtype) for i in inputs]
    else:
        logger.debug(f"Keeping arrays on CPU in type: {dtype}")
        out_arrays = [np.asarray(i, dtype=dtype) for i in inputs]

    # Return a single array if requested, otherwise return a tuple
    if return_single:
        return out_arrays[0]
    return tuple(out_arrays)


def plot_grid_with_density(all_center_coords: list[Any], all_densities: list[Any],
                           all_grid_coords: list[Any]):
    # --- VISUALIZATION BLOCK ---
    logger.info("Generating 2D grid projection plot...")
    grid_pts = np.vstack(all_grid_coords)
    center_pts = np.vstack(all_center_coords)

    plt.figure(figsize=(15, 15))

    # Plot the full Becke grid in light gray
    # plt.scatter(grid_pts[:, 0], grid_pts[:, 1], , s=1, alpha=0.3, label='Full Becke Grid')
    if len(all_densities) > 0:
        densities_flat = np.concatenate(all_densities)
        densities_flat = np.clip(densities_flat, a_min=1e-12, a_max=None)
        scatter = plt.scatter(grid_pts[:, 0], grid_pts[:, 1],
                              c=densities_flat, cmap='viridis', norm=LogNorm(),
                              s=1, alpha=0.6, label='Grid Points (Colored by Density)')
        plt.colorbar(scatter, label='Density Value (Log Scale)')
    else:
        plt.scatter(grid_pts[:, 0], grid_pts[:, 1], c='lightgray', s=1, alpha=0.3, label='Full Becke Grid')

    # Plot the selected KMeans centers in red
    plt.scatter(center_pts[:, 0], center_pts[:, 1], c='red', s=1, alpha=0.8, label='Selected Centers')

    # Plot the atomic nuclei as large black X's
    # plt.scatter(nuclei[:, 0], nuclei[:, 1], c='black', s=150, marker='X', label='Nuclei')

    plt.title('2D Projection (XY Plane) of Grid Points and KMeans Centers')
    plt.xlabel('X coordinate (Bohr)')
    plt.ylabel('Y coordinate (Bohr)')
    plt.axis('equal')  # Crucial to ensure the molecule's geometry isn't visually warped
    plt.legend()
    plt.show()
    # ---------------------------

def plot_sparsity(A):
    sorted_vals = np.sort(np.abs(A.ravel()))
    sorted_vals = sorted_vals[sorted_vals > 1e-12]

    # 2. Calculate cumulative percentages for the y-axis
    percentages = 100. * np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

    # 3. Plot
    plt.plot(sorted_vals, percentages)
    plt.xscale('log')
    plt.xlabel('Absolute Value |A_ij|')
    plt.ylabel('Percentage of Elements Below Value (%)')
    plt.grid(True, which="both", ls="--")
    plt.show()

def get_dfo(mol, auxbasis):
    dfo = df.DF(mol, auxbasis=auxbasis)
    dfo.build()

    return dfo

