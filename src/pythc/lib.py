import importlib.util
import json
import logging
import os
import subprocess
from typing import Optional

import cotengra as ctg
import numpy as np
import opt_einsum as oe
import psutil
import pyscf.lib as pyscflib
import pytblis as pt
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

def _read_text_file(path: str):
    with open(path, 'r') as f:
        return f.read().strip()


def _cgroup_v2_job_avail():
    """Return (avail, current) from the nearest ancestor cgroup with a limit, or None.

    Reads the process's own cgroup path from /proc/self/cgroup and walks up
    towards the root.  Reading only /sys/fs/cgroup/memory.max sees the host
    root (usually "max") on systemd machines; the SLURM job limit lives in an
    ancestor like /sys/fs/cgroup/system.slice/slurmstepd.scope/job_<id>/.
    """
    try:
        with open('/proc/self/cgroup', 'r') as f:
            lines = f.read().splitlines()
    except (FileNotFoundError, PermissionError):
        return None
    rel = None
    for line in lines:
        parts = line.strip().split(':')
        # cgroup v2 line looks like "0::/system.slice/..."
        if len(parts) == 3 and parts[0] == '0' and parts[1] == '':
            rel = parts[2]
            break
    if rel is None:
        return None
    root = '/sys/fs/cgroup'
    rel = rel.strip()
    if rel in ('', '/'):
        ancestors = ['']
    else:
        sub = rel.lstrip('/').split('/')
        ancestors = ['/'.join(sub[:i]) for i in range(len(sub), -1, -1)]
    for anc in ancestors:
        base = os.path.join(root, anc) if anc else root
        try:
            limit_s = _read_text_file(os.path.join(base, 'memory.max'))
        except (FileNotFoundError, PermissionError, ValueError, OSError):
            continue
        if limit_s == 'max':
            continue
        try:
            limit = int(limit_s)
            current = int(_read_text_file(os.path.join(base, 'memory.current')))
        except (FileNotFoundError, PermissionError, ValueError, OSError):
            continue
        return max(0, limit - current), current
    return None


def _v1_memory_mount():
    """Find the cgroup v1 memory controller mountpoint."""
    try:
        with open('/proc/self/mountinfo', 'r') as f:
            for line in f:
                # ... mountpoint ... - fstype source superoptions
                if ' - ' not in line:
                    continue
                pre, post = line.split(' - ', 1)
                pre_fields = pre.split()
                post_fields = post.split()
                if len(pre_fields) < 5 or len(post_fields) < 3:
                    continue
                mountpoint = pre_fields[4]
                fstype = post_fields[0]
                superopts = post_fields[2]
                if fstype == 'cgroup' and 'memory' in superopts.split(','):
                    return mountpoint
    except (FileNotFoundError, PermissionError, OSError):
        pass
    return '/sys/fs/cgroup/memory'


def _cgroup_v1_job_avail():
    """Return (avail, current) from the nearest ancestor v1 memory cgroup, or None."""
    try:
        with open('/proc/self/cgroup', 'r') as f:
            lines = f.read().splitlines()
    except (FileNotFoundError, PermissionError):
        return None
    rel = None
    for line in lines:
        parts = line.strip().split(':')
        if len(parts) != 3:
            continue
        controllers = parts[1].split(',')
        if 'memory' in controllers:
            rel = parts[2]
            break
    if rel is None:
        return None
    mount = _v1_memory_mount()
    rel = rel.strip()
    if rel in ('', '/'):
        ancestors = ['']
    else:
        sub = rel.lstrip('/').split('/')
        ancestors = ['/'.join(sub[:i]) for i in range(len(sub), -1, -1)]
    for anc in ancestors:
        base = os.path.join(mount, anc) if anc else mount
        try:
            limit = int(_read_text_file(os.path.join(base, 'memory.limit_in_bytes')))
            current = int(_read_text_file(os.path.join(base, 'memory.usage_in_bytes')))
        except (FileNotFoundError, PermissionError, ValueError, OSError):
            continue
        # Untouched v1 limit is ~2^63-1 (or 2^64-1); treat as unlimited
        if limit >= 2**62:
            continue
        return max(0, limit - current), current
    return None


def _cgroup_job_avail():
    """Best-effort (avail, current-usage) from process-aware cgroup lookup."""
    for fn in (_cgroup_v2_job_avail, _cgroup_v1_job_avail):
        try:
            res = fn()
        except Exception:
            continue
        if res is not None:
            return res
    # Legacy fallback: root-level files (docker without systemd, old layouts)
    try:
        cg_limit = _read_text_file('/sys/fs/cgroup/memory.max')
        if cg_limit != 'max':
            cg_current = int(_read_text_file('/sys/fs/cgroup/memory.current'))
            return max(0, int(cg_limit) - cg_current), cg_current
    except (FileNotFoundError, PermissionError, ValueError, OSError):
        pass
    try:
        cg_limit = int(_read_text_file('/sys/fs/cgroup/memory/memory.limit_in_bytes'))
        if cg_limit < 2**62:
            cg_current = int(_read_text_file('/sys/fs/cgroup/memory/memory.usage_in_bytes'))
            return max(0, cg_limit - cg_current), cg_current
    except (FileNotFoundError, PermissionError, ValueError, OSError):
        pass
    return None


def _parse_slurm_mem_mb(value) -> Optional[float]:
    """Parse a SLURM memory value (MB, optionally with K/M/G/T suffix) to MB."""
    import re
    if value is None:
        return None
    s = str(value).strip().upper().rstrip('B')
    if not s or s == '0':
        return None
    m = re.match(r'^([\d.]+)\s*([KMGT]?)M?$', s)
    if not m:
        try:
            v = float(s)
            return v if v > 0 else None
        except ValueError:
            return None
    num, suffix = float(m.group(1)), m.group(2)
    mult = {'': 1.0, 'M': 1.0, 'K': 1.0 / 1024.0, 'G': 1024.0, 'T': 1024.0 * 1024.0}[suffix]
    mb = num * mult
    return mb if mb > 0 else None


def _expand_slurm_count_list(s: str) -> list:
    """Expand SLURM compressed host lists like '64(x2),32' -> [64, 64, 32]."""
    import re
    out = []
    for part in str(s).split(','):
        part = part.strip()
        m = re.match(r'^(\d+)(?:\(x(\d+)\))?$', part)
        if not m:
            continue
        val, rep = int(m.group(1)), int(m.group(2)) if m.group(2) else 1
        out.extend([val] * rep)
    return out


def _slurm_cpus_on_node() -> Optional[int]:
    try:
        v = os.environ.get('SLURM_CPUS_ON_NODE')
        if v and int(v) > 0:
            return int(v)
    except ValueError:
        pass
    # Per-node list, pick entry for this node when possible
    lst = _expand_slurm_count_list(os.environ.get('SLURM_JOB_CPUS_PER_NODE', ''))
    if lst:
        try:
            nodeid = int(os.environ.get('SLURM_NODEID', '0'))
            if 0 <= nodeid < len(lst) and lst[nodeid] > 0:
                return lst[nodeid]
        except ValueError:
            pass
        return min(lst)  # heterogeneous allocation: be conservative
    try:
        per_task = int(os.environ.get('SLURM_CPUS_PER_TASK', '0'))
    except ValueError:
        per_task = 0
    if per_task and per_task > 0:
        ntasks = 0
        for var in ('SLURM_NTASKS_PER_NODE', 'SLURM_TASKS_PER_NODE'):
            raw = os.environ.get(var, '')
            # may be "8(x2)" style or plain int
            vals = _expand_slurm_count_list(raw)
            if vals:
                ntasks = vals[0]
                break
            try:
                ntasks = int(raw)
                break
            except (ValueError, TypeError):
                continue
        if not ntasks or ntasks <= 0:
            try:
                ntotal = int(os.environ.get('SLURM_NTASKS', '1'))
                nnodes = int(os.environ.get('SLURM_JOB_NUM_NODES',
                                            os.environ.get('SLURM_NNODES', '1')))
                ntasks = max(1, ntotal // max(1, nnodes))
            except ValueError:
                ntasks = 1
        return per_task * max(1, ntasks)
    return None


def _slurm_gpus_on_node() -> Optional[int]:
    for var in ('SLURM_GPUS_ON_NODE', 'SLURM_JOB_GPUS', 'SLURM_GPUS'):
        vals = _expand_slurm_count_list(os.environ.get(var, ''))
        if vals:
            return sum(vals) if 'JOB' in var or ',' in os.environ.get(var, '') else vals[0]
        try:
            v = int(os.environ.get(var, ''))
            if v > 0:
                return v
        except (ValueError, TypeError):
            continue
    try:
        per_task = int(os.environ.get('SLURM_GPUS_PER_TASK', '0'))
        if per_task > 0:
            cpus = _slurm_cpus_on_node() or 1
            per_cpu = int(os.environ.get('SLURM_CPUS_PER_TASK', '1') or 1)
            return max(1, per_task * max(1, cpus // max(1, per_cpu)))
    except ValueError:
        pass
    return None


def _slurm_limit_bytes() -> Optional[int]:
    """Job memory limit on this node from SLURM env vars, or None if unknown."""
    MB = 1024 * 1024
    per_node = _parse_slurm_mem_mb(os.environ.get('SLURM_MEM_PER_NODE'))
    if per_node:
        return int(per_node * MB)
    per_cpu = _parse_slurm_mem_mb(os.environ.get('SLURM_MEM_PER_CPU'))
    if per_cpu:
        cpus = _slurm_cpus_on_node()
        if cpus:
            return int(per_cpu * MB * cpus)
        return None  # know mem-per-cpu but not cpu count: don't guess
    per_gpu = _parse_slurm_mem_mb(os.environ.get('SLURM_MEM_PER_GPU'))
    if per_gpu:
        gpus = _slurm_gpus_on_node()
        if gpus:
            return int(per_gpu * MB * gpus)
        return None
    # Newer SLURM TRES strings, e.g. "cpu=8,mem=32000M"
    import re
    for var in ('SLURM_TRES_PER_NODE', 'SLURM_TRES_PER_TASK', 'SLURM_TRES_PER_JOB'):
        tres = os.environ.get(var, '')
        m = re.search(r'mem(?:ory)?=([\d.]+[KMGT]?)', tres, re.IGNORECASE)
        if not m:
            continue
        mb = _parse_slurm_mem_mb(m.group(1))
        if not mb:
            continue
        if 'TASK' in var:
            ntasks = _slurm_cpus_on_node() or 1
            try:
                per_task_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', '1') or 1)
                ntasks = max(1, ntasks // max(1, per_task_cpus))
            except ValueError:
                pass
            return int(mb * MB * max(1, ntasks))
        return int(mb * MB)
    return None


def available_memory_bytes():
    """Available memory for this job minus safety reserves.

    Takes the minimum of host-wide ``psutil`` available, the process-aware
    cgroup remaining (walks up from /proc/self/cgroup instead of reading
    only the host root, which is typically "max" on systemd/cgroup-v2
    machines), and — when SLURM enforces memory without a matching cgroup
    limit — the SLURM allocation (``SLURM_MEM_PER_NODE`` /
    ``SLURM_MEM_PER_CPU`` x on-node CPUs / ``SLURM_MEM_PER_GPU``) minus
    current job usage.
    """
    available = psutil.virtual_memory().available
    sources = {'host': available}

    cg = _cgroup_job_avail()
    cg_current = None
    if cg is not None:
        cg_avail, cg_current = cg
        available = min(available, cg_avail)
        sources['cgroup'] = cg_avail

    slurm_limit = None
    try:
        slurm_limit = _slurm_limit_bytes()
    except Exception:
        slurm_limit = None
    if slurm_limit is not None:
        usage = cg_current
        if usage is None:
            try:
                usage = psutil.Process(os.getpid()).memory_info().rss
            except Exception:
                usage = 0
        slurm_avail = max(0, slurm_limit - (usage or 0))
        available = min(available, slurm_avail)
        sources['slurm_limit'] = slurm_limit
        sources['slurm'] = slurm_avail

    logger.debug("available_memory_bytes sources (bytes): %s", sources)

    try:
        thread_count = int(os.environ.get('OMP_NUM_THREADS')
                           or os.environ.get('SLURM_CPUS_PER_TASK')
                           or '')
        if thread_count <= 0:
            raise ValueError
    except ValueError:
        thread_count = _slurm_cpus_on_node() or (os.cpu_count() or 1)
    thread_reserve = thread_count * 15 * 1024 * 1024
    return max(0, available - thread_reserve - 4 * 1024**3)


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
    """PySCF max_memory in MB, honouring the env overrides set on clusters.

    Job scripts (e.g. ``setup.sh``: ``MAX_MEMORY=$SLURM_CPUS_PER_TASK *
    SLURM_MEM_PER_CPU``) export the allocation as ``PYSCF_MAX_MEMORY`` /
    ``MAX_MEMORY``; plain ``pyscflib.param.MAX_MEMORY`` (default 4000) never
    sees it, so the J-tiling path would size tiles for 4 GB on a 256 GB job.
    """
    for var in ('PYSCF_MAX_MEMORY', 'MAX_MEMORY'):
        raw = os.environ.get(var)
        if raw:
            try:
                val = float(raw)
            except ValueError:
                continue
            if val > 0:
                return val
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

def get_dfo(mol, auxbasis):
    dfo = df.DF(mol, auxbasis=auxbasis)
    dfo.build()

    return dfo

