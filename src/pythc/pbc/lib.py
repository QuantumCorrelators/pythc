import numpy as np
import pyscf.lib as pyscflib
from pyscf.pbc import tools as pbctools
from scipy.linalg import svd


def get_supercell_phase(cell, kpts):
    """
    Compute the Fourier phase factor matrix for the k-space grid to supercell mapping.

    Wraps pyscf.pbc.tools.k2gamma.get_phase.
    
    Args:
        cell (gto.Cell): PySCF PBC Cell object.
        kpts (ndarray): k-points array, shape (nkpt, 3).

    Returns:
        phase (ndarray): 2D phase factor matrix, shape (nspc, nkpt) where nspc == nkpt.
    """
    if not isinstance(kpts, np.ndarray):
        kpts = np.asarray(kpts.kpts)

    kmesh = pbctools.k2gamma.kpts_to_kmesh(cell, kpts - kpts[0])
    is_wrap_around = np.allclose(kpts, cell.get_kpts(kmesh, wrap_around=True))
    assert np.allclose(kpts, cell.get_kpts(kmesh, wrap_around=is_wrap_around))

    phase = pbctools.k2gamma.get_phase(cell, kpts, kmesh, is_wrap_around)[1]
    return phase


def spc_to_kpt(m_spc, phase):
    """
    Convert a matrix from real-space supercell stripe representation to k-space.

    Args:
        m_spc (ndarray): Supercell array, shape (nspc, ...).
        phase (ndarray): Phase factor matrix, shape (nspc, nkpt).

    Returns:
        m_kpt (ndarray): k-space array, shape (nkpt, ...).
    """
    nspc, nkpt = phase.shape
    m_kpt = pyscflib.dot(phase.conj().T, m_spc.reshape(nspc, -1))
    return m_kpt.reshape((nkpt,) + m_spc.shape[1:])


def kpt_to_spc(m_kpt, phase):
    """
    Convert a matrix from k-space representation to real-space supercell stripe form.

    Args:
        m_kpt (ndarray): k-space array, shape (nkpt, ...).
        phase (ndarray): Phase factor matrix, shape (nspc, nkpt).

    Returns:
        m_spc (ndarray): Supercell array, shape (nspc, ...), real-valued.
    """
    nspc, nkpt = phase.shape
    m_spc = pyscflib.dot(phase, m_kpt.reshape(nkpt, -1))
    m_spc = m_spc.reshape((nspc,) + m_kpt.shape[1:])
    return m_spc.real


def contract_fft_k(f_kpt, g_kpt, phase):
    r"""
    Contract two k-space AO arrays into an AO pair-density cross-correlation tensor:
        [1] Matrix inner product:        T_{ml}^k = \sum_n F_{mn}^k* G_{ln}^k
        [2] k-space -> supercell:        T_{ml}^s = \text{kpt\_to\_spc}(T_{ml}^k)
        [3] Element-wise square:         X_{ml}^s = T_{ml}^s \cdot T_{ml}^s
        [4] supercell -> k-space:        X_{ml}^k = \text{spc\_to\_kpt}(X_{ml}^s)

    Args:
        f_kpt (ndarray): First k-space array, shape (k, m, n).
        g_kpt (ndarray): Second k-space array, shape (k, l, n).
        phase (ndarray): Phase factor matrix, shape (nspc, nkpt).

    Returns:
        x_kpt (ndarray): Contracted pair-density tensor in k-space, shape (k, m, l).
    """
    k, m, n = f_kpt.shape
    l = g_kpt.shape[1]
    assert f_kpt.shape == (k, m, n)
    assert g_kpt.shape == (k, l, n)

    f_kpt = np.ascontiguousarray(f_kpt)
    g_kpt = np.ascontiguousarray(g_kpt)

    # [1] Matrix multiplication per k-point
    t_kpt = [pyscflib.dot(fk.conj(), gk.T) for fk, gk in zip(f_kpt, g_kpt)]
    t_kpt = np.array(t_kpt).reshape(k, m, l)

    # [2] k-space to supercell transform
    t_spc = kpt_to_spc(t_kpt, phase)

    # [3] Element-wise square in real-space supercell
    x_spc = t_spc * t_spc

    # [4] Supercell to k-space transform
    x_kpt = spc_to_kpt(x_spc, phase)
    return x_kpt


def lstsq_svd(a, b, tol=1e-8):
    r"""
    Solve the Hermitian sandwich least-squares problem A @ X @ A \approx B using SVD.

    Args:
        a (ndarray): Hermitian metric matrix, shape (m, m).
        b (ndarray): Right-hand side Coulomb matrix, shape (m, m).
        tol (float): Tolerance for SVD singular values.

    Returns:
        x (ndarray): Solution matrix X, shape (m, m).
        rank (int): Effective numerical rank of A.
    """
    # [1] SVD of A
    u, s, vh = svd(a, full_matrices=False)

    # [2] Compute R[i, j] = 1 / (s[i] * s[j]) where s[i]*s[j] > tol^2
    r = s[None, :] * s[:, None]
    mask = np.abs(r) > tol * tol

    # [3] Compute T = (U^H @ B @ U) / r
    bu = pyscflib.dot(b, u)
    uh = u.conj().T
    t = pyscflib.dot(uh, bu)
    t[mask] /= r[mask]
    t[~mask] = 0.0

    # [4] Compute X = V @ T @ V^H
    v = vh.conj().T
    vt = pyscflib.dot(v, t)
    x = pyscflib.dot(vt, vh)

    rank = int(np.sum(s > tol))
    return x, rank
