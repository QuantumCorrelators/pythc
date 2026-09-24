"""THC integral object with the :class:`pyscf.df.DF` interface.

This module implements a single density-fitted-style J/K engine backed by
tensor hypercontraction (THC) factors::

    (mu nu | lam sig) ~= sum_PQ X[P,mu] X[P,nu] Z[P,Q] X[Q,lam] X[Q,sig]

with ``X : (M, N)`` collocation and symmetric ``Z : (M, M)`` core.

Design points:

* :class:`THCDF` subclasses :class:`pyscf.df.DF`, so it can be used anywhere
  a DF object is expected, e.g. (``density_fit`` re-classes the SCF object
  so ``get_jk`` routes through ``with_df``)::

      mf = scf.RHF(mol).density_fit(with_df=THCDF(mol, eri_thc, auxbasis))
      mf.kernel()

  Spin-block splitting (UHF/GHF) and the RHF/UHF/GHF Fock factors are then
  owned by PySCF, not by this module.
* The object internally holds a conventional DF object (the DF-ERI).  Each
  of J and K can independently be routed to THC or to DF via
  ``with_j_thc`` / ``with_k_thc``.  In particular ``with_j_thc=False``
  gives the "only-K" mode: Coulomb through DF/RI, exchange through THC.
* All densities are treated as flat stacks of spatial densities
  ``(S, N, N)``; arbitrary leading batch dimensions (including the
  ``(nblocks, n_dm, N, N)`` stacks built by ``pyscf.scf.ghf.get_jk``) are
  flattened on entry and restored on exit.
"""

import logging

import numpy as np
from pyscf import df, gto

import pythc.lib as lib
from pythc.thc.thc_base import ThcEri, THC

logger = logging.getLogger()

# NOTE on separable cores: some factorizations (LS-snRI) also provide
# Z = D @ D.T, D : (M, R). An RI-like reformulation with precomputed
# V[t] = X.T @ diag(D[:, t]) @ X costs R*M*N^2 once plus 2*R*N^3 per
# iteration, vs 2*M^2*N per iteration here. That only wins for R*N < M,
# but measured snRI sizes (M ~ 10*N, R ~ 3.5*N, robust across pruning
# thresholds) miss it by ~8x even amortized over 100 SCF iterations, and
# general THC cores are full-rank anyway. So the dense core is used
# unconditionally; verified numerically, do not "optimize" this.
#
# Fused path for one stack entry D_s (symmetric/Hermitian), sharing
# a single B intermediate between J and K:
#   B_s  = X @ D_s @ X.T                     # (M, M),  N^2*M + M^2*N
#   E_s  = diag(B_s)                          # free
#   F_s  = E_s @ Z                            # M^2
#   J_s  = X.T @ diag(F_s) @ X                # N^2*M
#   C_s  = Z * B_s  (elementwise, symmetric)  # M^2
#   K_s  = X.T @ C_s @ X                      # N*M^2 + N^2*M
# J is linear, so J(Da+Db) = J(Da)+J(Db).


def _backend():
    if lib.has_cuda_gpu():
        import cupy as cp
        return cp
    return np


def _to_device_stack(dms_host: np.ndarray):
    xp = _backend()
    if xp is np:
        if np.iscomplexobj(dms_host):
            return np.asarray(dms_host)
        return lib.to_backend(dms_host)
    import cupy as cp
    if np.iscomplexobj(dms_host):
        return cp.asarray(dms_host)
    return lib.to_backend(dms_host)


def _to_host(x):
    if lib.has_cuda_gpu():
        import cupy as cp
        return cp.asnumpy(x)
    return np.asarray(x)


def _thc_jk_stack(X, Z, dms, hermi=1, with_j=True, with_k=True):
    """Fused THC J/K kernel for a stack of spatial densities.

    Args:
        X: (M, N) collocation matrix (backend array).
        Z: (M, M) symmetric core (backend array).
        dms: host array of shape (S, N, N). Complex Hermitian supported
            (GHF off-diagonal blocks); pass ``hermi=0`` for non-Hermitian
            entries such as ``Dab``.
    Returns:
        (vj, vk) host arrays of shape (S, N, N), or None for a disabled
        term. vj[s] = J(dms[s]), vk[s] = K(dms[s]).
    """
    xp = _backend()
    M, N = X.shape

    dms_host = np.asarray(dms)
    S = dms_host.shape[0]
    if S == 0:
        z = np.zeros((0, N, N), dtype=dms_host.dtype)
        return (z if with_j else None, z if with_k else None)

    D = _to_device_stack(dms_host)
    # Exploit DM symmetry (Hermitian): cleans up difference densities
    # and guarantees B / J / K symmetry up to roundoff.
    if hermi == 1:
        D = 0.5 * (D + D.transpose(0, 2, 1).conj())

    Xt = X.T
    itemsize = np.dtype(X.dtype).itemsize if hasattr(X, 'dtype') else 8
    # Keep ~512 MB peak for the (S, M, M) intermediates; chunk over S.
    chunk = max(1, min(S, (512 * 1024**2) // max(1, 2 * M * M * itemsize)))

    vj_chunks, vk_chunks = [], []
    for s0 in range(0, S, chunk):
        Dc = D[s0:s0 + chunk]
        if with_k:
            # B = X @ D @ X.T shared by J and K.
            A = xp.matmul(Dc, Xt)      # (Sc, N, M)
            B = xp.matmul(X, A)       # (Sc, M, M)
            if hermi == 1:
                B = 0.5 * (B + B.transpose(0, 2, 1).conj())
            if with_j:
                E = xp.einsum('sii->si', B)  # diag, free
                F = xp.matmul(E, Z)          # (Sc, M)
                Xw = X[None, :, :] * F[:, :, None]  # (Sc, M, N)
                Vj = xp.matmul(Xt, Xw)       # (Sc, N, N)
                if hermi == 1:
                    Vj = 0.5 * (Vj + Vj.transpose(0, 2, 1).conj())
                vj_chunks.append(Vj)
            B = B * Z  # C = Z (.) B in place, symmetric
            T = xp.matmul(B, X)    # (Sc, M, N)
            Vk = xp.matmul(Xt, T)  # (Sc, N, N)
            if hermi == 1:
                Vk = 0.5 * (Vk + Vk.transpose(0, 2, 1).conj())
            vk_chunks.append(Vk)
        else:
            # J-only: diag(X D X.T) without forming the full M x M.
            A = xp.matmul(Dc, Xt)  # (Sc, N, M)
            E = xp.einsum('snp,np->sp', A, Xt)  # (Sc, M)
            F = xp.matmul(E, Z)
            Xw = X[None, :, :] * F[:, :, None]
            Vj = xp.matmul(Xt, Xw)
            if hermi == 1:
                Vj = 0.5 * (Vj + Vj.transpose(0, 2, 1).conj())
            vj_chunks.append(Vj)

    vj = _to_host(xp.concatenate(vj_chunks, axis=0)) if with_j else None
    vk = _to_host(xp.concatenate(vk_chunks, axis=0)) if with_k else None
    return vj, vk


class THCDF(df.DF):
    """THC J/K engine with the :class:`pyscf.df.DF` interface.

    Args:
        mol: PySCF molecule.
        eri_thc: THC factorization (:class:`ThcEri`).
        auxbasis: auxiliary basis for the internally held conventional DF
            object (exact baseline / non-THC terms).
        baseline: optional prebuilt :class:`pyscf.df.DF` to use instead of
            building the internal one (avoids building the DF-ERI twice
            when the caller already owns one).
        with_j_thc: route J through THC (False: J through DF/RI).
        with_k_thc: route K through THC (False: K through DF/RI).
            ``with_j_thc=False`` selects the "only-K" mode.

    The flags are plain attributes and may be flipped at any time, e.g.
    ``mf.with_df.with_j_thc = False``.  A temporal RI-then-THC switch can
    wrap this object (or drive these flags) from the SCF level, where the
    cycle count, energy and DIIS state are visible.
    """

    def __init__(self, mol: gto.Mole, thc: THC, auxbasis: str = None,
                 baseline: df.DF = None,
                 with_j_thc: bool = True, with_k_thc: bool = True):
        df.DF.__init__(self, mol, auxbasis)
        if baseline is None:
            baseline = df.DF(mol)
            if auxbasis is not None:
                baseline.auxbasis = auxbasis
        self._baseline = baseline

        self.thc = thc
        self.X, self.Z = None, None
        self.with_j_thc = with_j_thc
        self.with_k_thc = with_k_thc

    def build(self):
        eri_thc = self.thc.build(mode="ao")
        eri_thc.to_backend()

        X, Z = eri_thc.get_X_Z()
        Z = 0.5 * (Z + Z.T)
        self.X, self.Z = X, Z

        return self

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        auxbasis = self.auxbasis
        df.DF.reset(self, mol)
        self._baseline = df.DF(self.mol)
        if auxbasis is not None:
            self._baseline.auxbasis = auxbasis
        return self

    def get_jk(self, dm, hermi=1, with_j=True, with_k=True,
               direct_scf_tol=1e-13, omega=None):
        """Compute J/K for spatial densities, routing each term to THC/DF.

        Accepts arbitrary leading batch dimensions (flattened to (S, N, N)
        internally, e.g. GHF's ``(nblocks, n_dm, N, N)`` stacks) and complex
        densities. Returns vj/vk in the input's leading shape (None for a
        disabled term).
        """
        if self.Z is None:
            self.build()

        if omega is not None and omega != 0:
            # No range separation in THC; exact DF handles it.
            vj, vk = self._baseline.get_jk(dm, hermi, with_j, with_k,
                                           direct_scf_tol, omega)
            return vj, vk

        dm = np.asarray(dm)
        in_shape = dm.shape
        N = in_shape[-1]
        flat = dm.reshape(-1, N, N)

        want_j_thc = with_j and self.with_j_thc
        want_k_thc = with_k and self.with_k_thc
        want_j_df = with_j and not self.with_j_thc
        want_k_df = with_k and not self.with_k_thc

        vj = vk = None
        if want_j_thc or want_k_thc:
            tj, tk = _thc_jk_stack(self.X, self.Z, flat, hermi=hermi,
                                   with_j=want_j_thc, with_k=want_k_thc)
            if want_j_thc:
                vj = tj
            if want_k_thc:
                vk = tk
        if want_j_df or want_k_df:
            bj, bk = self._baseline.get_jk(flat, hermi=hermi,
                                           with_j=want_j_df, with_k=want_k_df)
            if want_j_df:
                vj = bj
            if want_k_df:
                vk = bk

        if vj is not None:
            vj = vj.reshape(in_shape)
        if vk is not None:
            vk = vk.reshape(in_shape)
        return vj, vk
