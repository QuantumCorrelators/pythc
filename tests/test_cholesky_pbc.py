"""Regression tests for SymMetricPBC (k-point Cholesky metric).

Covers the single-threading culprit fixed in ``pythc.decomp.cholesky``:
``_inner_products`` used ``np.einsum('kip,kjp->kij', ...)``, whose default
implementation is a single-threaded C loop that ignores
OPENBLAS_NUM_THREADS/OMP_NUM_THREADS and serialized the RPCholesky
grid-pruning step on one core. The hot path must stay on BLAS matmul.
"""
import numpy as np
import pytest

from pythc.decomp.cholesky import SymMetricPBC


def _random_pbc_data(rng, k=3, n=25, p_o=7, p_v=5):
    X = rng.standard_normal((k, n, p_o)) + 1j * rng.standard_normal((k, n, p_o))
    X_v = rng.standard_normal((k, n, p_v)) + 1j * rng.standard_normal((k, n, p_v))
    return np.ascontiguousarray(X), np.ascontiguousarray(X_v)


def _reference_mtx(X, X_v, vec_i, vec_j):
    """Naive triple-loop reference via np.vdot (no einsum/matmul batching)."""
    vec_i = list(vec_i)
    vec_j = list(vec_j)
    dtype = np.float64 if X_v is None else np.complex128
    out = np.zeros((len(vec_i), len(vec_j)), dtype=dtype)
    for a, i in enumerate(vec_i):
        for b, j in enumerate(vec_j):
            acc = 0.0
            for kk in range(X.shape[0]):
                d_o = np.vdot(X[kk, i], X[kk, j])
                if X_v is None:
                    acc += (d_o.real ** 2 + d_o.imag ** 2)
                else:
                    d_v = np.vdot(X_v[kk, i], X_v[kk, j])
                    acc += d_o * d_v.conj()
            out[a, b] = acc
    return out


@pytest.fixture(params=["ao", "ov"])
def pbc_metric(request):
    rng = np.random.default_rng(42)
    X, X_v = _random_pbc_data(rng)
    if request.param == "ao":
        return SymMetricPBC(X), X, None
    return SymMetricPBC(X, X_v=X_v), X, X_v


def test_function_mtx_matches_reference(pbc_metric):
    metric, X, X_v = pbc_metric
    rng = np.random.default_rng(1)
    n = X.shape[1]
    vec_i = rng.choice(n, size=9, replace=False)
    vec_j = rng.choice(n, size=13, replace=False)
    got = metric._function_mtx(vec_i, vec_j)
    expected = _reference_mtx(X, X_v, vec_i, vec_j)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_function_vec_matches_paired_reference(pbc_metric):
    metric, X, X_v = pbc_metric
    rng = np.random.default_rng(2)
    n = X.shape[1]
    vec = rng.choice(n, size=11, replace=False)
    got = metric._function_vec(vec, vec)
    expected = np.diag(_reference_mtx(X, X_v, vec, vec))
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_function_matches_reference(pbc_metric):
    metric, X, X_v = pbc_metric
    expected = _reference_mtx(X, X_v, [3], [7])[0, 0]
    assert metric._function([3], [7]) == pytest.approx(expected)


def test_diag_helper_matches_reference(pbc_metric):
    metric, X, X_v = pbc_metric
    n = X.shape[1]
    expected = np.diag(_reference_mtx(X, X_v, range(n), range(n)))
    np.testing.assert_allclose(metric._diag_helper(), expected, rtol=1e-12, atol=1e-12)


def test_inner_products_matches_reference(pbc_metric):
    metric, X, X_v = pbc_metric
    Xi = X[:, [0, 2, 4], :]
    Xj = X[:, [1, 3, 5, 6], :]
    Xi_v = None if X_v is None else X_v[:, [0, 2, 4], :]
    Xj_v = None if X_v is None else X_v[:, [1, 3, 5, 6], :]
    got = metric._inner_products(Xi, Xj, Xi_v, Xj_v).sum(axis=0)
    expected = _reference_mtx(X, X_v, [0, 2, 4], [1, 3, 5, 6])
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_hot_path_does_not_use_einsum(pbc_metric, monkeypatch):
    """Guard against reintroducing the single-threaded einsum path.

    The batched ``einsum('kip,kjp->kij')`` ignores BLAS threading; the
    pruning hot path (mtx/vec/inner products) must not call it.
    """
    metric, _, _ = pbc_metric

    def _forbidden(*args, **kwargs):
        raise AssertionError("hot path must not use np.einsum (single-threaded)")

    monkeypatch.setattr(np, "einsum", _forbidden)
    rng = np.random.default_rng(3)
    n = metric.n
    vec_i = rng.choice(n, size=6, replace=False)
    vec_j = rng.choice(n, size=8, replace=False)
    metric._function_mtx(vec_i, vec_j)
    metric._function_vec(vec_i[:6], vec_j[:6])
    metric._inner_products(
        metric.X[:, vec_i[:2], :], metric.X[:, vec_j[:2], :],
        None if metric.X_v is None else metric.X_v[:, vec_i[:2], :],
        None if metric.X_v is None else metric.X_v[:, vec_j[:2], :],
    )
