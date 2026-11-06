"""Unit tests for the memory-safe Laplace-MP2 K term.

These cover two things that previously failed silently in production:

* the K memory model must bound *every* array the tile loop holds at its peak
  (a missing full-tile temporary pushed a 20-water cluster OOM), and
* the tiled/chunked transform must reproduce the same K as a direct
  brute-force contraction, for both the full-B and the blocked-B paths.
"""
import numpy as np
import pytest

from pythc.methods import mp2


def _random_inputs(n_grid=37, n_occ=5, n_virt=7, n_lapl=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_grid, n_occ + n_virt))
    Z = rng.standard_normal((n_grid, n_grid))
    Z = Z + Z.T
    tau_o = rng.random((n_lapl, n_occ)) + 0.5
    tau_v = rng.random((n_lapl, n_virt)) + 0.5
    return tau_o, tau_v, X[:, :n_occ], X[:, n_occ:], Z


def _k_reference(tau_o, tau_v, X_o, X_v, Z):
    """Direct contraction of the K term, no tiling, no symmetry tricks."""
    n_lapl = tau_o.shape[0]
    B = np.einsum('PQ,Qj,Qb->Pjb', Z, X_o, X_v, optimize=True)

    k = 0.0
    for v in range(n_lapl):
        X_v_tau = X_v * tau_v[v][None, :]
        S = (X_o * tau_o[v][None, :]) @ X_o.T
        E_pr = np.einsum('Pjb,rb->Prj', B, X_v_tau, optimize=True)
        E_rp = np.einsum('Pjb,pb->Ppj', B, X_v_tau, optimize=True)
        k += np.einsum('pr,prj,rpj,j->', S, E_pr, E_rp, tau_o[v], optimize=True)
    return k


@pytest.mark.parametrize("tile,scratch_rows", [(64, 8), (16, 4), (37, 37), (10, 3)])
def test_full_b_matches_reference(tile, scratch_rows):
    tau_o, tau_v, X_o, X_v, Z = _random_inputs()
    ref = _k_reference(tau_o, tau_v, X_o, X_v, Z)
    got = mp2._mp2_K_full_B(tau_o, tau_v, X_o, X_v, Z, tile, scratch_rows)
    assert got == pytest.approx(ref, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("j_block", [1, 2, 3, 5])
def test_blocked_matches_reference(j_block):
    tau_o, tau_v, X_o, X_v, Z = _random_inputs()
    ref = _k_reference(tau_o, tau_v, X_o, X_v, Z)
    got = mp2._mp2_K_blocked(tau_o, tau_v, X_o, X_v, Z, j_block, 16, 4)
    assert got == pytest.approx(ref, rel=1e-12, abs=1e-12)


def _plan(n_grid, n_occ, n_virt, available_gb, monkeypatch, budget_gb=None,
          tile_max=None):
    monkeypatch.setattr(mp2.lib, "available_memory_bytes",
                        lambda: available_gb * 1e9)
    monkeypatch.delenv("PYTHC_MP2_BUDGET_GB", raising=False)
    monkeypatch.delenv("PYTHC_MP2_K_TILE_MAX", raising=False)
    if budget_gb is not None:
        monkeypatch.setenv("PYTHC_MP2_BUDGET_GB", str(budget_gb))
    if tile_max is not None:
        monkeypatch.setenv("PYTHC_MP2_K_TILE_MAX", str(tile_max))
    return mp2._plan_k_memory(n_grid, n_occ, n_virt, 8)


@pytest.mark.parametrize("available_gb", [2.0, 20.0, 400.0])
@pytest.mark.parametrize("shape", [(500, 4, 40), (5000, 20, 300), (40000, 100, 1000)])
def test_plan_peak_never_exceeds_budget(shape, available_gb, monkeypatch):
    plan = _plan(*shape, available_gb, monkeypatch)
    assert plan["peak"] <= plan["budget"] * (1 + 1e-12)
    assert plan["budget"] <= plan["limit"] * mp2._K_BUDGET_FRACTION * (1 + 1e-12)
    assert plan["tile"] <= shape[0]
    assert plan["tile"] % 64 == 0 or plan["tile"] == min(shape[0], mp2._K_MAX_TILE)
    assert 1 <= plan["j_block"] <= shape[1]


def test_plan_counts_both_tile_buffers(monkeypatch):
    """The peak must budget two full tile buffers, not one.

    This is the regression that caused the OOM: the model used to describe the
    steady state (4 arrays) while the loop transiently held 5.
    """
    n_grid, n_occ, n_virt = 20000, 50, 400
    plan = _plan(n_grid, n_occ, n_virt, 200.0, monkeypatch)
    assert plan["mode"] == "full"

    tile = plan["tile"]
    one_buffer = tile * tile * n_occ * 8
    work = plan["peak"] - plan["fixed"] - plan["b_bytes"]
    assert work >= 2 * one_buffer


def test_issue_numbers_now_fit():
    """The water_cluster_20 run that OOMed (381.73 GB free, 32840/100/1060).

    The old planner chose tile=9600 and needed 397.5 GB. The new plan must fit
    inside the same free memory while still using most of it.
    """
    plan = mp2._plan_k_memory(32840, 100, 1060, 8, available=381.73e9)
    assert plan["mode"] == "full"
    assert plan["peak"] <= plan["budget"]
    assert plan["b_bytes"] / 1e9 == pytest.approx(27.85, rel=0.01)
    # still allowed to fill the machine
    assert plan["peak"] / 1e9 > 300.0


def test_tile_max_override(monkeypatch):
    plan = _plan(40000, 100, 1000, 400.0, monkeypatch, tile_max=2048)
    assert plan["tile"] <= 2048
    assert plan["peak"] < 400e9


def test_blocked_path_is_reachable(monkeypatch):
    """A budget too small for full B must fall back, not silently pick tile=64."""
    plan = _plan(40000, 400, 3000, 30.0, monkeypatch)
    assert plan["mode"] == "blocked"
    assert plan["b_bytes"] < 40000 * 400 * 3000 * 8
    assert plan["peak"] <= plan["budget"]


def test_actual_allocation_stays_under_plan(monkeypatch):
    """Measure the real peak against the predicted one.

    Allocates ~200 MB of tile buffers so the (ru_maxrss) high-water mark is
    dominated by the K working set and the comparison is meaningful.
    """
    import resource

    n_grid, n_occ, n_virt = 2000, 4, 16
    monkeypatch.setattr(mp2.lib, "available_memory_bytes", lambda: 0.20 * 1e9)
    monkeypatch.delenv("PYTHC_MP2_BUDGET_GB", raising=False)
    monkeypatch.delenv("PYTHC_MP2_K_TILE_MAX", raising=False)
    plan = mp2._plan_k_memory(n_grid, n_occ, n_virt, 8)
    assert plan["mode"] == "full"

    tau_o, tau_v, X_o, X_v, Z = _random_inputs(n_grid, n_occ, n_virt, n_lapl=2)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    mp2._mp2_K_full_B(tau_o, tau_v, X_o, X_v, Z, plan["tile"],
                      plan["scratch_rows"], plan)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    # numpy/numba baselines are already resident; only the delta is ours.
    assert after - before <= plan["peak"] * 1.05
