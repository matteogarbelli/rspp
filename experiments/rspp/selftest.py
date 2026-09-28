"""Correctness checks for the trajectory kernels and the resampling bound.

Run:  python -m experiments.rspp.selftest
"""
from __future__ import annotations

import numpy as np

from . import kernels
from .geometry import resample, resample_fixed


def check_kernels(n_trials=8, tol=1e-9):
    rng = np.random.default_rng(20260818)
    worst = 0.0
    for _ in range(n_trials):
        P = rng.normal(0, 500, (rng.integers(10, 60), 2))
        Q = rng.normal(0, 500, (rng.integers(10, 60), 2))
        for name in ("frechet", "dtw", "hausdorff"):
            a = kernels.METRICS[name](P, Q)
            b = kernels.PY_REFERENCE[name](P, Q)
            worst = max(worst, abs(a - b))
    assert worst < tol, f"C kernel disagrees with reference by {worst}"
    return worst


def check_metric_axioms(n_trials=20):
    rng = np.random.default_rng(7)
    for _ in range(n_trials):
        A = rng.normal(0, 300, (rng.integers(5, 40), 2))
        B = rng.normal(0, 300, (rng.integers(5, 40), 2))
        C = rng.normal(0, 300, (rng.integers(5, 40), 2))
        assert kernels.frechet(A, A) == 0.0, "identity of indiscernibles"
        assert abs(kernels.frechet(A, B) - kernels.frechet(B, A)) < 1e-12, "symmetry"
        assert (kernels.frechet(A, C)
                <= kernels.frechet(A, B) + kernels.frechet(B, C) + 1e-9), "triangle"
    return True


def check_vertex_preservation():
    """Subdivision must retain every original vertex, or the lower bound fails."""
    rng = np.random.default_rng(3)
    P = rng.normal(0, 1000, (9, 2))
    S = resample(P, 300.0)
    for v in P:
        assert any(np.allclose(v, w) for w in S), "original vertex dropped"
    seg = np.sqrt(((S[1:] - S[:-1]) ** 2).sum(axis=1))
    assert seg.max() <= 300.0 + 1e-9, "spacing exceeds the requested step"
    return float(seg.max())


def check_resampling_bound(n_trials=12):
    """Proposition 5: delta_F <= delta_dF(P_h, Q_h) <= delta_F + h.

    The continuous distance is approached from above as h -> 0, so the h = 1
    value stands in for delta_F; vertex-preserving subdivision is what makes
    the lower bound hold.
    """
    rng = np.random.default_rng(11)
    worst_ratio = 0.0
    for _ in range(n_trials):
        m = int(rng.integers(4, 12))
        P = np.column_stack([np.linspace(0, 20000, m), rng.normal(0, 1500, m)])
        Q = P + rng.normal(0, 400, (m, 2)).cumsum(axis=0) * 0.3
        fine = kernels.frechet(resample(P, 1.0), resample(Q, 1.0))
        for h in (200.0, 500.0, 1000.0, 2000.0):
            d = kernels.frechet(resample(P, h), resample(Q, h))
            assert d >= fine - 1e-6, "discrete below continuous limit"
            assert d <= fine + h + 1e-6, f"bound violated: {d-fine} > {h}"
            worst_ratio = max(worst_ratio, (d - fine) / h)
    return worst_ratio


def check_naive_underreport(h=2000.0):
    """The failure Remark 5 is about: naive resampling under-reports deviation.

    ``resample_fixed`` is arc-length-uniform and does *not* retain the original
    vertices, so it cuts corners and yields a different curve.  The measured
    deviation then falls below the true geometric deviation -- the one direction
    of error a stability-preserving method cannot tolerate, because it reports a
    plan as more faithful to the baseline than it is.

    Returns the numbers the manuscript quotes, so that they are generated rather
    than transcribed.
    """
    rng = np.random.default_rng(20260820)
    # A 10 m reference step stands in for the continuous distance: the discrete
    # Frechet computation is quadratic in the sample count, so a metre-scale
    # reference on a 20 km curve costs 10^8 cell evaluations per pair.
    ref_step = 10.0
    n_trials, under, worst = 60, 0, None
    for _ in range(n_trials):
        m = int(rng.integers(6, 14))
        P = np.column_stack([np.linspace(0, 20000, m), rng.normal(0, 1500, m)])
        Q = P + rng.normal(0, 400, (m, 2)).cumsum(axis=0) * 0.3
        length = float(np.sqrt(((P[1:] - P[:-1]) ** 2).sum(axis=1)).sum())
        true = kernels.frechet(resample(P, ref_step), resample(Q, ref_step))
        keep = kernels.frechet(resample(P, h), resample(Q, h))
        # Vertex preservation is what the proposition assumes; without it the
        # lower bound is not merely loose, it is false.
        assert keep >= true - 1e-6, "vertex-preserving resampling broke the bound"
        n_pts = max(2, int(round(length / h)) + 1)
        naive = kernels.frechet(resample_fixed(P, n_pts), resample_fixed(Q, n_pts))
        if naive < true - 1e-6:
            under += 1
            gap = true - naive
            if worst is None or gap > worst["gap_m"]:
                worst = dict(length_km=length / 1000.0, true_m=true,
                             naive_m=naive, preserving_m=keep, gap_m=gap)
    assert under > 0, "expected naive resampling to under-report somewhere"
    return dict(h_m=h, n_trials=n_trials, n_under=under,
                pct_under=100.0 * under / n_trials, **worst)


if __name__ == "__main__":
    import json
    import os

    w = check_kernels()
    print(f"PASS  C kernels match NumPy reference (max abs deviation {w:.2e})")
    check_metric_axioms()
    print("PASS  discrete Frechet satisfies identity, symmetry, triangle inequality")
    mx = check_vertex_preservation()
    print(f"PASS  subdivision retains all vertices, max spacing {mx:.1f} m <= h")
    r = check_resampling_bound()
    print(f"PASS  resampling bound delta_F <= delta_dF <= delta_F + h "
          f"(worst case used {r*100:.1f}% of the allowance)")
    u = check_naive_underreport()
    print(f"PASS  naive resampling under-reports in {u['n_under']}/{u['n_trials']} "
          f"pairs at h = {u['h_m']:.0f} m; worst case {u['naive_m']:.0f} m "
          f"against a true {u['true_m']:.0f} m on a {u['length_km']:.0f} km pair "
          f"(vertex-preserving: {u['preserving_m']:.0f} m)")
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/selftest.json", "w") as fh:
        json.dump(dict(kernel_max_abs=w, max_spacing=mx, bound_worst_ratio=r,
                       **u), fh, indent=2)
