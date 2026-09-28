"""Polyline construction, metric projection, and controlled arc-length resampling.

Reviewer #4 asked for "a metric projection or geodesic distance and a
controlled resampling procedure".  Benchmark instances live natively in a
planar metric frame (metres), so projection is the identity there;
``project_wgs84_to_enu`` supplies a local East-North frame for the industrial
instances.  Resampling is by arc length, which is what makes Proposition 4
(discrete-to-continuous Frechet convergence) applicable.
"""
from __future__ import annotations

import numpy as np

_EARTH_R = 6_371_008.8  # mean Earth radius, metres (IUGG)


def project_wgs84_to_enu(latlng, origin=None):
    """Equirectangular East-North projection about a local origin.

    Accurate to well under a metre over a service region of a few tens of km,
    which is the regime of a single work shift.  Crucially it is *isotropic*:
    the raw-degree Euclidean distance it replaces stretches longitude by
    1/cos(lat) ~ 1.4 at 45 deg N, silently distorting every Frechet value
    computed in the previous version of this work.
    """
    arr = np.asarray(latlng, dtype=np.float64)
    lat, lng = arr[:, 0], arr[:, 1]
    lat0, lng0 = origin if origin is not None else (float(lat.mean()), float(lng.mean()))
    east = np.radians(lng - lng0) * _EARTH_R * np.cos(np.radians(lat0))
    north = np.radians(lat - lat0) * _EARTH_R
    return np.column_stack([east, north])


def route_polyline(route, coords, depot=0):
    """Vertex polyline of a route, depot-anchored at both ends.

    On the benchmark instances travel is straight-line, so the vertex polyline
    *is* the travelled path; on the industrial instances the same function is
    fed road-network geometry decoded from the routing provider's polylines.
    """
    idx = [depot] + list(route) + [depot]
    return np.asarray(coords)[idx]


def resample(curve, step):
    """Subdivide a polyline so consecutive samples are at most ``step`` apart.

    Every original vertex is retained and each segment is split into equal
    pieces of length at most ``step``.  Retaining the vertices is essential and
    not merely tidy: uniform arc-length resampling that ignores them cuts
    corners, so the resampled polyline is a *different curve* from the original.
    Under corner cutting the discrete Frechet distance can fall below the
    continuous Frechet distance of the true curves, and Proposition 5 does not
    hold.  With vertex-preserving subdivision the sampled polyline is
    geometrically identical to the original, and the proposition applies.
    """
    curve = np.asarray(curve, dtype=np.float64)
    if len(curve) < 2:
        return curve.copy()
    seg = curve[1:] - curve[:-1]
    lens = np.sqrt((seg ** 2).sum(axis=1))
    pieces = [curve[:1]]
    for i, L in enumerate(lens):
        if L <= 0.0:
            continue
        k = int(np.ceil(L / step))
        t = np.arange(1, k + 1, dtype=np.float64) / k
        pieces.append(curve[i] + t[:, None] * seg[i])
    if len(pieces) == 1:
        return curve[:1].copy()
    return np.vstack(pieces)


def resample_fixed(curve, n_pts):
    """Resample to exactly ``n_pts`` arc-length-uniform points."""
    curve = np.asarray(curve, dtype=np.float64)
    if len(curve) < 2:
        return np.repeat(curve[:1], n_pts, axis=0)
    seg = np.sqrt(((curve[1:] - curve[:-1]) ** 2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 0.0:
        return np.repeat(curve[:1], n_pts, axis=0)
    targets = np.linspace(0.0, cum[-1], n_pts)
    return np.column_stack([np.interp(targets, cum, curve[:, 0]),
                            np.interp(targets, cum, curve[:, 1])])


def diameter(coords):
    """Geometric diameter of the service region, in metres.

    The previous implementation used the largest *cost-matrix* entry instead,
    which is a travel cost, not a geometric diameter -- the source of the
    unsupported scale-invariance claim flagged by Reviewer #5.
    """
    pts = np.asarray(coords, dtype=np.float64)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    return float(d.max())
