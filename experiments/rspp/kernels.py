"""ctypes bindings for the C trajectory-distance kernels, with NumPy fallbacks.

The fallbacks are intentionally kept: they are slow, but they are the reference
implementation the C code is validated against, and they let the engine run on
a machine with no compiler.
"""
from __future__ import annotations

import ctypes
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_HERE, "libkernels.so")

_LIB = None
if os.path.exists(_LIB_PATH):
    _LIB = ctypes.CDLL(_LIB_PATH)
    _dbl_p = ctypes.POINTER(ctypes.c_double)
    for _name in ("discrete_frechet", "dtw_distance", "hausdorff_distance"):
        _fn = getattr(_LIB, _name)
        _fn.argtypes = [_dbl_p, ctypes.c_long, _dbl_p, ctypes.c_long]
        _fn.restype = ctypes.c_double
    _LIB.lcss_distance.argtypes = [_dbl_p, ctypes.c_long, _dbl_p, ctypes.c_long,
                                   ctypes.c_double, ctypes.c_long]
    _LIB.lcss_distance.restype = ctypes.c_double


def _as_c(curve):
    arr = np.ascontiguousarray(curve, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"curve must have shape (m, 2), got {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("curve must contain at least one point")
    return arr, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), arr.shape[0]


# ------------------------------------------------------------- reference
def _frechet_numpy(P, Q):
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    m, n = len(P), len(Q)
    D = np.sqrt(((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1))
    prev = np.maximum.accumulate(D[0])
    cur = np.empty(n)
    for i in range(1, m):
        cur[0] = max(prev[0], D[i, 0])
        for j in range(1, n):
            cur[j] = max(D[i, j], min(prev[j], prev[j - 1], cur[j - 1]))
        prev = cur.copy()
    return float(prev[-1])


def _dtw_numpy(P, Q):
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    m, n = len(P), len(Q)
    D = np.sqrt(((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1))
    prev = np.cumsum(D[0])
    cur = np.empty(n)
    for i in range(1, m):
        cur[0] = prev[0] + D[i, 0]
        for j in range(1, n):
            cur[j] = D[i, j] + min(prev[j], prev[j - 1], cur[j - 1])
        prev = cur.copy()
    return float(prev[-1] / (m + n))


def _hausdorff_numpy(P, Q):
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    D = np.sqrt(((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1))
    return float(max(D.min(axis=1).max(), D.min(axis=0).max()))


def _lcss_numpy(P, Q, eps, delta):
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    m, n = len(P), len(Q)
    D = np.sqrt(((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1))
    prev = np.zeros(n + 1, dtype=np.int64)
    cur = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, m + 1):
        cur[0] = 0
        for j in range(1, n + 1):
            if abs(i - j) <= delta and D[i - 1, j - 1] < eps:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur.copy()
    return float(eps * (1.0 - prev[n] / min(m, n)))


# ------------------------------------------------------------ public API
def frechet(P, Q):
    """Discrete Frechet distance between two planar polylines, in input units."""
    if _LIB is None:
        return _frechet_numpy(P, Q)
    _a, pp, m = _as_c(P); _b, qq, n = _as_c(Q)
    return float(_LIB.discrete_frechet(pp, m, qq, n))


def dtw(P, Q):
    if _LIB is None:
        return _dtw_numpy(P, Q)
    _a, pp, m = _as_c(P); _b, qq, n = _as_c(Q)
    return float(_LIB.dtw_distance(pp, m, qq, n))


def hausdorff(P, Q):
    if _LIB is None:
        return _hausdorff_numpy(P, Q)
    _a, pp, m = _as_c(P); _b, qq, n = _as_c(Q)
    return float(_LIB.hausdorff_distance(pp, m, qq, n))


def lcss(P, Q, eps=250.0, delta=25):
    if _LIB is None:
        return _lcss_numpy(P, Q, eps, delta)
    _a, pp, m = _as_c(P); _b, qq, n = _as_c(Q)
    return float(_LIB.lcss_distance(pp, m, qq, n, eps, delta))


METRICS = {"frechet": frechet, "dtw": dtw, "hausdorff": hausdorff, "lcss": lcss}
PY_REFERENCE = {"frechet": _frechet_numpy, "dtw": _dtw_numpy,
                "hausdorff": _hausdorff_numpy}
