"""Route Similarity Preservation Problem instances and disruption models.

An RSPP instance is a *pair*: a baseline problem P together with the plan
X_baseline that was in force when the shift started, and a disrupted problem P'
that must be replanned.  This pairing is precisely what public VRP benchmarks
do not provide, and it is what RSPP-Bench supplies.

Coordinates are planar and in metres throughout; there is no geodetic step and
therefore no projection error.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

import numpy as np


@dataclass
class Instance:
    """A capacitated routing instance with a fixed crew and duration limits."""

    name: str
    coords: np.ndarray          # (n+1, 2) metres; row 0 is the depot
    demand: np.ndarray          # (n+1,) units; demand[0] == 0
    service: np.ndarray         # (n+1,) seconds; service[0] == 0
    capacity: float             # per-vehicle capacity, units
    n_vehicles: int             # crew size K (fixed: this is workforce management)
    max_duration: float         # per-route duration cap, seconds
    speed: float = 8.33         # m/s (~30 km/h urban average)
    available: np.ndarray = None  # (K,) bool; False = vehicle withdrawn
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.coords = np.asarray(self.coords, dtype=np.float64)
        self.demand = np.asarray(self.demand, dtype=np.float64)
        self.service = np.asarray(self.service, dtype=np.float64)
        if self.available is None:
            self.available = np.ones(self.n_vehicles, dtype=bool)
        else:
            self.available = np.asarray(self.available, dtype=bool)
        self._dist = None

    @property
    def n_customers(self) -> int:
        return len(self.coords) - 1

    @property
    def dist(self) -> np.ndarray:
        """Euclidean travel-cost matrix, metres. Cached; O(n^2) memory."""
        if self._dist is None:
            c = self.coords
            self._dist = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        return self._dist

    def customers(self) -> np.ndarray:
        return np.arange(1, len(self.coords))

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("coords", "demand", "service", "available"):
            d[k] = np.asarray(getattr(self, k)).tolist()
        d.pop("_dist", None)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Instance":
        d = dict(d)
        d.pop("_dist", None)
        return Instance(**d)


@dataclass
class RSPPPair:
    """Baseline plan + disrupted instance: one RSPP-Bench problem."""

    name: str
    base: Instance              # instance as planned at shift start
    disrupted: Instance         # instance after the disruption
    baseline_routes: list       # list of K lists of customer indices, in `base`
    disruption: dict            # machine-readable manifest of what changed
    seed: int

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "base": self.base.to_dict(),
            "disrupted": self.disrupted.to_dict(),
            "baseline_routes": [list(map(int, r)) for r in self.baseline_routes],
            "disruption": self.disruption,
            "seed": self.seed,
        }

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh)

    @staticmethod
    def load(path: str) -> "RSPPPair":
        with open(path) as fh:
            d = json.load(fh)
        return RSPPPair(
            name=d["name"],
            base=Instance.from_dict(d["base"]),
            disrupted=Instance.from_dict(d["disrupted"]),
            baseline_routes=[list(r) for r in d["baseline_routes"]],
            disruption=d["disruption"],
            seed=d["seed"],
        )


# ------------------------------------------------------------------ layouts
def _layout_clustered(rng, n, extent, n_clusters=6):
    centres = rng.uniform(-extent * 0.7, extent * 0.7, (n_clusters, 2))
    which = rng.integers(0, n_clusters, n)
    return centres[which] + rng.normal(0.0, extent * 0.07, (n, 2))


def _layout_random(rng, n, extent):
    return rng.uniform(-extent, extent, (n, 2))


def _layout_mixed(rng, n, extent, n_clusters=4):
    n_clu = n // 2
    a = _layout_clustered(rng, n_clu, extent, n_clusters)
    b = _layout_random(rng, n - n_clu, extent)
    return np.vstack([a, b])


LAYOUTS = {"C": _layout_clustered, "R": _layout_random, "RC": _layout_mixed}


def make_instance(name, n, n_vehicles, seed, layout="R", extent=12_000.0,
                  capacity_slack=1.35, max_hours=8.0):
    """Generate a base instance. ``extent`` is the half-width of the service
    region in metres, so a default instance spans roughly 24 x 24 km."""
    rng = np.random.default_rng(seed)
    pts = LAYOUTS[layout](rng, n, extent)
    coords = np.vstack([np.zeros((1, 2)), pts])          # depot at the origin
    demand = np.concatenate([[0.0], rng.integers(1, 20, n).astype(float)])
    service = np.concatenate([[0.0], rng.integers(300, 1200, n).astype(float)])
    capacity = float(np.ceil(demand.sum() / n_vehicles * capacity_slack))
    return Instance(name=name, coords=coords, demand=demand, service=service,
                    capacity=capacity, n_vehicles=n_vehicles,
                    max_duration=max_hours * 3600.0,
                    meta={"layout": layout, "extent": extent, "seed": seed})
