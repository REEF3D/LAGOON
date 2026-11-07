# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Foam, spray and bubbles: white water from the free surface, for showcase films.

REEF3D does not resolve the air in breaking waves; this adds it the way film
effects do (diffuse particles after Ihmsen et al. 2012, "Unified spray, foam and
bubbles for particle-based fluids"), driven by the free surface and its velocity:

* particles are born where the surface breaks: where REEF3D flags breaking
  (NHFLOW's ``breaking``), on sharp crests moving forward, and where the surface
  flow converges and traps air; more of them the faster the water moves;
* they then live by where they are: **spray** above the surface flies on under
  gravity, **foam** on the surface drifts with its velocity and fades, **bubbles**
  below rise and turn into foam.

Everything scales with the case (Froude): the cell size sets the particles' size
and spacing, the water depth ``d`` the speeds (sqrt(g d)) and times (sqrt(d / g)),
so a lab flume and a coast get the same look. NumPy only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

G = 9.81

SPRAY, FOAM, BUBBLE = 0, 1, 2
#: grid points along the domain's edges where no white water is born
EDGE = 2
KINDS = {SPRAY: "spray", FOAM: "foam", BUBBLE: "bubbles"}


@dataclass(frozen=True)
class FoamSettings:
    """How much white water, how big and for how long (scaled to the case)."""

    #: particles born per cell and per unit of time sqrt(d/g) where the potential is 1
    rate: float = 20.0
    #: how long foam lasts on the surface, in units of sqrt(d / g) (randomly 0.5-1.5x)
    lifetime: float = 12.0
    #: particle radius in cell sizes (foam; spray and bubbles a little smaller)
    radius: float = 0.12
    #: at most this many particles (the oldest go first)
    limit: int = 400_000
    seed: int = 1

    def scaled(self, amount: float) -> FoamSettings:
        """Less (or more) white water: as many particles born and as long a life times
        ``amount`` (1: as above)."""
        return replace(self, rate=self.rate * amount, lifetime=self.lifetime * amount)


#: how much white water a path-traced render shows (1: FoamSettings as they are)
WHITE_WATER_AMOUNTS = {"none": 0.0, "light": 0.35, "medium": 0.6, "heavy": 1.0}
DEFAULT_WHITE_WATER = "medium"


def white_water_amount(foam) -> float:
    """The amount of white water ``foam`` asks for: True the default ("medium"),
    False or None none, a name of WHITE_WATER_AMOUNTS, or a number (0 - 2)."""
    if foam is True:
        return WHITE_WATER_AMOUNTS[DEFAULT_WHITE_WATER]
    if foam is False or foam is None:
        return 0.0
    if isinstance(foam, str):
        try:
            return WHITE_WATER_AMOUNTS[foam.lower()]
        except KeyError:
            raise ValueError(
                f"white water: one of {', '.join(WHITE_WATER_AMOUNTS)}, got {foam!r}"
            ) from None
    return min(max(float(foam), 0.0), 2.0)


# ======================================================================== surface
class SurfaceGrid:
    """The free surface on its rectilinear grid (REEF3D writes it on one): heights,
    velocity and REEF3D's breaking flag per grid point (NaN where dry)."""

    def __init__(self, xs, ys, z, u, v, w, breaking=None) -> None:
        self.xs, self.ys = np.asarray(xs, float), np.asarray(ys, float)
        self.z, self.u, self.v, self.w = (np.asarray(a, float) for a in (z, u, v, w))
        self.breaking = None if breaking is None else np.asarray(breaking, float)
        self.h = float(
            np.sqrt(np.mean(np.diff(self.xs)) * np.mean(np.diff(self.ys)))
            if len(self.xs) > 1 and len(self.ys) > 1
            else 1.0
        )

    @classmethod
    def from_points(cls, points, velocity, breaking=None) -> SurfaceGrid | None:
        """The grid of surface points (any order); None when they are not on one."""
        p = np.asarray(points, float).reshape(-1, 3)
        ok = np.isfinite(p).all(axis=1)
        if ok.sum() < 4:
            return None
        span = max(float(np.ptp(p[ok, :2], axis=0).max()), 1e-9)
        quantum = 1e-6 * span
        kx = np.round(p[ok, 0] / quantum).astype(np.int64)
        ky = np.round(p[ok, 1] / quantum).astype(np.int64)
        ux, ix = np.unique(kx, return_inverse=True)
        uy, iy = np.unique(ky, return_inverse=True)
        if len(ux) < 2 or len(uy) < 2 or len(ux) * len(uy) > 4 * ok.sum():
            return None  # scattered points (AMR, an unstructured surface)
        shape = (len(ux), len(uy))

        def grid(values):
            out = np.full(shape, np.nan)
            out[ix, iy] = np.asarray(values, float)[ok]
            return out

        vel = np.asarray(velocity, float).reshape(-1, 3)
        return cls(
            ux * quantum,
            uy * quantum,
            grid(p[:, 2]),
            grid(vel[:, 0]),
            grid(vel[:, 1]),
            grid(vel[:, 2]),
            None if breaking is None else grid(breaking),
        )

    @classmethod
    def from_scattered(
        cls,
        points,
        velocity,
        cell: float | None = None,
        triangles=None,
        fill: int = 2,
        like: SurfaceGrid | None = None,
    ) -> SurfaceGrid | None:
        """A surface whose points are on no grid (CFD's free surface, an iso-surface
        of the volume; AMR) put on one: grid points ``cell`` apart (default: the
        median plan length of the triangles' edges, about the volume's cell size),
        each the highest point near it (the top of an overturning crest) with its
        velocity. Holes of up to ``fill`` grid points (no point of the surface came
        near) take their neighbours' mean, so they make no false crests; the rest
        stays dry (NaN). ``like``: on that surface's grid (the frames of a film on
        one grid, so the white water carries on from frame to frame). None with too
        few points."""
        full = np.asarray(points, float).reshape(-1, 3)
        speed = np.asarray(velocity, float).reshape(-1, 3)
        ok = np.isfinite(full).all(axis=1) & np.isfinite(speed).all(axis=1)
        if ok.sum() < 4:
            return None
        t = None
        if triangles is not None and len(triangles):
            t = np.asarray(triangles, np.int64).reshape(-1, 3)
            t = t[ok[t].all(axis=1)]
        if like is not None and len(like.xs) > 1 and len(like.ys) > 1:
            xs, ys = like.xs, like.ys
            cell = float(xs[1] - xs[0])
        else:
            xs = ys = None
        if not cell and t is not None and len(t):
            edges = np.concatenate(
                [full[t[:, a], :2] - full[t[:, b], :2] for a, b in ((0, 1), (1, 2), (2, 0))]
            )
            lengths = np.linalg.norm(edges, axis=1)
            lengths = lengths[lengths > 0]
            cell = float(np.median(lengths)) if len(lengths) else None
        p, vel = full[ok], speed[ok]
        if xs is None:
            low, high = p[:, :2].min(axis=0), p[:, :2].max(axis=0)
            if not cell or cell <= 0:
                cell = float(max(high - low)) / max(math.sqrt(len(p)), 2.0)
            n = np.maximum(np.round((high - low) / cell).astype(np.int64) + 1, 2)
            if n[0] * n[1] > 4_000_000:  # a coarser grid for a very large surface
                cell *= math.sqrt(n[0] * n[1] / 4_000_000)
                n = np.maximum(np.round((high - low) / cell).astype(np.int64) + 1, 2)
            xs = low[0] + cell * np.arange(n[0])
            ys = low[1] + cell * np.arange(n[1])
        low = np.array([xs[0], ys[0]])
        n = np.array([len(xs), len(ys)])
        if t is not None and len(t):
            # points across the triangles too: a coarse cell (AMR's level 0 next to a
            # refined block) leaves no grid point without one
            extra, extra_vel = _triangle_points(full, speed, t, 0.5 * cell)
            p, vel = np.concatenate([p, extra]), np.concatenate([vel, extra_vel])
        inside = (
            (p[:, 0] >= xs[0] - cell / 2)
            & (p[:, 0] <= xs[-1] + cell / 2)
            & (p[:, 1] >= ys[0] - cell / 2)
            & (p[:, 1] <= ys[-1] + cell / 2)
        )
        p, vel = p[inside], vel[inside]
        i = np.clip(np.round((p[:, 0] - low[0]) / cell).astype(np.int64), 0, n[0] - 1)
        j = np.clip(np.round((p[:, 1] - low[1]) / cell).astype(np.int64), 0, n[1] - 1)
        # the highest point of each grid point: sorted by height, the last one wins
        order = np.argsort(p[:, 2], kind="stable")
        z = np.full(tuple(n), np.nan)
        u, v, w = (np.full(tuple(n), np.nan) for _ in range(3))
        z[i[order], j[order]] = p[order, 2]
        for field, k in ((u, 0), (v, 1), (w, 2)):
            field[i[order], j[order]] = vel[order, k]
        for _ in range(max(int(fill), 0)):
            holes = ~np.isfinite(z)
            if not holes.any():
                break
            fields = [z, u, v, w]
            sums = [np.zeros_like(z) for _ in fields]
            count = np.zeros_like(z)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                shifted = np.full_like(z, np.nan)
                src = (
                    slice(max(-di, 0), n[0] - max(di, 0)),
                    slice(max(-dj, 0), n[1] - max(dj, 0)),
                )
                dst = (
                    slice(max(di, 0), n[0] - max(-di, 0)),
                    slice(max(dj, 0), n[1] - max(-dj, 0)),
                )
                shifted[dst] = z[src]
                there = np.isfinite(shifted)
                count += there
                for total, field in zip(sums, fields, strict=True):
                    moved = np.zeros_like(z)
                    moved[dst] = field[src]
                    total += np.where(there, np.nan_to_num(moved), 0.0)
            # a hole between water on two sides at least (not the dry land's edge)
            filled = holes & (count >= 2)
            for total, field in zip(sums, fields, strict=True):
                field[filled] = total[filled] / count[filled]
        return cls(xs, ys, z, u, v, w)

    def blend(self, other: SurfaceGrid, s: float) -> SurfaceGrid:
        """The surface a fraction ``s`` of the way to ``other`` (same grid)."""
        if s <= 0.0 or other.z.shape != self.z.shape:
            return self
        if s >= 1.0:
            return other

        def mix(a, b):
            return (1 - s) * a + s * b

        breaking = self.breaking if s < 0.5 else other.breaking
        return SurfaceGrid(
            self.xs,
            self.ys,
            mix(self.z, other.z),
            mix(self.u, other.u),
            mix(self.v, other.v),
            mix(self.w, other.w),
            breaking,
        )

    def sample(self, xy: np.ndarray, *fields: np.ndarray) -> list[np.ndarray]:
        """Bilinear values of grid fields at points (n, 2); NaN outside and where dry."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        nx, ny = len(self.xs), len(self.ys)
        i = np.clip(np.searchsorted(self.xs, xy[:, 0]) - 1, 0, nx - 2)
        j = np.clip(np.searchsorted(self.ys, xy[:, 1]) - 1, 0, ny - 2)
        fx = (xy[:, 0] - self.xs[i]) / (self.xs[i + 1] - self.xs[i])
        fy = (xy[:, 1] - self.ys[j]) / (self.ys[j + 1] - self.ys[j])
        outside = (fx < -1e-9) | (fx > 1 + 1e-9) | (fy < -1e-9) | (fy > 1 + 1e-9)
        out = []
        for f in fields:
            value = (
                (1 - fx) * (1 - fy) * f[i, j]
                + fx * (1 - fy) * f[i + 1, j]
                + (1 - fx) * fy * f[i, j + 1]
                + fx * fy * f[i + 1, j + 1]
            )
            out.append(np.where(outside, np.nan, value))
        return out


def _triangle_points(points, velocity, triangles, spacing: float, most: int = 32):
    """Points inside the triangles at most ``spacing`` apart (barycentric steps, up to
    ``most`` along an edge) with their interpolated velocity; the corners are not
    among them."""
    corners = points[triangles]  # (n, 3, 3)
    longest = np.max(
        np.linalg.norm(corners[:, :, :2] - np.roll(corners[:, :, :2], 1, axis=1), axis=2), axis=1
    )
    steps = np.clip(np.ceil(longest / max(spacing, 1e-12)), 1, most).astype(np.int64)
    out, out_vel = [np.zeros((0, 3))], [np.zeros((0, 3))]
    for m in np.unique(steps):
        if m < 2:
            continue
        chosen = triangles[steps == m]
        a, b = np.meshgrid(np.arange(m + 1), np.arange(m + 1), indexing="ij")
        keep = (a + b <= m) & ~(((a == 0) | (a == m)) & ((b == 0) | (b == m)))
        a, b = a[keep] / m, b[keep] / m
        weights = np.stack([1.0 - a - b, a, b], axis=1)  # (k, 3)
        out.append(np.einsum("kc,ncd->nkd", weights, points[chosen]).reshape(-1, 3))
        out_vel.append(np.einsum("kc,ncd->nkd", weights, velocity[chosen]).reshape(-1, 3))
    return np.concatenate(out), np.concatenate(out_vel)


def _ramp(value, low: float, high: float) -> np.ndarray:
    """0 below ``low``, 1 above ``high``, linear between (Ihmsen's clamping)."""
    return np.clip((np.nan_to_num(value, nan=0.0) - low) / (high - low), 0.0, 1.0)


def potentials(grid: SurfaceGrid, depth: float) -> tuple[np.ndarray, np.ndarray]:
    """Where white water is born, per grid point (0..1): at breaking crests (REEF3D's
    flag, or a sharp crest moving forward, fast enough) and where the surface flow
    converges and traps air."""
    xs, ys = grid.xs, grid.ys
    z = np.where(np.isfinite(grid.z), grid.z, np.nanmean(grid.z))
    zx, zy = np.gradient(z, xs, ys)
    zxx = np.gradient(zx, xs, axis=0)
    zyy = np.gradient(zy, ys, axis=1)
    norm = np.sqrt(zx**2 + zy**2 + 1.0)
    nx, ny, nz = -zx / norm, -zy / norm, 1.0 / norm
    u, v, w = (np.nan_to_num(a) for a in (grid.u, grid.v, grid.w))
    speed = np.sqrt(u**2 + v**2 + w**2)
    facing = (u * nx + v * ny + w * nz) / np.maximum(speed, 1e-9)
    # how sharply the slope turns over one cell at a crest (dimensionless)
    crest = np.maximum(-(zxx + zyy), 0.0) * grid.h
    # fast water (the speed against the shallow water wave speed sqrt(g d), breaking
    # near 0.8), or a steep face (in deep water the waves are slower than sqrt(g d))
    energy = _ramp(speed / math.sqrt(G * depth), 0.15, 0.5)
    steep = _ramp(np.sqrt(zx**2 + zy**2), 0.35, 0.7)
    # a sharp crest whose water moves forward, not one falling back
    breaking_crest = (
        _ramp(crest, 0.03, 0.15) * _ramp(facing, -0.05, 0.25) * np.maximum(energy, steep)
    )
    if grid.breaking is not None:
        breaking_crest = np.maximum(breaking_crest, (np.nan_to_num(grid.breaking) > 0) * 1.0)
    divergence = np.gradient(u, xs, axis=0) + np.gradient(v, ys, axis=1)
    trapped = _ramp(-divergence * math.sqrt(depth / G), 0.05, 0.5) * energy
    dry = ~np.isfinite(grid.z)
    breaking_crest[dry] = 0.0
    trapped[dry] = 0.0
    # none along the domain's edges: the slopes there are one-sided, and a surface
    # cut from the volume (CFD's) bends up or down in the last cells at an outlet
    for field in (breaking_crest, trapped):
        field[:EDGE], field[-EDGE:], field[:, :EDGE], field[:, -EDGE:] = 0.0, 0.0, 0.0, 0.0
    return breaking_crest, trapped


# ======================================================================== particles
class FoamSimulation:
    """White water over time: :meth:`advance` from one surface to the next spawns,
    moves and ages the particles; :meth:`particles` gives them for rendering."""

    def __init__(self, depth: float, settings: FoamSettings | None = None) -> None:
        self.settings = settings or FoamSettings()
        self.depth = max(float(depth), 1e-6)
        self.rng = np.random.default_rng(self.settings.seed)
        self.position = np.zeros((0, 3))
        self.velocity = np.zeros((0, 3))
        self.kind = np.zeros(0, np.int8)
        self.life = np.zeros(0)  # what is left of a foam particle's life [s]
        self.size = np.zeros(0)  # each particle's size, a factor around 1
        self.h = 1.0
        self.time: float | None = None

    @property
    def unit_time(self) -> float:
        return math.sqrt(self.depth / G)

    def __len__(self) -> int:
        return len(self.position)

    # ------------------------------------------------------------- stepping
    def advance(self, start: SurfaceGrid, end: SurfaceGrid, t0: float, t1: float) -> None:
        """From time t0 (surface ``start``) to t1 (surface ``end``), in substeps."""
        span = float(t1 - t0)
        self.h = start.h
        if span <= 0:
            self.time = t1
            return
        flow = np.abs(np.stack([start.u, start.v, start.w]))
        fastest = float(np.nanmax(flow)) if np.isfinite(flow).any() else 0.0
        fastest = max(fastest, 0.2 * math.sqrt(G * self.depth))
        dt = min(0.5 * start.h / fastest, 0.1 * self.unit_time)
        steps = int(min(max(math.ceil(span / dt), 1), 200))
        dt = span / steps
        b0, t0_ = potentials(start, self.depth)
        b1, t1_ = potentials(end, self.depth)
        for k in range(steps):
            s = (k + 0.5) / steps
            surface = start.blend(end, s)
            self._spawn(surface, (1 - s) * b0 + s * b1, (1 - s) * t0_ + s * t1_, dt)
            self._move(surface, dt)
        self.time = t1

    def _spawn(self, surface: SurfaceGrid, crest, trapped, dt: float) -> None:
        settings = self.settings
        per_cell = settings.rate * dt / self.unit_time
        rng = self.rng
        new = []
        for potential, kind in ((crest, SPRAY), (trapped, BUBBLE)):
            expected = potential * per_cell
            counts = rng.poisson(expected) if expected.max() > 0 else None
            if counts is None or not counts.any():
                continue
            i, j = np.nonzero(counts)
            n = counts[i, j]
            i, j = np.repeat(i, n), np.repeat(j, n)
            m = len(i)
            # anywhere in the cell around the grid point
            x = surface.xs[i] + (rng.random(m) - 0.5) * surface.h
            y = surface.ys[j] + (rng.random(m) - 0.5) * surface.h
            z = surface.z[i, j]
            vel = np.stack([surface.u[i, j], surface.v[i, j], surface.w[i, j]], 1)
            speed = np.linalg.norm(vel, axis=1)
            r = settings.radius * surface.h
            if kind == SPRAY:
                # thrown up and forward off the crest
                z = z + r * (1.0 + rng.random(m))
                vel = vel * (0.8 + 0.4 * rng.random((m, 1)))
                vel[:, 2] += speed * (0.2 + 0.4 * rng.random(m))
            else:
                z = z - surface.h * (0.5 + rng.random(m))  # air drawn under
            new.append((np.stack([x, y, z], 1), vel, np.full(m, kind, np.int8)))
        if not new:
            return
        positions = np.concatenate([p for p, _, _ in new])
        ok = np.isfinite(positions).all(axis=1)
        life = self.unit_time * settings.lifetime * (0.5 + self.rng.random(int(ok.sum())))
        self.position = np.concatenate([self.position, positions[ok]])
        self.velocity = np.concatenate([self.velocity, np.concatenate([v for _, v, _ in new])[ok]])
        self.kind = np.concatenate([self.kind, np.concatenate([k for _, _, k in new])[ok]])
        self.life = np.concatenate([self.life, life])
        self.size = np.concatenate([self.size, 0.6 + 0.8 * self.rng.random(int(ok.sum()))])
        extra = len(self.position) - settings.limit
        if extra > 0:  # the oldest go
            self._keep(np.arange(len(self.position)) >= extra)

    def _move(self, surface: SurfaceGrid, dt: float) -> None:
        if not len(self.position):
            return
        p, vel = self.position, self.velocity
        z, u, v, w = surface.sample(p[:, :2], surface.z, surface.u, surface.v, surface.w)
        alive = np.isfinite(z)
        r = self.settings.radius * surface.h
        height = p[:, 2] - z
        kind = np.where(height > r, SPRAY, np.where(height < -2 * r, BUBBLE, FOAM)).astype(np.int8)
        flow = np.stack([u, v, w], 1)
        spray, foam, bubble = kind == SPRAY, kind == FOAM, kind == BUBBLE
        # spray: thrown, falling under gravity
        vel[spray, 2] -= G * dt
        # foam: carried on the surface
        vel[foam] = flow[foam]
        # bubbles: dragged along by the water, rising
        drag = min(dt / (0.2 * self.unit_time), 1.0)
        vel[bubble] += drag * (flow[bubble] - vel[bubble])
        vel[bubble, 2] += 0.5 * G * dt
        vel[bubble, 2] = np.minimum(vel[bubble, 2], 0.3 * math.sqrt(G * self.depth))
        p += vel * dt
        # foam stays on the surface
        if foam.any():
            zf = surface.sample(p[foam, :2], surface.z)[0]
            p[foam, 2] = np.where(np.isfinite(zf), zf + 0.3 * r, p[foam, 2])
        self.life[foam] -= dt
        self.kind = kind
        alive &= np.isfinite(p).all(axis=1) & (self.life > 0)
        self._keep(alive)

    def _keep(self, mask: np.ndarray) -> None:
        self.position = self.position[mask]
        self.velocity = self.velocity[mask]
        self.kind = self.kind[mask]
        self.life = self.life[mask]
        self.size = self.size[mask]

    # ------------------------------------------------------------- result
    def particles(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Per kind ("spray", "foam", "bubbles"): positions (n, 3) and radii (n,)."""
        out = {}
        base = self.settings.radius * self.h
        size = {SPRAY: 0.6, FOAM: 1.0, BUBBLE: 0.5}
        for kind, name in KINDS.items():
            mine = self.kind == kind
            if not mine.any():
                continue
            radius = base * size[kind] * self.size[mine]
            if kind == FOAM:  # foam thins out as it fades
                fading = 0.3 * self.unit_time * self.settings.lifetime
                radius = radius * (0.4 + 0.6 * np.clip(self.life[mine] / fading, 0, 1))
            out[name] = (self.position[mine].astype(np.float32), radius.astype(np.float32))
        return out
