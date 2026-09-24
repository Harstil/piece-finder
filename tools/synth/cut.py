"""Jigsaw cut generator: turns a motif rectangle into cols x rows interlocking piece outlines.

Why it matters: the engine finds a piece's 4 core corners and classifies each side as flat, tab
or blank, then canonicalises the piece by its corners. So the tabs must look like real die-cut
tabs (round head, narrow neck, undercut), corners must be exact, and neighbouring pieces must
share one identical curve, or every shape metric measured on this data is wrong.

Geometry (all in motif.png pixel-centre coordinates, see src/engine/types.ts):
- The motif area is the pixel-edge rectangle [-0.5, W-0.5] x [-0.5, H-0.5]. Lattice point
  (c, r) sits at (-0.5 + c*W/cols, -0.5 + r*H/rows) before jitter.
- "grid" (ribbon cut): lattice untouched. "irregular": interior lattice points are jittered by
  up to IRREGULAR_JITTER of the cell, border points only along their border line, so pieces stay
  4-connected quads with straight outer borders.
- Each internal edge is sampled once, as a polyline from its start lattice point to its end.
  Both neighbours use that same polyline (one of them reversed), so shared edges are identical
  by construction and tab <-> blank is automatic.
- Piece outline = top edge (TL->TR), right (TR->BR), bottom (BR->BL), left (BL->TL): clockwise
  on screen, no repeated closing point, and the 4 core corners are exact outline vertices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .common import signed_area

# Interior lattice jitter for the "irregular" cut, as a fraction of the cell. From the task
# spec; measured to keep every piece a simple polygon and all tabs clear of each other (test_synth).
IRREGULAR_JITTER = 0.12

# Outline sampling density: points per edge chord, scaled by the edge's arc length. A flat edge
# gets EDGE_POINTS points, a tabbed one ~1.6x that, so an outline has ~256-420 points (the contract
# asks for >= 200). Guessed: dense enough that the polygon is smooth at 300 px cores.
EDGE_POINTS = 64

# Maximum core-cell aspect mismatch accepted when choosing a grid (cells are "about square").
# Guessed from real boxes: Ravensburger 1000 = 38x27 or 36x28, cells up to ~1.1 aspect.
MAX_CELL_ASPECT = 1.25


# ----------------------------------------------------------------------------------------------
# Grid choice
# ----------------------------------------------------------------------------------------------


def choose_grid(n_pieces: int, aspect: float) -> tuple[int, int]:
    """(cols, rows) with cols*rows close to n_pieces and near-square cells for motif aspect W/H.

    Cost = relative piece-count error + |log cell aspect| / 4, so a 2 % count error is worth the
    same as an 8 % aspect error; commercial grids behave like this (500 = 25x20 for a 4:3 motif,
    1000 = 40x25 for 3:2).
    """
    best: tuple[float, int, int] | None = None
    for rows in range(1, n_pieces + 1):
        for cols in {max(1, math.floor(n_pieces / rows)), max(1, math.ceil(n_pieces / rows))}:
            cell_aspect = (aspect / cols) / (1.0 / rows)
            if not (1 / MAX_CELL_ASPECT <= cell_aspect <= MAX_CELL_ASPECT):
                continue
            cost = abs(cols * rows - n_pieces) / n_pieces + abs(math.log(cell_aspect)) / 4.0
            if best is None or cost < best[0]:
                best = (cost, cols, rows)
    if best is None:  # extreme aspect: fall back to the count-only answer
        cols = max(1, round(math.sqrt(n_pieces * aspect)))
        return cols, max(1, round(n_pieces / cols))
    return best[1], best[2]


# ----------------------------------------------------------------------------------------------
# Tab curve
# ----------------------------------------------------------------------------------------------


def _cubic(p0, c1, c2, p1, n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    mt = 1.0 - t
    return (mt**3) * p0 + 3 * (mt**2) * t * c1 + 3 * mt * (t**2) * c2 + (t**3) * p1


def _tab_params(rng: np.random.Generator) -> dict[str, float]:
    """One tab's shape in "tab units" (1 = the local cell size).

    Ranges are guessed from photos of Ravensburger / Clementoni / Educa pieces: the head is a
    near-circle of ~0.24 cell diameter, the neck ~0.15 wide, the knob protrudes ~0.26 of the core.
    """
    p: dict[str, float] = {}
    for side in ("l", "r"):
        nw = rng.uniform(0.062, 0.084)            # neck half-width at its narrowest
        hr = rng.uniform(0.108, 0.132)            # head radius (half-width at its widest)
        hr = max(hr, nw + 0.034)                  # always a visible undercut
        nh = rng.uniform(0.035, 0.065)            # height of the neck's narrowest point
        hy = nh + rng.uniform(0.07, 0.10)         # height of the head's widest point
        p["nw" + side] = nw
        p["hr" + side] = hr
        p["nh" + side] = nh
        p["hy" + side] = hy
        p["fl" + side] = rng.uniform(0.022, 0.042)  # fillet where shoulder turns into the neck
        p["sh" + side] = rng.uniform(-0.012, 0.012)  # neck-base height (shoulder dip or rise)
        p["w" + side] = rng.uniform(-0.07, 0.07)   # shoulder tangent slope at the corner
    # Symmetric-ish: the right half copies the left with a few percent of mismatch.
    for k in ("nw", "hr", "nh", "hy"):
        p[k + "r"] = p[k + "l"] * rng.uniform(0.93, 1.07)
    p["hrr"] = max(p["hrr"], p["nwr"] + 0.034)
    top = max(p["hyl"] + p["hrl"], p["hyr"] + p["hrr"]) * rng.uniform(0.96, 1.02)
    p["ht"] = top                                 # knob top height
    p["sk"] = rng.uniform(-0.018, 0.018)          # sideways lean of the knob top
    p["c"] = 0.5 + rng.uniform(-0.045, 0.045)     # knob position along the edge (fraction)
    p["bow"] = rng.uniform(-0.012, 0.012)         # whole-edge waviness (fraction of chord)
    return p


def _tab_curve(p: dict[str, float], length: float, unit: float) -> np.ndarray:
    """Dense tab polyline in edge-local coords: x along the edge in [0, length], y = bulge (>0).

    `unit` is the tab size in px (the local cell size). Built from 8 cubic Beziers joined with
    matching tangents: shoulder, fillet, neck, head, head, neck, fillet, shoulder.
    """
    cx = p["c"] * length

    def L(u: float, v: float) -> np.ndarray:  # tab units -> edge-local px
        return np.array([cx + u * unit, v * unit])

    nwl, hrl, nhl, hyl, fll, shl = p["nwl"], p["hrl"], p["nhl"], p["hyl"], p["fll"], p["shl"]
    nwr, hrr, nhr, hyr, flr, shr = p["nwr"], p["hrr"], p["nhr"], p["hyr"], p["flr"], p["shr"]
    ht, sk = p["ht"], p["sk"]

    a = np.array([0.0, 0.0])
    b = np.array([length, 0.0])
    s_l = L(-(nwl + fll), shl)
    n_l = L(-nwl, nhl)
    h_l = L(-hrl, hyl)
    t = L(sk, ht)
    h_r = L(hrr, hyr)
    n_r = L(nwr, nhr)
    s_r = L(nwr + flr, shr)

    k = 0.5523  # quarter-circle Bezier constant
    segs = []
    d0 = s_l[0] - a[0]
    segs.append((a, a + np.array([d0 / 3, p["wl"] * d0 / 3]), s_l - np.array([d0 / 3, 0]), s_l))
    segs.append((s_l, s_l + np.array([k * fll * unit, 0]), n_l - np.array([0, k * (nhl - shl) * unit]), n_l))
    m = 0.5 * (hyl - nhl) * unit
    segs.append((n_l, n_l + np.array([0, m]), h_l - np.array([0, m]), h_l))
    segs.append((h_l, h_l + np.array([0, k * (ht - hyl) * unit]), t - np.array([k * (sk + hrl) * unit, 0]), t))
    segs.append((t, t + np.array([k * (hrr - sk) * unit, 0]), h_r + np.array([0, k * (ht - hyr) * unit]), h_r))
    m = 0.5 * (hyr - nhr) * unit
    segs.append((h_r, h_r - np.array([0, m]), n_r + np.array([0, m]), n_r))
    segs.append((n_r, n_r - np.array([0, k * (nhr - shr) * unit]), s_r - np.array([k * flr * unit, 0]), s_r))
    d1 = b[0] - s_r[0]
    segs.append((s_r, s_r + np.array([d1 / 3, 0]), b - np.array([d1 / 3, -p["wr"] * d1 / 3]), b))

    pts = [_cubic(*s, 48)[:-1] for s in segs]
    pts.append(b[None, :])
    curve = np.concatenate(pts, axis=0)
    curve[:, 1] += p["bow"] * length * np.sin(np.pi * np.clip(curve[:, 0] / length, 0.0, 1.0))
    return curve


def _resample(poly: np.ndarray, n: int) -> np.ndarray:
    """n points at uniform arc length along an open polyline, both endpoints kept exactly."""
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, s[-1], n)
    x = np.interp(targets, s, poly[:, 0])
    y = np.interp(targets, s, poly[:, 1])
    out = np.stack([x, y], axis=1)
    out[0] = poly[0]
    out[-1] = poly[-1]
    return out


def _edge_polyline(a: np.ndarray, b: np.ndarray, normal: np.ndarray, sign: int, p: dict[str, float] | None,
                   unit: float) -> np.ndarray:
    """Edge from lattice point a to b; bulges toward `normal` if sign > 0. Flat when p is None."""
    d = b - a
    length = float(np.linalg.norm(d))
    u = d / length
    if p is None:
        return _resample(np.stack([a, b]), EDGE_POINTS + 1)
    local = _tab_curve(p, length, unit)
    arc = float(np.linalg.norm(np.diff(local, axis=0), axis=1).sum())
    n_pts = int(round(EDGE_POINTS * arc / length)) + 1
    local = _resample(local, n_pts)
    world = a[None, :] + local[:, :1] * u[None, :] + (sign * local[:, 1:2]) * normal[None, :]
    world[0] = a
    world[-1] = b
    return world


# ----------------------------------------------------------------------------------------------
# Puzzle cut
# ----------------------------------------------------------------------------------------------


@dataclass
class Piece:
    id: int
    col: int
    row: int
    sides: list[str]            # top, right, bottom, left: "flat" | "tab" | "blank"
    corners: np.ndarray         # (4, 2) TL, TR, BR, BL in motif coords
    outline: np.ndarray         # (N, 2) clockwise, no closing duplicate

    @property
    def cell(self) -> int:
        return self.id


@dataclass
class Cut:
    cols: int
    rows: int
    width: int
    height: int
    kind: str                   # "grid" | "irregular"
    pieces: list[Piece]
    lattice: np.ndarray         # (rows+1, cols+1, 2)

    @property
    def cell_size(self) -> float:
        return math.sqrt((self.width / self.cols) * (self.height / self.rows))


def make_cut(rng: np.random.Generator, width: int, height: int, cols: int, rows: int, kind: str) -> Cut:
    if kind not in ("grid", "irregular"):
        raise ValueError(f"unknown cut kind {kind!r}")
    cw = width / cols
    ch = height / rows
    xs = -0.5 + np.arange(cols + 1) * cw
    ys = -0.5 + np.arange(rows + 1) * ch
    lat = np.stack(np.meshgrid(xs, ys), axis=-1).astype(np.float64)  # (rows+1, cols+1, 2)
    if kind == "irregular":
        j = rng.uniform(-IRREGULAR_JITTER, IRREGULAR_JITTER, size=lat.shape) * np.array([cw, ch])
        j[0, :, 1] = 0.0
        j[-1, :, 1] = 0.0
        j[:, 0, 0] = 0.0
        j[:, -1, 0] = 0.0
        lat = lat + j
        # Border corners of the whole puzzle stay put (both components zeroed above).
    unit_nominal = math.sqrt(cw * ch)

    # Horizontal edges: hedge[r][c] runs lat[r, c] -> lat[r, c+1]; normal points down (+y side).
    # Vertical edges:   vedge[r][c] runs lat[r, c] -> lat[r+1, c]; normal points right.
    # sign +1 = the bulge goes into the lower / right cell (so that cell has a blank there).
    hsign = np.where(rng.random((rows + 1, cols)) < 0.5, 1, -1)
    vsign = np.where(rng.random((rows, cols + 1)) < 0.5, 1, -1)
    hedge: list[list[np.ndarray]] = []
    for r in range(rows + 1):
        row_edges = []
        for c in range(cols):
            a, b = lat[r, c], lat[r, c + 1]
            d = (b - a) / np.linalg.norm(b - a)
            normal = np.array([-d[1], d[0]])  # (1,0) -> (0,1): down
            border = r == 0 or r == rows
            params = None if border else _tab_params(rng)
            unit = min(float(np.linalg.norm(b - a)), unit_nominal)
            row_edges.append(_edge_polyline(a, b, normal, int(hsign[r, c]), params, unit))
        hedge.append(row_edges)
    vedge: list[list[np.ndarray]] = []
    for r in range(rows):
        row_edges = []
        for c in range(cols + 1):
            a, b = lat[r, c], lat[r + 1, c]
            d = (b - a) / np.linalg.norm(b - a)
            normal = np.array([d[1], -d[0]])  # (0,1) -> (1,0): right
            border = c == 0 or c == cols
            params = None if border else _tab_params(rng)
            unit = min(float(np.linalg.norm(b - a)), unit_nominal)
            row_edges.append(_edge_polyline(a, b, normal, int(vsign[r, c]), params, unit))
        vedge.append(row_edges)

    pieces: list[Piece] = []
    for r in range(rows):
        for c in range(cols):
            top = hedge[r][c]
            right = vedge[r][c + 1]
            bottom = hedge[r + 1][c][::-1]
            left = vedge[r][c][::-1]
            outline = np.concatenate([top[:-1], right[:-1], bottom[:-1], left[:-1]], axis=0)
            sides = [
                "flat" if r == 0 else ("blank" if hsign[r, c] > 0 else "tab"),
                "flat" if c == cols - 1 else ("tab" if vsign[r, c + 1] > 0 else "blank"),
                "flat" if r == rows - 1 else ("tab" if hsign[r + 1, c] > 0 else "blank"),
                "flat" if c == 0 else ("blank" if vsign[r, c] > 0 else "tab"),
            ]
            corners = np.stack([lat[r, c], lat[r, c + 1], lat[r + 1, c + 1], lat[r + 1, c]])
            assert signed_area(outline) > 0
            pieces.append(Piece(id=r * cols + c, col=c, row=r, sides=sides, corners=corners, outline=outline))
    return Cut(cols=cols, rows=rows, width=width, height=height, kind=kind, pieces=pieces, lattice=lat)


def round_cut(cut: Cut, decimals: int = 2) -> Cut:
    """The cut exactly as it is written to pieces.json (rounded coordinates).

    Scenes and tests are built from this rounded version so the ground truth written to disk and
    the pixels rendered from it come from the very same numbers.
    """
    pieces = [Piece(id=p.id, col=p.col, row=p.row, sides=list(p.sides),
                    corners=np.round(p.corners, decimals) + 0.0, outline=np.round(p.outline, decimals) + 0.0)
              for p in cut.pieces]
    return Cut(cols=cut.cols, rows=cut.rows, width=cut.width, height=cut.height, kind=cut.kind,
               pieces=pieces, lattice=np.round(cut.lattice, decimals) + 0.0)
