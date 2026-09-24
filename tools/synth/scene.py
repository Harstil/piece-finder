"""Composes camera frames of loose pieces on a table, with exact ground truth.

Why this is the heart of the generator: every matching, shape and segmentation metric is
computed against what this module writes, so the geometry is done once, analytically, and the
pixels are rendered *from* that geometry — never the other way round.

Model: a flat table plane seen by a phone camera. Plane coordinates are what the camera would see
looking straight down (principal point at the frame centre); a tilted phone is the homography
H = K R K^-1 (common.camera_rotation_homography). Each piece is placed on the plane by a
similarity A (rotation, scale, position; mirrored if face-down), so motif -> frame is H @ A,
exactly, for the outline, the corners and the printed texture alike.

Ground truth per piece (docs/DATASET.md): corners in MOTIF order TL, TR, BR, BL mapped into the
frame; the top-face outline; upAngleDeg = frame direction of the motif-up vector at the core centre
(Jacobian of H @ A, clockwise from image-up); visibleFraction = visible top-face pixels / all
top-face pixels (occlusion by later pieces and the frame edge); faceUp. The 8-bit mask holds the
1-based paint-order index of the piece covering each pixel (top face plus visible side wall).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .backgrounds import make_background
from .camera import apply_camera, lighting_map, phone_shadow
from .common import (apply_h, choice_weighted, clip_bbox, camera_rotation_homography, local_scale, poly_bbox,
                     pts_json, raster_alpha, raster_binary, rot_mat, scale_mat, translate, up_angle_deg, write_jpg,
                     write_json, write_png)
from .cut import Cut, Piece
from .render import (PrintStyle, contact_and_drop_shadow, glare_field, surface_finish, sweep_alpha, wall_colour,
                     warp_surface)

# ---- Frame and camera. Guessed from iPhone 1080p video (26 mm-equivalent main lens).
PORTRAIT = (1080, 1920)            # (width, height): the phone is normally held upright
LANDSCAPE = (1920, 1080)
P_PORTRAIT = 0.75
FOCAL_PX = (1250.0, 1550.0)        # ~65-75 deg across the long side
MAX_TILT_DEG = 25.0                # from the task spec
TILT_SCALE_DEG = 9.0               # |N(0, 9)| clipped: most frames are held nearly flat
# ---- Scale: frame px per piece core side, tied to piece count (see core_size_range).
CORE_PX_LIMITS = (60.0, 280.0)     # from the task spec
CORE_PX_K = 2600.0                 # core ~= K / sqrt(pieces): 100 -> 260 px, 500 -> 116, 1000 -> 82
# ---- Layout. Rates guessed to cover real sorting trays: most pieces apart, some touching.
MAX_PIECES = 25
P_TOUCH = 0.18                     # share of pieces placed touching a neighbour
P_OVERLAP = 0.08                   # share lying partly on top of a neighbour
OVERLAP_FRACTION = (0.03, 0.25)    # of the lower piece's area
P_FRAME_EDGE = 0.12                # share cut off by the frame edge
FREE_GAP_PX = 3.0                  # minimum gap for pieces placed "apart"
FACE_DOWN_MEAN = 0.5               # Poisson mean of face-down distractors per frame (+1..4 on clutter)
# ---- Lighting.
P_GLARE = 0.4
P_PHONE_SHADOW = 0.2
LIGHT_ELEVATION_DEG = (35.0, 80.0)
SHADOW_STRENGTH = (0.2, 0.5)
JPEG_QUALITY = (60, 95)            # from the task spec
PLACE_RES = 2                      # occupancy grid is 1/PLACE_RES of the frame


@dataclass
class PuzzleForScenes:
    pid: str
    cut: Cut                 # rounded exactly as written to pieces.json
    printed: np.ndarray      # motif after the puzzle's print style (uint8 RGB)
    style: PrintStyle
    n_pieces_nominal: int


def core_size_range(n_pieces: int) -> tuple[float, float]:
    mid = CORE_PX_K / math.sqrt(max(1, n_pieces))
    lo = float(np.clip(mid * 0.7, *CORE_PX_LIMITS))
    hi = float(np.clip(mid * 1.3, *CORE_PX_LIMITS))
    if hi - lo < 10.0:
        lo = max(CORE_PX_LIMITS[0], hi - 10.0) if hi >= CORE_PX_LIMITS[1] else lo
        hi = max(hi, lo + 10.0)
    return lo, hi


@dataclass
class _Placed:
    piece: Piece
    face_up: bool
    A: np.ndarray            # motif -> plane
    frame_poly: np.ndarray   # top-face outline in frame px
    centre: np.ndarray       # frame px
    radius: float
    angle: float             # in-plane rotation in degrees


def render_scene(rng: np.random.Generator, puzzle: PuzzleForScenes) -> tuple[np.ndarray, np.ndarray, dict, int]:
    """(frame uint8 RGB, mask uint8, scene JSON dict, JPEG quality)."""
    W, Hh = PORTRAIT if rng.random() < P_PORTRAIT else LANDSCAPE
    f = rng.uniform(*FOCAL_PX)
    tilt = float(min(MAX_TILT_DEG, abs(rng.normal(0.0, TILT_SCALE_DEG))))
    tilt_dir = rng.uniform(0, 360)
    H = camera_rotation_homography(W, Hh, f, tilt, tilt_dir)
    Hinv = np.linalg.inv(H)
    centre_plane = apply_h(Hinv, np.array([[(W - 1) / 2, (Hh - 1) / 2]]))[0]
    ls = local_scale(H, centre_plane)
    lo, hi = core_size_range(puzzle.n_pieces_nominal)
    core_px = rng.uniform(lo, hi)
    s = (core_px / ls) / puzzle.cut.cell_size         # motif px -> plane px
    thick = puzzle.style.thickness * core_px / ls      # plane px
    O = np.array([(W - 1) / 2, (Hh - 1) / 2])         # frontal principal point, plane coords
    D = f                                              # camera height in plane px

    placed, occ = _place_pieces(rng, puzzle, H, Hinv, W, Hh, s)

    # ---- Background, warped from the table plane.
    corners_plane = apply_h(Hinv, np.array([[0, 0], [W - 1, 0], [W - 1, Hh - 1], [0, Hh - 1]], np.float64))
    px0, py0 = corners_plane.min(axis=0) - 8
    px1, py1 = corners_plane.max(axis=0) + 8
    area = (px1 - px0) * (py1 - py0)
    bs = min(1.0, math.sqrt(3.2e6 / area))
    bw, bh = int(math.ceil((px1 - px0) * bs)), int(math.ceil((py1 - py0) * bs))
    small_motif = cv2.resize(puzzle.printed, (64, 48), interpolation=cv2.INTER_AREA)
    bg, bg_label = make_background(rng, bh, bw, motif=small_motif)
    clutter_extra = bg_label == "clutter"
    Mbg = H @ translate(px0, py0) @ scale_mat(1.0 / bs)
    canvas = cv2.warpPerspective(bg, Mbg, (W, Hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    placed = _add_face_down(rng, puzzle, placed, occ, H, Hinv, W, Hh, s, extra=clutter_extra)

    # ---- Pieces, in paint order.
    light_az = rng.uniform(0, 2 * math.pi)
    to_light = np.array([math.cos(light_az), math.sin(light_az)])
    elev = math.radians(rng.uniform(*LIGHT_ELEVATION_DEG))
    shadow_off = -to_light * thick / math.tan(elev)
    shadow_blur = thick * rng.uniform(0.3, 1.0) * ls
    shadow_strength = rng.uniform(*SHADOW_STRENGTH)
    labels = np.zeros((Hh, W), np.uint8)
    gloss = np.zeros((Hh, W), np.float32)
    records = []
    for k, pl in enumerate(placed, start=1):
        top_plane = apply_h(pl.A, pl.piece.outline)
        foot_plane = O + (top_plane - O) * ((D - thick) / D)
        top_f = apply_h(H, top_plane)
        foot_f = apply_h(H, foot_plane)
        shadow_f = apply_h(H, foot_plane + shadow_off)
        pad = 3.0 * shadow_blur + 3.0
        full = poly_bbox(np.concatenate([top_f, foot_f, shadow_f]), pad)
        bb = clip_bbox(full, W, Hh)
        tb = poly_bbox(top_f, 1.0)
        alpha_top_full = raster_alpha([top_f], tb[0], tb[1], tb[2] - tb[0], tb[3] - tb[1])
        full_count = int((alpha_top_full >= 0.5).sum())
        rec = {"placed": pl, "top_f": top_f, "full_count": full_count, "bbox": bb, "top_bin": None}
        records.append(rec)
        if bb[2] <= bb[0] or bb[3] <= bb[1]:
            continue
        x0, y0, x1, y1 = bb
        bwid, bhei = x1 - x0, y1 - y0
        alpha_top = _crop_alpha(alpha_top_full, tb, bb)
        alpha_foot = raster_alpha([foot_f], x0, y0, bwid, bhei)
        disp = (foot_f - top_f).mean(axis=0)
        wall = np.maximum(sweep_alpha(alpha_top, disp), alpha_foot)
        drop, contact = contact_and_drop_shadow(foot_f, shadow_f, bb, shadow_blur)
        region = canvas[y0:y1, x0:x1]
        region = region * np.clip(1.0 - shadow_strength * drop - 0.3 * contact, 0, 1)[..., None]
        to_cam = O - apply_h(pl.A, pl.piece.corners.mean(axis=0)[None, :])[0]
        facing = 0.5 + 0.5 * float(np.dot(to_cam / (np.linalg.norm(to_cam) + 1e-6), to_light))
        wall_rgb = wall_colour(rng, puzzle.style, facing)
        region = region * (1 - wall[..., None]) + wall_rgb[None, None, :] * wall[..., None]
        if pl.face_up:
            top_rgb = warp_surface(puzzle.printed, H @ pl.A, pl.piece.outline, bb)
        else:
            top_rgb = np.zeros((bhei, bwid, 3), np.float32)
        top_rgb = surface_finish(rng, top_rgb, alpha_top, puzzle.style, core_px, pl.angle, pl.face_up)
        region = region * (1 - alpha_top[..., None]) + top_rgb * alpha_top[..., None]
        canvas[y0:y1, x0:x1] = region
        cover = np.maximum(alpha_top, wall)
        lab = labels[y0:y1, x0:x1]
        lab[cover >= 0.5] = k
        g = gloss[y0:y1, x0:x1]
        g_k = puzzle.style.gloss * rng.uniform(0.3, 1.0) if pl.face_up else 0.15
        gloss[y0:y1, x0:x1] = g * (1 - cover) + g_k * alpha_top
        rec["top_bin"] = alpha_top >= 0.5

    # ---- Lighting, glare, phone shadow, camera.
    canvas = canvas * lighting_map(rng, Hh, W)[..., None]
    has_glare = rng.random() < P_GLARE
    if has_glare:
        gl = np.clip(glare_field(rng, Hh, W) * gloss, 0, 1)[..., None]
        canvas = canvas + (1.0 - canvas) * gl
    has_phone_shadow = rng.random() < P_PHONE_SHADOW
    if has_phone_shadow:
        canvas = canvas * phone_shadow(rng, Hh, W)[..., None]
    frame, cam = apply_camera(rng, np.clip(canvas, 0, 1), tilt_dir_deg=tilt_dir, tilt_deg=tilt)
    quality = int(rng.integers(JPEG_QUALITY[0], JPEG_QUALITY[1] + 1))

    # ---- Ground truth.
    pieces_json = []
    for k, rec in enumerate(records, start=1):
        pl: _Placed = rec["placed"]
        visible = 0
        if rec["top_bin"] is not None:
            x0, y0, x1, y1 = rec["bbox"]
            visible = int(((labels[y0:y1, x0:x1] == k) & rec["top_bin"]).sum())
        vis = min(1.0, visible / max(1, rec["full_count"]))
        corners_f = apply_h(H @ pl.A, pl.piece.corners)
        outline_f = rec["top_f"] if pl.face_up else rec["top_f"][::-1]
        pieces_json.append({
            "index": k, "pieceId": pl.piece.id, "col": pl.piece.col, "row": pl.piece.row, "cell": pl.piece.cell,
            "upAngleDeg": round(up_angle_deg(H @ pl.A, pl.piece.corners), 2),
            "corners": pts_json(corners_f), "outline": pts_json(outline_f),
            "visibleFraction": round(vis, 4), "faceUp": pl.face_up,
        })
    scene = {
        "puzzleId": puzzle.pid, "width": W, "height": Hh, "background": bg_label, "pieces": pieces_json,
        "render": {
            "corePx": round(core_px, 1), "tiltDeg": round(tilt, 2), "focalPx": round(f, 1),
            "jpegQuality": quality, "glare": has_glare, "phoneShadow": has_phone_shadow, **cam,
        },
    }
    return frame, labels, scene, quality


def _crop_alpha(alpha: np.ndarray, src: tuple[int, int, int, int], dst: tuple[int, int, int, int]) -> np.ndarray:
    """Re-window an alpha rasterised over bbox `src` onto bbox `dst` (zeros outside src)."""
    out = np.zeros((dst[3] - dst[1], dst[2] - dst[0]), np.float32)
    ix0, iy0 = max(src[0], dst[0]), max(src[1], dst[1])
    ix1, iy1 = min(src[2], dst[2]), min(src[3], dst[3])
    if ix1 > ix0 and iy1 > iy0:
        out[iy0 - dst[1]:iy1 - dst[1], ix0 - dst[0]:ix1 - dst[0]] = alpha[iy0 - src[1]:iy1 - src[1], ix0 - src[0]:ix1 - src[0]]
    return out


# ----------------------------------------------------------------------------------------------
# Placement
# ----------------------------------------------------------------------------------------------


class _Occupancy:
    """Coarse frame-space occupancy (1/PLACE_RES px), padded so off-frame parts still count."""

    def __init__(self, W: int, Hh: int):
        self.pad = 400 // PLACE_RES
        self.grid = np.zeros((Hh // PLACE_RES + 2 * self.pad, W // PLACE_RES + 2 * self.pad), np.uint8)

    def _raster(self, poly_f: np.ndarray, dilate_px: float = 0.0):
        q = poly_f / PLACE_RES + self.pad
        x0, y0, x1, y1 = poly_bbox(q, dilate_px / PLACE_RES + 1)
        gh, gw = self.grid.shape
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, gw), min(y1, gh)
        if x1 <= x0 or y1 <= y0:
            return None
        m = raster_binary([q], x0, y0, x1 - x0, y1 - y0)
        if dilate_px > 0:
            r = max(1, int(round(dilate_px / PLACE_RES)))
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        return m, (x0, y0, x1, y1)

    def overlap(self, poly_f: np.ndarray, dilate_px: float = 0.0) -> tuple[int, int]:
        """(overlapping cells, candidate cells)."""
        r = self._raster(poly_f, dilate_px)
        if r is None:
            return 0, 0
        m, (x0, y0, x1, y1) = r
        return int((m & self.grid[y0:y1, x0:x1]).sum()), int(m.sum())

    def add(self, poly_f: np.ndarray) -> None:
        r = self._raster(poly_f)
        if r is not None:
            m, (x0, y0, x1, y1) = r
            self.grid[y0:y1, x0:x1] |= m


def _similarity(piece: Piece, s: float, angle: float, pos_plane: np.ndarray, mirror: bool) -> np.ndarray:
    c = piece.corners.mean(axis=0)
    M = translate(*pos_plane) @ rot_mat(angle) @ scale_mat(s)
    if mirror:
        M = M @ np.diag([-1.0, 1.0, 1.0])
    return M @ translate(-c[0], -c[1])


def _candidate(piece: Piece, s: float, angle: float, centre_f: np.ndarray, H: np.ndarray, Hinv: np.ndarray,
               mirror: bool) -> tuple[np.ndarray, np.ndarray]:
    pos_plane = apply_h(Hinv, centre_f[None, :])[0]
    A = _similarity(piece, s, angle, pos_plane, mirror)
    return A, apply_h(H @ A, piece.outline)


def _place_pieces(rng: np.random.Generator, puzzle: PuzzleForScenes, H: np.ndarray, Hinv: np.ndarray, W: int,
                  Hh: int, s: float) -> tuple[list[_Placed], _Occupancy]:
    cut = puzzle.cut
    core_f = s * cut.cell_size * local_scale(H, apply_h(Hinv, np.array([[W / 2, Hh / 2]]))[0])
    piece_area = 1.45 * core_f ** 2
    fill = rng.uniform(0.12, 0.45)
    capacity = int(np.clip(W * Hh * fill / piece_area, 1, MAX_PIECES))
    n = int(rng.integers(1, capacity + 1))
    ids = rng.choice(len(cut.pieces), size=min(n, len(cut.pieces)), replace=False)
    occ = _Occupancy(W, Hh)
    placed: list[_Placed] = []
    for pid in ids:
        pl = _place_one(rng, cut.pieces[int(pid)], True, placed, occ, H, Hinv, W, Hh, s)
        if pl is not None:
            placed.append(pl)
            occ.add(pl.frame_poly)
    return placed, occ


def _add_face_down(rng: np.random.Generator, puzzle: PuzzleForScenes, placed: list[_Placed], occ: _Occupancy,
                   H: np.ndarray, Hinv: np.ndarray, W: int, Hh: int, s: float, extra: bool) -> list[_Placed]:
    """Insert face-down distractors (other pieces of the same puzzle) at random paint positions."""
    used = {p.piece.id for p in placed}
    n = min(4, int(rng.poisson(FACE_DOWN_MEAN)) + (int(rng.integers(1, 5)) if extra else 0))
    free = [p for p in puzzle.cut.pieces if p.id not in used]
    out = list(placed)
    for _ in range(n):
        if not free:
            break
        piece = free.pop(int(rng.integers(len(free))))
        pl = _place_one(rng, piece, False, out, occ, H, Hinv, W, Hh, s)
        if pl is None:
            continue
        occ.add(pl.frame_poly)
        out.insert(int(rng.integers(0, len(out) + 1)), pl)
    return out


def _place_one(rng: np.random.Generator, piece: Piece, face_up: bool, placed: list[_Placed], occ: _Occupancy,
               H: np.ndarray, Hinv: np.ndarray, W: int, Hh: int, s: float) -> _Placed | None:
    angle = rng.uniform(0, 360)
    mirror = not face_up
    mode = "free"
    if face_up and placed:
        mode = choice_weighted(rng, ["free", "touch", "overlap", "edge"],
                               [1 - P_TOUCH - P_OVERLAP - P_FRAME_EDGE, P_TOUCH, P_OVERLAP, P_FRAME_EDGE])
    elif face_up and rng.random() < P_FRAME_EDGE:
        mode = "edge"
    # Frame-space radius of this piece at this scale (for margins and search distances).
    _, poly0 = _candidate(piece, s, angle, np.array([W / 2, Hh / 2]), H, Hinv, mirror)
    radius = float(np.max(np.linalg.norm(poly0 - poly0.mean(axis=0), axis=1)))

    def make(centre: np.ndarray) -> _Placed:
        A, poly = _candidate(piece, s, angle, centre, H, Hinv, mirror)
        return _Placed(piece, face_up, A, poly, centre, radius, angle)

    if mode in ("touch", "overlap"):
        anchor = placed[int(rng.integers(len(placed)))]
        if anchor.face_up:
            target = 0.0 if mode == "touch" else rng.uniform(*OVERLAP_FRACTION)
            res = _slide(rng, anchor, radius, target, make, occ, W, Hh)
            if res is not None:
                return res
        mode = "free"
    for _ in range(40):
        if mode == "edge":
            side = int(rng.integers(4))
            off = rng.uniform(-0.35, 0.45) * radius  # >0: centre inside the frame
            along = rng.uniform(0.1, 0.9)
            centre = [np.array([along * W, off]), np.array([W - 1 - off, along * Hh]),
                      np.array([along * W, Hh - 1 - off]), np.array([off, along * Hh])][side]
        else:
            m = radius + FREE_GAP_PX
            if W - 2 * m <= 0 or Hh - 2 * m <= 0:
                return None
            centre = np.array([rng.uniform(m, W - m), rng.uniform(m, Hh - m)])
        cand = make(centre)
        inter, _ = occ.overlap(cand.frame_poly, dilate_px=FREE_GAP_PX)
        if inter == 0:
            return cand
    return None


def _slide(rng: np.random.Generator, anchor: _Placed, radius: float, target: float, make, occ: _Occupancy,
           W: int, Hh: int) -> _Placed | None:
    """Move away from `anchor` along a random direction until the overlap with everything already
    placed drops to `target` (0 = just touching). Binary search on the distance."""
    for _ in range(6):
        a = rng.uniform(0, 2 * math.pi)
        u = np.array([math.cos(a), math.sin(a)])
        lo, hi = 0.0, anchor.radius + radius + 4.0
        cand_hi = make(anchor.centre + u * hi)
        inter, tot = occ.overlap(cand_hi.frame_poly)
        if inter > 0 or tot == 0:
            continue  # something else is in the way
        for _it in range(12):
            mid = (lo + hi) / 2
            inter, tot = occ.overlap(make(anchor.centre + u * mid).frame_poly)
            frac = inter / max(1, tot)
            if (target == 0.0 and inter == 0) or (target > 0.0 and frac <= target):
                hi = mid
            else:
                lo = mid
        d = hi + (rng.uniform(0.0, 1.5) if target == 0.0 else 0.0)
        cand = make(anchor.centre + u * d)
        c = cand.centre
        if -0.3 * radius <= c[0] <= W + 0.3 * radius and -0.3 * radius <= c[1] <= Hh + 0.3 * radius:
            return cand
    return None


def write_scene(frame: np.ndarray, labels: np.ndarray, scene: dict, quality: int, out_dir: Path, scene_id: str) -> None:
    write_jpg(out_dir / f"{scene_id}.jpg", frame, quality)
    write_png(out_dir / f"{scene_id}_mask.png", labels)
    write_json(out_dir / f"{scene_id}.json", scene)
