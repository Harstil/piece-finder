"""
Golden fixtures for the engine primitives: OpenCV is the oracle, the TypeScript engine is the student.

The phone engine re-implements a handful of classic OpenCV operations in pure TypeScript (contours,
morphology, homographies, warps, Lab, area resize ...). This script builds small deterministic inputs,
runs the real opencv-python on them and writes the inputs plus OpenCV's outputs as JSON next to itself.
The Vitest suites under src/engine/image and src/engine/geom load these files and compare within
documented tolerances, so a behavioural drift from OpenCV fails a test instead of silently skewing
accuracy.

Run from anywhere:  python src/engine/__golden__/make_goldens.py
Everything is seeded; re-running produces byte-identical JSON (checked in, < 1 MB total).
Masks are stored as lists of '0'/'1' row strings, images as flat row-major number lists.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.RandomState(20260923)


def r5(values) -> list:
    """Floats rounded to 5 decimals: plenty for the test tolerances, keeps the JSON small."""
    return [round(float(v), 5) for v in np.asarray(values, dtype=np.float64).ravel()]


def ints(values) -> list:
    return [int(v) for v in np.asarray(values).ravel()]


def mask_rows(mask: np.ndarray) -> list:
    return [''.join('1' if v else '0' for v in row) for row in mask]


def write(name: str, payload: dict) -> None:
    path = os.path.join(OUT_DIR, name)
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(payload, fh, separators=(',', ':'))
        fh.write('\n')
    print(f'wrote {name}: {os.path.getsize(path)} bytes')


# ----------------------------------------------------------------------------------------------
# Masks: shapes with holes, nested blobs, touching blobs, 1-pixel lines, border contact, noise.
# ----------------------------------------------------------------------------------------------

def mask_blobs() -> np.ndarray:
    m = np.zeros((32, 40), np.uint8)
    cv2.circle(m, (8, 8), 5, 1, -1)
    cv2.rectangle(m, (18, 2), (25, 7), 1, -1)
    cv2.ellipse(m, (32, 22), (6, 4), 30, 0, 360, 1, -1)
    m[12:16, 2:6] = 1          # two squares touching only diagonally -> one 8-connected blob
    m[16:20, 6:10] = 1
    m[24, 3:15] = 1            # horizontal 1-pixel line
    for i in range(7):         # diagonal 1-pixel line
        m[20 + i, 20 + i] = 1
    m[28, 30] = 1              # isolated pixel
    m[0:5, 35:40] = 1          # touches the top-right image corner
    m[10:14, 30:31] = 1        # vertical 1-pixel line
    return m


def mask_holes() -> np.ndarray:
    m = np.zeros((32, 40), np.uint8)
    cv2.circle(m, (10, 10), 8, 1, -1)
    cv2.circle(m, (10, 10), 4, 0, -1)       # ring
    m[9:12, 9:12] = 1                        # blob nested inside the ring's hole
    m[4:28, 22:37] = 1
    m[8:14, 26:33] = 0                       # rectangular hole
    m[18:24, 26:29] = 0                      # second hole
    m[20:22, 29:31] = 0
    m[22:30, 2:14] = 1                       # U shape
    m[22:27, 5:11] = 0
    return m


def mask_random() -> np.ndarray:
    return (RNG.rand(28, 36) < 0.42).astype(np.uint8)


def mask_piece() -> np.ndarray:
    """A jigsaw-like piece: rotated square core, round tab out, round blank in, touching nothing."""
    m = np.zeros((48, 48), np.uint8)
    core = cv2.boxPoints(((23.5, 24.0), (26, 24), 17.0))
    cv2.fillPoly(m, [np.round(core).astype(np.int32)], 1)
    cv2.circle(m, (38, 20), 5, 1, -1)      # tab on the right
    cv2.circle(m, (21, 12), 4, 0, -1)      # blank at the top
    return m


def mask_border() -> np.ndarray:
    m = np.ones((20, 24), np.uint8)
    m[5:9, 6:12] = 0
    m[12:15, 15:20] = 0
    m[16, 2] = 0
    return m


MASKS = {
    'blobs': mask_blobs(),
    'holes': mask_holes(),
    'random': mask_random(),
    'piece': mask_piece(),
    'border': mask_border(),
}


def contour_list(contours) -> list:
    return [ints(c.reshape(-1, 2)) for c in contours]


def outer_contours_all_depths(mask: np.ndarray) -> list:
    """Outer borders of every component, nested ones included (RETR_TREE, even depths)."""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    h = hierarchy[0]
    out = []
    for i, c in enumerate(contours):
        depth, p = 0, h[i][3]
        while p >= 0:
            depth += 1
            p = h[p][3]
        if depth % 2 == 0:
            out.append(c)
    return contour_list(out)


def build_masks_and_contours() -> None:
    write('masks.json', {name: {'width': m.shape[1], 'height': m.shape[0], 'rows': mask_rows(m)}
                         for name, m in MASKS.items()})
    payload = {}
    for name, m in MASKS.items():
        external, _ = cv2.findContours(m.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        payload[name] = {
            'external': contour_list(external),
            'allOuter': outer_contours_all_depths(m.copy()),
            # OpenCV's own orientation, recorded so the test can assert the documented flip.
            'externalOrientedArea': [round(float(cv2.contourArea(c, oriented=True)), 3) for c in external],
        }
    write('contours.json', payload)

    approx = []
    for name in ('piece', 'blobs', 'holes', 'random'):
        external, _ = cv2.findContours(MASKS[name].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in external:
            if len(c) < 3:
                continue
            for eps in (0.5, 1.0, 2.0, 3.5):
                a = cv2.approxPolyDP(c, eps, True)
                approx.append({'mask': name, 'epsilon': eps, 'input': ints(c.reshape(-1, 2)),
                               'output': ints(a.reshape(-1, 2))})
    write('approx.json', {'cases': approx})


def build_components_and_distance() -> None:
    comps = {}
    dist = {}
    for name, m in MASKS.items():
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
        comps[name] = {
            'count': int(n - 1),
            'labels': ints(labels),
            'stats': [ints(s) for s in stats[1:]],          # x, y, w, h, area
            'centroids': [r5(c) for c in centroids[1:]],
        }
        if (m == 0).any():
            d = cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            dist[name] = r5(d)
    write('components.json', comps)
    write('distance.json', dist)


def build_morphology() -> None:
    kernels = [
        ('rect', 3, 3), ('rect', 5, 3), ('rect', 4, 4), ('rect', 1, 6),
        ('ellipse', 7, 7), ('ellipse', 6, 4), ('ellipse', 5, 9),
    ]
    cases = []
    for name in ('blobs', 'holes', 'random', 'border'):
        m = MASKS[name]
        for shape, kw, kh in kernels:
            k = cv2.getStructuringElement(cv2.MORPH_RECT if shape == 'rect' else cv2.MORPH_ELLIPSE, (kw, kh))
            cases.append({
                'mask': name, 'shape': shape, 'width': kw, 'height': kh,
                'kernel': mask_rows(k),
                'erode': mask_rows(cv2.erode(m, k)),
                'dilate': mask_rows(cv2.dilate(m, k)),
                'open': mask_rows(cv2.morphologyEx(m, cv2.MORPH_OPEN, k)),
                'close': mask_rows(cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)),
            })
    write('morph.json', {'cases': cases})


# ----------------------------------------------------------------------------------------------
# Images: smooth test pictures so interpolation differences stay far below one grey level.
# ----------------------------------------------------------------------------------------------

def smooth_gray(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = 110 + 60 * np.sin(xx / 5.3 + 0.4) * np.cos(yy / 7.1 - 0.2) + 1.6 * xx - 0.9 * yy
    for _ in range(4):
        cx, cy, s, a = RNG.rand() * w, RNG.rand() * h, 3 + RNG.rand() * 5, RNG.rand() * 60 - 30
        img += a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
    return np.clip(img, 0, 255)


def build_lab() -> None:
    cube = [[r, g, b] for r in (0, 255) for g in (0, 255) for b in (0, 255)]
    ramp = [[v, v, v] for v in range(0, 256, 15)]
    dark = [[v, (v * 3) % 7, v // 2] for v in range(0, 12)]      # exercises the linear branch of f(t)
    rand = RNG.randint(0, 256, size=(200, 3)).tolist()
    rgb = np.array(cube + ramp + dark + rand, np.uint8)
    lab = cv2.cvtColor(rgb.reshape(1, -1, 3).astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab).reshape(-1, 3)
    gray = cv2.cvtColor(rgb.reshape(1, -1, 3), cv2.COLOR_RGB2GRAY).reshape(-1)
    write('lab.json', {'rgb': ints(rgb), 'lab': r5(lab), 'gray8': ints(gray)})


def build_resize_blur_sobel() -> None:
    gray = smooth_gray(29, 37).astype(np.float32)
    gray += RNG.rand(*gray.shape).astype(np.float32) * 8     # a bit of texture
    area = []
    for (w, h) in ((12, 9), (18, 14), (9, 29), (20, 10), (37, 29), (5, 3)):
        area.append({'width': w, 'height': h, 'data': r5(cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA))})
    rgb = np.stack([smooth_gray(30, 40), 255 - smooth_gray(30, 40), smooth_gray(30, 40) * 0.5], axis=2)
    rgb = np.clip(rgb + RNG.rand(30, 40, 3) * 20, 0, 255).astype(np.uint8)
    area_rgb = []
    for (w, h) in ((13, 11), (20, 15), (7, 4)):
        area_rgb.append({'width': w, 'height': h,
                         'data': ints(cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA))})
    linear = []
    for (w, h) in ((50, 40), (20, 15), (37, 29), (11, 30)):
        linear.append({'width': w, 'height': h,
                       'data': r5(cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR))})
    write('resize.json', {
        'gray': {'width': 37, 'height': 29, 'data': r5(gray)},
        'area': area,
        'rgb': {'width': 40, 'height': 30, 'data': ints(rgb)},
        'areaRgb': area_rgb,
        'linear': linear,
    })

    img = (smooth_gray(24, 32) + RNG.rand(24, 32) * 30).astype(np.float32)
    blur = []
    for sigma in (0.8, 1.0, 2.0, 3.5):
        ksize = int(round(sigma * 4 * 2 + 1)) | 1          # OpenCV's automatic size for float images
        out = cv2.GaussianBlur(img, (ksize, ksize), sigma, borderType=cv2.BORDER_REFLECT_101)
        blur.append({'sigma': sigma, 'radius': ksize // 2, 'data': r5(out)})
    dx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    write('filters.json', {'image': {'width': 32, 'height': 24, 'data': r5(img)},
                           'blur': blur, 'sobelDx': r5(dx), 'sobelDy': r5(dy)})


def random_convex_quad(cx: float, cy: float, r: float) -> np.ndarray:
    angles = np.sort(RNG.rand(4) * 0.9 + np.arange(4)) * (np.pi / 2) + RNG.rand() * np.pi
    radii = r * (0.7 + 0.3 * RNG.rand(4))
    return np.stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)], axis=1).astype(np.float32)


def build_homography_and_warp() -> None:
    quads = []
    for _ in range(6):
        src = random_convex_quad(200, 150, 120)
        dst = random_convex_quad(64, 64, 60)
        M = cv2.getPerspectiveTransform(src, dst)
        quads.append({'src': r5(src), 'dst': r5(dst), 'H': [float(v) for v in M.ravel()]})

    point_sets = []
    for noise in (0.0, 0.3):
        src = (RNG.rand(20, 2) * [640, 480]).astype(np.float64)
        H = np.array([[0.9, 0.12, 30], [-0.08, 1.05, 12], [0.0004, -0.0002, 1.0]])
        p = np.hstack([src, np.ones((20, 1))]) @ H.T
        dst = p[:, :2] / p[:, 2:3] + RNG.randn(20, 2) * noise
        M, _ = cv2.findHomography(src, dst, 0)
        point_sets.append({'noise': noise, 'src': r5(src), 'dst': r5(dst), 'H': [float(v) for v in M.ravel()]})

    # Affine / similarity least squares oracles (numpy; OpenCV's estimators are RANSAC-only).
    src = RNG.rand(15, 2) * 300
    theta, s = 0.35, 1.3
    R = s * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    dst_sim = src @ R.T + [40, -25] + RNG.randn(15, 2) * 0.5
    A = np.hstack([src, np.ones((15, 1))])
    affine = np.linalg.lstsq(A, dst_sim, rcond=None)[0].T          # 2x3
    mu_s, mu_d = src.mean(0), dst_sim.mean(0)
    sc, dc = src - mu_s, dst_sim - mu_d
    U, S, Vt = np.linalg.svd(dc.T @ sc / 15)
    D = np.diag([1, np.sign(np.linalg.det(U @ Vt))])
    Rot = U @ D @ Vt
    scale = np.trace(np.diag(S) @ D) / (sc ** 2).sum(1).mean()
    t = mu_d - scale * Rot @ mu_s
    similarity = np.hstack([scale * Rot, t[:, None]])
    write('homography.json', {
        'quads': quads, 'pointSets': point_sets,
        'fit': {'src': r5(src), 'dst': r5(dst_sim), 'affine': r5(affine), 'similarity': r5(similarity)},
    })

    gray = smooth_gray(40, 48)
    gray8 = np.round(gray).astype(np.uint8)
    warps = []
    for _ in range(4):
        src_quad = random_convex_quad(24, 20, 17)
        dst_rect = np.array([[-0.5, -0.5], [23.5, -0.5], [23.5, 23.5], [-0.5, 23.5]], np.float32)
        dst_to_src = cv2.getPerspectiveTransform(dst_rect, src_quad).astype(np.float64)
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        out8 = cv2.warpPerspective(gray8, dst_to_src, (24, 24), flags=flags,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        outf = cv2.warpPerspective(gray8.astype(np.float32), dst_to_src, (24, 24), flags=flags,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warps.append({'dstToSrc': [float(v) for v in dst_to_src.ravel()], 'width': 24, 'height': 24,
                      'out8': ints(out8), 'outF': r5(outf)})
    # One warp that deliberately samples outside the source, for the validity mask.
    shift = np.array([[1.3, 0.1, -9.0], [-0.05, 1.2, -7.5], [0.0005, 0.0, 1.0]])
    out8 = cv2.warpPerspective(gray8, shift, (40, 36), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warps.append({'dstToSrc': [float(v) for v in shift.ravel()], 'width': 40, 'height': 36,
                  'out8': ints(out8), 'outF': None})
    write('warp.json', {'image': {'width': 48, 'height': 40, 'data': ints(gray8)}, 'warps': warps})


if __name__ == '__main__':
    build_masks_and_contours()
    build_components_and_distance()
    build_morphology()
    build_lab()
    build_resize_blur_sobel()
    build_homography_and_warp()
    total = sum(os.path.getsize(os.path.join(OUT_DIR, f)) for f in os.listdir(OUT_DIR) if f.endswith('.json'))
    print(f'total {total} bytes')
