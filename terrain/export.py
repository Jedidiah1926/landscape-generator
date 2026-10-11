"""높이맵 내보내기: 16비트 PNG, STL(3D 프린팅), OBJ."""

from __future__ import annotations

import io
import struct

import numpy as np
from PIL import Image


def heightmap_png(z: np.ndarray) -> bytes:
    """0~65535로 정규화한 16비트 그레이스케일 PNG (게임 엔진/Blender용)."""
    lo, hi = float(z.min()), float(z.max())
    norm = (z - lo) / (hi - lo) if hi > lo else np.zeros_like(z)
    img = Image.fromarray((norm * 65535).astype(np.uint16))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _mesh(z: np.ndarray, size_mm: float, height_mm: float, base_mm: float):
    """윗면 + 옆면 + 바닥을 가진 닫힌(watertight) 메시의 정점·삼각형."""
    h, w = z.shape
    lo, hi = float(z.min()), float(z.max())
    norm = (z - lo) / (hi - lo) if hi > lo else np.zeros_like(z)
    cell = size_mm / (max(w, h) - 1)
    xs = np.arange(w) * cell
    ys = (h - 1 - np.arange(h)) * cell  # 이미지 위쪽 = +Y
    X, Y = np.meshgrid(xs, ys)
    top = np.column_stack([X.ravel(), Y.ravel(), (base_mm + norm * height_mm).ravel()])
    bottom = top.copy()
    bottom[:, 2] = 0.0
    verts = np.vstack([top, bottom])
    n = h * w

    def idx(r, c):
        return r * w + c

    tris = []
    for r in range(h - 1):
        for c in range(w - 1):
            a, b, cc, d = idx(r, c), idx(r, c + 1), idx(r + 1, c), idx(r + 1, c + 1)
            tris += [(a, cc, b), (b, cc, d)]          # 윗면 (바깥쪽 = +Z)
            tris += [(a + n, b + n, cc + n), (b + n, d + n, cc + n)]  # 바닥
    # 테두리를 시계 방향 반대로 돌며 옆면 생성
    border = ([idx(0, c) for c in range(w)] + [idx(r, w - 1) for r in range(1, h)]
              + [idx(h - 1, c) for c in range(w - 2, -1, -1)] + [idx(r, 0) for r in range(h - 2, 0, -1)])
    for i in range(len(border)):
        p, q = border[i], border[(i + 1) % len(border)]
        tris += [(p, q, p + n), (q, q + n, p + n)]
    return verts, np.asarray(tris, dtype=np.int64)


def stl_bytes(z: np.ndarray, size_mm: float = 100.0, height_mm: float = 20.0, base_mm: float = 3.0) -> bytes:
    """바이너리 STL."""
    verts, tris = _mesh(z, size_mm, height_mm, base_mm)
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"] = normals
    rec["v"] = np.stack([v0, v1, v2], axis=1)
    header = b"landscape-generator terrain".ljust(80, b" ")
    return header + struct.pack("<I", len(tris)) + rec.tobytes()


def obj_bytes(z: np.ndarray, size_mm: float = 100.0, height_mm: float = 20.0, base_mm: float = 3.0) -> bytes:
    """Wavefront OBJ (Blender, 게임 엔진 등)."""
    verts, tris = _mesh(z, size_mm, height_mm, base_mm)
    lines = ["# landscape-generator terrain"]
    lines += [f"v {x:.4f} {y:.4f} {zz:.4f}" for x, y, zz in verts]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in tris]
    return ("\n".join(lines) + "\n").encode()
