"""AI가 추출한 고도 정보(제어점 + 거친 격자)를 촘촘한 높이맵으로 보간한다."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter
from scipy.interpolate import RBFInterpolator

from .ai_analyzer import TerrainAnalysis


def _grid_points(grid: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    """거친 격자를 (x, y) 셀 중심 좌표와 값으로 변환. 행 길이가 달라도 처리한다."""
    xy, vals = [], []
    rows = len(grid)
    for r, row in enumerate(grid):
        cols = len(row)
        for c, v in enumerate(row):
            xy.append(((c + 0.5) / cols, (r + 0.5) / rows))
            vals.append(v)
    return np.asarray(xy, dtype=float).reshape(-1, 2), np.asarray(vals, dtype=float)


def build_heightmap(
    analysis: TerrainAnalysis,
    width: int,
    height: int,
    smoothing: float = 1.0,
    image: Image.Image | None = None,
    detail: float = 0.0,
) -> np.ndarray:
    """(height, width) 크기의 높이맵(고도 단위 그대로)을 만든다.

    제어점은 정확히 지나도록(작은 smoothing), 격자 추정값은 느슨하게(큰 smoothing)
    맞추는 thin-plate RBF 보간을 사용한다. `detail` > 0이면 원본 이미지 밝기의
    고주파 성분을 더해 음영 기복도 같은 지도의 질감을 살린다.
    """
    cp = analysis.control_points
    cp_xy = np.array([(p.x, p.y) for p in cp], dtype=float).reshape(-1, 2)
    cp_v = np.array([p.elevation for p in cp], dtype=float)
    g_xy, g_v = _grid_points(analysis.elevation_grid)

    xy = np.vstack([cp_xy, g_xy])
    vals = np.concatenate([cp_v, g_v])
    if len(vals) == 0:
        raise ValueError("보간할 고도 데이터가 없습니다.")

    # 같은 위치의 중복 점은 RBF 행렬을 특이하게 만들므로 평균으로 합친다.
    keys = np.round(xy, 4)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    vals = np.bincount(inverse, weights=vals) / np.bincount(inverse)
    is_cp = np.zeros(len(uniq), dtype=bool)
    is_cp[inverse[: len(cp_v)]] = True
    xy = uniq

    span = float(np.ptp(vals)) or 1.0
    norm = (vals - vals.min()) / span
    per_point_smoothing = np.where(is_cp, 1e-4, 1e-2) * smoothing

    gx, gy = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    query = np.column_stack([gx.ravel(), gy.ravel()])

    if len(norm) < 3:
        z = np.full(len(query), norm.mean())
    else:
        rbf = RBFInterpolator(
            xy, norm,
            kernel="thin_plate_spline",
            smoothing=per_point_smoothing,
            neighbors=min(len(norm), 80),
        )
        z = rbf(query)
    z = z.reshape(height, width)

    if detail > 0 and image is not None:
        z += detail * image_detail(image, width, height)

    # 보간 과정의 오버슈트를 AI가 보고한 범위(또는 데이터 범위)로 제한
    lo = min(analysis.min_elevation, vals.min()) if analysis.max_elevation > analysis.min_elevation else vals.min()
    hi = max(analysis.max_elevation, vals.max()) if analysis.max_elevation > analysis.min_elevation else vals.max()
    out = z * span + vals.min()
    margin = 0.05 * (hi - lo)
    return np.clip(out, lo - margin, hi + margin)


def image_detail(image: Image.Image, width: int, height: int) -> np.ndarray:
    """이미지 밝기의 고주파 성분을 [-0.5, 0.5] 정도 범위로 정규화해 반환."""
    gray = image.convert("L").resize((width, height), Image.BILINEAR)
    base = np.asarray(gray, dtype=float) / 255.0
    blurred = np.asarray(gray.filter(ImageFilter.GaussianBlur(radius=max(width, height) / 40)), dtype=float) / 255.0
    hp = base - blurred
    scale = np.abs(hp).max() or 1.0
    return 0.5 * hp / scale


def resolution_for(image: Image.Image, max_edge: int) -> tuple[int, int]:
    """이미지 비율을 유지하는 (width, height) 해상도."""
    scale = max_edge / max(image.size)
    return max(8, round(image.width * scale)), max(8, round(image.height * scale))
