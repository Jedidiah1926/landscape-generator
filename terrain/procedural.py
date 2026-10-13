"""절차적 지형 생성: AI가 만든 높이맵(밑그림) 위에 노이즈 디테일과 침식을 더한다.

구성 요소 (numpy만 사용):
- Perlin 그래디언트 노이즈 기반 fBm / ridged multifractal (산 능선)
- 도메인 왜곡(domain warping)으로 자연스럽게 휘어진 지형
- 수력 침식(D8 유량 누적 + stream power 법칙)으로 하천 계곡 형성
- 열 침식(talus 각도 초과 경사의 붕괴)으로 경사면 정리
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_SQRT2 = np.sqrt(2.0)


@dataclass(frozen=True)
class ProceduralSettings:
    seed: int = 0
    detail: float = 0.08        # 디테일 진폭 (밑그림 고도 범위 대비 비율)
    feature_size: float = 0.25  # 가장 큰 노이즈 특징의 크기 (지도 긴 변 대비 비율)
    octaves: int = 6
    ridged: float = 0.4         # 0 = 둥근 언덕(fBm), 1 = 날카로운 능선(ridged)
    warp: float = 0.3           # 도메인 왜곡 강도
    height_mask: float = 0.6    # 높은 곳일수록 거칠게 (0 = 어디나 같은 강도)
    hydraulic: int = 20         # 수력 침식 반복 횟수
    erosion_strength: float = 0.5
    thermal: int = 10           # 열 침식 반복 횟수
    talus: float = 0.6          # 열 침식 임계 경사 (높이/셀, 정규화 단위 × 해상도)


PRESETS: dict[str, ProceduralSettings] = {
    "자연스러운 언덕": ProceduralSettings(detail=0.1, feature_size=0.3, ridged=0.1, warp=0.25,
                                     hydraulic=15, erosion_strength=0.4, thermal=8),
    "험준한 산맥": ProceduralSettings(detail=0.15, feature_size=0.2, ridged=0.85, warp=0.35,
                                  height_mask=0.8, hydraulic=25, erosion_strength=0.5, thermal=6),
    "침식된 계곡": ProceduralSettings(detail=0.1, feature_size=0.25, ridged=0.4, warp=0.3,
                                  hydraulic=50, erosion_strength=1.0, thermal=10),
    "부드러운 사구": ProceduralSettings(detail=0.08, feature_size=0.08, ridged=0.0, warp=0.8,
                                   height_mask=0.0, hydraulic=0, thermal=25, talus=0.3),
}


# ------------------------------------------------------------------ 노이즈
class GradientNoise:
    """시드 고정 2D Perlin 노이즈. 임의의 실수 좌표 배열을 한 번에 평가한다."""

    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        self.perm = np.tile(rng.permutation(256), 2)
        angles = rng.uniform(0, 2 * np.pi, 256)
        self.grads = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        tx, ty = x - x0, y - y0
        xi, yi = x0 & 255, y0 & 255

        def corner(dx: int, dy: int) -> np.ndarray:
            g = self.grads[self.perm[self.perm[(xi + dx) & 255] + ((yi + dy) & 255)]]
            return g[..., 0] * (tx - dx) + g[..., 1] * (ty - dy)

        u = tx * tx * tx * (tx * (tx * 6 - 15) + 10)
        v = ty * ty * ty * (ty * (ty * 6 - 15) + 10)
        top = corner(0, 0) + u * (corner(1, 0) - corner(0, 0))
        bottom = corner(0, 1) + u * (corner(1, 1) - corner(0, 1))
        return (top + v * (bottom - top)) * _SQRT2  # 대략 [-1, 1]


def fbm(noise: GradientNoise, x, y, octaves: int, ridged: float = 0.0,
        lacunarity: float = 2.0, gain: float = 0.5) -> np.ndarray:
    """fBm과 ridged multifractal을 ridged 비율로 섞은 값. 대략 [-1, 1]."""
    smooth = np.zeros_like(x)
    ridge = np.zeros_like(x)
    amp, freq, total = 1.0, 1.0, 0.0
    weight = np.ones_like(x)
    for i in range(octaves):
        # 옥타브마다 좌표를 어긋나게 해 격자 무늬가 겹치지 않게 한다
        n = noise(x * freq + 17.3 * i, y * freq - 31.7 * i)
        smooth += amp * n
        r = (1.0 - np.abs(n)) ** 2
        ridge += amp * r * weight
        weight = np.clip(r * 1.5, 0, 1)
        total += amp
        amp *= gain
        freq *= lacunarity
    smooth /= total
    ridge = ridge / total * 2.0 - 1.0
    return (1 - ridged) * smooth + ridged * ridge


def noise_field(shape: tuple[int, int], s: ProceduralSettings) -> np.ndarray:
    """도메인 왜곡을 적용한 디테일 노이즈 필드. 평균 0, 대략 [-1, 1]."""
    h, w = shape
    noise = GradientNoise(s.seed)
    edge = max(h, w)
    base_freq = 1.0 / max(s.feature_size, 1e-3)
    y, x = np.mgrid[0:h, 0:w] / edge * base_freq
    if s.warp > 0:
        wx = fbm(noise, x + 5.2, y + 1.3, 4)
        wy = fbm(noise, x - 8.1, y + 2.8, 4)
        x = x + s.warp * wx
        y = y + s.warp * wy
    field = fbm(noise, x, y, s.octaves, s.ridged)
    return field - field.mean()


# ------------------------------------------------------------------ 침식
_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _receivers(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """각 셀의 가장 가파른 하강 방향 이웃(D8)의 평탄 인덱스와 경사."""
    h, w = z.shape
    padded = np.pad(z, 1, mode="edge")
    best_slope = np.zeros_like(z)
    best_idx = np.arange(h * w).reshape(h, w)
    rows, cols = np.mgrid[0:h, 0:w]
    for dr, dc in _OFFSETS:
        neigh = padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        slope = (z - neigh) / np.hypot(dr, dc)
        nr = np.clip(rows + dr, 0, h - 1)
        nc = np.clip(cols + dc, 0, w - 1)
        better = slope > best_slope
        best_slope = np.where(better, slope, best_slope)
        best_idx = np.where(better, nr * w + nc, best_idx)
    return best_idx.ravel(), best_slope.ravel()


def flow_accumulation(z: np.ndarray, receivers: np.ndarray) -> np.ndarray:
    """높은 곳부터 물을 아래로 흘려 각 셀의 상류 면적(셀 수)을 구한다."""
    area = np.ones(z.size)
    order = np.argsort(z.ravel())[::-1]
    rec = receivers
    for i in order.tolist():
        r = rec[i]
        if r != i:
            area[r] += area[i]
    return area


def hydraulic_erosion(z: np.ndarray, iterations: int, strength: float = 0.5) -> np.ndarray:
    """Stream power 법칙 (E = K·A^0.5·S). 정규화된 높이(0~1)를 입력으로 받는다."""
    if iterations <= 0 or strength <= 0:
        return z
    z = z.copy()
    h, w = z.shape
    edge = max(h, w)
    k = 0.06 * strength
    for _ in range(iterations):
        rec, slope = _receivers(z)
        area = flow_accumulation(z, rec) / edge  # 해상도와 무관한 유역 크기
        flat = z.ravel()
        cut = k * np.sqrt(area) * slope * edge / 50
        # 하류 셀보다 낮게 파이지 않도록 제한 (구덩이 생성 방지)
        new = np.maximum(flat - cut, np.minimum(flat, flat[rec]))
        z = new.reshape(h, w)
    return z


def thermal_erosion(z: np.ndarray, iterations: int, talus: float = 0.6, rate: float = 0.25) -> np.ndarray:
    """임계 경사를 넘는 부분을 아래 이웃으로 무너뜨린다. 정규화된 높이 입력."""
    if iterations <= 0:
        return z
    z = z.copy()
    h, w = z.shape
    limit = talus / max(h, w)
    for _ in range(iterations):
        padded = np.pad(z, 1, mode="edge")
        delta = np.zeros_like(z)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neigh = padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
            excess = np.maximum(z - neigh - limit, 0) * rate / 4
            delta -= excess
            # 받는 쪽: 이웃 셀 입장에서 반대 방향에서 들어오는 양
            delta += np.pad(excess, 1)[1 - dr:1 - dr + h, 1 - dc:1 - dc + w]
        z = z + delta
    return z


# ------------------------------------------------------------------ 합성
def apply_procedural(base: np.ndarray, s: ProceduralSettings) -> np.ndarray:
    """밑그림 높이맵(고도 단위)에 절차적 디테일과 침식을 더해 같은 단위로 돌려준다."""
    lo, hi = float(base.min()), float(base.max())
    span = hi - lo or 1.0
    norm = (base - lo) / span

    if s.detail > 0:
        mask = (1 - s.height_mask) + s.height_mask * norm
        norm = norm + s.detail * mask * noise_field(base.shape, s)
    norm = hydraulic_erosion(norm, s.hydraulic, s.erosion_strength)
    norm = thermal_erosion(norm, s.thermal, s.talus)
    return lo + norm * span
