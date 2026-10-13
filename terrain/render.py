"""등고선 지도와 3D 지형 시각화."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from PIL import Image  # noqa: E402

from .ai_analyzer import TerrainAnalysis  # noqa: E402


def contour_levels(z: np.ndarray, interval: float, max_levels: int = 60) -> np.ndarray:
    """등고선 간격에 맞는 레벨. 간격이 없거나 너무 촘촘하면 자동으로 조정."""
    lo, hi = float(np.nanmin(z)), float(np.nanmax(z))
    if hi <= lo:
        return np.array([lo])
    if interval <= 0 or (hi - lo) / interval > max_levels:
        interval = _nice_interval((hi - lo) / 15)
    start = np.ceil(lo / interval) * interval
    return np.arange(start, hi + 1e-9, interval)


def _nice_interval(raw: float) -> float:
    exp = 10 ** np.floor(np.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * exp:
            return float(m * exp)
    return float(10 * exp)


def contour_figure(
    z: np.ndarray,
    analysis: TerrainAnalysis,
    interval: float,
    background: Image.Image | None = None,
    show_points: bool = False,
    filled: bool = True,
) -> bytes:
    """등고선 지도를 PNG 바이트로 렌더링."""
    h, w = z.shape
    levels = contour_levels(z, interval)
    fig, ax = plt.subplots(figsize=(10, 10 * h / w), dpi=120)
    extent = (0, w, h, 0)

    if background is not None:
        ax.imshow(background.convert("RGB").resize((w, h)), extent=extent, alpha=0.55)
    elif filled and len(levels) > 1:
        ax.contourf(z, levels=levels, cmap="terrain", extend="both")

    major = levels[::5] if len(levels) > 10 else levels
    ax.contour(z, levels=levels, colors="#7a4a1e", linewidths=0.6)
    cs = ax.contour(z, levels=major, colors="#5a2e0c", linewidths=1.3)
    ax.clabel(cs, fmt=lambda v: f"{v:g}", fontsize=7)

    if show_points:
        pts = analysis.control_points
        ax.scatter([p.x * w for p in pts], [p.y * h for p in pts], s=8, c="red", zorder=5)

    unit = analysis.elevation_unit
    ax.set_title(f"Contours ({levels[1] - levels[0] if len(levels) > 1 else 0:g} {unit} interval)")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def surface_figure(
    z: np.ndarray,
    analysis: TerrainAnalysis,
    interval: float,
    exaggeration: float = 1.0,
    texture: Image.Image | None = None,
) -> go.Figure:
    """Plotly 3D 지형. texture가 있으면 원본 지도를 표면에 입힌다."""
    h, w = z.shape
    levels = contour_levels(z, interval)
    step = float(levels[1] - levels[0]) if len(levels) > 1 else 1.0

    surface_kwargs: dict = {}
    if texture is not None:
        gray = np.asarray(texture.convert("L").resize((w, h)), dtype=float)
        surface_kwargs = {"surfacecolor": gray, "colorscale": "gray", "showscale": False}
    else:
        surface_kwargs = {"colorscale": "earth", "colorbar": {"title": analysis.elevation_unit}}

    fig = go.Figure(go.Surface(
        z=z,
        x=np.arange(w),
        y=np.arange(h),
        contours={"z": {"show": True, "start": float(levels[0]), "end": float(levels[-1]),
                        "size": step, "color": "#3b2410", "width": 1}},
        lighting={"ambient": 0.55, "diffuse": 0.8, "roughness": 0.9, "specular": 0.1},
        hovertemplate="고도 %{z:.1f}<extra></extra>",
        **surface_kwargs,
    ))
    fig.update_layout(
        height=650,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        scene={
            "xaxis": {"visible": False},
            "yaxis": {"visible": False, "autorange": "reversed"},
            "zaxis": {"title": analysis.elevation_unit},
            "aspectmode": "manual",
            "aspectratio": {"x": 1, "y": h / w, "z": 0.35 * exaggeration},
        },
    )
    return fig


def hillshade_png(z: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> bytes:
    """음영 기복도(hillshade) PNG. 밑그림과 결과 비교용."""
    from matplotlib.colors import LightSource, Normalize

    h, w = z.shape
    lo = float(z.min()) if vmin is None else vmin
    hi = float(z.max()) if vmax is None else vmax
    rgb = LightSource(azdeg=315, altdeg=45).shade(
        z, cmap=plt.cm.terrain, norm=Normalize(lo, hi), vert_exag=0.15 * max(h, w) / ((hi - lo) or 1.0),
        blend_mode="overlay",
    )
    buf = io.BytesIO()
    Image.fromarray((rgb[..., :3] * 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()
