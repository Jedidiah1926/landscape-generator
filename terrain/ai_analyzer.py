"""Claude 비전으로 지도 이미지를 스캔해 고도 정보를 구조화된 JSON으로 추출한다."""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field

import anthropic
from PIL import Image

MODEL = "claude-opus-5-5"
MAX_IMAGE_EDGE = 1568  # 비전 입력 권장 최대 변 길이

SYSTEM_PROMPT = """\
You are a cartography and terrain-analysis expert. You receive one map image
(topographic map, contour map, hiking map, shaded-relief map, satellite image,
or a hand-drawn sketch map) and must estimate the terrain elevation across it.

Coordinate system: x and y are fractions of the image size in [0, 1],
x = 0 at the left edge, y = 0 at the TOP edge.

Do this carefully:
1. Identify the map type and read any elevation labels, spot heights, peak
   names with heights, contour labels and the contour interval.
2. Trace the contour lines: place many control_points ALONG each contour line
   you can see (every few percent of the image), each with that line's
   elevation. Add peaks, saddles, valley floors, rivers, lakes and coastlines
   (water/coast are usually the lowest points).
3. If the map has no explicit numbers, infer plausible relative elevations from
   shading, colour tints (hypsometric colours), drainage patterns, rivers and
   symbols, and say so in `notes`.
4. Fill `elevation_grid` with a coarse row-major estimate of the whole map
   (about 16 rows x 16 columns; row 0 = top of image). It must agree with the
   control points.
Provide at least 40 control points when the map shows terrain detail; more
(up to ~300) is better for dense contour maps.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "map_type": {"type": "string"},
        "summary": {"type": "string"},
        "elevation_unit": {"type": "string", "enum": ["m", "ft", "relative"]},
        "contour_interval": {"type": "number"},
        "min_elevation": {"type": "number"},
        "max_elevation": {"type": "number"},
        "control_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "elevation": {"type": "number"},
                    "kind": {
                        "type": "string",
                        "enum": ["contour", "peak", "saddle", "valley", "water", "spot_height", "estimate"],
                    },
                    "label": {"type": "string"},
                },
                "required": ["x", "y", "elevation", "kind", "label"],
                "additionalProperties": False,
            },
        },
        "elevation_grid": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        },
        "notes": {"type": "string"},
    },
    "required": [
        "map_type", "summary", "elevation_unit", "contour_interval",
        "min_elevation", "max_elevation", "control_points", "elevation_grid", "notes",
    ],
    "additionalProperties": False,
}


class AnalysisError(RuntimeError):
    """AI 분석 실패(거절, 잘림, 파싱 오류 등)."""


@dataclass
class ControlPoint:
    x: float
    y: float
    elevation: float
    kind: str = "estimate"
    label: str = ""


@dataclass
class TerrainAnalysis:
    map_type: str
    summary: str
    elevation_unit: str
    contour_interval: float
    min_elevation: float
    max_elevation: float
    control_points: list[ControlPoint] = field(default_factory=list)
    elevation_grid: list[list[float]] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "TerrainAnalysis":
        points = [
            ControlPoint(
                x=min(max(float(p["x"]), 0.0), 1.0),
                y=min(max(float(p["y"]), 0.0), 1.0),
                elevation=float(p["elevation"]),
                kind=p.get("kind", "estimate"),
                label=p.get("label", ""),
            )
            for p in data.get("control_points", [])
        ]
        grid = [[float(v) for v in row] for row in data.get("elevation_grid", []) if row]
        return cls(
            map_type=data.get("map_type", ""),
            summary=data.get("summary", ""),
            elevation_unit=data.get("elevation_unit", "m"),
            contour_interval=float(data.get("contour_interval") or 0),
            min_elevation=float(data.get("min_elevation", 0)),
            max_elevation=float(data.get("max_elevation", 0)),
            control_points=points,
            elevation_grid=grid,
            notes=data.get("notes", ""),
        )

    def to_dict(self) -> dict:
        return {
            "map_type": self.map_type,
            "summary": self.summary,
            "elevation_unit": self.elevation_unit,
            "contour_interval": self.contour_interval,
            "min_elevation": self.min_elevation,
            "max_elevation": self.max_elevation,
            "control_points": [vars(p) for p in self.control_points],
            "elevation_grid": self.elevation_grid,
            "notes": self.notes,
        }


def prepare_image(image: Image.Image) -> tuple[str, str]:
    """이미지를 RGB PNG로 축소·인코딩해 (media_type, base64) 반환."""
    img = image.convert("RGB")
    scale = MAX_IMAGE_EDGE / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "image/png", base64.standard_b64encode(buf.getvalue()).decode("ascii")


def analyze_map(image: Image.Image, api_key: str, hint: str = "", effort: str = "high") -> TerrainAnalysis:
    """지도 이미지를 Claude에 보내 고도 분석 결과를 받는다. API 키는 필수."""
    if not api_key or not api_key.strip():
        raise ValueError("Anthropic API 키가 필요합니다.")

    media_type, data = prepare_image(image)
    user_text = "Analyze this map and estimate its terrain elevation."
    if hint.strip():
        user_text += f"\n\nExtra information from the user about this map:\n{hint.strip()}"

    client = anthropic.Anthropic(api_key=api_key.strip())
    # 응답이 길 수 있으므로 스트리밍으로 받아 HTTP 타임아웃을 피한다.
    with client.beta.messages.stream(
        model=MODEL,
        max_tokens=64000,
        system=SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
        # 안전 분류기가 거절하면 서버가 권장 모델로 자동 재시도
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}},
                {"type": "text", "text": user_text},
            ],
        }],
    ) as stream:
        response = stream.get_final_message()

    if response.stop_reason == "refusal":
        raise AnalysisError("모델이 이 이미지 분석을 거절했습니다.")
    if response.stop_reason == "max_tokens":
        raise AnalysisError("응답이 너무 길어 잘렸습니다. 더 작은 영역의 지도로 다시 시도하세요.")

    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise AnalysisError("모델 응답에 결과가 없습니다.")
    try:
        result = TerrainAnalysis.from_dict(json.loads(text))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise AnalysisError(f"모델 응답을 해석하지 못했습니다: {exc}") from exc
    if not result.control_points and not result.elevation_grid:
        raise AnalysisError("지도에서 고도 정보를 찾지 못했습니다.")
    return result
