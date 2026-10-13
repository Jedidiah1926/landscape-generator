import json
import struct
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from terrain import ai_analyzer
from terrain.ai_analyzer import AnalysisError, TerrainAnalysis, analyze_map
from terrain.export import heightmap_png, obj_bytes, stl_bytes
from terrain.heightmap import build_heightmap
from terrain.render import contour_figure, contour_levels, surface_figure


def hill_analysis() -> TerrainAnalysis:
    """중앙에 500m 봉우리가 있는 원뿔형 언덕."""
    points = [{"x": 0.5, "y": 0.5, "elevation": 500, "kind": "peak", "label": "정상"}]
    for r, elev in ((0.15, 400), (0.3, 250), (0.45, 100)):
        for t in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            points.append({"x": 0.5 + r * np.cos(t), "y": 0.5 + r * np.sin(t),
                           "elevation": elev, "kind": "contour", "label": ""})
    grid = [[max(0.0, 500 - 900 * np.hypot((c + .5) / 8 - .5, (r + .5) / 8 - .5)) for c in range(8)]
            for r in range(8)]
    return TerrainAnalysis.from_dict({
        "map_type": "test", "summary": "", "elevation_unit": "m", "contour_interval": 50,
        "min_elevation": 0, "max_elevation": 500, "control_points": points,
        "elevation_grid": grid, "notes": "",
    })


def test_heightmap_peak_in_center():
    z = build_heightmap(hill_analysis(), 60, 40)
    assert z.shape == (40, 60)
    peak = np.unravel_index(np.argmax(z), z.shape)
    assert abs(peak[0] - 20) <= 3 and abs(peak[1] - 30) <= 3
    assert z.max() == pytest.approx(500, abs=40)
    assert z[0, 0] < 200


def test_heightmap_handles_duplicates_and_ragged_grid():
    a = hill_analysis()
    a.control_points += a.control_points[:5]
    a.elevation_grid = [[1, 2, 3], [4, 5]]
    assert np.isfinite(build_heightmap(a, 20, 20)).all()


def test_contour_levels_auto_when_too_dense():
    z = np.linspace(0, 1000, 100).reshape(10, 10)
    assert len(contour_levels(z, 1)) <= 60
    assert np.allclose(np.diff(contour_levels(z, 100)), 100)


def test_renders_and_exports():
    a = hill_analysis()
    z = build_heightmap(a, 30, 30)
    img = Image.new("RGB", (90, 90), "white")
    assert contour_figure(z, a, 50).startswith(b"\x89PNG")
    assert contour_figure(z, a, 50, background=img, show_points=True).startswith(b"\x89PNG")
    surface_figure(z, a, 50, texture=img)
    assert heightmap_png(z).startswith(b"\x89PNG")

    stl = stl_bytes(z)
    (count,) = struct.unpack("<I", stl[80:84])
    assert len(stl) == 84 + 50 * count
    # 윗면/바닥 2*(29*29)*2 + 옆면 테두리 116*2
    assert count == 4 * 29 * 29 + 2 * 116
    assert obj_bytes(z).count(b"\nf ") == count


def test_analyze_map_requires_key():
    with pytest.raises(ValueError):
        analyze_map(Image.new("RGB", (10, 10)), "  ")


def _fake_client(stop_reason="end_turn", text=None):
    payload = text if text is not None else json.dumps(hill_analysis().to_dict())
    message = SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=payload)])
    stream = mock.MagicMock()
    stream.__enter__.return_value.get_final_message.return_value = message
    client = mock.MagicMock()
    client.beta.messages.stream.return_value = stream
    return client


def test_analyze_map_parses_response():
    client = _fake_client()
    with mock.patch.object(ai_analyzer.anthropic, "Anthropic", return_value=client) as ctor:
        result = analyze_map(Image.new("RGB", (3000, 1500)), "sk-test", hint="10m interval")
    ctor.assert_called_once_with(api_key="sk-test")
    kwargs = client.beta.messages.stream.call_args.kwargs
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    assert "10m interval" in kwargs["messages"][0]["content"][1]["text"]
    assert result.max_elevation == 500 and len(result.control_points) == 49


@pytest.mark.parametrize("stop_reason,text", [("refusal", None), ("max_tokens", None), ("end_turn", "not json")])
def test_analyze_map_errors(stop_reason, text):
    with mock.patch.object(ai_analyzer.anthropic, "Anthropic", return_value=_fake_client(stop_reason, text)):
        with pytest.raises(AnalysisError):
            analyze_map(Image.new("RGB", (10, 10)), "sk-test")


def test_prepare_image_downscales():
    media, data = ai_analyzer.prepare_image(Image.new("RGBA", (4000, 1000)))
    import base64, io
    img = Image.open(io.BytesIO(base64.b64decode(data)))
    assert media == "image/png" and max(img.size) == ai_analyzer.MAX_IMAGE_EDGE
