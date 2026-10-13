"""지도 → 등고선 / 3D 지형 생성기 (Streamlit 앱). Anthropic / OpenAI / Google 지원.

실행:  streamlit run app.py
"""

from __future__ import annotations

import json
import os

import streamlit as st
from PIL import Image

from terrain.ai_analyzer import AnalysisError, TerrainAnalysis, analyze_map
from terrain.providers import PROVIDERS, ProviderError
from terrain.export import heightmap_png, obj_bytes, stl_bytes
from terrain.heightmap import build_heightmap, resolution_for
from terrain.render import contour_figure, contour_levels, surface_figure

st.set_page_config(page_title="지도 → 3D 지형 생성기", page_icon="🏔️", layout="wide")

# ---------------------------------------------------------------- 사이드바
with st.sidebar:
    st.header("🤖 AI 제공자")
    provider_key = st.radio(
        "사용할 AI", list(PROVIDERS), format_func=lambda k: PROVIDERS[k].label,
        help="세 곳 중 API 키가 있는 곳을 고르세요.",
    )
    provider = PROVIDERS[provider_key]
    api_key = st.text_input(
        f"{provider.label} API 키",
        type="password",
        value=os.environ.get(provider.env_var, ""),
        placeholder=provider.key_placeholder,
        key=f"api_key_{provider_key}",
        help=f"{provider.key_url} 에서 발급. 키는 이 세션에서만 사용되고 저장되지 않습니다.",
    )
    model = st.text_input(
        "모델", value=provider.default_model, key=f"model_{provider_key}",
        help="비전(이미지 입력)과 JSON 스키마 출력을 지원하는 모델이어야 합니다.",
    )
    effort = st.select_slider(
        "분석 정밀도 (effort)", options=["low", "medium", "high", "xhigh"], value="high",
        help="높을수록 정확하지만 느리고 비용이 큽니다. (Gemini에서는 무시됩니다)",
    )

    st.header("⚙️ 지형 설정")
    resolution = st.slider("높이맵 해상도 (긴 변 픽셀)", 64, 400, 200, step=8)
    smoothing = st.slider("매끄러움", 0.1, 20.0, 1.0, help="클수록 지형이 부드러워집니다.")
    detail = st.slider("이미지 질감 반영", 0.0, 0.3, 0.0, step=0.01,
                       help="음영 기복도/위성사진에서 원본 밝기의 미세 기복을 더합니다.")
    exaggeration = st.slider("3D 높이 과장", 0.2, 5.0, 1.0, step=0.1)

st.title("🏔️ 지도 → 등고선 / 3D 지형 생성기")
st.caption("지도 이미지를 올리면 AI(Claude / GPT / Gemini)가 스캔해 고도를 읽고, 등고선 지도와 3D 지형을 만들어 줍니다.")

uploaded = st.file_uploader("지도 이미지 업로드 (PNG, JPG, WEBP)", type=["png", "jpg", "jpeg", "webp"])
hint = st.text_area(
    "지도에 대한 추가 정보 (선택)",
    placeholder="예: 북한산 일대 1:25,000 지형도, 등고선 간격 10m / 정상 높이 836m",
    height=70,
)

if uploaded is None:
    st.info("먼저 지도 이미지를 업로드하세요. 지형도·등산지도·음영기복도·손그림 지도 모두 가능합니다.")
    st.stop()

image = Image.open(uploaded)
image.load()
file_key = f"{uploaded.name}:{uploaded.size}"
if st.session_state.get("file_key") != file_key:
    st.session_state.pop("analysis", None)
    st.session_state["file_key"] = file_key

col_img, col_btn = st.columns([3, 1])
with col_img:
    st.image(image, caption="업로드한 지도", use_container_width=True)
with col_btn:
    run = st.button("🔍 AI로 지도 스캔", type="primary", use_container_width=True, disabled=not api_key)
    if not api_key:
        st.warning("사이드바에 API 키를 입력해야 스캔할 수 있습니다.")

if run:
    with st.spinner(f"{provider.label}가 지도를 분석하는 중... (최대 몇 분 걸릴 수 있습니다)"):
        try:
            st.session_state["analysis"] = analyze_map(
                image, api_key, hint, effort, provider=provider_key, model=model,
            ).to_dict()
        except (ProviderError, AnalysisError) as exc:
            st.error(str(exc))

if "analysis" not in st.session_state:
    st.stop()

analysis = TerrainAnalysis.from_dict(st.session_state["analysis"])

# ---------------------------------------------------------------- 분석 결과
st.subheader("📋 AI 분석 결과")
m1, m2, m3, m4 = st.columns(4)
m1.metric("지도 종류", analysis.map_type or "-")
m2.metric("최저 고도", f"{analysis.min_elevation:g} {analysis.elevation_unit}")
m3.metric("최고 고도", f"{analysis.max_elevation:g} {analysis.elevation_unit}")
m4.metric("추출한 제어점", len(analysis.control_points))
st.write(analysis.summary)
if analysis.notes:
    st.caption(f"참고: {analysis.notes}")

width, height = resolution_for(image, resolution)
z = build_heightmap(analysis, width, height, smoothing=smoothing, image=image, detail=detail)

levels = contour_levels(z, analysis.contour_interval)
default_interval = float(levels[1] - levels[0]) if len(levels) > 1 else 10.0
interval = st.number_input(
    f"등고선 간격 ({analysis.elevation_unit})", min_value=0.0, value=default_interval,
    help="0이면 자동으로 정합니다.",
)

tab3d, tab_contour, tab_overlay, tab_export = st.tabs(["🗻 3D 지형", "🗺️ 등고선 지도", "🔎 원본 위 등고선", "💾 내보내기"])

with tab3d:
    textured = st.toggle("원본 지도를 지형 표면에 입히기", value=False)
    st.plotly_chart(
        surface_figure(z, analysis, interval, exaggeration, texture=image if textured else None),
        use_container_width=True,
    )

contour_png = contour_figure(z, analysis, interval)
with tab_contour:
    st.image(contour_png, use_container_width=True)

with tab_overlay:
    show_points = st.toggle("AI가 찍은 제어점 표시", value=True)
    st.image(contour_figure(z, analysis, interval, background=image, show_points=show_points),
             use_container_width=True)

with tab_export:
    st.write("3D 프린팅·Blender·게임 엔진에서 사용할 수 있는 파일로 내보냅니다.")
    c1, c2, c3 = st.columns(3)
    size_mm = c1.number_input("모델 가로 크기 (mm)", 20.0, 500.0, 100.0)
    height_mm = c2.number_input("지형 높이 (mm)", 1.0, 200.0, 20.0 * exaggeration)
    base_mm = c3.number_input("받침 두께 (mm)", 0.0, 50.0, 3.0)
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.download_button("STL", stl_bytes(z, size_mm, height_mm, base_mm), "terrain.stl", "model/stl")
    d2.download_button("OBJ", obj_bytes(z, size_mm, height_mm, base_mm), "terrain.obj", "model/obj")
    d3.download_button("높이맵 PNG (16bit)", heightmap_png(z), "heightmap.png", "image/png")
    d4.download_button("등고선 PNG", contour_png, "contours.png", "image/png")
    d5.download_button("분석 JSON", json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2),
                       "analysis.json", "application/json")
