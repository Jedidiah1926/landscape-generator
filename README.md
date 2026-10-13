# 🏔️ Landscape Generator — 지도 → 등고선 / 3D 지형 AI

지도 이미지를 업로드하면 **AI 비전 모델(Claude / GPT / Gemini 중 선택)** 이 지도를 스캔해 고도를 읽어내고,
자동으로 **등고선 지도**와 **회전 가능한 3D 지형**을 만들어 줍니다.
결과는 STL/OBJ(3D 프린팅·Blender), 16비트 높이맵 PNG로 내보낼 수 있습니다.

> ⚠️ AI 분석에는 아래 세 곳 중 **한 곳의 API 키가 꼭 필요합니다.** 키가 없으면 스캔 버튼이 비활성화됩니다.

| 제공자 | 기본 모델 | 키 발급 | 환경 변수 |
|---|---|---|---|
| Anthropic (Claude) | `claude-opus-5-5` | [console.anthropic.com](https://console.anthropic.com) | `ANTHROPIC_API_KEY` |
| OpenAI (GPT) | `gpt-5.4` | [platform.openai.com](https://platform.openai.com/api-keys) | `OPENAI_API_KEY` |
| Google (Gemini) | `gemini-3.1-pro-preview` | [aistudio.google.com](https://aistudio.google.com/apikey) | `GEMINI_API_KEY` |

모델 이름은 사이드바에서 바꿀 수 있습니다. 이미지 입력과 JSON 스키마 출력을 지원하는 모델이면 됩니다.

## 동작 방식

```
지도 이미지 ──▶ Claude / GPT / Gemini (비전 + JSON 스키마 구조화 출력)
                 │  · 등고선 숫자 / 표고점 / 산 정상 높이 읽기
                 │  · 등고선을 따라 제어점(x, y, 고도) 수십~수백 개 추출
                 │  · 숫자가 없으면 음영·색상·하천으로 상대 고도 추정
                 ▼
         제어점 + 16×16 거친 고도 격자 (JSON)
                 ▼
     Thin-plate RBF 보간 ──▶ 촘촘한 높이맵
                 ▼
   등고선 지도 · 원본 위 등고선 · 3D 지형 · STL/OBJ/PNG 내보내기
```

지원 지도: 지형도, 등산 지도, 음영 기복도, 색으로 고도를 표시한 지도, 위성사진, 손으로 그린 지도 등.

## 설치 및 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

브라우저가 열리면:

1. 왼쪽 사이드바에서 **AI 제공자**(Anthropic / OpenAI / Google)를 고르고 그 API 키를 입력
   — 위 표의 환경 변수를 설정해 두면 자동으로 채워집니다.
2. 지도 이미지 업로드 (선택: "등고선 간격 10m" 같은 추가 정보 입력 → 정확도 향상)
3. **🔍 AI로 지도 스캔** 클릭
4. 탭에서 3D 지형 / 등고선 지도 / 원본 위 등고선 확인, **💾 내보내기**에서 파일 다운로드

API 키는 브라우저 세션에서 요청에만 쓰이고 디스크에 저장되지 않습니다.

## 설정 항목

| 항목 | 설명 |
|---|---|
| AI 제공자 / 모델 | Claude, GPT, Gemini 중 선택, 모델 이름 변경 가능 |
| 분석 정밀도 (effort) | 높을수록 꼼꼼히 분석(느리고 비용 증가). Gemini에서는 무시 |
| 높이맵 해상도 | 보간 결과 격자 크기 |
| 매끄러움 | 클수록 지형이 부드러움 |
| 이미지 질감 반영 | 음영 기복도·위성사진의 미세한 기복을 원본 밝기에서 추가 |
| 3D 높이 과장 | 평평한 지형을 강조 |
| 등고선 간격 | AI가 읽은 간격이 기본값, 0이면 자동 |

## 프로젝트 구조

```
app.py                  Streamlit 웹 UI
terrain/ai_analyzer.py  프롬프트·스키마, 이미지 → 고도 JSON
terrain/providers.py    Anthropic / OpenAI / Google API 호출과 오류 처리
terrain/heightmap.py    제어점 보간 → 높이맵
terrain/render.py       등고선(matplotlib) · 3D(plotly) 시각화
terrain/export.py       STL / OBJ / 16비트 PNG 내보내기
tests/                  pytest (API는 mock 처리)
```

## 테스트

```bash
pytest
```

## 한계

- AI가 이미지를 보고 **추정**한 고도이므로 측량 수준의 정확도는 아닙니다.
  등고선 숫자가 선명한 지형도일수록 정확합니다.
- 정확한 실제 지형이 필요하면 SRTM 같은 DEM 데이터를 쓰는 것이 좋습니다.
