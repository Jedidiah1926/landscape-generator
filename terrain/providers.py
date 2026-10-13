"""AI 제공자(Anthropic / OpenAI / Google)별 비전 + 구조화 출력 호출.

각 제공자 함수는 (이미지, 프롬프트, JSON 스키마)를 받아 스키마에 맞는 JSON 문자열을 돌려준다.
SDK별 예외는 ProviderError로 통일해 UI가 제공자와 무관하게 처리할 수 있게 한다.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Callable


class ProviderError(RuntimeError):
    """API 호출 실패. kind: auth | rate_limit | connection | refusal | truncated | api"""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class Provider:
    key: str
    label: str
    default_model: str
    env_var: str
    key_placeholder: str
    key_url: str
    call: Callable[..., str]


# ------------------------------------------------------------------ Anthropic
def _call_anthropic(*, api_key, model, system, user_text, media_type, image_b64, schema, effort) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    try:
        # 응답이 길 수 있으므로 스트리밍으로 받아 HTTP 타임아웃을 피한다.
        with client.beta.messages.stream(
            model=model,
            max_tokens=64000,
            system=system,
            thinking={"type": "adaptive"},
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
            # 안전 분류기가 거절하면 서버가 권장 모델로 자동 재시도
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
                    {"type": "text", "text": user_text},
                ],
            }],
        ) as stream:
            response = stream.get_final_message()
    except anthropic.AuthenticationError as exc:
        raise ProviderError("auth", "API 키가 올바르지 않습니다.") from exc
    except anthropic.RateLimitError as exc:
        raise ProviderError("rate_limit", "요청 한도를 초과했습니다. 잠시 후 다시 시도하세요.") from exc
    except anthropic.APIStatusError as exc:
        raise ProviderError("api", f"API 오류 ({exc.status_code}): {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise ProviderError("connection", "Anthropic API에 연결할 수 없습니다.") from exc

    if response.stop_reason == "refusal":
        raise ProviderError("refusal", "모델이 이 이미지 분석을 거절했습니다.")
    if response.stop_reason == "max_tokens":
        raise ProviderError("truncated", "응답이 너무 길어 잘렸습니다.")
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise ProviderError("api", "모델 응답에 결과가 없습니다.")
    return text


# ------------------------------------------------------------------ OpenAI
def _call_openai(*, api_key, model, system, user_text, media_type, image_b64, schema, effort) -> str:
    import openai

    client = openai.OpenAI(api_key=api_key)
    try:
        response = client.responses.create(
            model=model,
            instructions=system,
            reasoning={"effort": "high" if effort == "xhigh" else effort},
            max_output_tokens=64000,
            text={"format": {"type": "json_schema", "name": "terrain_analysis", "schema": schema, "strict": True}},
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": f"data:{media_type};base64,{image_b64}"},
                    {"type": "input_text", "text": user_text},
                ],
            }],
        )
    except openai.AuthenticationError as exc:
        raise ProviderError("auth", "API 키가 올바르지 않습니다.") from exc
    except openai.RateLimitError as exc:
        raise ProviderError("rate_limit", "요청 한도를 초과했거나 크레딧이 부족합니다.") from exc
    except openai.APIStatusError as exc:
        raise ProviderError("api", f"API 오류 ({exc.status_code}): {exc.message}") from exc
    except openai.APIConnectionError as exc:
        raise ProviderError("connection", "OpenAI API에 연결할 수 없습니다.") from exc

    if response.status == "incomplete":
        reason = getattr(response.incomplete_details, "reason", "")
        if reason == "content_filter":
            raise ProviderError("refusal", "모델이 이 이미지 분석을 거절했습니다.")
        raise ProviderError("truncated", "응답이 너무 길어 잘렸습니다.")
    for item in response.output:
        for part in getattr(item, "content", None) or []:
            if getattr(part, "type", "") == "refusal":
                raise ProviderError("refusal", f"모델이 분석을 거절했습니다: {part.refusal}")
    if not response.output_text:
        raise ProviderError("api", "모델 응답에 결과가 없습니다.")
    return response.output_text


# ------------------------------------------------------------------ Google
def _call_google(*, api_key, model, system, user_text, media_type, image_b64, schema, effort) -> str:
    from google import genai
    from google.genai import errors, types

    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=base64.b64decode(image_b64), mime_type=media_type),
                user_text,
            ],
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=64000,
                response_mime_type="application/json",
                response_json_schema=schema,
            ),
        )
    except errors.APIError as exc:
        if exc.code in (401, 403) or "API_KEY_INVALID" in str(exc):
            raise ProviderError("auth", "API 키가 올바르지 않습니다.") from exc
        if exc.code == 429:
            raise ProviderError("rate_limit", "요청 한도를 초과했습니다. 잠시 후 다시 시도하세요.") from exc
        raise ProviderError("api", f"API 오류 ({exc.code}): {exc.message}") from exc
    except OSError as exc:
        raise ProviderError("connection", "Google Gemini API에 연결할 수 없습니다.") from exc

    feedback = getattr(response, "prompt_feedback", None)
    if feedback is not None and getattr(feedback, "block_reason", None):
        raise ProviderError("refusal", f"모델이 이 이미지 분석을 거절했습니다 ({feedback.block_reason}).")
    candidate = (response.candidates or [None])[0]
    finish = str(getattr(candidate, "finish_reason", "") or "")
    if "MAX_TOKENS" in finish:
        raise ProviderError("truncated", "응답이 너무 길어 잘렸습니다.")
    if "SAFETY" in finish or "PROHIBITED" in finish:
        raise ProviderError("refusal", "모델이 이 이미지 분석을 거절했습니다.")
    if not response.text:
        raise ProviderError("api", "모델 응답에 결과가 없습니다.")
    return response.text


PROVIDERS: dict[str, Provider] = {
    p.key: p
    for p in (
        Provider("anthropic", "Anthropic (Claude)", "claude-opus-5-5", "ANTHROPIC_API_KEY",
                 "sk-ant-...", "https://console.anthropic.com", _call_anthropic),
        Provider("openai", "OpenAI (GPT)", "gpt-5.4", "OPENAI_API_KEY",
                 "sk-...", "https://platform.openai.com/api-keys", _call_openai),
        Provider("google", "Google (Gemini)", "gemini-3.1-pro-preview", "GEMINI_API_KEY",
                 "AIza...", "https://aistudio.google.com/apikey", _call_google),
    )
}
