"""LLM 호출 한 곳.

대시보드(app.py)와 수집기(collector.py)가 같은 모델·같은 상한을 쓰게 하려고 뺐다.
streamlit을 import하지 않으므로 수집기에서도 그대로 돈다.

제공자는 환경변수로 고른다(LLM_PROVIDER). 키가 있는 것만 쓸 수 있다.
  gemini    (기본) GEMINI_API_KEY   — 무료 등급, 모델별 하루 20회
  anthropic        ANTHROPIC_API_KEY — 별도 유료 API. Claude Code 구독과는 다른 계정/과금이다
  openai           OPENAI_API_KEY
"""
import os

PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()

# ── Gemini ────────────────────────────────────────────────────────────────────
# 같은 프롬프트로 네 모델을 재본 실측값은 app.py의 GEMINI_MODEL 주석에 있다.
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
# gemini-flash-latest는 목록에서 뺐다. 같은 프롬프트에서 10개 섹션 중 2개만 쓰고 187자에서 잘렸다.
# 잘린 답이 멀쩡한 척 나오는 것이 실패보다 나쁘다.
FALLBACK_MODELS = [
    m.strip() for m in os.environ.get(
        "GEMINI_FALLBACK_MODELS",
        "gemini-3.5-flash-lite,gemini-flash-lite-latest"
    ).split(",") if m.strip()
]
# 폴백으로 내려가기 전에 기본 모델을 이만큼 더 시도한다(한 번 지연했다고 품질을 포기하지 않게).
PRIMARY_RETRIES = int(os.environ.get("GEMINI_PRIMARY_RETRIES", "1"))
# 이보다 짧은 답은 잘린 것으로 보고 다음 모델로 넘어간다(정상은 1,500~3,300자).
MIN_USABLE = int(os.environ.get("LLM_MIN_USABLE", "600"))
# app.py와 같은 이유로 high를 쓴다(그쪽 GEMINI_THINKING_LEVEL 주석에 실측값이 있다).
THINKING_LEVEL = os.environ.get("GEMINI_THINKING_LEVEL", "high").strip()
# 말문을 못 떼고 붙잡고 있는 모델을 버리는 상한. 스트리밍이라 '조각 사이 간격' 상한이 된다.
# high는 말문을 떼기까지 오래 걸린다. 실측 21~50초로 그날그날 편차가 크다(app.py 주석 참고).
# 넉넉히 줘야 붐비는 날 정상 응답을 '지연'으로 오해해 버리지 않는다.
STALL_SEC = float(os.environ.get("GEMINI_STALL_SEC", "120"))

# ── 다른 제공자 ───────────────────────────────────────────────────────────────
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5")
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "4096"))


def available_providers() -> dict[str, bool]:
    """어떤 제공자를 쓸 수 있는지. 화면에서 안내할 때 쓴다."""
    return {
        "gemini": bool(os.environ.get("GEMINI_API_KEY")),
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
    }


def _call_gemini(prompt: str) -> tuple[str, str]:
    from google import genai

    client = genai.Client()
    cfg = ({"generation_config": {"thinking_level": THINKING_LEVEL}}
           if THINKING_LEVEL else {})
    tried: list[str] = []
    last_exc: Exception | None = None
    order = ([MODEL] * (1 + PRIMARY_RETRIES)
             + [m for m in FALLBACK_MODELS if m != MODEL])
    best = ""
    best_model = ""
    for model in order:
        tried.append(model)
        for kwargs in (cfg, {}):
            try:
                out = []
                for event in client.interactions.create(
                        model=model, input=prompt, stream=True,
                        timeout=STALL_SEC, **kwargs):
                    delta = getattr(event, "delta", None)
                    piece = getattr(delta, "text", None) if delta is not None else None
                    if piece:
                        out.append(piece)
                text = "".join(out)
                if len(text.strip()) >= MIN_USABLE:
                    return text, model
                if len(text.strip()) > len(best):
                    best, best_model = text.strip(), model
                break                       # 빈 답·잘린 답이면 다음 모델로
            except Exception as exc:
                last_exc = exc
                if kwargs and "thinking_level" in str(exc):
                    continue                # 이 옵션을 못 받는 모델이면 빼고 한 번 더
                break                       # 한도·지연 등은 다음 모델로
    # 전부 짧게 나왔더라도 그중 가장 긴 답은 돌려준다. 빈 화면보다는 낫다.
    if best:
        return best, best_model
    raise RuntimeError(
        f"사용 가능한 모델을 찾지 못했습니다. 시도한 모델: {', '.join(tried)}. "
        f"마지막 오류: {type(last_exc).__name__ if last_exc else '빈 응답'}")


def _call_anthropic(prompt: str) -> tuple[str, str]:
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=ANTHROPIC_MODEL, max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}])
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return text, ANTHROPIC_MODEL


def _call_openai(prompt: str) -> tuple[str, str]:
    from openai import OpenAI

    client = OpenAI()
    r = client.responses.create(model=OPENAI_MODEL, input=prompt,
                                max_output_tokens=MAX_TOKENS)
    return (r.output_text or ""), OPENAI_MODEL


def call(prompt: str) -> tuple[str, str]:
    """설정된 제공자로 부른다. 반환: (본문, 실제로 답한 모델명)."""
    if PROVIDER == "anthropic":
        return _call_anthropic(prompt)
    if PROVIDER == "openai":
        return _call_openai(prompt)
    return _call_gemini(prompt)
