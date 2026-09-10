"""애널리스트 리포트 목록과 그 AI 요약.

대시보드(app.py)와 수집기(collector.py)가 같이 쓴다. 그래서 streamlit을 import하지 않는다.

원래는 app.py 안에만 있었는데, 그러면 요약이 '대시보드를 열었을 때'만 갱신된다.
아침에 새 리포트가 올라와도 화면을 열기 전까지는 옛 요약이 그대로 보인다.
24시간 도는 수집기가 아침에 한 번 확인하도록 이쪽으로 옮겼다.
"""
import datetime as dt
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

import analyst_targets
import llm

KST = dt.timezone(dt.timedelta(hours=9))
NAVER_RESEARCH_URL = "https://finance.naver.com/research/company_list.naver"
DIGEST_FILE = os.environ.get("ANALYST_DIGEST_FILE", "data/analyst_digest.json")
DIGEST_COUNT = 12                 # 요약에 넣을 최근 리포트 수

# 리포트는 장 시작 전에 몰리지만 장중·마감 후에도 올라온다. 아침에 한 번만 보면
# 10시에 올라온 리포트가 다음 날 아침까지 요약에 안 들어간다.
# 그래서 평일 08~19시 사이 2시간마다 확인한다. 지문이 같으면 생성하지 않으므로
# 확인을 자주 해도 무료 한도(하루 20회)를 더 쓰지는 않는다.
CHECK_FROM = dt.time(8, 0)
CHECK_TO = dt.time(19, 0)
CHECK_EVERY_MIN = 120

# 목록에는 PDF 주소가 없어 건별로 상세를 열어야 한다. 40건을 순차로 열면 4초쯤이라
# 병렬로 받는다(실측 0.6초). 네이버에 부담을 주지 않는 선에서 8개면 충분하다.
DETAIL_WORKERS = 8


def fetch_reports(ticker: str, count: int = 40) -> pd.DataFrame:
    """네이버가 모아주는 증권사 리포트 목록. PDF 주소는 상세에서 붙인다.

    2026-09-10 네이버가 PC 리서치 페이지를 표 없는 새 SPA로 바꿔서 표 스크래핑이
    끊겼다. 모바일 JSON API로 옮겼고, 예전에 '제목↔PDF 짝 맞추기'로 애먹던 부분은
    상세가 attachUrl을 직접 주므로 사라졌다.
    """
    pages = max(1, -(-count // analyst_targets.PAGE_SIZE))
    items = analyst_targets.list_reports(ticker, pages=pages)[:count]
    if not items:
        return pd.DataFrame(columns=["제목", "증권사", "작성일", "url", "조회수"])

    with ThreadPoolExecutor(max_workers=DETAIL_WORKERS) as pool:
        details = list(pool.map(lambda it: analyst_targets.fetch_detail(it["nid"]), items))

    rows = []
    for item, detail in zip(items, details):
        pdf = str(detail.get("attachUrl") or "")
        rows.append({
            "제목": item["제목"],
            "증권사": item["증권사"],
            "작성일": item["작성일"],
            # fetch_report_body가 .pdf만 받으므로 여기서 걸러 둔다.
            "url": pdf if pdf.lower().endswith(".pdf") else None,
            "조회수": item.get("조회수", ""),
        })
    return pd.DataFrame(rows)


# ── 리포트 본문(PDF) ──────────────────────────────────────────────────────────
# 제목만 읽히면 "40조원은 시작" 같은 홍보성 문구밖에 못 본다. 본문에는 목표주가,
# P/E·P/B, 영업이익 추정치가 그대로 들어 있다(실측: 5페이지에서 6,435자, 1.5초).
# 한 번 받은 본문은 파일에 남겨 다시 내려받지 않는다.
BODY_CACHE_FILE = os.environ.get("REPORT_BODY_FILE", "data/report_bodies.json")
BODY_COUNT = 6            # 본문까지 읽을 최근 리포트 수
BODY_CHARS = 2000         # 리포트당 본문에서 가져올 글자 수
BODY_PAGES = 6            # 앞쪽 몇 페이지만 읽는다(뒤는 면책·표가 대부분)
# 이보다 짧으면 본문을 못 뽑은 것으로 보고 캐시에 남기지 않는다. 표지가 이미지인
# 리포트는 6페이지를 다 읽어도 143자뿐이라 OCR 없이는 방법이 없지만, 그날 잠깐
# 실패한 건(0바이트 응답)은 다음에 다시 받아야 한다.
BODY_MIN_CHARS = 300


def _load_bodies() -> dict:
    try:
        with open(BODY_CACHE_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save_bodies(d: dict) -> None:
    try:
        os.makedirs(os.path.dirname(BODY_CACHE_FILE) or ".", exist_ok=True)
        # 무한정 쌓이지 않게 최근 40건만 남긴다
        if len(d) > 40:
            d = dict(list(d.items())[-40:])
        with open(BODY_CACHE_FILE, "w", encoding="utf-8") as fh:
            json.dump(d, fh, ensure_ascii=False)
    except Exception:
        pass


def fetch_report_body(url: str) -> str:
    """리포트 PDF에서 앞쪽 몇 페이지의 글을 뽑는다. 실패하면 빈 문자열."""
    if not url or not str(url).lower().endswith(".pdf"):
        return ""
    try:
        import io as _io

        from pypdf import PdfReader

        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
        if not r.content:
            return ""              # 0바이트로 오는 날이 있다. 캐시에 남기지 않고 다음에 다시 받는다.
        reader = PdfReader(_io.BytesIO(r.content))
        text = "".join((p.extract_text() or "") for p in reader.pages[:BODY_PAGES])
        # 줄바꿈이 많아 그대로 넣으면 프롬프트만 커진다. 공백을 정리한다.
        return re.sub(r"[ \t]*\n[ \t]*", " ", text).strip()
    except Exception:
        return ""


def collect_bodies(df: pd.DataFrame) -> dict:
    """상위 리포트의 본문을 모은다. 캐시에 있으면 다시 받지 않는다."""
    if "url" not in df.columns:
        return {}
    cache = _load_bodies()
    changed = False
    out = {}
    for _, r in df.head(BODY_COUNT).iterrows():
        url = r.get("url")
        if not url or not isinstance(url, str):
            continue
        # 캐시에 '있어도' 껍데기면 다시 받는다. 예전에 실패한 결과(0바이트로 내려온 날의
        # 64자짜리)가 그대로 굳어 있는 경우가 있어서, 없을 때만 받으면 영영 그 상태다.
        if len(cache.get(url, "")) < BODY_MIN_CHARS:
            body = fetch_report_body(url)
            if len(body) >= BODY_MIN_CHARS:
                cache[url] = body
                changed = True
            else:
                # 이번에도 실패. 표지가 이미지인 리포트는 몇 번을 받아도 마찬가지지만,
                # 그날만의 장애일 수도 있으니 캐시에 남기지 않고 다음에 또 시도한다.
                cache.pop(url, None)
                continue
        out[url] = cache[url]
    if changed:
        _save_bodies(cache)
    return out


def fingerprint(df: pd.DataFrame) -> str:
    """리포트 목록이 바뀌었는지 판단할 지문. 새 리포트가 뜨면 값이 달라진다."""
    if df.empty:
        return ""
    head = df.head(DIGEST_COUNT)
    return "|".join(f"{r['작성일']}~{r['증권사']}~{r['제목']}" for _, r in head.iterrows())


def load() -> dict:
    try:
        with open(DIGEST_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def save(payload: dict) -> None:
    try:
        os.makedirs(os.path.dirname(DIGEST_FILE) or ".", exist_ok=True)
        with open(DIGEST_FILE, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
    except Exception:
        pass


def _prompt(stock_label: str, df: pd.DataFrame, bodies: dict | None = None) -> str:
    bodies = bodies or {}
    has_url = "url" in df.columns
    rows, n_body = [], 0
    for _, r in df.head(DIGEST_COUNT).iterrows():
        line = f"- [{r['증권사']}] {r['제목']} ({r['작성일']})"
        body = bodies.get(r.get("url")) if has_url else None
        if body:
            n_body += 1
            line += "\n  본문: " + body[:BODY_CHARS]
        rows.append(line)
    listing = "\n".join(rows)
    return f"""다음은 {stock_label}에 대해 최근 나온 증권사 리포트입니다.
{n_body}건은 PDF 본문 앞부분까지 함께 넣었고(`본문:`), 나머지는 제목만 있습니다.
본문이 있는 건 본문의 숫자(목표주가·PER·추정 실적 등)를 근거로 쓰고,
본문이 없는 건 제목에서 읽어낼 수 있는 것만 써라. 없는 숫자를 지어내지 마라.

{listing}

아래 형식으로 한국어로 작성해줘.

## 지금 증권가의 시각
제목들을 관통하는 공통된 논지를 2~3문장으로. 어느 쪽으로 기울어 있는지 분명히 써라.

## 반복해서 나오는 주제
가장 자주 등장하는 화두 3개를 뽑아 각각 한 줄로. 어느 증권사가 그 이야기를 했는지 붙여라.

## 시각이 갈리는 지점
증권사마다 다르게 보는 부분이 있으면 짚어라. 없으면 "제목만으로는 뚜렷한 이견이 안 보인다"고 써라.

## 시점 흐름
날짜 순으로 논조가 달라졌는지. 최근 리포트가 이전과 다른 이야기를 하면 그 변화를 짚어라.

작성 규칙:
- 위 목록에 없는 리포트나 숫자를 지어내지 마라.
- **본문이 있는 리포트는 목표주가·투자의견·추정 실적 같은 구체 숫자를 인용해라.**
- 본문이 없는 리포트는 제목만 근거다. 그 리포트의 목표주가를 말하지 마라.
- 매수/매도 추천은 하지 마라. 증권가가 무슨 말을 하는지만 옮겨라.
- 제목은 홍보성으로 붙는 경우가 많다. 단정적으로 읽지 말고 '~라는 시각'처럼 써라.
- LaTeX 수식 기호를 쓰지 마라. 화살표는 →, 곱셈은 x로 그냥 써라."""


# 모델이 가끔 LaTeX을 섞어 쓴다. 화면은 $를 글자로 escape하므로 그대로 두면
# 기호가 날것으로 보인다. 자주 나오는 것만 평문으로 바꾼다.
_LATEX_MAP = {
    "\\rightarrow": "→", "\\to": "→", "\\Rightarrow": "⇒",
    "\\times": "x", "\\approx": "≈", "\\uparrow": "↑", "\\downarrow": "↓",
}


def _strip_latex(text: str) -> str:
    for k, v in _LATEX_MAP.items():
        text = text.replace("$" + k + "$", v).replace(k, v)
    return text


def refresh(ticker: str, stock_label: str, df: pd.DataFrame | None = None,
            force: bool = False) -> tuple[dict, bool]:
    """필요하면 요약을 새로 만든다.

    반환: (저장된 내용, 새로 만들었는지)
    새 리포트가 없으면 다시 만들지 않는다. 내용이 같은데 매일 부르면
    하루 20회뿐인 무료 한도를 그냥 태우게 된다.
    """
    if df is None:
        df = fetch_reports(ticker)
    if df.empty:
        return load(), False
    fp = fingerprint(df)
    saved = load()
    if not force and saved.get("text") and saved.get("fingerprint") == fp:
        return saved, False

    bodies = collect_bodies(df)
    text, model = llm.call(_prompt(stock_label, df, bodies))
    payload = {
        "text": text,
        "note": (None if model == llm.MODEL else f"{model} 모델로 정리했습니다."),
        "fingerprint": fp,
        "date": dt.datetime.now(KST).date().isoformat(),
        "count": int(min(len(df), DIGEST_COUNT)),
    }
    save(payload)
    return payload, True


# 수집기 루프에서 부르는 부분 ────────────────────────────────────────────────
_last_check_at: dt.datetime | None = None


def tick(now: dt.datetime, ticker: str, stock_label: str, log=print) -> None:
    """평일 장중에 2시간마다 확인한다. 조건이 안 맞으면 즉시 돌아간다."""
    global _last_check_at
    if now.weekday() >= 5 or not (CHECK_FROM <= now.time() <= CHECK_TO):
        return
    if _last_check_at and (now - _last_check_at).total_seconds() < CHECK_EVERY_MIN * 60:
        return
    _last_check_at = now
    try:
        payload, made = refresh(ticker, stock_label)
    except Exception as exc:
        log(f"[리포트요약] 실패: {type(exc).__name__}: {exc}")
        return
    if made:
        log(f"[리포트요약] 새 리포트를 반영해 다시 정리함 ({payload.get('count')}건)")
    else:
        log("[리포트요약] 새 리포트 없음 - 그대로 둠")
