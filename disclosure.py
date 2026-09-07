"""전자공시 목록·본문과 그 AI 요약.

대시보드(app.py)와 수집기(collector.py)가 같이 쓴다. 그래서 streamlit을 import하지 않는다.

공시는 뉴스보다 빠르고 숫자가 확정적이다. 자기주식 취득·소각, 시설투자, 조회공시 답변이
주가를 움직인 원인인 경우가 많은데 제목만으로는 규모를 알 수 없다. 본문까지 받아서 넘긴다.

app.py에도 fetch_disclosures()가 있지만 그건 AI 프롬프트에 끼워 넣을 마크다운 한 덩어리를
만드는 함수다. 여기는 화면에 표로 뿌리고 따로 요약까지 만드는 쪽이라 목적이 다르다.
출처(네이버가 중계하는 KOSCOM 공시)는 같으므로 값이 어긋날 일은 없다.
"""
import datetime as dt
import json
import os
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

import llm

KST = dt.timezone(dt.timedelta(hours=9))
LIST_URL = "https://m.stock.naver.com/api/stock/{code}/disclosure"
_UA = {"User-Agent": "Mozilla/5.0", "Referer": "https://m.stock.naver.com/"}

DIGEST_FILE = os.environ.get("DISCLOSURE_DIGEST_FILE", "data/disclosure_digest.json")
BODY_CACHE_FILE = os.environ.get("DISCLOSURE_BODY_FILE", "data/disclosure_bodies.json")

LIST_COUNT = 40           # 화면 표에 뿌릴 공시 수
DIGEST_COUNT = 15         # 요약에 넣을 최근 공시 수
BODY_COUNT = 10           # 본문까지 읽을 최근 공시 수
BODY_CHARS = 900          # 공시 하나에서 가져올 본문 글자 수
# 이보다 짧으면 본문을 못 받은 것으로 보고 캐시에 남기지 않는다(다음에 다시 받는다).
# 공시 본문은 짧은 것도 있어서 리포트(300)보다 낮게 잡는다.
BODY_MIN_CHARS = 80


def fetch_list(ticker: str, count: int = LIST_COUNT) -> pd.DataFrame:
    """공시 목록. 열: 일시 · 제목 · 공시ID · 출처."""
    try:
        r = requests.get(LIST_URL.format(code=ticker), headers=_UA,
                         params={"pageSize": count}, timeout=10)
        r.raise_for_status()
        items = r.json() or []
    except Exception:
        return pd.DataFrame(columns=["일시", "제목", "공시ID", "출처"])
    if not isinstance(items, list):
        return pd.DataFrame(columns=["일시", "제목", "공시ID", "출처"])
    rows = []
    for d in items:
        title = str(d.get("title") or "").strip()
        # 모든 제목이 '에스케이하이닉스(주) '로 시작해서 표에서 자리만 먹는다.
        title = re.sub(r"^[가-힣A-Za-z().\s]{2,20}\(주\)\s*", "", title) or title
        rows.append({
            "일시": str(d.get("datetime") or "").replace("T", " ")[:16],
            "제목": title,
            "공시ID": d.get("disclosureId"),
            "출처": d.get("author") or "",
        })
    return pd.DataFrame(rows)


def _load_bodies() -> dict:
    try:
        with open(BODY_CACHE_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save_bodies(d: dict) -> None:
    try:
        os.makedirs(os.path.dirname(BODY_CACHE_FILE) or ".", exist_ok=True)
        if len(d) > 120:                      # 무한정 쌓이지 않게 최근 것만
            d = dict(list(d.items())[-120:])
        with open(BODY_CACHE_FILE, "w", encoding="utf-8") as fh:
            json.dump(d, fh, ensure_ascii=False)
    except Exception:
        pass


def fetch_body(ticker: str, disclosure_id) -> str:
    """공시 본문 텍스트. 실패하면 빈 문자열."""
    if not disclosure_id:
        return ""
    try:
        r = requests.get(f"{LIST_URL.format(code=ticker)}/{disclosure_id}",
                         headers=_UA, timeout=10)
        r.raise_for_status()
        html = ((r.json() or {}).get("disclosure") or {}).get("contents") or ""
    except Exception:
        return ""
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def collect_bodies(ticker: str, df: pd.DataFrame, count: int = BODY_COUNT) -> dict:
    """최근 공시의 본문을 모은다. 한 번 받은 건 다시 받지 않는다(공시는 내용이 안 바뀐다)."""
    if df.empty or "공시ID" not in df.columns:
        return {}
    cache = _load_bodies()
    changed = False
    out: dict[str, str] = {}
    for did in df.head(count)["공시ID"]:
        key = str(did)
        if not did:
            continue
        # 캐시에 '있어도' 껍데기면 다시 받는다. 없을 때만 받으면, 한 번 실패해 짧게 저장된
        # 값이 영영 그대로 남는다(analyst_digest에서 실제로 그 일이 있었다).
        if len(cache.get(key, "")) < BODY_MIN_CHARS:
            body = fetch_body(ticker, did)
            if len(body) >= BODY_MIN_CHARS:
                cache[key] = body
                changed = True
            else:
                cache.pop(key, None)          # 실패는 캐시에 안 남긴다. 다음에 다시 받는다
                continue
        out[key] = cache[key]
    if changed:
        _save_bodies(cache)
    return out


def fingerprint(df: pd.DataFrame) -> str:
    """새 공시가 떴는지 판단할 지문."""
    if df.empty:
        return ""
    head = df.head(DIGEST_COUNT)
    return "|".join(f"{r['일시']}~{r['제목']}" for _, r in head.iterrows())


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


def _prompt(stock_label: str, df: pd.DataFrame, bodies: dict) -> str:
    rows, n_body = [], 0
    for _, r in df.head(DIGEST_COUNT).iterrows():
        line = f"- {r['일시']} {r['제목']}"
        body = bodies.get(str(r.get("공시ID")))
        if body:
            n_body += 1
            line += "\n  본문: " + body[:BODY_CHARS]
        rows.append(line)
    listing = "\n".join(rows)
    today = dt.datetime.now(KST).strftime("%Y-%m-%d")
    return f"""오늘은 {today}입니다. 다음은 {stock_label}의 최근 전자공시입니다.
{n_body}건은 본문까지 함께 넣었고(`본문:`), 나머지는 제목만 있습니다.

{listing}

아래 형식 그대로, 한국어로 작성해줘.

## 한 줄 요약
최근 공시 흐름을 한 문장으로. 가장 무거운 공시 하나를 짚어라.

## 주목할 공시
중요한 것 3~5개를 골라 각각 한 줄. 날짜와 **본문의 숫자**(금액·주식수·기간·비율)를 반드시 넣어라.
본문이 없는 건은 제목에서 읽히는 것만 쓰고 숫자를 지어내지 마라.

## 주주환원 관련
자기주식 취득·처분·소각, 배당 공시가 있으면 규모와 기간을 숫자로 정리해라. 없으면 "해당 공시 없음".

## 확인이 필요한 것
조회공시 요구·해명(미확정)·풍문 관련 공시가 있으면 무엇에 대한 것인지, 회사가 뭐라고 했는지.
'미확정'은 확정된 사실이 아니라는 점을 분명히 써라. 없으면 "해당 공시 없음".

## 읽는 법
이 공시들을 주가와 연결해 어떻게 봐야 하는지 2~3문장. 단정하지 말고 무엇을 더 봐야 하는지 써라.

작성 규칙:
- 위 목록에 없는 공시를 지어내지 마라. 모르면 "공시에 없음".
- **본문이 있는 건은 숫자를 그대로 인용해라.** 어림잡지 마라.
- 매수/매도 추천은 하지 마라. 공시가 무슨 내용인지와 무엇을 봐야 하는지만 써라.
- '2단계 가격제한폭 확대요건 도달' 같은 시장조치 공시는 회사가 낸 것이 아니라
  거래소의 기계적 공시다. 회사의 의사결정으로 읽지 마라.
- 전체 1,500자 안쪽. 서론·맺음말 금지.
- LaTeX 수식 기호를 쓰지 마라. 화살표는 →, 곱셈은 x로 그냥 써라."""


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
    """새 공시가 있을 때만 요약을 다시 만든다. 반환: (저장된 payload, 새로 만들었는지)."""
    if df is None:
        df = fetch_list(ticker)
    if df.empty:
        return load(), False
    fp = fingerprint(df)
    saved = load()
    if not force and saved.get("fingerprint") == fp and saved.get("text"):
        return saved, False

    bodies = collect_bodies(ticker, df)
    text, model = llm.call(_prompt(stock_label, df, bodies))
    payload = {
        "text": _strip_latex(text),
        "fingerprint": fp,
        "date": dt.datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
        "count": int(min(len(df), DIGEST_COUNT)),
        "body_count": len(bodies),
        "model": model,
        "latest": str(df.iloc[0]["일시"]) if len(df) else "",
    }
    save(payload)
    return payload, True


# 공시는 장 시작 전(06~07시)과 장 마감 후(16~18시)에 몰리지만 장중에도 뜬다.
# 하루 한 번만 보면 아침에 뜬 자사주 공시를 저녁까지 놓친다.
CHECK_FROM = dt.time(7, 0)
CHECK_TO = dt.time(20, 0)
CHECK_EVERY_MIN = 120
_last_check_at: dt.datetime | None = None


def tick(now: dt.datetime, ticker: str, stock_label: str, log=print) -> None:
    """수집기에서 주기적으로 부른다. 지문이 같으면 AI를 부르지 않는다."""
    global _last_check_at
    if now.weekday() >= 5 or not (CHECK_FROM <= now.time() <= CHECK_TO):
        return
    if _last_check_at and (now - _last_check_at).total_seconds() < CHECK_EVERY_MIN * 60:
        return
    _last_check_at = now
    try:
        payload, made = refresh(ticker, stock_label)
    except Exception as exc:
        log(f"[공시] 요약 실패: {type(exc).__name__}: {exc}")
        return
    if made:
        log(f"[공시] 요약 갱신 (공시 {payload.get('count')}건, "
            f"본문 {payload.get('body_count')}건, 모델 {payload.get('model')})")
