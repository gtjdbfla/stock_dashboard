"""FnGuide(Company Guide)에서 재무·밸류에이션 값을 뽑는다.

wcomp.fnguide.com은 화면을 JS로 그리지만, 값은 HTML 안에 `snpFinancial : {...}` 형태의
JSON으로 통째로 박혀 있다. 그래서 표를 긁는 대신 그 JSON을 잘라 쓴다.
(SVD_Main.asp 같은 옛 경로는 1,829바이트짜리 차단 페이지만 돌려준다. cmp_cd 파라미터가
 붙은 루트 경로만 동작한다.)

streamlit을 import하지 않는다. 대시보드와 수집기가 같이 쓴다.
"""
import datetime as dt
import json
import re

import pandas as pd
import requests

URL = "https://wcomp.fnguide.com/"
_UA = {"User-Agent": "Mozilla/5.0", "Referer": URL}

# 재무 표에 담을 항목과 단위. fnguide 원본 이름 그대로 골라 쓴다.
ROWS = [
    ("매출액", "억원"),
    ("영업이익", "억원"),
    ("영업이익률", "%"),
    ("당기순이익", "억원"),
    ("순이익률(지배)", "%"),
    ("ROE", "%"),
    ("EPS", "원"),
    ("BPS", "원"),
    ("PER", "배"),
    ("PBR", "배"),
    ("부채비율", "%"),
    ("현금배당수익률", "%"),
]


def to_number(text: object) -> float | None:
    if text is None:
        return None
    cleaned = re.sub(r"[^0-9.\-]", "", str(text))
    if cleaned in ("", "-", "."):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def fetch_page(ticker: str) -> str:
    r = requests.get(URL, params={"cmp_cd": "A" + ticker}, headers=_UA, timeout=15)
    r.raise_for_status()
    return r.text


def embedded_json(html: str, name: str) -> dict | None:
    """`name : { ... }` 로 박힌 JSON을 괄호 균형으로 잘라낸다."""
    m = re.search(re.escape(name) + r"\s*:\s*\{", html)
    if not m:
        return None
    start = html.index("{", m.start())
    depth = 0
    for i in range(start, len(html)):
        if html[i] == "{":
            depth += 1
        elif html[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(html[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def financial_frames(html: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """snpFinancial을 연간/분기 두 표로 정리한다.

    header의 YYMM이 각 열의 기간이고, EP_CHK == 'E'면 추정치다. 열 순서만 보고
    연도를 짐작하면 안 된다(연간 4열 + 분기 4열이 한 덩어리로 들어온다).
    """
    js = embedded_json(html, "snpFinancial")
    if not js or not js.get("header"):
        return pd.DataFrame(), pd.DataFrame()

    cols = []
    for h in js["header"]:
        yymm = (h.get("YYMM") or "").strip()
        if not yymm:
            continue
        cols.append({
            "cd": h.get("CD"),
            "기간": yymm + ("(E)" if (h.get("EP_CHK") or "").strip() == "E" else ""),
        })
    half = len(cols) // 2
    by_name = {(row.get("NAME") or "").strip(): row for row in (js.get("data") or [])}

    def build(colset) -> pd.DataFrame:
        out = {}
        for label, unit in ROWS:
            row = by_name.get(label)
            if row is None:
                continue
            out[f"{label} ({unit})"] = [to_number(row.get(c["cd"])) for c in colset]
        if not out:
            return pd.DataFrame()
        return pd.DataFrame(out, index=[c["기간"] for c in colset]).T

    return build(cols[:half]), build(cols[half:])


def target_price_history(html: str) -> pd.DataFrame:
    """컨센서스 목표주가 주간 추이(약 1년)."""
    js = embedded_json(html, "snpTargetChart")
    rows = (js or {}).get("data") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([{
        "일자": pd.to_datetime(r["TRD_DT"], format="%Y/%m/%d", errors="coerce"),
        "목표주가": to_number(r.get("TRGT_PRC")),
        "투자의견": to_number(r.get("DEG")),
        "주가": to_number(r.get("CLS_PRC")),
    } for r in rows if r.get("TRD_DT")])
    return df.dropna(subset=["일자"]).sort_values("일자").reset_index(drop=True)


def short_balance(html: str) -> pd.DataFrame:
    """차입공매도 비중 주간 추이(약 1년)."""
    js = embedded_json(html, "snpShortChart")
    rows = (js or {}).get("data") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([{
        "일자": pd.to_datetime(r["TRD_DT"], format="%Y/%m/%d", errors="coerce"),
        "차입공매도비중": to_number(r.get("VAL")),
        "수정주가": to_number(r.get("ADJ_PRC")),
    } for r in rows if r.get("TRD_DT")])
    return df.dropna(subset=["일자"]).sort_values("일자").reset_index(drop=True)
