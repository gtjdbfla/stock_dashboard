"""재무 데이터 AI 요약.

원래 대시보드 렌더 안에서 만들었는데 두 가지가 문제였다.
  1) 생성이 화면을 붙잡는다. Streamlit은 보이는 탭을 매 리런마다 다시 그리므로,
     생성이 필요한 순간에 페이지 전체가 20~60초 멈춘다.
  2) 실패하면 저장할 게 없어서 다음 리런에 또 시도한다. 실제로 모델 4개가 모두 막힌 날
     한 번 시도에 100초가 걸렸고, 그 상태에서는 클릭할 때마다 100초씩 멈춘다.
그래서 생성은 수집기가 정해진 시각에 맡고, 화면은 저장된 결과를 읽기만 한다.
"""
import datetime as dt
import json
import os

import pandas as pd

import fnguide
import llm

KST = dt.timezone(dt.timedelta(hours=9))
DIGEST_FILE = os.environ.get("FINANCIAL_DIGEST_FILE", "data/financial_digest.json")

# 재무는 분기마다, 목표주가·공매도는 주 단위로 바뀐다. 아침에 한 번 확인하면 충분하다.
MORNING_FROM = dt.time(8, 20)
MORNING_TO = dt.time(9, 40)


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


def fingerprint(annual: pd.DataFrame, quarter: pd.DataFrame,
                tp: pd.DataFrame, sb: pd.DataFrame) -> str:
    """재무·밸류에이션 값이 하나라도 바뀌면 달라지는 지문."""
    parts = []
    for df in (annual, quarter):
        if df is not None and not df.empty:
            parts.append("|".join(f"{i}:{','.join(str(v) for v in df.loc[i])}" for i in df.index))
    for df, col in ((tp, "목표주가"), (sb, "차입공매도비중")):
        if df is not None and not df.empty and col in df.columns:
            parts.append(f"{col}:{df.iloc[-1][col]}@{df.iloc[-1]['일자']:%Y-%m-%d}")
    return "~".join(parts)


def _table_to_text(df: pd.DataFrame, title: str) -> str:
    if df is None or df.empty:
        return f"[{title}] (수집 실패)"
    head = " | ".join(str(c) for c in df.columns)
    rows = [f"  {i}: " + " | ".join("-" if pd.isna(v) else f"{v:,.2f}" for v in df.loc[i])
            for i in df.index]
    return f"[{title}]\n  항목 | {head}\n" + "\n".join(rows)


def _prompt(stock_label: str, annual: pd.DataFrame, quarter: pd.DataFrame,
            tp: pd.DataFrame, sb: pd.DataFrame) -> str:
    extra = []
    if tp is not None and not tp.empty:
        first, last = tp.iloc[0], tp.iloc[-1]
        extra.append(f"[컨센서스 목표주가] {first['일자']:%Y-%m-%d} {first['목표주가']:,.0f}원 -> "
                     f"{last['일자']:%Y-%m-%d} {last['목표주가']:,.0f}원 "
                     f"({last['목표주가'] / first['목표주가'] - 1:+.1%})")
        if len(tp) >= 5:
            m1 = tp.iloc[-5]
            extra.append(f"  최근 4주 변화: {m1['목표주가']:,.0f}원 -> {last['목표주가']:,.0f}원 "
                         f"({last['목표주가'] / m1['목표주가'] - 1:+.1%})")
    if sb is not None and not sb.empty:
        extra.append(f"[차입공매도 비중] 최근 {sb.iloc[-1]['일자']:%Y-%m-%d} "
                     f"{sb.iloc[-1]['차입공매도비중']:.2f}% "
                     f"(1년 평균 {sb['차입공매도비중'].mean():.2f}%)")

    return f"""다음은 {stock_label}의 재무 데이터입니다(FnGuide, IFRS 연결).
매출·이익은 억원, EPS·BPS는 원, 나머지는 %/배입니다.

{_table_to_text(annual, "연간")}

{_table_to_text(quarter, "분기")}

{chr(10).join(extra)}

아래 형식으로 한국어로 짧게 정리해줘. 각 항목 2~3문장.

## 실적이 어디까지 왔나
매출·이익·이익률이 어떻게 변해왔는지. 방향이 바뀐 지점이 있으면 그 해를 짚어라.

## 지금 밸류에이션
PER·PBR이 과거 대비 어느 위치인지. 이익이 늘어서 오른 건지 주가가 더 빨리 올라서 오른 건지 구분해라.

## 재무 체력
ROE·부채비율·배당으로 버는 힘과 안정성을 짚어라.

## 눈여겨볼 지점
숫자에서 읽히는 주의할 점 2~3개. 추정치가 실제와 크게 벌어져 있으면 그것도 짚어라.

작성 규칙:
- **(E)가 붙은 열은 확정 실적이 아니라 증권사 추정치 평균이다.** 확정치와 섞어서 추세를 말하지 마라.
  추정치를 인용할 때는 반드시 '추정'임을 밝혀라.
- 위 표에 없는 숫자를 지어내지 마라. 값이 '-'면 없는 것이다.
- 매수/매도 추천을 하지 마라. 숫자가 무슨 뜻인지만 설명해라.
- 주가 전망을 하지 마라.
- 순이익률·ROE·BPS는 지배주주 기준이다."""


def refresh(ticker: str, stock_label: str, force: bool = False) -> tuple[dict, bool]:
    """필요하면 요약을 새로 만든다. 반환: (저장된 내용, 새로 만들었는지)."""
    html = fnguide.fetch_page(ticker)
    annual, quarter = fnguide.financial_frames(html)
    tp = fnguide.target_price_history(html)
    sb = fnguide.short_balance(html)
    if annual.empty and quarter.empty:
        return load(), False

    fp = fingerprint(annual, quarter, tp, sb)
    saved = load()
    if not force and saved.get("text") and saved.get("fingerprint") == fp:
        return saved, False

    text, model = llm.call(_prompt(stock_label, annual, quarter, tp, sb))
    payload = {
        "text": text,
        "note": (None if model == llm.MODEL else f"{model} 모델로 정리했습니다."),
        "fingerprint": fp,
        "date": dt.datetime.now(KST).date().isoformat(),
    }
    save(payload)
    return payload, True


# 수집기 루프에서 부르는 부분 ────────────────────────────────────────────────
_last_check_date: dt.date | None = None


def tick(now: dt.datetime, ticker: str, stock_label: str, log=print) -> None:
    """아침 시간대에 하루 한 번만 확인한다. 조건이 안 맞으면 즉시 돌아간다."""
    global _last_check_date
    if now.weekday() >= 5 or not (MORNING_FROM <= now.time() <= MORNING_TO):
        return
    if _last_check_date == now.date():
        return
    _last_check_date = now.date()
    try:
        _, made = refresh(ticker, stock_label)
    except Exception as exc:
        log(f"[재무요약] 실패: {type(exc).__name__}: {exc}")
        return
    log("[재무요약] 값이 바뀌어 다시 정리함" if made else "[재무요약] 변화 없음 - 그대로 둠")
