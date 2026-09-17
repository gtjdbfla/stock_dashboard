"""코스피200 선물 외국인 순매수 일별 이력을 파일로 들고 있는다. **streamlit을 import하지 않는다.**

옛 소스(finance.naver.com/sise/investorDealTrendDay.naver, 한 번에 여러 날 반환)가
2026-09월 중 HTTP 410으로 완전히 사라졌다(CONTEXT.md §5). 새 소스
(m.stock.naver.com/api/index/FUT/trend?bizdate=)는 하루치만 준다 - 700일을 채우려면
700번을 걸어야 해서, daily_history.py와 같은 파일 누적 방식으로 하루 한 번씩만 받아
쌓는다. 시장 전체 지표라 종목과 무관하게 파일 하나로 관리한다.

거래일이 아닌 날(주말·공휴일)은 이 API가 개인·외국인·기관 모두 문자열 "0"으로
돌려준다. 실제 거래일에 셋 다 정확히 0일 확률은 사실상 없으므로, 이를 '거래 없음'
신호로 보고 건너뛴다.
"""
import datetime as dt
import os
import time

import pandas as pd
import requests

STORE = os.environ.get("FUTURES_HISTORY_FILE", "data/futures_foreign_history.csv")
COLUMNS = ["날짜", "선물외국인"]

# 파일이 이보다 오래되면 못 믿는다. daily_history.py와 같은 기준.
FRESH_DAYS = 5

TREND_URL = "https://m.stock.naver.com/api/index/FUT/trend"
_UA = {"User-Agent": "Mozilla/5.0", "Referer": "https://m.stock.naver.com/"}


def _to_number(text) -> float | None:
    if text is None:
        return None
    s = str(text).replace(",", "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fetch_one_day(bizdate: dt.date) -> float | None:
    """그 날짜의 선물 외국인 순매수(계약수). 비거래일이거나 실패하면 None."""
    try:
        r = requests.get(TREND_URL, params={"bizdate": bizdate.strftime("%Y%m%d")},
                          headers=_UA, timeout=8)
        r.raise_for_status()
        js = r.json() or {}
    except Exception:
        return None
    personal = _to_number(js.get("personalValue"))
    foreign = _to_number(js.get("foreignValue"))
    inst = _to_number(js.get("institutionalValue"))
    if personal == 0 and foreign == 0 and inst == 0:
        return None            # 비거래일(주말·공휴일) 신호
    return foreign


def fetch_range(end_date: dt.date, days: int, log=print) -> pd.DataFrame:
    """end_date부터 거슬러 올라가며 거래일 days개를 모은다.

    하루 한 번씩 걸기 때문에 days가 크면 오래 걸린다(700일 기준 실측 약 4~5분).
    최초 백필에서만 크게 쓰고, 그 뒤로는 매일 며칠치만 덧붙인다.
    """
    rows = []
    cur = end_date
    tried = 0
    # 주말·연휴가 이어져도 포기하지 않게 넉넉히 허용하되, 무한루프는 막는다.
    max_calendar_days = days * 2 + 30
    while len(rows) < days and tried < max_calendar_days:
        tried += 1
        value = fetch_one_day(cur)
        if value is not None:
            rows.append({"날짜": pd.Timestamp(cur), "선물외국인": value})
        cur -= dt.timedelta(days=1)
        time.sleep(0.12)        # 하루 한 번이라도 수백 번이면 예의상 간격을 둔다
    if len(rows) < days:
        log(f"[선물이력] 목표 {days}일 중 {len(rows)}일만 모으고 중단(달력 {tried}일 시도)")
    return pd.DataFrame(rows).sort_values("날짜").reset_index(drop=True)


def load(min_days: int) -> pd.DataFrame | None:
    """파일이 쓸 만하면 최근 min_days개를 돌려주고, 아니면 None."""
    if not os.path.exists(STORE):
        return None
    try:
        df = pd.read_csv(STORE, parse_dates=["날짜"])
    except Exception:
        return None
    if df.empty or not set(COLUMNS).issubset(df.columns):
        return None
    if len(df) < min_days:
        return None
    newest = df["날짜"].max()
    if (pd.Timestamp(dt.date.today()) - newest).days > FRESH_DAYS:
        return None
    return df.sort_values("날짜").tail(min_days).reset_index(drop=True)


def save(df: pd.DataFrame) -> None:
    """받아온 이력을 파일에 합친다. 기존 줄과 날짜로 합쳐서 더 긴 이력을 유지한다."""
    if df is None or df.empty:
        return
    keep = [c for c in COLUMNS if c in df.columns]
    if "날짜" not in keep:
        return
    incoming = df[keep].copy()
    incoming["날짜"] = pd.to_datetime(incoming["날짜"])

    try:
        os.makedirs(os.path.dirname(STORE) or ".", exist_ok=True)
        if os.path.exists(STORE):
            old = pd.read_csv(STORE, parse_dates=["날짜"])
            incoming = pd.concat([old, incoming], ignore_index=True)
        merged = (incoming.drop_duplicates(subset="날짜", keep="last")
                          .sort_values("날짜").reset_index(drop=True))
        tmp = f"{STORE}.tmp"
        merged.to_csv(tmp, index=False)
        os.replace(tmp, STORE)      # 원자적 교체 — 상대가 반쪽 파일을 읽지 않게
    except OSError:
        pass


def status() -> str:
    """수집기 로그에 남길 한 줄."""
    if not os.path.exists(STORE):
        return "파일 없음"
    try:
        df = pd.read_csv(STORE, parse_dates=["날짜"])
        return f"{len(df)}행, 최신 {df['날짜'].max():%Y-%m-%d}"
    except Exception as exc:
        return f"읽기 실패: {type(exc).__name__}"


# ── 수집기용 ────────────────────────────────────────────────────────────────
# 선물 정규장은 15:45 마감. 그 값이 반영될 시각까지 여유를 두고 확인한다.
REFRESH_FROM = dt.time(16, 0)
REFRESH_TO = dt.time(23, 0)
TARGET_DAYS = int(os.environ.get("FUTURES_HISTORY_DAYS", "700"))

_last_refresh_date: dt.date | None = None


def tick(now: dt.datetime, log=print) -> None:
    """조건이 맞을 때만 갱신한다. 파일이 없으면(최초 배포) 시각과 무관하게 전체를 채운다."""
    global _last_refresh_date
    if _last_refresh_date == now.date():
        return

    first_build = load(TARGET_DAYS) is None
    in_window = now.weekday() < 5 and REFRESH_FROM <= now.time() <= REFRESH_TO
    if not (first_build or in_window):
        return

    _last_refresh_date = now.date()
    try:
        # 평소엔 최근 며칠만 덧붙인다. 주말을 지나도 하루 이틀 정도면 넉넉하다.
        fresh = fetch_range(now.date(), TARGET_DAYS if first_build else 5, log=log)
    except Exception as exc:
        _last_refresh_date = None       # 실패는 오늘 안에 다시 시도할 수 있게 둔다
        log(f"[선물이력] 받기 실패: {type(exc).__name__}: {exc}")
        return
    save(fresh)
    log(f"[선물이력] {'최초 생성' if first_build else '갱신'} — {status()}")
