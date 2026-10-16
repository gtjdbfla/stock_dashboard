"""장기(10년 이상) 일별 시세·수급 이력 파일. **streamlit을 import하지 않는다.**

daily_history(네이버, 700거래일)로는 5년 남짓밖에 못 본다. 하락·상승 조기신호와 괴리율을
10년·21년으로 검증해 보니 700일 결론이 일부 뒤집혔다(CONTEXT.md §5) — 그래서 화면과 AI가 같은
긴 이력을 보게 한다. 출처는 KIS '종목별 투자자매매동향(일별)'(FHPTJ04160001)로 2004-12부터
종목별 외국인·기관·개인 순매수와 종가·거래량을 30행씩 준다. 첫 생성은 페이지가 180장쯤이라
1~2분 걸리므로 수집기가 별도 스레드로 돌린다. 이후에는 마감 뒤 최근 며칠만 덧붙인다.

네트워크 코드는 ai_inputs에 있고(순환 import 방지, daily_history와 같은 구조) 여기는 저장·판정만 한다.
"""
import datetime as dt
import os

import pandas as pd

STORE_DIR = os.environ.get("LONG_HISTORY_DIR", "data")
COLUMNS = ["날짜", "종가", "거래량", "기관", "외국인", "개인"]
SINCE = dt.date(2005, 1, 1)
FRESH_DAYS = 5               # 파일이 이보다 낡았으면 못 믿는다(주말·연휴를 넘기게 넉넉히)
MIN_ROWS = 700               # 이보다 짧으면 daily_history와 다를 바 없어 쓰지 않는다
# 상·하한가가 ±30%라 하루에 35% 넘게 움직였다면 액면분할·병합으로 가격이 불연속이 된 것이다
# (삼성전자 2018-05 50:1 분할 등). 그 이전 구간은 버린다 — 섞으면 -98% '폭락'이 백테스트에 들어간다.
MAX_DAILY_JUMP = 0.35


def path_for(ticker: str) -> str:
    return os.path.join(STORE_DIR, f"long_history_{ticker}.csv")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """날짜순 정렬·중복 제거 후, 마지막 가격 불연속(액면분할 등) 이후 구간만 남긴다."""
    d = df[[c for c in COLUMNS if c in df.columns]].copy()
    d["날짜"] = pd.to_datetime(d["날짜"])
    d = d.dropna(subset=["날짜", "종가"]).drop_duplicates(subset="날짜", keep="last").sort_values("날짜").reset_index(drop=True)
    d = d[d["종가"] > 0].reset_index(drop=True)
    if len(d) < 2:
        return d
    jump = d["종가"].pct_change().abs() > MAX_DAILY_JUMP
    if jump.any():
        d = d.iloc[int(jump[jump].index[-1]):].reset_index(drop=True)
    return d


def _read(ticker: str) -> pd.DataFrame | None:
    p = path_for(ticker)
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_csv(p, parse_dates=["날짜"])
    except Exception:
        return None
    if df.empty or not set(COLUMNS).issubset(df.columns):
        return None
    return df


def load(ticker: str, min_days: int) -> pd.DataFrame | None:
    """최근 min_days개(파일이 더 짧으면 있는 만큼). 파일이 없다·너무 짧다·낡았다면 None."""
    df = _read(ticker)
    if df is None:
        return None
    df = clean(df)
    if len(df) < MIN_ROWS:
        return None
    if (pd.Timestamp(dt.date.today()) - df["날짜"].max()).days > FRESH_DAYS:
        return None
    return df.tail(min_days).reset_index(drop=True)


def save(ticker: str, df: pd.DataFrame) -> None:
    """받아온 줄을 기존 파일과 날짜로 합친다(새 값이 이긴다 — 마감 직후 잠정치를 확정치로 덮는다)."""
    if df is None or df.empty:
        return
    new = df[[c for c in COLUMNS if c in df.columns]].copy()
    new["날짜"] = pd.to_datetime(new["날짜"])
    p = path_for(ticker)
    try:
        os.makedirs(STORE_DIR, exist_ok=True)
        old = _read(ticker)
        merged = new if old is None else pd.concat([old, new], ignore_index=True)
        merged = clean(merged)
        tmp = f"{p}.tmp"
        merged.to_csv(tmp, index=False)
        os.replace(tmp, p)          # 원자적 교체 — 상대(대시보드)가 반쪽 파일을 읽지 않게
    except OSError:
        pass


# ── 수집기용 ────────────────────────────────────────────────────────────────
# KIS 확정 수급은 마감 12분 뒤(15:42)에 이미 나오지만 당일 조회는 15:40 이전에 TIME LIMIT 오류를 낸다.
REFRESH_FROM = dt.time(15, 50)
REFRESH_TO = dt.time(23, 0)
RETRY_MIN = 30
MAX_ATTEMPTS_PER_DAY = 4
OVERLAP_DAYS = 7             # 최근 며칠은 확정값으로 다시 덮어쓴다

_last_attempt: dict[str, dt.datetime] = {}
_attempts: dict[tuple[str, dt.date], int] = {}


def newest(ticker: str) -> dt.date | None:
    df = _read(ticker)
    return None if df is None or df.empty else df["날짜"].max().date()


def tick(now: dt.datetime, ticker: str, log=print) -> None:
    """파일이 없으면 곧바로 전체 이력을 만들고, 있으면 마감 뒤 최근 며칠만 덧붙인다."""
    last = _last_attempt.get(ticker)
    if last is not None and (now - last) < dt.timedelta(minutes=RETRY_MIN):
        return
    have = newest(ticker)
    first_build = have is None
    in_window = now.weekday() < 5 and REFRESH_FROM <= now.time() <= REFRESH_TO
    if not (first_build or in_window):
        return
    if not first_build and have == now.date():
        return
    if not first_build and _attempts.get((ticker, now.date()), 0) >= MAX_ATTEMPTS_PER_DAY:
        return                       # 평일 휴장일엔 오늘 치가 영영 안 나온다
    _last_attempt[ticker] = now
    _attempts[(ticker, now.date())] = _attempts.get((ticker, now.date()), 0) + 1

    import ai_inputs                 # 순환 import 방지: ai_inputs가 이 모듈을 import한다
    floor = SINCE if first_build else have - dt.timedelta(days=OVERLAP_DAYS)
    try:
        fresh = ai_inputs.fetch_kis_stock_daily_long(ticker, floor)
    except Exception as exc:
        log(f"[장기이력] {ticker} 받기 실패: {type(exc).__name__}: {exc}")
        return
    save(ticker, fresh)
    df = _read(ticker)
    log(f"[장기이력] {ticker} {'최초 생성' if first_build else '갱신'} — "
        f"{0 if df is None else len(df)}행, 최신 {newest(ticker)}")
