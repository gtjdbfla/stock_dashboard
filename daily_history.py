"""일별 시세·수급 이력을 파일로 들고 있는다. **streamlit을 import하지 않는다.**

대시보드가 700일 백테스트 이력을 볼 때마다 네이버 frgn 페이지를 35장 받았고, 그게
5~7초였다 — 콜드 로드에서 제일 큰 한 덩어리였다. 같은 데이터를 수집기가 하루 한 번만
받아 여기 쌓아 두면 화면은 파일만 읽는다.

여기에는 네트워크 코드가 없다. 받아오는 건 `ai_inputs.fetch_backtest_history`가 하고,
이 모듈은 그 결과를 저장하고 되돌려주는 일만 한다(순환 import 방지).

대시보드와 수집기가 같은 `data/` 볼륨을 함께 보므로 쓰기는 임시파일 + os.replace로
원자적으로 한다. 반쯤 쓰인 CSV를 상대가 읽는 일이 없어야 한다.
"""
import datetime as dt
import os

import pandas as pd

STORE_DIR = os.environ.get("DAILY_HISTORY_DIR", "data")
# 개인도 저장한다. 예전 수급 소스(frgn.naver)는 개인을 안 줘서 -(기관+외국인)으로
# 유도했는데, 기타법인이 빠져 실제와 크게 어긋났다. 새 소스(trend API)는 직접 준다.
COLUMNS = ["날짜", "종가", "거래량", "기관", "외국인", "개인"]

# 파일이 이보다 오래되면 못 믿는다. 주말·연휴를 건너뛰어야 하므로 달력 기준으로 넉넉히
# 잡는다. 오늘·어제치는 어차피 fetch_latest_bars(ttl=60)가 위에 덧씌운다.
FRESH_DAYS = 5


def path_for(ticker: str) -> str:
    return os.path.join(STORE_DIR, f"daily_history_{ticker}.csv")


def load(ticker: str, min_days: int) -> pd.DataFrame | None:
    """파일이 쓸 만하면 최근 min_days개를 돌려주고, 아니면 None.

    None을 주는 경우는 셋이다: 파일이 없다 / 줄이 모자란다 / 너무 오래됐다.
    부르는 쪽은 그때만 네트워크로 간다.
    """
    p = path_for(ticker)
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_csv(p, parse_dates=["날짜"])
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


def save(ticker: str, df: pd.DataFrame) -> None:
    """받아온 이력을 파일에 합친다. 기존 줄과 날짜로 합쳐서 더 긴 이력을 유지한다.

    한 번에 700일을 받아도, 다음에 250일만 받는 호출이 파일을 250일로 깎지 않게 한다.
    """
    if df is None or df.empty:
        return
    keep = [c for c in COLUMNS if c in df.columns]
    if "날짜" not in keep:
        return
    incoming = df[keep].copy()
    incoming["날짜"] = pd.to_datetime(incoming["날짜"])

    p = path_for(ticker)
    try:
        os.makedirs(STORE_DIR, exist_ok=True)
        if os.path.exists(p):
            old = pd.read_csv(p, parse_dates=["날짜"])
            incoming = pd.concat([old, incoming], ignore_index=True)
        merged = (incoming.drop_duplicates(subset="날짜", keep="last")
                          .sort_values("날짜").reset_index(drop=True))
        tmp = f"{p}.tmp"
        merged.to_csv(tmp, index=False)
        os.replace(tmp, p)          # 원자적 교체 — 상대가 반쪽 파일을 읽지 않게
    except OSError:
        pass                        # 저장 실패가 화면을 막지는 않는다


# ── 수집기용 ────────────────────────────────────────────────────────────────
# 네이버 일별 수급은 마감(15:30) 한참 뒤에 올라온다 — flow_probe 실측으로 18:25였다.
# 그 뒤에 하루 한 번만 받아 둔다. 파일이 아직 없으면 시각과 상관없이 바로 만든다.
REFRESH_FROM = dt.time(18, 30)
REFRESH_TO = dt.time(23, 0)
TARGET_DAYS = int(os.environ.get("DAILY_HISTORY_DAYS", "700"))

_last_refresh_date: dt.date | None = None


def tick(now: dt.datetime, ticker: str, log=print) -> None:
    """조건이 맞을 때만 네이버에서 받아 파일을 갱신한다. 아니면 즉시 돌아간다."""
    global _last_refresh_date
    if _last_refresh_date == now.date():
        return

    first_build = load(ticker, TARGET_DAYS) is None
    in_window = now.weekday() < 5 and REFRESH_FROM <= now.time() <= REFRESH_TO
    if not (first_build or in_window):
        return

    _last_refresh_date = now.date()
    # ai_inputs가 이 모듈을 import하므로 여기서 늦게 부른다(순환 방지).
    import ai_inputs
    try:
        fresh = ai_inputs._fetch_backtest_history_web(ticker, TARGET_DAYS)
    except Exception as exc:
        _last_refresh_date = None       # 실패는 오늘 안에 다시 시도할 수 있게 둔다
        log(f"[일별이력] 받기 실패: {type(exc).__name__}: {exc}")
        return
    save(ticker, fresh)
    log(f"[일별이력] {'최초 생성' if first_build else '갱신'} — {status(ticker)}")


def status(ticker: str) -> str:
    """수집기 로그에 남길 한 줄."""
    p = path_for(ticker)
    if not os.path.exists(p):
        return "파일 없음"
    try:
        df = pd.read_csv(p, parse_dates=["날짜"])
        return f"{len(df)}행, 최신 {df['날짜'].max():%Y-%m-%d}"
    except Exception as exc:
        return f"읽기 실패: {type(exc).__name__}"
