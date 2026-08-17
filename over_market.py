"""프리장/애프터장(NXT) 시세 수집·저장 로직.

app.py(화면)와 collector.py(백그라운드 수집기)가 함께 쓴다. 한쪽에만 고치는 일이 없도록
파일 형식과 파싱 규칙을 여기 한 곳에 모아둔다.

네이버는 시간외 '분봉'을 제공하지 않는다(파라미터·경로·호스트를 두루 확인함).
대신 실시간 시세 응답의 overMarketPriceInfo에 현재 시간외 체결가가 들어 있어서,
그 값을 주기적으로 찍어 1분 단위로 쌓는다.
"""
import datetime as dt
import os
import threading

import pandas as pd
import requests

TICK_FILE = os.environ.get("OVER_MARKET_TICK_FILE", "data/over_market_ticks.csv")
TICK_KEEP_DAYS = int(os.environ.get("OVER_MARKET_TICK_KEEP_DAYS", "7"))
POLL_SEC = int(os.environ.get("OVER_MARKET_POLL_SEC", "20"))
IDLE_SEC = int(os.environ.get("OVER_MARKET_IDLE_SEC", "300"))
COLLECT_TICKERS = [
    t.strip() for t in os.environ.get("OVER_MARKET_COLLECT_TICKERS", "000660").split(",") if t.strip()
]
KST = dt.timezone(dt.timedelta(hours=9))

# 네이버가 주는 시간외 세션 구분값 -> 화면에 쓸 이름
SESSION_LABELS = {
    "PRE_MARKET": "프리장",
    "BEFORE_MARKET": "프리장",
    "AFTER_MARKET": "애프터장",
    "OVER_MARKET": "시간외",
}

# 수집기와 화면이 같은 파일을 건드리므로 읽기-수정-쓰기를 잠근다
_LOCK = threading.Lock()


def fetch_current_price_raw(ticker: str) -> dict:
    """캐시 없는 순수 조회. Streamlit 밖(수집기 프로세스)에서도 그대로 쓸 수 있다."""
    url = f"https://polling.finance.naver.com/api/realtime/domestic/stock/{ticker}"
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.naver.com/"}
    resp = requests.get(url, headers=headers, timeout=5)
    resp.raise_for_status()
    return resp.json()["datas"][0]


def parse_price_number(value: object) -> float | None:
    """'1,565,000'처럼 콤마가 들어간 문자열을 숫자로. 시간외 시세에는 raw 필드가 없다."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(",", "").strip()
    if not text or text in ("-", "N/A"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def save_tick(ticker: str, price: float, traded_at: object, session: str) -> None:
    """시간외 체결가를 1분 단위로 쌓는다. 같은 분에 여러 번 들어오면 마지막 값만 남긴다."""
    try:
        stamp = pd.to_datetime(traded_at)
    except (ValueError, TypeError):
        stamp = pd.Timestamp.now()
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    stamp = stamp.floor("min")

    row = pd.DataFrame([{"시각": stamp, "종목": str(ticker).zfill(6), "가격": float(price), "세션": session}])
    with _LOCK:
        _write(row)


def _write(row: pd.DataFrame) -> None:
    try:
        os.makedirs(os.path.dirname(TICK_FILE) or ".", exist_ok=True)
        if os.path.exists(TICK_FILE):
            # 종목코드를 문자열로 읽어야 한다. 그냥 두면 '000660'이 정수 660이 되어
            # 앞의 0이 날아가고, 중복 제거·조회에서 다른 종목처럼 취급된다.
            ticks = pd.read_csv(TICK_FILE, parse_dates=["시각"], dtype={"종목": str})
            ticks["종목"] = ticks["종목"].astype(str).str.zfill(6)
            ticks = pd.concat([ticks, row], ignore_index=True)
        else:
            ticks = row
        ticks = ticks.drop_duplicates(subset=["시각", "종목"], keep="last")
        cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=TICK_KEEP_DAYS)
        ticks = ticks[ticks["시각"] >= cutoff].sort_values("시각")
        ticks.to_csv(TICK_FILE, index=False)
    except (OSError, ValueError):
        pass  # 기록 실패가 화면을 막지 않게 한다


def load_ticks(ticker: str, since_date: dt.date) -> pd.DataFrame:
    """since_date 이후로 쌓인 시간외 체결가.

    하루만 뽑지 않는 이유: 프리장(08:00~09:00)에는 분봉 API가 아직 당일 데이터를 주지 않아
    '정규장 날짜'가 전 거래일로 잡힌다. 그 날짜로 걸러버리면 정작 오늘 아침 프리장이 빠진다.
    """
    # 빈 결과도 dtype을 맞춰서 돌려준다. 그냥 columns만 준 빈 DataFrame은 object 타입이라
    # 호출부에서 .dt.date 같은 걸 쓰면 AttributeError가 나고, 기록이 없는 평상시에
    # 그래프가 통째로 사라지는 문제가 생긴다.
    empty = pd.DataFrame({
        "시각": pd.Series(dtype="datetime64[ns]"),
        "가격": pd.Series(dtype="float64"),
        "세션": pd.Series(dtype="object"),
    })
    if not os.path.exists(TICK_FILE):
        return empty
    try:
        ticks = pd.read_csv(TICK_FILE, parse_dates=["시각"], dtype={"종목": str})
    except (OSError, ValueError):
        return empty
    if ticks.empty:
        return empty
    codes = ticks["종목"].astype(str).str.zfill(6)
    same = ticks[(codes == str(ticker).zfill(6)) & (ticks["시각"].dt.date >= since_date)]
    if same.empty:
        return empty
    return same[["시각", "가격", "세션"]].sort_values("시각").reset_index(drop=True)


def record_once(ticker: str) -> bool:
    """시간외 장이 열려 있으면 현재 체결가를 한 번 기록하고 True를 돌려준다."""
    data = fetch_current_price_raw(ticker)
    over = data.get("overMarketPriceInfo") or {}
    if over.get("overMarketStatus") != "OPEN":
        return False
    price = parse_price_number(over.get("overPrice"))
    if price is None:
        return False
    label = SESSION_LABELS.get(over.get("tradingSessionType"), "시간외")
    save_tick(ticker, price, over.get("localTradedAt"), label)
    return True


def in_collect_window(now_kst: dt.datetime) -> bool:
    """NXT 시간외 거래가 있을 만한 시간대인지(평일 07:30~20:30 KST)."""
    if now_kst.weekday() >= 5:
        return False
    return dt.time(7, 30) <= now_kst.time() <= dt.time(20, 30)
