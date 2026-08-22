"""장 마감 후 투자자 수급이 어느 경로에 가장 먼저 올라오는지 기록한다.

대시보드는 지금 네이버 frgn.naver에서 수급을 받는데, 마감(15:30) 후 한참 뒤에야
올라온다. 어느 경로가 가장 빠른지 실측하려고 브라우저 쪽에서 감시를 걸었더니
세션이 끊기면서 첫 샘플만 남고 죽었다. 그래서 24시간 도는 수집기에 붙인다.

하는 일은 단순하다. 평일 마감 후 5분마다 각 경로의 '최신 날짜'를 확인해서,
오늘 날짜가 처음 보이는 순간을 CSV에 한 줄 남긴다. 경로마다 한 번씩만 기록하고,
전부 잡히면 그 날은 더 확인하지 않는다.
"""
import csv
import datetime as dt
import io
import os

import pandas as pd
import requests

KST = dt.timezone(dt.timedelta(hours=9))
LOG_FILE = os.environ.get("FLOW_PROBE_LOG", "data/flow_publish_log.csv")
TICKER = os.environ.get("FLOW_PROBE_TICKER", "000660")

# 마감 15:30 직후부터 저녁까지. 이 바깥에서는 아무것도 하지 않는다.
WATCH_FROM = dt.time(15, 35)
WATCH_TO = dt.time(21, 0)
PROBE_SEC = 300

_UA = {"User-Agent": "Mozilla/5.0"}


def _naver_frgn(ticker: str) -> dt.date | None:
    r = requests.get("https://finance.naver.com/item/frgn.naver",
                     params={"code": ticker, "page": 1},
                     headers={**_UA, "Referer": "https://finance.naver.com/"}, timeout=8)
    r.raise_for_status()
    r.encoding = "euc-kr"
    table = pd.read_html(io.StringIO(r.text))[3]
    table.columns = ["날짜", "종가", "전일비", "등락률", "거래량",
                     "기관", "외국인순매매", "외국인보유", "외국인비율"]
    table = table.dropna(subset=["날짜"])
    if table.empty:
        return None
    return pd.to_datetime(str(table.iloc[0]["날짜"]), format="%Y.%m.%d").date()


def _daum(ticker: str) -> dt.date | None:
    r = requests.get("https://finance.daum.net/api/investor/days",
                     params={"symbolCode": f"A{ticker}", "page": 1, "perPage": 2, "pagination": "true"},
                     headers={**_UA, "Referer": f"https://finance.daum.net/quotes/A{ticker}"}, timeout=8)
    r.raise_for_status()
    rows = (r.json() or {}).get("data") or []
    if not rows:
        return None
    return dt.date.fromisoformat(rows[0]["date"][:10])


def _naver_mobile(path: str):
    def fn(ticker: str) -> dt.date | None:
        r = requests.get(f"https://m.stock.naver.com/api/stock/{ticker}/{path}",
                         headers={**_UA, "Referer": "https://m.stock.naver.com/"}, timeout=8)
        r.raise_for_status()
        js = r.json()
        rows = js if isinstance(js, list) else ((js or {}).get("dealTrendInfos") or [])
        if not rows:
            return None
        return dt.datetime.strptime(str(rows[0]["bizdate"]), "%Y%m%d").date()
    return fn


def _naver_daily(ticker: str) -> dt.date | None:
    """종가·거래량 경로. 수급은 없지만 '얼마나 먼저 확정되는지' 비교 기준으로 같이 잰다."""
    today = dt.datetime.now(KST)
    r = requests.get(f"https://api.stock.naver.com/chart/domestic/item/{ticker}/day",
                     params={"startDateTime": (today - dt.timedelta(days=7)).strftime("%Y%m%d0000"),
                             "endDateTime": today.strftime("%Y%m%d2359")},
                     headers={**_UA, "Referer": "https://m.stock.naver.com/"}, timeout=8)
    r.raise_for_status()
    rows = r.json()
    if not isinstance(rows, list) or not rows:
        return None
    return dt.datetime.strptime(str(rows[-1]["localDate"]), "%Y%m%d").date()


SOURCES = {
    "naver_frgn": _naver_frgn,          # 대시보드가 지금 쓰는 경로
    "daum_investor": _daum,
    "naver_trend": _naver_mobile("trend"),
    "naver_integration": _naver_mobile("integration"),
    "naver_daily_ohlcv": _naver_daily,  # 수급 아님 (거래량·종가 확정 시점 비교용)
}

# 그 날 이미 잡은 경로. {거래일: {소스명, ...}}
_seen: dict[dt.date, set] = {}
_last_probe: dt.datetime | None = None


def _append_log(trade_date: dt.date, source: str, when: dt.datetime) -> None:
    os.makedirs(os.path.dirname(LOG_FILE) or ".", exist_ok=True)
    is_new = not os.path.exists(LOG_FILE)
    delay = (when - dt.datetime.combine(trade_date, dt.time(15, 30), KST)).total_seconds() / 60
    with open(LOG_FILE, "a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        if is_new:
            w.writerow(["거래일", "소스", "최초확인시각", "마감후_분"])
        w.writerow([trade_date.isoformat(), source, when.strftime("%H:%M"), f"{delay:.0f}"])


def tick(now: dt.datetime, log=print) -> None:
    """수집기 루프에서 매번 불린다. 조건이 안 맞으면 즉시 돌아간다."""
    global _last_probe
    if now.weekday() >= 5 or not (WATCH_FROM <= now.time() <= WATCH_TO):
        return
    if _last_probe and (now - _last_probe).total_seconds() < PROBE_SEC:
        return
    _last_probe = now

    today = now.date()
    done = _seen.setdefault(today, set())
    if len(done) >= len(SOURCES):
        return

    for name, fetch in SOURCES.items():
        if name in done:
            continue
        try:
            latest = fetch(TICKER)
        except Exception as exc:
            log(f"[수급탐침] {name} 조회 실패: {type(exc).__name__}")
            continue
        if latest == today:
            done.add(name)
            _append_log(today, name, now)
            delay = (now - dt.datetime.combine(today, dt.time(15, 30), KST)).total_seconds() / 60
            log(f"[수급탐침] {name} 최초 확인 {now:%H:%M} (마감 +{delay:.0f}분)")

    if len(done) >= len(SOURCES):
        log(f"[수급탐침] {today} 전 경로 확인 완료: {', '.join(sorted(done))}")


def summary(path: str = None) -> str:
    """쌓인 기록에서 경로별 평균 지연을 요약한다."""
    path = path or LOG_FILE
    if not os.path.exists(path):
        return "기록 없음"
    df = pd.read_csv(path)
    if df.empty:
        return "기록 없음"
    g = df.groupby("소스")["마감후_분"].agg(["count", "mean", "min", "max"]).sort_values("mean")
    lines = [f"{'소스':<20} {'일수':>4} {'평균':>7} {'최소':>6} {'최대':>6}  (마감 후 분)"]
    for name, r in g.iterrows():
        lines.append(f"{name:<20} {int(r['count']):>4} {r['mean']:>7.0f} {r['min']:>6.0f} {r['max']:>6.0f}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
