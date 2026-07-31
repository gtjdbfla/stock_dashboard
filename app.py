import datetime as dt
import os
from io import StringIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from bs4 import BeautifulSoup
from google import genai
from plotly.subplots import make_subplots

DEFAULT_TICKER = "000660"
DEFAULT_STOCK_NAME = "SK하이닉스"
NAVER_SEARCH_URL = "https://ac.stock.naver.com/ac"
NAVER_FRGN_URL = "https://finance.naver.com/item/frgn.naver"
NAVER_NEWS_URL = "https://search.naver.com/search.naver"
NAVER_RESEARCH_URL = "https://finance.naver.com/research/company_list.naver"
NAVER_BOARD_URL = "https://finance.naver.com/item/board.naver"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
YAHOO_SYMBOLS = {
    "SOX": "%5ESOX",
    "DXY": "DX-Y.NYB",
}
DRAMEXCHANGE_URL = "https://www.dramexchange.com/"
MEMORY_SEMICONDUCTOR_TICKERS = {"000660": "SK하이닉스", "005930": "삼성전자"}

POSITIVE_KEYWORDS = [
    "상승", "오른다", "올랐", "급등", "떡상", "가즈아", "가보자", "존버", "매수",
    "저점매수", "반등", "호재", "강세", "상한가", "신고가", "돌파", "추매", "줍줍", "익절",
]
NEGATIVE_KEYWORDS = [
    "하락", "내린다", "내렸", "급락", "떡락", "손절", "물렸", "물림", "개미지옥", "지옥",
    "악재", "약세", "하한가", "신저가", "붕괴", "패닉", "팔아", "매도", "손실", "탈출",
    "폭락", "마이너스", "마이나스", "개미눈물",
]
DRAM_HISTORY_FILE = os.environ.get("DRAM_HISTORY_FILE", "data/dram_spot_history.csv")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


@st.cache_data(ttl=3600)
def fetch_stock_search(query: str) -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(NAVER_SEARCH_URL, params={"q": query, "target": "stock"}, headers=headers, timeout=10)
    resp.raise_for_status()
    items = resp.json().get("items", [])
    rows = [
        {"code": it["code"], "name": it["name"]}
        for it in items
        if it.get("typeCode") == "KOSPI" and it.get("code", "").isdigit() and len(it.get("code", "")) == 6
    ]
    return pd.DataFrame(rows)


if "ticker" not in st.session_state:
    st.session_state.ticker = DEFAULT_TICKER
    st.session_state.stock_name = DEFAULT_STOCK_NAME

st.set_page_config(page_title=f"{st.session_state.stock_name} 대시보드", layout="wide")

with st.sidebar:
    st.header("종목 검색 (코스피)")
    search_query = st.text_input("종목명 검색", placeholder="예: 삼성전자")
    if search_query:
        search_results = fetch_stock_search(search_query)
        if search_results.empty:
            st.caption("검색 결과가 없습니다.")
        else:
            options = [f"{row['name']} ({row['code']})" for _, row in search_results.iterrows()]
            choice = st.selectbox("검색 결과", options)
            if st.button("이 종목으로 변경"):
                idx = options.index(choice)
                st.session_state.ticker = search_results.iloc[idx]["code"]
                st.session_state.stock_name = search_results.iloc[idx]["name"]
                st.session_state.pop("ai_analysis", None)
                st.rerun()
    st.caption(f"현재 선택: {st.session_state.stock_name} ({st.session_state.ticker})")

    st.divider()
    st.header("설정")
    refresh_sec = st.slider("시세 갱신 주기(초)", min_value=5, max_value=120, value=5, step=5)
    lookback_days = st.slider("수급 분석 기간(일)", min_value=10, max_value=180, value=30, step=10)
    community_post_count = st.slider("커뮤니티 게시글 조회 개수", min_value=20, max_value=300, value=60, step=20)

TICKER = st.session_state.ticker
STOCK_NAME = st.session_state.stock_name

st.title(f"{STOCK_NAME}({TICKER}) 대시보드")


@st.cache_data(ttl=5)
def fetch_current_price(ticker: str) -> dict:
    url = f"https://polling.finance.naver.com/api/realtime/domestic/stock/{ticker}"
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.naver.com/"}
    resp = requests.get(url, headers=headers, timeout=5)
    resp.raise_for_status()
    return resp.json()["datas"][0]


def _fetch_frgn_page(ticker: str, page: int) -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.naver.com/"}
    resp = requests.get(
        NAVER_FRGN_URL, params={"code": ticker, "page": page}, headers=headers, timeout=10
    )
    resp.raise_for_status()
    resp.encoding = "euc-kr"
    tables = pd.read_html(StringIO(resp.text))
    df = tables[3]
    df.columns = ["날짜", "종가", "전일비", "등락률", "거래량", "기관", "외국인_순매매량", "외국인_보유주수", "외국인_보유율"]
    df = df.dropna(subset=["날짜"]).copy()
    df["날짜"] = pd.to_datetime(df["날짜"], format="%Y.%m.%d")
    df["종가"] = df["종가"].astype(float)
    df["기관"] = df["기관"].astype(float)
    df["거래량"] = pd.to_numeric(df["거래량"].astype(str).str.replace(",", ""), errors="coerce")
    return df[["날짜", "종가", "거래량", "기관", "외국인_순매매량"]].rename(columns={"외국인_순매매량": "외국인"})


@st.cache_data(ttl=3600)
def fetch_investor_netbuy(ticker: str, days: int) -> pd.DataFrame:
    cutoff = pd.Timestamp(dt.date.today() - dt.timedelta(days=days))
    frames = []
    max_pages = 25
    for page in range(1, max_pages + 1):
        page_df = _fetch_frgn_page(ticker, page)
        if page_df.empty:
            break
        frames.append(page_df)
        if page_df["날짜"].min() <= cutoff:
            break

    if not frames:
        return pd.DataFrame(columns=["개인", "외국인", "기관"])

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="날짜")
    df = df[df["날짜"] >= cutoff].sort_values("날짜")
    df["개인"] = -(df["기관"] + df["외국인"])
    df = df.set_index("날짜")[["개인", "외국인", "기관"]]
    return df


def calc_slope(cum_series: pd.Series) -> float:
    y = cum_series.to_numpy()
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    cum = series.cumsum()
    return cum.rolling(window).apply(lambda w: calc_slope(pd.Series(w)), raw=False)


@st.cache_data(ttl=24 * 3600)
def fetch_backtest_history(ticker: str, target_days: int = 500) -> pd.DataFrame:
    frames = []
    max_pages = 60
    for page in range(1, max_pages + 1):
        page_df = _fetch_frgn_page(ticker, page)
        if page_df.empty:
            break
        frames.append(page_df)
        if sum(len(f) for f in frames) >= target_days:
            break

    if not frames:
        return pd.DataFrame(columns=["날짜", "종가", "기관", "외국인", "개인"])

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="날짜").sort_values("날짜")
    df["개인"] = -(df["기관"] + df["외국인"])
    return df.reset_index(drop=True)


@st.cache_data(ttl=60)
def fetch_latest_bars(ticker: str) -> pd.DataFrame:
    """장중 계속 바뀌는 최근 1페이지(며칠치)만 짧은 캐시로 빠르게 가져온다."""
    page_df = _fetch_frgn_page(ticker, 1)
    if page_df.empty:
        return pd.DataFrame(columns=["날짜", "종가", "거래량", "기관", "외국인", "개인"])
    page_df = page_df.copy()
    page_df["개인"] = -(page_df["기관"] + page_df["외국인"])
    return page_df.reset_index(drop=True)


def fetch_backtest_history_live(ticker: str, target_days: int = 700) -> pd.DataFrame:
    """24시간 캐시된 과거 이력에 오늘자를 포함한 최근 며칠치를 실시간(1분 캐시)으로 덧씌워 반환한다."""
    hist = fetch_backtest_history(ticker, target_days=target_days)
    latest = fetch_latest_bars(ticker)
    if latest.empty:
        return hist
    merged = pd.concat([hist, latest], ignore_index=True).drop_duplicates(subset="날짜", keep="last")
    return merged.sort_values("날짜").reset_index(drop=True)


def forward_max_drawdown(price: pd.Series, horizon: int) -> pd.Series:
    """t 시점 가격 대비, 이후 horizon일 내 최저가까지의 낙폭(최대 하락률, 음수)."""
    values = price.to_numpy()
    n = len(values)
    result = np.full(n, np.nan)
    for i in range(n - horizon):
        window = values[i + 1 : i + 1 + horizon]
        result[i] = window.min() / values[i] - 1
    return pd.Series(result, index=price.index)


def two_proportion_ztest(x1: float, n1: float, x2: float, n2: float):
    if n1 == 0 or n2 == 0:
        return None
    from scipy import stats as scistats
    p_pool = (x1 + x2) / (n1 + n2)
    se = (p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)) ** 0.5
    if se == 0:
        return None
    z = (x1 / n1 - x2 / n2) / se
    return float(2 * (1 - scistats.norm.cdf(abs(z))))


def _level_slope(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).apply(lambda w: calc_slope(pd.Series(w)), raw=False)


def run_overheat_backtest(
    df: pd.DataFrame, price_col: str, ma_window: int, horizon: int,
    quantile: float = 0.2, drawdown_threshold: float = 0.05,
) -> dict:
    """종가가 ma_window일 이동평균 대비 상위 quantile(과열) 구간일 때, 향후 horizon일 내
    drawdown_threshold 이상 하락(하락장) 확률이 나머지 구간보다 높은지 검증한다."""
    d = df.copy()
    d["ma"] = d[price_col].rolling(ma_window).mean()
    d["deviation"] = d[price_col] / d["ma"] - 1
    d["drawdown"] = forward_max_drawdown(d[price_col], horizon)
    valid = d.dropna(subset=["deviation", "drawdown"])

    result = {
        "ma_window": ma_window, "horizon": horizon, "quantile": quantile,
        "drawdown_threshold": drawdown_threshold, "n": len(valid),
        "hi_n": 0, "rest_n": 0, "hi_rate": None, "rest_rate": None, "base_rate": None,
        "p_value": None, "hi_cutoff": None,
        "current_deviation": None, "current_regime": None,
    }
    # 현재 상태는 향후 수익률 계산 없이 전체 이력에서 바로 판단한다 (최근 horizon일은 drawdown이 아직 계산 안 돼 valid에서 빠짐).
    deviation_all = d["deviation"].dropna()
    if len(deviation_all) > 0:
        result["current_deviation"] = float(deviation_all.iloc[-1])

    if len(valid) < 30:
        return result

    valid = valid.assign(downtrend=(valid["drawdown"] <= -drawdown_threshold).astype(float))
    hi_cutoff = valid["deviation"].quantile(1 - quantile)
    hi_group = valid[valid["deviation"] >= hi_cutoff]
    rest_group = valid[valid["deviation"] < hi_cutoff]

    result["hi_n"] = len(hi_group)
    result["rest_n"] = len(rest_group)
    result["hi_rate"] = float(hi_group["downtrend"].mean()) if len(hi_group) else None
    result["rest_rate"] = float(rest_group["downtrend"].mean()) if len(rest_group) else None
    result["base_rate"] = float(valid["downtrend"].mean())
    result["hi_cutoff"] = float(hi_cutoff)

    if result["hi_rate"] is not None and result["rest_rate"] is not None:
        result["p_value"] = two_proportion_ztest(
            hi_group["downtrend"].sum(), len(hi_group), rest_group["downtrend"].sum(), len(rest_group)
        )

    result["current_regime"] = (
        f"과열 (상위 {quantile:.0%})" if result["current_deviation"] is not None and result["current_deviation"] >= hi_cutoff
        else "평상시"
    )
    return result


def run_overheat_threshold_strategy(
    df_with_deviation: pd.DataFrame, entry_threshold: float, period_start: pd.Timestamp,
) -> dict:
    """괴리율이 entry_threshold 이상일 때 매수, 0% 이하로 돌아오면 매도하는 단순 전략을
    period_start 이후 구간에서 시뮬레이션하고 buy & hold와 비교한다.
    이동평균/괴리율은 전체 이력 기준으로 계산된 값을 그대로 사용해 lookback 손실이 없다."""
    test_df = df_with_deviation[df_with_deviation["날짜"] >= period_start].dropna(subset=["괴리율"]).reset_index(drop=True)
    result = {
        "n_days": len(test_df), "trades": [], "cum_return": None, "buy_hold_return": None,
        "still_open": False, "unrealized_return": None, "equity_curve": None,
        "period_start": None, "period_end": None,
    }
    if test_df.empty:
        return result

    result["period_start"] = test_df.iloc[0]["날짜"]
    result["period_end"] = test_df.iloc[-1]["날짜"]
    result["buy_hold_return"] = float(test_df.iloc[-1]["종가"] / test_df.iloc[0]["종가"] - 1)

    position = None
    trades = []
    equity = []
    equity_base = 1.0
    for _, row in test_df.iterrows():
        date, price, deviation = row["날짜"], row["종가"], row["괴리율"]
        if position is None:
            equity.append({"날짜": date, "자산가치": equity_base})
            if deviation >= entry_threshold:
                position = {"buy_date": date, "buy_price": price}
        else:
            mark = equity_base * (price / position["buy_price"])
            equity.append({"날짜": date, "자산가치": mark})
            if deviation <= 0:
                equity_base = mark
                trades.append({
                    "buy_date": position["buy_date"], "sell_date": date,
                    "buy_price": position["buy_price"], "sell_price": price,
                    "ret": price / position["buy_price"] - 1,
                })
                position = None

    result["trades"] = trades
    result["still_open"] = position is not None
    cum_return = 1.0
    for t in trades:
        cum_return *= (1 + t["ret"])
    if position is not None:
        result["unrealized_return"] = float(test_df.iloc[-1]["종가"] / position["buy_price"] - 1)
        cum_return *= (1 + result["unrealized_return"])
    result["cum_return"] = float(cum_return - 1) if trades or position is not None else None
    result["equity_curve"] = pd.DataFrame(equity)
    return result


FUTURES_DEAL_TREND_URL = "https://finance.naver.com/sise/investorDealTrendDay.naver"


@st.cache_data(ttl=24 * 3600)
def fetch_futures_foreign_history(target_days: int = 700) -> pd.DataFrame:
    """코스피200 선물 외국인 순매수(계약수) 일별 이력. 특정 종목이 아닌 시장 전체 지표라 티커와 무관하게 캐시된다."""
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.naver.com/sise/sise_trans_style.naver?code=FUT"}
    frames = []
    seen_dates = set()
    bizdate = dt.date.today().strftime("%Y%m%d")
    for _ in range(90):
        resp = requests.get(FUTURES_DEAL_TREND_URL, params={"bizdate": bizdate, "code": "FUT"}, headers=headers, timeout=10)
        resp.raise_for_status()
        resp.encoding = "euc-kr"
        try:
            tables = pd.read_html(StringIO(resp.text))
        except ValueError:
            break
        t = tables[0]
        t.columns = ["날짜", "개인", "외국인", "기관계", "금융투자", "보험", "투신", "은행", "기타금융", "연기금", "기타법인"]
        t = t.dropna(subset=["날짜"]).copy()
        if t.empty:
            break
        t["날짜"] = pd.to_datetime(t["날짜"], format="%y.%m.%d")
        t["외국인"] = pd.to_numeric(t["외국인"], errors="coerce")
        new_rows = t[~t["날짜"].isin(seen_dates)]
        if new_rows.empty:
            break
        seen_dates.update(new_rows["날짜"])
        frames.append(new_rows[["날짜", "외국인"]])
        if sum(len(f) for f in frames) >= target_days:
            break
        bizdate = (new_rows["날짜"].min() - pd.Timedelta(days=1)).strftime("%Y%m%d")
    if not frames:
        return pd.DataFrame(columns=["날짜", "선물외국인"])
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset="날짜").sort_values("날짜")
    return out.reset_index(drop=True).rename(columns={"외국인": "선물외국인"})


@st.cache_data(ttl=60)
def fetch_latest_futures_bars() -> pd.DataFrame:
    """장중 계속 바뀌는 코스피200 선물 최근 며칠치만 짧은 캐시로 빠르게 가져온다."""
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.naver.com/sise/sise_trans_style.naver?code=FUT"}
    bizdate = dt.date.today().strftime("%Y%m%d")
    resp = requests.get(FUTURES_DEAL_TREND_URL, params={"bizdate": bizdate, "code": "FUT"}, headers=headers, timeout=10)
    resp.raise_for_status()
    resp.encoding = "euc-kr"
    try:
        tables = pd.read_html(StringIO(resp.text))
    except ValueError:
        return pd.DataFrame(columns=["날짜", "선물외국인"])
    t = tables[0]
    t.columns = ["날짜", "개인", "외국인", "기관계", "금융투자", "보험", "투신", "은행", "기타금융", "연기금", "기타법인"]
    t = t.dropna(subset=["날짜"]).copy()
    if t.empty:
        return pd.DataFrame(columns=["날짜", "선물외국인"])
    t["날짜"] = pd.to_datetime(t["날짜"], format="%y.%m.%d")
    t["외국인"] = pd.to_numeric(t["외국인"], errors="coerce")
    return t[["날짜", "외국인"]].reset_index(drop=True).rename(columns={"외국인": "선물외국인"})


def fetch_futures_foreign_history_live(target_days: int = 700) -> pd.DataFrame:
    """24시간 캐시된 과거 이력에 오늘자를 포함한 최근 며칠치를 실시간(1분 캐시)으로 덧씌워 반환한다."""
    hist = fetch_futures_foreign_history(target_days=target_days)
    latest = fetch_latest_futures_bars()
    if latest.empty:
        return hist
    merged = pd.concat([hist, latest], ignore_index=True).drop_duplicates(subset="날짜", keep="last")
    return merged.sort_values("날짜").reset_index(drop=True)


def run_futures_decline_backtest(
    price: pd.Series, dates: pd.Series, flow: pd.Series, window: int, horizon: int,
    quantile: float = 0.2, drawdown_threshold: float = 0.07,
) -> dict:
    """코스피200 선물 외국인 누적 순매수 기울기가 하위 quantile(강한 매도)일 때, 현재 종목의 향후 horizon일 내
    drawdown_threshold 이상 하락 확률이 나머지 구간보다 높은지 검증한다. 하락 예측 전용이며 매수 신호로는 쓰지 않는다."""
    d = pd.DataFrame({"날짜": dates, "종가": price, "선물외국인": flow}).dropna(subset=["선물외국인"])
    d["slope"] = _rolling_slope(d["선물외국인"], window)
    d["drawdown"] = forward_max_drawdown(d["종가"], horizon)
    valid = d.dropna(subset=["slope", "drawdown"])

    result = {
        "n": len(valid), "lo_n": 0, "rest_n": 0, "lo_rate": None, "rest_rate": None, "base_rate": None,
        "p_value": None, "lo_cutoff": None, "current_slope": None, "current_regime": None,
    }
    slope_all = d["slope"].dropna()
    if len(slope_all) > 0:
        result["current_slope"] = float(slope_all.iloc[-1])

    if len(valid) < 30:
        return result

    valid = valid.assign(downtrend=(valid["drawdown"] <= -drawdown_threshold).astype(float))
    lo_cutoff = valid["slope"].quantile(quantile)
    lo_group = valid[valid["slope"] <= lo_cutoff]
    rest_group = valid[valid["slope"] > lo_cutoff]

    result["lo_n"] = len(lo_group)
    result["rest_n"] = len(rest_group)
    result["lo_rate"] = float(lo_group["downtrend"].mean()) if len(lo_group) else None
    result["rest_rate"] = float(rest_group["downtrend"].mean()) if len(rest_group) else None
    result["base_rate"] = float(valid["downtrend"].mean())
    result["lo_cutoff"] = float(lo_cutoff)

    if result["lo_rate"] is not None and result["rest_rate"] is not None:
        result["p_value"] = two_proportion_ztest(
            lo_group["downtrend"].sum(), len(lo_group), rest_group["downtrend"].sum(), len(rest_group)
        )

    result["current_regime"] = (
        f"강한 매도 경고 (하위 {quantile:.0%})"
        if result["current_slope"] is not None and result["current_slope"] <= lo_cutoff
        else "평상시"
    )
    return result


@st.cache_data(ttl=24 * 3600)
def fetch_yahoo_history(label: str) -> pd.DataFrame:
    symbol = YAHOO_SYMBOLS[label]
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(
        YAHOO_CHART_URL.format(symbol=symbol), params={"range": "2y", "interval": "1d"}, headers=headers, timeout=15
    )
    resp.raise_for_status()
    data = resp.json()
    result = data["chart"]["result"][0]
    ts = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]
    df = pd.DataFrame({"날짜": pd.to_datetime(ts, unit="s").normalize(), label: closes})
    return df.dropna().reset_index(drop=True)


def build_composite_dataset(ticker: str) -> pd.DataFrame:
    """무거운 하위 fetch들(기관/외국인 이력, SOX/DXY)은 각자 24시간 캐시되고 최근 며칠치만 1분 캐시로
    실시간 반영되므로, 이 함수 자체는 캐시하지 않고 매번 가볍게 재조립한다."""
    stock_hist = fetch_backtest_history_live(ticker)

    df = stock_hist.copy()
    for label in YAHOO_SYMBOLS:
        df = df.merge(fetch_yahoo_history(label), on="날짜", how="left")
    df = df.sort_values("날짜").reset_index(drop=True)

    for label in YAHOO_SYMBOLS:
        df[label] = df[label].ffill()

    df["기관_기울기"] = _rolling_slope(df["기관"], 20)
    df["SOX_기울기"] = _level_slope(df["SOX"], 20)
    df["DXY_기울기"] = _level_slope(df["DXY"], 20)
    return df


def backtest_signal(df: pd.DataFrame, signal_col: str, horizon: int) -> dict:
    d = df.copy()
    d["fwd_return"] = d["종가"].shift(-horizon) / d["종가"] - 1
    valid = d.dropna(subset=[signal_col, "fwd_return"])

    result = {
        "signal": signal_col, "horizon": horizon,
        "n": len(valid), "pos_n": 0, "neg_n": 0,
        "pos_mean": None, "neg_mean": None, "p_value": None, "corr": None,
        "current_value": None, "current_regime": None,
    }
    if len(valid) < 30:
        return result

    pos = valid[valid[signal_col] > 0]["fwd_return"]
    neg = valid[valid[signal_col] < 0]["fwd_return"]
    result["pos_n"], result["neg_n"] = len(pos), len(neg)
    result["pos_mean"] = pos.mean() if len(pos) else None
    result["neg_mean"] = neg.mean() if len(neg) else None
    result["corr"] = float(valid[signal_col].corr(valid["fwd_return"]))

    if len(pos) >= 2 and len(neg) >= 2:
        from scipy import stats as scistats
        _, pval = scistats.ttest_ind(pos, neg, equal_var=False)
        result["p_value"] = float(pval)

    current_value = float(valid[signal_col].iloc[-1])
    result["current_value"] = current_value
    result["current_regime"] = "양수" if current_value > 0 else "음수"
    return result


def compute_composite(df: pd.DataFrame, signal_cols: list[str], results: dict[str, dict]) -> tuple[pd.Series, dict]:
    raw_weights = {}
    total_abs = 0.0
    for col in signal_cols:
        r = results[col]
        w = abs(r["corr"]) if (r["p_value"] is not None and r["p_value"] < 0.05 and r["corr"] is not None) else 0.0
        raw_weights[col] = w
        total_abs += w

    composite = pd.Series(0.0, index=df.index)
    if total_abs == 0:
        return composite, raw_weights

    weights = {col: raw_weights[col] / total_abs for col in signal_cols}
    for col in signal_cols:
        r = results[col]
        sign_flip = -1.0 if (r["corr"] is not None and r["corr"] < 0) else 1.0
        composite = composite + weights[col] * sign_flip * np.sign(df[col].fillna(0))
    return composite, weights


@st.cache_data(ttl=6 * 3600)
def fetch_news_headlines(query: str, count: int = 6) -> list[str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(
        NAVER_NEWS_URL, params={"where": "news", "query": query, "sort": "1"}, headers=headers, timeout=10
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    headlines = [el.get_text(strip=True) for el in soup.select("span.sds-comps-text-type-headline1")]
    return headlines[:count]


@st.cache_data(ttl=6 * 3600)
def fetch_analyst_reports(ticker: str, count: int = 5) -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(
        NAVER_RESEARCH_URL, params={"searchType": "itemCode", "itemCode": ticker}, headers=headers, timeout=10
    )
    resp.encoding = "euc-kr"
    tables = pd.read_html(StringIO(resp.text))
    df = tables[0].dropna(subset=["제목"]).copy()
    df["작성일"] = pd.to_datetime(df["작성일"], format="%y.%m.%d").dt.strftime("%Y-%m-%d")
    return df[["제목", "증권사", "작성일"]].head(count)


@st.cache_data(ttl=1800)
def _fetch_board_page(ticker: str, page: int) -> list[dict]:
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.naver.com/"}
    resp = requests.get(
        NAVER_BOARD_URL, params={"code": ticker, "page": page}, headers=headers, timeout=10
    )
    resp.raise_for_status()
    resp.encoding = "utf-8"
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", class_="type2")
    if table is None:
        return []

    posts = []
    for tr in table.find_all("tr"):
        title_td = tr.find("td", class_="title")
        if title_td is None:
            continue
        a = title_td.find("a")
        if a is None:
            continue
        tds = tr.find_all("td")
        date_text = tds[0].get_text(strip=True) if tds else ""
        date_only = date_text.split(" ")[0].replace(".", "-") if date_text else ""
        posts.append({"날짜": date_only, "제목": a.get_text(strip=True)})
    return posts


@st.cache_data(ttl=1800)
def fetch_community_posts(ticker: str, count: int = 60) -> pd.DataFrame:
    all_posts: list[dict] = []
    max_pages = 20
    for page in range(1, max_pages + 1):
        posts = _fetch_board_page(ticker, page)
        if not posts:
            break
        all_posts.extend(posts)
        if len(all_posts) >= count:
            break
    return pd.DataFrame(all_posts[:count])


def classify_sentiment(posts_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, post in posts_df.iterrows():
        title = post["제목"]
        pos_hits = sum(1 for kw in POSITIVE_KEYWORDS if kw in title)
        neg_hits = sum(1 for kw in NEGATIVE_KEYWORDS if kw in title)
        if pos_hits > neg_hits:
            label = "긍정"
        elif neg_hits > pos_hits:
            label = "부정"
        else:
            label = "중립"
        rows.append({"날짜": post["날짜"], "제목": title, "심리": label})
    return pd.DataFrame(rows)


def _parse_dram_table(soup: BeautifulSoup, tbody_id: str, item_filter: set[str] | None = None) -> pd.DataFrame:
    tbody = soup.find("tbody", id=tbody_id)
    if tbody is None:
        return pd.DataFrame(columns=["품목", "평균가(USD)", "변동률(%)", "방향"])

    rows = []
    for tr in tbody.find_all("tr")[1:]:
        tds = tr.find_all("td")
        if len(tds) < 7:
            continue
        item = tds[0].get_text(strip=True)
        if item_filter is not None and item not in item_filter:
            continue
        avg_price = float(tds[5].get_text(strip=True))
        change_text = tds[6].get_text(strip=True).replace("%", "").strip()
        change_pct = float(change_text) if change_text else 0.0
        img = tds[6].find("img")
        src = (img.get("src") or "") if img else ""
        direction = "하락" if "down" in src else ("상승" if "up" in src else "보합")
        rows.append({"품목": item, "평균가(USD)": avg_price, "변동률(%)": change_pct, "방향": direction})
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def fetch_dram_module_prices() -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(DRAMEXCHANGE_URL, headers=headers, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    return _parse_dram_table(soup, "tb_ModuleSpotPrice")


@st.cache_data(ttl=3600)
def fetch_dram_chip_prices() -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(DRAMEXCHANGE_URL, headers=headers, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    return _parse_dram_table(
        soup, "tb_NationalDramSpotPrice",
        item_filter={
            "DDR5 16Gb (2Gx8) 4800/5600",
            "DDR5 16Gb (2Gx8) eTT",
            "DDR4 16Gb (2Gx8) 3200",
            "DDR4 16Gb (2Gx8) eTT",
        },
    )


def _signed_pct(row: pd.Series) -> str:
    sign = "+" if row["방향"] == "상승" else ("-" if row["방향"] == "하락" else "")
    return f"{sign}{row['변동률(%)']:.2f}%"


def save_dram_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    today = dt.date.today().isoformat()
    snapshot = df.copy()
    snapshot.insert(0, "날짜", today)

    os.makedirs(os.path.dirname(DRAM_HISTORY_FILE) or ".", exist_ok=True)
    try:
        if os.path.exists(DRAM_HISTORY_FILE):
            history = pd.read_csv(DRAM_HISTORY_FILE)
            history = pd.concat([history, snapshot], ignore_index=True)
            history = history.drop_duplicates(subset=["날짜", "품목"], keep="last")
        else:
            history = snapshot
        history.to_csv(DRAM_HISTORY_FILE, index=False)
        return history
    except OSError:
        return snapshot


BIGTECH_CIKS = {
    "Microsoft": "0000789019",
    "Alphabet(Google)": "0001652044",
    "Amazon": "0001018724",
    "Meta": "0001326801",
}
CAPEX_TAGS = ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"]
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


def _fetch_company_standalone_capex_quarters(cik: str) -> list[dict]:
    """SEC XBRL companyfacts에서 분기별 단독 capex(설비투자)를 계산한다.
    일부 기업은 분기 단독 수치를, 일부는 연초 누적(YTD) 수치를 보고하므로,
    같은 회계연도 시작일을 공유하는 누적치들을 서로 빼서 분기 단독값을 구한다."""
    headers = {"User-Agent": "PersonalDashboard contact@example.com"}
    resp = requests.get(SEC_COMPANYFACTS_URL.format(cik=cik), headers=headers, timeout=20)
    resp.raise_for_status()
    facts = resp.json().get("facts", {}).get("us-gaap", {})

    dedup = {}
    for tag in CAPEX_TAGS:
        if tag not in facts:
            continue
        for e in facts[tag]["units"].get("USD", []):
            if e.get("form") not in ("10-Q", "10-K"):
                continue
            start, end = e.get("start"), e.get("end")
            if not start or not end:
                continue
            days = (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days
            if days < 60 or days > 380:
                continue
            key = (start, end)
            if key not in dedup or e.get("filed", "") > dedup[key].get("filed", ""):
                dedup[key] = e

    by_start = {}
    for (start, end), e in dedup.items():
        by_start.setdefault(start, []).append({"end": end, "val": e["val"], "filed": e.get("filed", "")})

    candidates = []
    for start, group in by_start.items():
        group_sorted = sorted(group, key=lambda g: g["end"])
        prev_val, prev_end = 0.0, start
        for g in group_sorted:
            duration = (dt.date.fromisoformat(g["end"]) - dt.date.fromisoformat(prev_end)).days
            candidates.append({"end": g["end"], "val": g["val"] - prev_val, "duration": duration, "filed": g["filed"]})
            prev_val, prev_end = g["val"], g["end"]

    by_end = {}
    for c in candidates:
        if 75 <= c["duration"] <= 105:
            by_end.setdefault(c["end"], []).append(c)

    quarters = [min(group, key=lambda c: (abs(c["duration"] - 91), -len(c["filed"]))) for group in by_end.values()]
    return sorted(quarters, key=lambda q: q["end"])


@st.cache_data(ttl=24 * 3600)
def fetch_bigtech_capex() -> pd.DataFrame:
    """빅테크(마이크로소프트/구글/아마존/메타)의 분기별 설비투자(capex) 실적을 SEC 공시(XBRL)에서 가져온다."""
    rows = []
    for name, cik in BIGTECH_CIKS.items():
        try:
            quarters = _fetch_company_standalone_capex_quarters(cik)
        except Exception:
            continue
        for q in quarters:
            rows.append({"기업": name, "분기말": q["end"], "capex_USD": q["val"]})
    if not rows:
        return pd.DataFrame(columns=["기업", "분기말", "capex_USD"])
    df = pd.DataFrame(rows)
    df["분기말"] = pd.to_datetime(df["분기말"])
    df = df[df["분기말"] >= "2022-01-01"].sort_values(["분기말", "기업"])
    return df.reset_index(drop=True)


@st.cache_data(ttl=3600)
def generate_ai_analysis(
    stock_label: str,
    time_label: str,
    price_summary: str,
    supply_summary: str,
    headlines: list[str],
    reports_md: str,
    dram_summary: str,
    community_summary: str,
    composite_summary: str,
    overheat_summary: str,
    futures_summary: str,
) -> str:
    client = genai.Client()
    prompt = f"""다음은 {time_label} 기준 {stock_label} 관련 데이터입니다.

[오늘 주가]
{price_summary}

[최근 수급 동향]
{supply_summary}

[통합 매수/매도 신호 — 실험적 백테스트, 매매 신호 아님]
{composite_summary}

[가격 과열도 백테스트 — 과거 통계 참고용, 매매 신호 아님]
{overheat_summary}

[코스피200 선물 외국인 순매도 하락 경보 — 과거 통계 참고용, 매매 신호 아님]
{futures_summary}

[DRAM 현물가]
{dram_summary}

[관련 뉴스 헤드라인]
{chr(10).join(f"- {h}" for h in headlines) if headlines else "(수집된 뉴스 없음)"}

[최근 애널리스트 리포트]
{reports_md if reports_md else "(수집된 리포트 없음)"}

[투자자 커뮤니티 심리 — 참고용, 사실 아님]
{community_summary}

위 정보를 종합해서 오늘 {stock_label} 주가 변동에 영향을 준 것으로 보이는 주요 요인들을 3~5개 항목으로 간결하게 분석해줘.
통합 신호, 가격 과열도, 선물 하락 경보, 커뮤니티 심리는 참고 정도로만 가볍게 언급하고, 핵심 근거로 삼지 마.
투자 조언이나 매수/매도 추천은 하지 말고, 객관적인 요인 분석만 한국어로 작성해줘."""

    interaction = client.interactions.create(model=GEMINI_MODEL, input=prompt)
    return interaction.output_text or "AI가 응답을 생성하지 못했습니다."


@st.fragment(run_every=refresh_sec)
def render_current_price():
    st.subheader("현재가 (시세 지연)")
    try:
        data = fetch_current_price(TICKER)
        close_price = int(data["closePriceRaw"])
        change = int(data["compareToPreviousClosePriceRaw"])
        change_pct = float(data["fluctuationsRatioRaw"])
        market_status = data["marketStatus"]
        try:
            updated_at = pd.to_datetime(data["localTradedAt"]).strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            updated_at = data["localTradedAt"]

        price_info = data.get("integratedPriceInfo", {})
        open_p = price_info.get("openPrice", "-")
        high_p = price_info.get("highPrice", "-")
        low_p = price_info.get("lowPrice", "-")
        volume = price_info.get("accumulatedTradingVolume", "-")

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(
                label=f"{data['stockName']} ({TICKER})",
                value=f"{close_price:,}원",
                delta=f"{change:+,}원 ({change_pct:+.2f}%)",
                delta_color="inverse",
            )
            st.caption(f"시장상태: {market_status} · 갱신시각: {updated_at}")
        with col2:
            st.table(
                pd.DataFrame(
                    {"시가": [open_p], "고가": [high_p], "저가": [low_p], "거래량": [volume]}
                ).T.rename(columns={0: "값"})
            )
        st.session_state["current_price_summary"] = f"종가 {close_price:,}원, 전일대비 {change:+,}원 ({change_pct:+.2f}%)"
    except Exception as e:
        st.error(f"현재가 조회에 실패했습니다: {e}")
        st.session_state["current_price_summary"] = "현재가 데이터를 가져오지 못함"


render_current_price()

REFRESH_CHECK_INTERVAL_SEC = 60  # 정시가 됐는지 확인하는 주기 (실제 새로고침은 매 정시에 한 번만)


def _refresh_all_indicator_caches(progress_bar=None) -> None:
    """모든 지표용 캐시를 비우고 다시 채운다. progress_bar가 주어지면 단계별 진행률을 표시한다."""
    fetch_investor_netbuy.clear()
    fetch_backtest_history.clear()
    fetch_latest_bars.clear()
    fetch_futures_foreign_history.clear()
    fetch_latest_futures_bars.clear()
    fetch_yahoo_history.clear()
    fetch_dram_module_prices.clear()
    fetch_dram_chip_prices.clear()
    fetch_bigtech_capex.clear()

    steps = [
        ("투자자별 순매수 이력", lambda: fetch_investor_netbuy(TICKER, lookback_days)),
        ("종목 시세·수급 이력", lambda: fetch_backtest_history(TICKER, target_days=700)),
        ("최근 시세(오늘자)", lambda: fetch_latest_bars(TICKER)),
        ("코스피200 선물 외국인 이력", lambda: fetch_futures_foreign_history(target_days=700)),
        ("코스피200 선물 외국인 오늘자", lambda: fetch_latest_futures_bars()),
        ("미국 반도체지수(SOX)", lambda: fetch_yahoo_history("SOX")),
        ("달러인덱스(DXY)", lambda: fetch_yahoo_history("DXY")),
        ("DRAM 모듈 시세", lambda: fetch_dram_module_prices()),
        ("DRAM 칩 시세", lambda: fetch_dram_chip_prices()),
        ("빅테크 Capex", lambda: fetch_bigtech_capex()),
    ]
    for i, (label, fetch_fn) in enumerate(steps):
        if progress_bar is not None:
            progress_bar.progress(i / len(steps), text=f"{label} 수집 중... ({i + 1}/{len(steps)})")
        try:
            fetch_fn()
        except Exception:
            pass  # 개별 항목이 실패해도 새로고침은 계속 진행하고, 각 탭에서 자체적으로 에러를 표시한다.
    if progress_bar is not None:
        progress_bar.progress(1.0, text="완료! 화면을 갱신합니다...")


def _floor_to_hour(t: dt.datetime) -> dt.datetime:
    return t.replace(minute=0, second=0, microsecond=0)


@st.cache_resource
def _get_global_refresh_state() -> dict:
    """세션마다 따로 있는 st.session_state와 달리, 서버 전체에서 공유되는 새로고침 상태.
    브라우저를 새로고침하거나 새 세션이 열려도 '진짜 마지막 데이터 갱신 시각'을 그대로 유지한다."""
    now = dt.datetime.now()
    return {"last_hour": _floor_to_hour(now), "last_time": now}


@st.fragment(run_every=REFRESH_CHECK_INTERVAL_SEC)
def _auto_refresh_indicators():
    state = _get_global_refresh_state()
    current_hour = _floor_to_hour(dt.datetime.now())
    if current_hour > state["last_hour"]:
        state["last_hour"] = current_hour
        state["last_time"] = dt.datetime.now()
        _refresh_all_indicator_caches()
        st.rerun(scope="app")


_auto_refresh_indicators()

st.caption(
    f"현재가는 {refresh_sec}초마다, 나머지 지표/그래프는 서버시간 기준 매 정시(예: 1:00, 2:00)에 자동으로 최신 데이터로 갱신됩니다. "
    "지금 바로 갱신하려면 아래 버튼을 눌러주세요."
)
st.caption(f"🕒 지표 마지막 새로고침: {_get_global_refresh_state()['last_time'].strftime('%Y-%m-%d %H:%M:%S')}")
if st.button("🔄 지표 새로고침 (모든 그래프를 현재 시점 데이터로 갱신)"):
    state = _get_global_refresh_state()
    state["last_hour"] = _floor_to_hour(dt.datetime.now())
    state["last_time"] = dt.datetime.now()
    progress_bar = st.progress(0, text="새로고침 준비 중...")
    _refresh_all_indicator_caches(progress_bar)
    st.rerun()

st.divider()


def _add_regime_shading(fig, dates: pd.Series, is_active: pd.Series, color: str, opacity: float = 0.08) -> None:
    """is_active가 True인 연속 구간들을 색상 음영으로 표시한다."""
    dates = dates.reset_index(drop=True)
    is_active = is_active.fillna(False).reset_index(drop=True)
    start = None
    for i in range(len(dates)):
        if is_active.iloc[i] and start is None:
            start = dates.iloc[i]
        elif not is_active.iloc[i] and start is not None:
            fig.add_vrect(x0=start, x1=dates.iloc[i], fillcolor=color, opacity=opacity, line_width=0, layer="below")
            start = None
    if start is not None:
        fig.add_vrect(x0=start, x1=dates.iloc[len(dates) - 1], fillcolor=color, opacity=opacity, line_width=0, layer="below")


def _add_downtrend_shading(fig, dates: pd.Series, window_decline: pd.Series, threshold: float) -> None:
    """직전 DOWNTREND_WINDOW일 대비 하락률(window_decline)이 threshold보다 더 떨어진 구간(하락장)을 음영 처리한다.
    단순 기울기 부호(<0)만 쓰면 횡보장의 미세한 노이즈도 하락장으로 잡히므로, 하락 '강도'를 threshold로 걸러낸다."""
    _add_regime_shading(fig, dates, window_decline < threshold, "red", opacity=0.08)


def _window_decline(price: pd.Series, window: int) -> pd.Series:
    return price / price.shift(window) - 1


def _render_dram_trend_chart(history: pd.DataFrame, items: list[str], key_prefix: str, chart_key: str) -> None:
    """가격대가 서로 다른 여러 품목을 한 그래프에 함께 그리면 스케일 차이로 잘 안 보이므로,
    그래프 우측에서 품목을 하나 골라 해당 품목만 그린다."""
    chart_col, toggle_col = st.columns([5, 1])
    with toggle_col:
        st.caption("표시 품목")
        selected = st.radio("표시 품목", items, key=f"{key_prefix}_select", label_visibility="collapsed")
    with chart_col:
        hist = history[history["품목"] == selected]
        fig = px.line(hist, x="날짜", y="평균가(USD)", markers=True)
        fig.update_layout(title=selected, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key=chart_key)


DOWNTREND_WINDOW = 20

tab_labels = [
    "수급 현황",
    "가격 과열도",
    "선물 경보",
    "통합 신호",
    "하락 조기신호",
    "상승 조기신호",
    "DRAM 시세",
    "빅테크 Capex",
    "커뮤니티",
    "AI 분석",
]
tabs = st.tabs(tab_labels)

with tabs[0]:

    investor_df = pd.DataFrame()

    st.subheader(f"최근 {lookback_days}일 투자자별 순매수 거래량")
    st.caption("개인 순매수는 기관·외국인 합산의 잔차로 추정한 값입니다 (기타법인 등 소액 오차 포함 가능).")
    try:
        df = fetch_investor_netbuy(TICKER, lookback_days)
        investor_df = df
        if df.empty:
            st.warning("투자자 수급 데이터를 가져오지 못했습니다. 잠시 후 다시 시도해 주세요.")
        else:
            df_long = df.reset_index().melt(id_vars="날짜", var_name="투자자", value_name="순매수")
            fig_bar = px.bar(
                df_long, x="날짜", y="순매수", color="투자자", barmode="group", title="일별 순매수 거래량(주)"
            )
            st.plotly_chart(fig_bar, use_container_width=True, key="chart_investor_bar")

            df_cum = df.cumsum()
            df_cum_long = df_cum.reset_index().melt(id_vars="날짜", var_name="투자자", value_name="누적 순매수")
            fig_line = px.line(
                df_cum_long, x="날짜", y="누적 순매수", color="투자자", title="누적 순매수 추세"
            )
            st.plotly_chart(fig_line, use_container_width=True, key="chart_investor_line")

            slopes = {col: calc_slope(df_cum[col]) for col in df_cum.columns}
            slope_df = pd.DataFrame(
                {
                    "투자자": list(slopes.keys()),
                    "일평균 추세(기울기, 주/일)": [f"{v:,.0f}" for v in slopes.values()],
                    "방향": [
                        "매수 우위" if v > 0 else ("매도 우위" if v < 0 else "중립")
                        for v in slopes.values()
                    ],
                }
            )
            st.dataframe(slope_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"투자자 수급 데이터 조회에 실패했습니다: {e}")

with tabs[1]:

    OVERHEAT_MA_WINDOW = 60
    OVERHEAT_HORIZON = 20
    OVERHEAT_QUANTILE = 0.2
    OVERHEAT_DRAWDOWN_THRESHOLD = 0.07

    overheat_summary = "가격 과열도 백테스트 미실행"

    st.subheader("가격 과열도 백테스트 (참고용)")
    st.caption(
        f"종가가 {OVERHEAT_MA_WINDOW}일 이동평균 대비 상위 {OVERHEAT_QUANTILE:.0%}(과열)로 괴리되어 있을 때, "
        f"향후 {OVERHEAT_HORIZON}거래일 내 {OVERHEAT_DRAWDOWN_THRESHOLD:.0%} 이상 하락할 확률을 나머지 구간과 비교합니다. "
        "코스피 시가총액 상위 10개 종목 전체(51개 큰폭 하락 사례)에서 공통적으로 확인된 종목 무관 범용 지표입니다. "
        "매매 신호가 아니라 참고용 통계입니다."
    )
    try:
        overheat_hist = fetch_backtest_history_live(TICKER, target_days=700)
        if len(overheat_hist) < 80:
            st.warning("백테스트에 충분한 과거 데이터가 없습니다.")
        else:
            overheat_result = run_overheat_backtest(
                overheat_hist, "종가", OVERHEAT_MA_WINDOW, OVERHEAT_HORIZON,
                quantile=OVERHEAT_QUANTILE, drawdown_threshold=OVERHEAT_DRAWDOWN_THRESHOLD,
            )
            if overheat_result["n"] < 30:
                st.warning("백테스트에 충분한 표본이 없습니다.")
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("표본 수", f"{overheat_result['n']}일")
                col2.metric(
                    f"과열(상위 {OVERHEAT_QUANTILE:.0%})일 때 하락장 확률",
                    f"{overheat_result['hi_rate']:.1%}" if overheat_result["hi_rate"] is not None else "N/A",
                )
                col3.metric(
                    "나머지 구간 하락장 확률",
                    f"{overheat_result['rest_rate']:.1%}" if overheat_result["rest_rate"] is not None else "N/A",
                )
                st.caption(f"참고: 전체 기간 기저 하락장 확률 {overheat_result['base_rate']:.1%} (표본 {overheat_result['hi_n']}일/{overheat_result['rest_n']}일)")
                if overheat_result["p_value"] is not None:
                    sig_note = "통계적으로 유의함" if overheat_result["p_value"] < 0.05 else "통계적으로 유의하지 않음 — 참고만"
                    st.caption(f"p-value: {overheat_result['p_value']:.4f} ({sig_note})")

                overheat_hist_ma = overheat_hist.copy()
                overheat_hist_ma["MA"] = overheat_hist_ma["종가"].rolling(OVERHEAT_MA_WINDOW).mean()
                overheat_hist_ma["괴리율"] = overheat_hist_ma["종가"] / overheat_hist_ma["MA"] - 1
                overheat_chart_df = overheat_hist_ma.dropna(subset=["괴리율"])

                show_overheat_shading = st.checkbox(
                    "과열 구간 음영 표시 (상위 5/10/15/20% 그라데이션)", value=True, key="show_overheat_shading"
                )
                fig_overheat = make_subplots(specs=[[{"secondary_y": True}]])
                fig_overheat.add_trace(
                    go.Scatter(x=overheat_chart_df["날짜"], y=overheat_chart_df["괴리율"], name=f"종가/{OVERHEAT_MA_WINDOW}일선 괴리율", line=dict(color="#1f77b4")),
                    secondary_y=False,
                )
                fig_overheat.add_trace(
                    go.Scatter(x=overheat_chart_df["날짜"], y=overheat_chart_df["종가"], name="종가", line=dict(color="#d62728")),
                    secondary_y=True,
                )
                fig_overheat.add_hline(y=0, line_dash="dash", line_color="gray", secondary_y=False)
                OVERHEAT_GRADIENT_BANDS = [0.20, 0.15, 0.10, 0.05]
                if len(overheat_chart_df) > 0:
                    band_cutoffs = {
                        q: float(overheat_chart_df["괴리율"].quantile(1 - q)) for q in OVERHEAT_GRADIENT_BANDS
                    }
                    for q in OVERHEAT_GRADIENT_BANDS:
                        fig_overheat.add_hline(
                            y=band_cutoffs[q], line_dash="dot", line_color="orange", secondary_y=False,
                            annotation_text=f"상위 {q:.0%}", annotation_position="top right",
                        )
                    if show_overheat_shading:
                        for q in OVERHEAT_GRADIENT_BANDS:
                            _add_regime_shading(
                                fig_overheat, overheat_chart_df["날짜"],
                                overheat_chart_df["괴리율"] >= band_cutoffs[q], "orange", opacity=0.06,
                            )
                fig_overheat.update_layout(title=f"종가/{OVERHEAT_MA_WINDOW}일 이동평균 괴리율 vs 주가")
                fig_overheat.update_yaxes(title_text="괴리율", secondary_y=False)
                fig_overheat.update_yaxes(title_text="종가(원)", secondary_y=True)
                fig_overheat.update_xaxes(rangeslider_visible=True)
                st.plotly_chart(fig_overheat, use_container_width=True, key="chart_overheat")
                caption = "차트 하단 슬라이더를 드래그하면 보고 싶은 기간만 확대해서 볼 수 있습니다."
                if show_overheat_shading:
                    caption += " 주황 음영은 괴리율 상위 20/15/10/5% 구간이 겹겹이 쌓인 것으로, 상위 5%에 가까울수록 색이 진해집니다."
                st.caption(caption)

                st.info(f"현재 상태: **{overheat_result['current_regime']}** (괴리율 {overheat_result['current_deviation']:+.1%}, 과거 통계 대비 참고용, 매매 신호 아님)")

                overheat_summary = (
                    f"가격 과열도 백테스트({OVERHEAT_MA_WINDOW}일선 괴리율, 표본 {overheat_result['n']}일): "
                    f"과열(상위 {OVERHEAT_QUANTILE:.0%})일 때 향후 {OVERHEAT_HORIZON}일 내 {OVERHEAT_DRAWDOWN_THRESHOLD:.0%} 이상 하락 확률 "
                    f"{overheat_result['hi_rate']:.1%} (나머지 구간 {overheat_result['rest_rate']:.1%}, p={overheat_result['p_value']:.4f}). "
                    f"현재 상태: {overheat_result['current_regime']} (괴리율 {overheat_result['current_deviation']:+.1%}). (통계 참고용, 매매 신호 아님)"
                )

                st.divider()
                st.subheader("과열도 임계값 매수·매도 전략 백테스트 (참고용)")
                st.caption(
                    "괴리율이 임계값 이상일 때 매수하고, 0%로 되돌아오면 매도하는 단순 전략을 buy & hold와 비교합니다. "
                    "기간과 임계값을 직접 조절해볼 수 있습니다. 거래비용·세금·슬리피지는 반영되지 않았고, 표본이 적어 참고용입니다."
                )
                strat_col1, strat_col2 = st.columns(2)
                with strat_col1:
                    strategy_threshold_pct = st.slider(
                        "매수 임계값 (괴리율 ≥)", min_value=5, max_value=40, value=15, step=1,
                        format="%d%%", key="overheat_strategy_threshold",
                    )
                    strategy_threshold = strategy_threshold_pct / 100
                with strat_col2:
                    strategy_months = st.slider(
                        "백테스트 기간 (최근 N개월)", min_value=1, max_value=24, value=6, step=1,
                        key="overheat_strategy_months",
                    )
                period_start = overheat_hist_ma["날짜"].max() - pd.Timedelta(days=strategy_months * 30.44)
                strategy_result = run_overheat_threshold_strategy(overheat_hist_ma, strategy_threshold, period_start)

                if strategy_result["n_days"] < 10 or strategy_result["buy_hold_return"] is None:
                    st.warning("선택한 기간에 데이터가 부족합니다.")
                else:
                    scol1, scol2, scol3 = st.columns(3)
                    scol1.metric(
                        "buy & hold 수익률",
                        f"{strategy_result['buy_hold_return']:+.1%}",
                    )
                    if strategy_result["cum_return"] is not None:
                        delta = strategy_result["cum_return"] - strategy_result["buy_hold_return"]
                        scol2.metric(
                            "전략 누적 수익률", f"{strategy_result['cum_return']:+.1%}",
                            delta=f"{delta:+.1%} vs buy&hold",
                        )
                    else:
                        scol2.metric("전략 누적 수익률", "거래 없음")
                    scol3.metric("거래 횟수", f"{len(strategy_result['trades'])}회")

                    st.caption(
                        f"기간: {strategy_result['period_start'].date()} ~ {strategy_result['period_end'].date()}"
                        + (
                            f" (마지막 매수 포지션 미청산, 평가손익 {strategy_result['unrealized_return']:+.1%} 포함)"
                            if strategy_result["still_open"] and strategy_result["unrealized_return"] is not None
                            else ""
                        )
                    )

                    if strategy_result["trades"]:
                        trades_df = pd.DataFrame(strategy_result["trades"])
                        trades_df_display = trades_df.assign(
                            매수일=trades_df["buy_date"].dt.strftime("%Y-%m-%d"),
                            매도일=trades_df["sell_date"].dt.strftime("%Y-%m-%d"),
                            매수가=trades_df["buy_price"].map(lambda v: f"{v:,.0f}"),
                            매도가=trades_df["sell_price"].map(lambda v: f"{v:,.0f}"),
                            수익률=trades_df["ret"].map(lambda v: f"{v:+.1%}"),
                        )[["매수일", "매수가", "매도일", "매도가", "수익률"]]
                        st.dataframe(trades_df_display, use_container_width=True, hide_index=True)

                    equity_curve = strategy_result["equity_curve"]
                    if equity_curve is not None and not equity_curve.empty:
                        buy_hold_curve = test_period_df = overheat_hist_ma[
                            overheat_hist_ma["날짜"] >= period_start
                        ].reset_index(drop=True)
                        fig_strategy = go.Figure()
                        fig_strategy.add_trace(go.Scatter(
                            x=equity_curve["날짜"], y=(equity_curve["자산가치"] - 1) * 100,
                            name="전략", line=dict(color="#2ca02c"),
                        ))
                        if not buy_hold_curve.empty:
                            bh_base = buy_hold_curve.iloc[0]["종가"]
                            fig_strategy.add_trace(go.Scatter(
                                x=buy_hold_curve["날짜"], y=(buy_hold_curve["종가"] / bh_base - 1) * 100,
                                name="buy & hold", line=dict(color="#888888", dash="dot"),
                            ))
                        fig_strategy.update_layout(title="전략 vs buy & hold 누적 수익률 (%)")
                        fig_strategy.update_yaxes(title_text="누적 수익률 (%)")
                        st.plotly_chart(fig_strategy, use_container_width=True, key="chart_overheat_strategy")
    except Exception as e:
        st.error(f"가격 과열도 백테스트에 실패했습니다: {e}")

with tabs[2]:

    FUTURES_WINDOW = 15
    FUTURES_HORIZON = 10
    FUTURES_QUANTILE = 0.2
    FUTURES_DRAWDOWN_THRESHOLD = 0.07

    futures_summary = "코스피200 선물 하락 신호 백테스트 미실행"

    st.subheader("코스피200 선물 외국인 순매도 — 하락 조기 경보 (참고용)")
    st.caption(
        f"코스피200 선물 시장에서 외국인이 {FUTURES_WINDOW}일간 강하게 순매도(하위 {FUTURES_QUANTILE:.0%})할 때, "
        f"현재 종목이 향후 {FUTURES_HORIZON}거래일 내 {FUTURES_DRAWDOWN_THRESHOLD:.0%} 이상 하락할 확률을 나머지 구간과 비교합니다. "
        "자체 검증 결과 하락 예측에는 유의미했지만, 반대(외국인 순매수)가 상승을 예측하지는 못해 "
        "하락 경보 용도로만 제공합니다. 매매 신호가 아닙니다."
    )
    try:
        futures_hynix_hist = fetch_backtest_history_live(TICKER, target_days=700)
        futures_hist = fetch_futures_foreign_history_live(target_days=700)
        if len(futures_hynix_hist) < 80 or len(futures_hist) < 80:
            st.warning("백테스트에 충분한 과거 데이터가 없습니다.")
        else:
            futures_result = run_futures_decline_backtest(
                futures_hynix_hist["종가"], futures_hynix_hist["날짜"], futures_hist.set_index("날짜")["선물외국인"].reindex(futures_hynix_hist["날짜"]).reset_index(drop=True),
                FUTURES_WINDOW, FUTURES_HORIZON, quantile=FUTURES_QUANTILE, drawdown_threshold=FUTURES_DRAWDOWN_THRESHOLD,
            )
            if futures_result["n"] < 30:
                st.warning("백테스트에 충분한 표본이 없습니다.")
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("표본 수", f"{futures_result['n']}일")
                col2.metric(
                    f"선물 외국인 강한 매도(하위 {FUTURES_QUANTILE:.0%})일 때 하락 확률",
                    f"{futures_result['lo_rate']:.1%}" if futures_result["lo_rate"] is not None else "N/A",
                )
                col3.metric(
                    "나머지 구간 하락 확률",
                    f"{futures_result['rest_rate']:.1%}" if futures_result["rest_rate"] is not None else "N/A",
                )
                st.caption(f"참고: 전체 기간 기저 하락 확률 {futures_result['base_rate']:.1%} (표본 {futures_result['lo_n']}일/{futures_result['rest_n']}일)")
                if futures_result["p_value"] is not None:
                    sig_note = "통계적으로 유의함" if futures_result["p_value"] < 0.05 else "통계적으로 유의하지 않음 — 참고만"
                    st.caption(f"p-value: {futures_result['p_value']:.4f} ({sig_note})")

                futures_flow_aligned = futures_hist.set_index("날짜")["선물외국인"].reindex(futures_hynix_hist["날짜"]).reset_index(drop=True)
                futures_chart_df = pd.DataFrame(
                    {
                        "날짜": futures_hynix_hist["날짜"],
                        "기울기": _rolling_slope(futures_flow_aligned, FUTURES_WINDOW),
                        "종가": futures_hynix_hist["종가"],
                    }
                ).dropna(subset=["기울기"])

                show_futures_shading = st.checkbox(
                    f"강한 매도(하위 {FUTURES_QUANTILE:.0%}) 구간 음영 표시", value=True, key="show_futures_shading"
                )
                fig_futures = make_subplots(specs=[[{"secondary_y": True}]])
                fig_futures.add_trace(
                    go.Scatter(x=futures_chart_df["날짜"], y=futures_chart_df["기울기"], name="선물 외국인 순매수 기울기", line=dict(color="#1f77b4")),
                    secondary_y=False,
                )
                fig_futures.add_trace(
                    go.Scatter(x=futures_chart_df["날짜"], y=futures_chart_df["종가"], name="종가", line=dict(color="#d62728")),
                    secondary_y=True,
                )
                fig_futures.add_hline(y=0, line_dash="dash", line_color="gray", secondary_y=False)
                if futures_result["lo_cutoff"] is not None:
                    fig_futures.add_hline(
                        y=futures_result["lo_cutoff"], line_dash="dot", line_color="orange", secondary_y=False,
                        annotation_text=f"하위 {FUTURES_QUANTILE:.0%} 기준", annotation_position="bottom right",
                    )
                    if show_futures_shading:
                        _add_regime_shading(
                            fig_futures, futures_chart_df["날짜"],
                            futures_chart_df["기울기"] <= futures_result["lo_cutoff"], "orange", opacity=0.15,
                        )
                fig_futures.update_layout(title=f"코스피200 선물 외국인 {FUTURES_WINDOW}일 누적 순매수 기울기 vs 주가")
                fig_futures.update_yaxes(title_text="기울기", secondary_y=False)
                fig_futures.update_yaxes(title_text="종가(원)", secondary_y=True)
                fig_futures.update_xaxes(rangeslider_visible=True)
                st.plotly_chart(fig_futures, use_container_width=True, key="chart_futures")
                futures_caption = "차트 하단 슬라이더를 드래그하면 보고 싶은 기간만 확대해서 볼 수 있습니다."
                if show_futures_shading:
                    futures_caption += f" 주황 음영은 선물 외국인 기울기가 하위 {FUTURES_QUANTILE:.0%}(강한 매도) 구간입니다."
                st.caption(futures_caption)

                st.info(
                    f"현재 상태: **{futures_result['current_regime']}** "
                    f"(선물 외국인 {FUTURES_WINDOW}일 누적 순매수 기울기: {futures_result['current_slope']:,.0f}, 과거 통계 대비 참고용, 매매 신호 아님)"
                )

                futures_summary = (
                    f"코스피200 선물 외국인 순매도 하락 신호 백테스트(표본 {futures_result['n']}일): "
                    f"강한 매도(하위 {FUTURES_QUANTILE:.0%})일 때 향후 {FUTURES_HORIZON}일 내 {FUTURES_DRAWDOWN_THRESHOLD:.0%} 이상 하락 확률 "
                    f"{futures_result['lo_rate']:.1%} (나머지 구간 {futures_result['rest_rate']:.1%}, p={futures_result['p_value']:.4f}). "
                    f"현재 상태: {futures_result['current_regime']}. (하락 경보 전용, 매매 신호 아님)"
                )
    except Exception as e:
        st.error(f"코스피200 선물 하락 신호 백테스트에 실패했습니다: {e}")

with tabs[3]:

    COMPOSITE_HORIZON = 10
    COMPOSITE_SIGNAL_LABELS = {
        "기관_기울기": "기관 수급",
        "SOX_기울기": "미국 반도체지수(SOX)",
        "DXY_기울기": "달러인덱스(DXY)",
    }

    composite_summary = "통합 신호 미실행"

    st.subheader("통합 매수/매도 신호 (실험적)")
    st.caption(
        "기관 수급, 미국 반도체지수(SOX), 달러인덱스(DXY)를 통계적 유의성에 따라 가중합산한 실험적 종합 신호입니다 "
        "(다른 지표들은 백테스트 결과 상관관계가 낮거나 정보가 중복돼 제외했습니다). "
        "매매 신호가 아니라 백테스트 기반 참고 자료이며, 동일 기간 데이터로 가중치를 정한 인-샘플 결과라 미래 예측력을 보장하지 않습니다."
    )

    try:
        dataset = build_composite_dataset(TICKER)
        signal_cols = list(COMPOSITE_SIGNAL_LABELS.keys())
        results = {col: backtest_signal(dataset, col, COMPOSITE_HORIZON) for col in signal_cols}
        composite_series, weights = compute_composite(dataset, signal_cols, results)
        dataset["composite"] = composite_series
        composite_result = backtest_signal(dataset, "composite", COMPOSITE_HORIZON)
        composite_df = dataset[["날짜", "composite", "종가"]].dropna()

        rows = [
            {
                "지표": label,
                "상관계수": f"{results[col]['corr']:.3f}" if results[col]["corr"] is not None else "N/A",
                "p-value": f"{results[col]['p_value']:.4f}" if results[col]["p_value"] is not None else "N/A",
                "가중치": f"{weights.get(col, 0):.0%}",
                "현재 상태": results[col]["current_regime"] or "N/A",
            }
            for col, label in COMPOSITE_SIGNAL_LABELS.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("**종합 신호 (통계적 유의성 기반 가중합산, 자체 백테스트 검증)**")
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "종합 점수", f"{composite_result['current_value']:+.2f}" if composite_result["current_value"] is not None else "N/A"
        )
        col2.metric(
            f"종합 신호 양수일 때 {COMPOSITE_HORIZON}일 후 평균수익률",
            f"{composite_result['pos_mean']:+.2%}" if composite_result["pos_mean"] is not None else "N/A",
        )
        col3.metric(
            f"종합 신호 음수일 때 {COMPOSITE_HORIZON}일 후 평균수익률",
            f"{composite_result['neg_mean']:+.2%}" if composite_result["neg_mean"] is not None else "N/A",
        )

        if composite_result["p_value"] is not None:
            sig_note = "통계적으로 유의함" if composite_result["p_value"] < 0.05 else "통계적으로 유의하지 않음 — 참고만"
            corr_text = f"{composite_result['corr']:.4f}" if composite_result["corr"] is not None else "N/A"
            st.caption(f"종합 신호 상관계수: {corr_text} · p-value: {composite_result['p_value']:.4f} ({sig_note})")

        if len(composite_df) > 0:
            show_downtrend_composite = st.checkbox(
                "하락장 구간 음영 표시", value=True, key="show_downtrend_composite"
            )
            downtrend_pct_composite = st.slider(
                f"하락장 판단 기준 (최근 {DOWNTREND_WINDOW}일간 하락률)",
                min_value=1, max_value=20, value=5, step=1, format="%d%%",
                key="downtrend_pct_composite", disabled=not show_downtrend_composite,
            )
            fig_composite = make_subplots(specs=[[{"secondary_y": True}]])
            fig_composite.add_trace(
                go.Scatter(x=composite_df["날짜"], y=composite_df["composite"], name="종합 신호", line=dict(color="#1f77b4")),
                secondary_y=False,
            )
            fig_composite.add_trace(
                go.Scatter(x=composite_df["날짜"], y=composite_df["종가"], name="종가", line=dict(color="#d62728")),
                secondary_y=True,
            )
            fig_composite.add_hline(y=0, line_dash="dash", line_color="gray", secondary_y=False)
            if show_downtrend_composite:
                decline = _window_decline(composite_df["종가"], DOWNTREND_WINDOW)
                _add_downtrend_shading(fig_composite, composite_df["날짜"], decline, -downtrend_pct_composite / 100)
            fig_composite.update_layout(title="종합 매수/매도 신호 vs 주가")
            fig_composite.update_yaxes(title_text="종합 신호", secondary_y=False)
            fig_composite.update_yaxes(title_text="종가(원)", secondary_y=True)
            fig_composite.update_xaxes(rangeslider_visible=True)
            st.plotly_chart(fig_composite, use_container_width=True, key="chart_composite")
            composite_caption = "차트 하단 슬라이더를 드래그하면 보고 싶은 기간만 확대해서 볼 수 있습니다."
            if show_downtrend_composite:
                composite_caption += f" 빨간 음영 구간은 주가가 최근 {DOWNTREND_WINDOW}일간 {downtrend_pct_composite}% 이상 하락한 하락장 구간입니다."
            st.caption(composite_caption)

        regime_label = "매수 우위" if (composite_result["current_value"] or 0) > 0 else "매도 우위"
        st.info(f"현재 종합 신호: **{regime_label}** (여러 지표의 통계적 가중합산, 매매 신호 아님 — 참고용)")

        if composite_result["p_value"] is not None:
            composite_summary = (
                f"통합 신호(기관 수급+미국 반도체지수 SOX, 유의성 가중합산): 현재 {regime_label}. "
                f"상관계수 {composite_result['corr']:.3f}, "
                f"종합 신호 양수일 때 {COMPOSITE_HORIZON}일 후 평균 {composite_result['pos_mean']:+.2%}, "
                f"음수일 때 {composite_result['neg_mean']:+.2%} (p={composite_result['p_value']:.4f}, 인-샘플 결과)."
            )
        else:
            composite_summary = "통합 신호 표본 부족으로 계산되지 않음"
    except Exception as e:
        st.error(f"통합 신호 백테스트에 실패했습니다: {e}")

with tabs[4]:

    DECLINE_PATTERN_WINDOW = 20
    DECLINE_PATTERN_VOL_WINDOW = 20

    st.subheader("큰폭 하락 조기 신호 (SK하이닉스 전용, 참고용)")
    if TICKER != DEFAULT_TICKER:
        st.caption("이 지표는 SK하이닉스에서만 검증되어 SK하이닉스에서만 표시됩니다 (다른 종목에는 일반화되지 않는 것으로 자체 검증됨).")
    else:
        st.caption(
            "과거 SK하이닉스의 큰폭 하락(고점 대비 15% 이상) 8건을 분석한 결과, 하락 초기 10거래일 동안 "
            "외국인 순매도가 8건 중 7건, 거래량 증가가 7건 중 7건에서 공통적으로 나타났습니다. "
            "코스피 시가총액 상위 10개 종목 전체로는 이 패턴이 일반화되지 않는 것으로 확인되어, SK하이닉스 개별 참고용으로만 제공합니다. "
            "매매 신호가 아닙니다."
        )
        try:
            decline_hist = fetch_backtest_history_live(TICKER, target_days=700)
            if len(decline_hist) < 40:
                st.warning("데이터가 부족합니다.")
            else:
                foreign_slope = _rolling_slope(decline_hist["외국인"], DECLINE_PATTERN_WINDOW)
                volume_avg = decline_hist["거래량"].rolling(DECLINE_PATTERN_VOL_WINDOW).mean()
                volume_ratio = decline_hist["거래량"] / volume_avg

                current_foreign_slope = float(foreign_slope.dropna().iloc[-1])
                current_volume_ratio = float(volume_ratio.dropna().iloc[-1])
                foreign_selling = current_foreign_slope < 0
                volume_spike = current_volume_ratio > 1.0

                col1, col2 = st.columns(2)
                col1.metric(
                    f"외국인 {DECLINE_PATTERN_WINDOW}일 누적 순매수 기울기",
                    f"{current_foreign_slope:,.0f}",
                    delta="순매도 우위" if foreign_selling else "순매수 우위",
                    delta_color="inverse" if foreign_selling else "normal",
                )
                col2.metric(
                    f"거래량 / 직전 {DECLINE_PATTERN_VOL_WINDOW}일 평균",
                    f"{current_volume_ratio:.2f}배",
                    delta="평균 이상" if volume_spike else "평균 이하",
                    delta_color="inverse" if volume_spike else "normal",
                )

                if foreign_selling and volume_spike:
                    st.warning(
                        "현재 외국인 순매도 + 거래량 증가가 동시에 나타나고 있습니다. "
                        "과거 큰폭 하락 초기와 유사한 패턴이지만, 확정적 신호가 아니라 참고용입니다."
                    )
                else:
                    st.info("현재는 과거 큰폭 하락 초기 패턴(외국인 순매도 + 거래량 증가 동시 발생)과 일치하지 않습니다. (참고용)")

                decline_chart_df = pd.DataFrame(
                    {
                        "날짜": decline_hist["날짜"],
                        "외국인기울기": foreign_slope,
                        "거래량비율": volume_ratio,
                        "종가": decline_hist["종가"],
                    }
                )

                fig_decline_foreign = make_subplots(specs=[[{"secondary_y": True}]])
                fig_decline_foreign.add_trace(
                    go.Scatter(x=decline_chart_df["날짜"], y=decline_chart_df["외국인기울기"], name="외국인 순매수 기울기", line=dict(color="#1f77b4")),
                    secondary_y=False,
                )
                fig_decline_foreign.add_trace(
                    go.Scatter(x=decline_chart_df["날짜"], y=decline_chart_df["종가"], name="종가", line=dict(color="#d62728")),
                    secondary_y=True,
                )
                fig_decline_foreign.add_hline(y=0, line_dash="dash", line_color="gray", secondary_y=False)
                fig_decline_foreign.update_layout(title=f"외국인 {DECLINE_PATTERN_WINDOW}일 누적 순매수 기울기 vs 주가")
                fig_decline_foreign.update_yaxes(title_text="기울기", secondary_y=False)
                fig_decline_foreign.update_yaxes(title_text="종가(원)", secondary_y=True)
                fig_decline_foreign.update_xaxes(rangeslider_visible=True)
                st.plotly_chart(fig_decline_foreign, use_container_width=True, key="chart_decline_foreign")

                fig_decline_volume = make_subplots(specs=[[{"secondary_y": True}]])
                fig_decline_volume.add_trace(
                    go.Scatter(x=decline_chart_df["날짜"], y=decline_chart_df["거래량비율"], name="거래량/20일평균", line=dict(color="#1f77b4")),
                    secondary_y=False,
                )
                fig_decline_volume.add_trace(
                    go.Scatter(x=decline_chart_df["날짜"], y=decline_chart_df["종가"], name="종가", line=dict(color="#d62728")),
                    secondary_y=True,
                )
                fig_decline_volume.add_hline(y=1.0, line_dash="dash", line_color="gray", secondary_y=False)
                fig_decline_volume.update_layout(title=f"거래량 / 직전 {DECLINE_PATTERN_VOL_WINDOW}일 평균 vs 주가")
                fig_decline_volume.update_yaxes(title_text="거래량비율(배)", secondary_y=False)
                fig_decline_volume.update_yaxes(title_text="종가(원)", secondary_y=True)
                fig_decline_volume.update_xaxes(rangeslider_visible=True)
                st.plotly_chart(fig_decline_volume, use_container_width=True, key="chart_decline_volume")
                st.caption("차트 하단 슬라이더를 드래그하면 보고 싶은 기간만 확대해서 볼 수 있습니다.")
        except Exception as e:
            st.error(f"조기 신호 조회에 실패했습니다: {e}")

with tabs[5]:

    RALLY_PATTERN_WINDOW = 20
    RALLY_PATTERN_VOL_WINDOW = 20

    st.subheader("큰폭 상승 조기 신호 (SK하이닉스 전용, 참고용)")
    if TICKER != DEFAULT_TICKER:
        st.caption("이 지표는 SK하이닉스에서만 검증되어 SK하이닉스에서만 표시됩니다 (다른 종목에는 일반화되지 않는 것으로 자체 검증됨).")
    else:
        st.caption(
            "과거 SK하이닉스의 큰폭 상승(저점 대비 15% 이상) 16건을 분석한 결과, 상승 초기 10거래일 동안 "
            "개인 순매도가 16건 중 15건, 기관 순매수가 13건, 거래량 증가가 16건 전부에서 공통적으로 나타났습니다. "
            "코스피 시가총액 상위 10개 종목 전체로는 이 세 조건이 함께 일반화되지 않는 것으로 확인되어 "
            "(특히 거래량 조건은 다른 종목에서 오히려 반대로 나타남), SK하이닉스 개별 참고용으로만 제공합니다. "
            "매매 신호가 아닙니다."
        )
        try:
            rally_hist = fetch_backtest_history_live(TICKER, target_days=700)
            if len(rally_hist) < 40:
                st.warning("데이터가 부족합니다.")
            else:
                inst_slope = _rolling_slope(rally_hist["기관"], RALLY_PATTERN_WINDOW)
                retail_slope = _rolling_slope(rally_hist["개인"], RALLY_PATTERN_WINDOW)
                volume_avg = rally_hist["거래량"].rolling(RALLY_PATTERN_VOL_WINDOW).mean()
                volume_ratio = rally_hist["거래량"] / volume_avg

                current_inst_slope = float(inst_slope.dropna().iloc[-1])
                current_retail_slope = float(retail_slope.dropna().iloc[-1])
                current_volume_ratio = float(volume_ratio.dropna().iloc[-1])
                inst_buying = current_inst_slope > 0
                retail_selling = current_retail_slope < 0
                volume_spike = current_volume_ratio > 1.0

                col1, col2, col3 = st.columns(3)
                col1.metric(
                    f"기관 {RALLY_PATTERN_WINDOW}일 누적 순매수 기울기",
                    f"{current_inst_slope:,.0f}",
                    delta="순매수 우위" if inst_buying else "순매도 우위",
                    delta_color="normal" if inst_buying else "inverse",
                )
                col2.metric(
                    f"개인 {RALLY_PATTERN_WINDOW}일 누적 순매수 기울기",
                    f"{current_retail_slope:,.0f}",
                    delta="순매도 우위" if retail_selling else "순매수 우위",
                    delta_color="normal" if retail_selling else "inverse",
                )
                col3.metric(
                    f"거래량 / 직전 {RALLY_PATTERN_VOL_WINDOW}일 평균",
                    f"{current_volume_ratio:.2f}배",
                    delta="평균 이상" if volume_spike else "평균 이하",
                    delta_color="normal" if volume_spike else "inverse",
                )

                if inst_buying and retail_selling and volume_spike:
                    st.success(
                        "현재 기관 순매수 + 개인 순매도 + 거래량 증가가 동시에 나타나고 있습니다. "
                        "과거 큰폭 상승 초기와 유사한 패턴이지만, 확정적 신호가 아니라 참고용입니다."
                    )
                else:
                    st.info("현재는 과거 큰폭 상승 초기 패턴(기관 순매수 + 개인 순매도 + 거래량 증가 동시 발생)과 일치하지 않습니다. (참고용)")

                rally_chart_df = pd.DataFrame(
                    {
                        "날짜": rally_hist["날짜"],
                        "기관기울기": inst_slope,
                        "개인기울기": retail_slope,
                        "거래량비율": volume_ratio,
                        "종가": rally_hist["종가"],
                    }
                )

                fig_rally_flow = make_subplots(specs=[[{"secondary_y": True}]])
                fig_rally_flow.add_trace(
                    go.Scatter(x=rally_chart_df["날짜"], y=rally_chart_df["기관기울기"], name="기관 순매수 기울기", line=dict(color="#1f77b4")),
                    secondary_y=False,
                )
                fig_rally_flow.add_trace(
                    go.Scatter(x=rally_chart_df["날짜"], y=rally_chart_df["개인기울기"], name="개인 순매수 기울기", line=dict(color="#2ca02c")),
                    secondary_y=False,
                )
                fig_rally_flow.add_trace(
                    go.Scatter(x=rally_chart_df["날짜"], y=rally_chart_df["종가"], name="종가", line=dict(color="#d62728")),
                    secondary_y=True,
                )
                fig_rally_flow.add_hline(y=0, line_dash="dash", line_color="gray", secondary_y=False)
                fig_rally_flow.update_layout(title=f"기관/개인 {RALLY_PATTERN_WINDOW}일 누적 순매수 기울기 vs 주가")
                fig_rally_flow.update_yaxes(title_text="기울기", secondary_y=False)
                fig_rally_flow.update_yaxes(title_text="종가(원)", secondary_y=True)
                fig_rally_flow.update_xaxes(rangeslider_visible=True)
                st.plotly_chart(fig_rally_flow, use_container_width=True, key="chart_rally_flow")

                fig_rally_volume = make_subplots(specs=[[{"secondary_y": True}]])
                fig_rally_volume.add_trace(
                    go.Scatter(x=rally_chart_df["날짜"], y=rally_chart_df["거래량비율"], name="거래량/20일평균", line=dict(color="#1f77b4")),
                    secondary_y=False,
                )
                fig_rally_volume.add_trace(
                    go.Scatter(x=rally_chart_df["날짜"], y=rally_chart_df["종가"], name="종가", line=dict(color="#d62728")),
                    secondary_y=True,
                )
                fig_rally_volume.add_hline(y=1.0, line_dash="dash", line_color="gray", secondary_y=False)
                fig_rally_volume.update_layout(title=f"거래량 / 직전 {RALLY_PATTERN_VOL_WINDOW}일 평균 vs 주가")
                fig_rally_volume.update_yaxes(title_text="거래량비율(배)", secondary_y=False)
                fig_rally_volume.update_yaxes(title_text="종가(원)", secondary_y=True)
                fig_rally_volume.update_xaxes(rangeslider_visible=True)
                st.plotly_chart(fig_rally_volume, use_container_width=True, key="chart_rally_volume")
                st.caption("차트 하단 슬라이더를 드래그하면 보고 싶은 기간만 확대해서 볼 수 있습니다.")
        except Exception as e:
            st.error(f"조기 신호 조회에 실패했습니다: {e}")

with tabs[6]:

    dram_summary = "해당 없음 (메모리 반도체 관련주가 아니라 DRAM 시세를 표시하지 않음)"

    st.subheader("DRAM 현물가 (모듈 + 칩)")
    if TICKER not in MEMORY_SEMICONDUCTOR_TICKERS:
        st.caption("이 지표는 메모리 반도체 관련주(SK하이닉스, 삼성전자)에서만 제공됩니다.")
    else:
        st.caption(
            "DRAMeXchange 기준 오늘 시세입니다. 과거 이력은 무료로 제공되지 않아, "
            "대시보드를 사용할 때마다 그날 시세를 기록해 추이를 직접 쌓아갑니다."
        )
        try:
            module_df = fetch_dram_module_prices()
            chip_df = fetch_dram_chip_prices()
            combined_df = pd.concat([module_df, chip_df], ignore_index=True)

            if combined_df.empty:
                st.warning("DRAM 현물가 데이터를 가져오지 못했습니다.")
            else:
                dram_summary = "\n".join(
                    f"- {row['품목']}: ${row['평균가(USD)']:,.3f} ({_signed_pct(row)})" for _, row in combined_df.iterrows()
                )
                history = save_dram_snapshot(combined_df)
                has_trend = len(history["날짜"].unique()) >= 2

                st.markdown("**칩 현물가 (DDR5 16Gb, DDR4 16Gb)**")
                chip_display = chip_df.copy()
                chip_display["평균가(USD)"] = chip_display["평균가(USD)"].map(lambda v: f"${v:,.3f}")
                chip_display["변동률(%)"] = chip_df.apply(_signed_pct, axis=1)
                st.dataframe(chip_display, use_container_width=True, hide_index=True)
                if has_trend:
                    st.markdown("**칩 현물가 추이 (누적 기록)**")
                    _render_dram_trend_chart(history, list(chip_df["품목"]), "dram_chip_toggle", "chart_dram_chip")
                else:
                    st.caption("아직 하루치 데이터만 있어서 추이 그래프는 다음날부터 표시됩니다.")

                st.divider()

                st.markdown("**모듈 현물가 (DDR5 UDIMM/RDIMM, DDR4 UDIMM)**")
                module_display = module_df.copy()
                module_display["평균가(USD)"] = module_display["평균가(USD)"].map(lambda v: f"${v:,.2f}")
                module_display["변동률(%)"] = module_df.apply(_signed_pct, axis=1)
                st.dataframe(module_display, use_container_width=True, hide_index=True)
                if has_trend:
                    st.markdown("**모듈 현물가 추이 (누적 기록)**")
                    _render_dram_trend_chart(history, list(module_df["품목"]), "dram_module_toggle", "chart_dram_module")
                else:
                    st.caption("아직 하루치 데이터만 있어서 추이 그래프는 다음날부터 표시됩니다.")
        except Exception as e:
            st.error(f"DRAM 현물가 조회에 실패했습니다: {e}")

with tabs[7]:
    st.subheader("빅테크 분기별 설비투자(Capex)")
    st.caption(
        "마이크로소프트·구글(알파벳)·아마존·메타의 분기별 설비투자 실적입니다 (SEC 공시 XBRL 데이터, 미국 회계기준). "
        "AI/데이터센터向 capex가 HBM·DRAM 수요의 핵심 동력이라는 점에서 참고용으로 제공하며, "
        "백테스트나 예측 신호가 아니라 분기별 데이터 흐름을 보여드리는 것이 목적입니다. 분기 실적 발표 주기상 최대 며칠~몇 주 지연될 수 있습니다."
    )
    try:
        capex_df = fetch_bigtech_capex()
        if capex_df.empty:
            st.warning("Capex 데이터를 가져오지 못했습니다.")
        else:
            capex_df = capex_df.copy()
            capex_df["capex_B"] = capex_df["capex_USD"] / 1e9

            fig_capex = px.bar(
                capex_df, x="분기말", y="capex_B", color="기업", barmode="stack",
                labels={"capex_B": "Capex (10억달러)", "분기말": "분기"},
            )
            totals = capex_df.groupby("분기말")["capex_B"].sum().reset_index().sort_values("분기말")
            totals["qoq_pct"] = totals["capex_B"].pct_change() * 100
            fig_capex.add_trace(
                go.Scatter(x=totals["분기말"], y=totals["capex_B"], name="4개사 합계", mode="lines+markers", line=dict(color="black", dash="dot")),
            )
            fig_capex.add_trace(
                go.Scatter(
                    x=totals["분기말"], y=totals["qoq_pct"], name="4개사 합계 전분기 대비 증감률(%)",
                    mode="lines+markers", line=dict(color="#d62728"), yaxis="y2",
                ),
            )
            fig_capex.update_layout(
                title="빅테크 4개사 분기별 Capex (누적 막대 + 합계 추세 + 전분기 대비 증감률)",
                yaxis=dict(title="Capex (10억달러)"),
                yaxis2=dict(title="전분기 대비 증감률(%)", overlaying="y", side="right", showgrid=False),
            )
            st.plotly_chart(fig_capex, use_container_width=True, key="chart_bigtech_capex")

            company_count = capex_df.groupby("분기말")["기업"].nunique()
            n_companies = capex_df["기업"].nunique()
            complete_qs = sorted(company_count[company_count == n_companies].index)
            incomplete_latest = capex_df["분기말"].max() not in complete_qs
            if incomplete_latest:
                missing = set(capex_df["기업"].unique()) - set(
                    capex_df[capex_df["분기말"] == capex_df["분기말"].max()]["기업"]
                )
                st.caption(f"⚠️ 최근 분기는 {', '.join(missing)}의 실적 발표 전이라 그래프의 마지막 막대는 아직 미완성입니다.")

            if complete_qs:
                latest_q = complete_qs[-1]
                latest_total = totals[totals["분기말"] == latest_q]["capex_B"].iloc[0]
                if len(complete_qs) >= 5:
                    yoy_q = complete_qs[-5]
                    yoy_total = totals[totals["분기말"] == yoy_q]["capex_B"].iloc[0]
                    yoy_change = (latest_total / yoy_total - 1) * 100
                    st.info(
                        f"가장 최근 4개사 실적이 모두 발표된 분기({latest_q.strftime('%Y-%m')}) 합계 capex: **${latest_total:,.0f}B** "
                        f"(전년 동기 대비 {yoy_change:+.0f}%). 참고용 데이터이며 매매 신호가 아닙니다."
                    )
                else:
                    st.info(f"가장 최근 4개사 실적이 모두 발표된 분기({latest_q.strftime('%Y-%m')}) 합계 capex: **${latest_total:,.0f}B** (참고용, 매매 신호 아님)")

            with st.expander("분기별 상세 수치 보기"):
                pivot = capex_df.pivot(index="분기말", columns="기업", values="capex_B").sort_index(ascending=False)
                pivot["합계"] = pivot.sum(axis=1)
                qoq_pivot = pivot.pct_change(periods=-1) * 100
                pivot.index = pivot.index.strftime("%Y-%m")
                qoq_pivot.index = qoq_pivot.index.strftime("%Y-%m")

                st.markdown("**Capex (10억달러)**")
                st.dataframe(pivot.round(1), use_container_width=True)

                st.markdown("**전분기 대비 증감률(%)**")
                st.dataframe(qoq_pivot.round(1), use_container_width=True)
    except Exception as e:
        st.error(f"빅테크 Capex 조회에 실패했습니다: {e}")

with tabs[8]:

    community_summary = "커뮤니티 심리 데이터를 가져오지 못함"

    st.subheader("커뮤니티 심리 (네이버 종목토론방)")
    st.caption(
        f"최근 게시글 {community_post_count}건을 키워드 기반으로 긍정/부정/중립 분류한 결과입니다. "
        "여론일 뿐 사실이 아니며, 매매 신호로 쓰지 마세요."
    )
    try:
        posts_df = fetch_community_posts(TICKER, community_post_count)
        if posts_df.empty:
            st.warning("커뮤니티 게시글을 가져오지 못했습니다.")
        else:
            sentiment_df = classify_sentiment(posts_df)
            counts = sentiment_df["심리"].value_counts()
            total = len(sentiment_df)
            pos_n = int(counts.get("긍정", 0))
            neg_n = int(counts.get("부정", 0))
            neu_n = int(counts.get("중립", 0))

            col1, col2, col3 = st.columns(3)
            col1.metric("긍정", f"{pos_n}건", f"{pos_n / total:.0%}")
            col2.metric("부정", f"{neg_n}건", f"{neg_n / total:.0%}", delta_color="inverse")
            col3.metric("중립", f"{neu_n}건", f"{neu_n / total:.0%}")

            daily = sentiment_df.groupby(["날짜", "심리"]).size().reset_index(name="건수")
            num_days = sentiment_df["날짜"].nunique()
            if num_days >= 2:
                st.markdown("**일자별 심리 추이**")
                fig_sentiment = px.bar(
                    daily, x="날짜", y="건수", color="심리", barmode="stack",
                    color_discrete_map={"긍정": "#d62728", "부정": "#1f77b4", "중립": "#7f7f7f"},
                )
                st.plotly_chart(fig_sentiment, use_container_width=True, key="chart_sentiment")
            else:
                st.caption(f"조회된 게시글이 전부 {sentiment_df['날짜'].iloc[0]} 하루에 몰려 있어 일자별 비교는 아직 어렵습니다 (조회 개수를 늘려보세요).")

            with st.expander(f"게시글 {total}건 상세 보기 (날짜별)"):
                st.dataframe(
                    sentiment_df.sort_values("날짜", ascending=False),
                    use_container_width=True, hide_index=True,
                )

            community_summary = (
                f"최근 게시글 {total}건({num_days}일치) 중 긍정 {pos_n}건({pos_n / total:.0%}), "
                f"부정 {neg_n}건({neg_n / total:.0%}), 중립 {neu_n}건({neu_n / total:.0%}). "
                "(키워드 기반 단순 분류이며 여론 참고용)"
            )
    except Exception as e:
        st.error(f"커뮤니티 심리 분석에 실패했습니다: {e}")

with tabs[9]:

    st.subheader("AI 분석: 오늘의 주가 변동 요인")
    st.caption(
        "뉴스 헤드라인·애널리스트 리포트·수급 데이터를 종합해 AI가 분석합니다. "
        "버튼을 누를 때만 실행됩니다 (자동 갱신 없음). 투자 조언이 아닌 참고용 요인 분석입니다."
    )

    if not os.environ.get("GEMINI_API_KEY"):
        st.info("AI 분석을 사용하려면 GEMINI_API_KEY 환경변수를 설정해주세요.")
    else:
        if st.button("지금 바로 분석하기"):
            try:
                price_summary = st.session_state.get("current_price_summary", "현재가 데이터를 가져오지 못함")

                if not investor_df.empty:
                    recent = investor_df.tail(5)
                    supply_summary = "\n".join(
                        f"- 최근 {len(recent)}거래일 {col} 순매수 합계: {recent[col].sum():+,.0f}주"
                        for col in recent.columns
                    )
                else:
                    supply_summary = "수급 데이터를 가져오지 못함"

                headlines = fetch_news_headlines(STOCK_NAME)
                reports_df = fetch_analyst_reports(TICKER)
                reports_md = "\n".join(
                    f"- [{row['증권사']}] {row['제목']} ({row['작성일']})" for _, row in reports_df.iterrows()
                )

                time_label = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
                analysis = generate_ai_analysis(
                    f"{STOCK_NAME}({TICKER})",
                    time_label, price_summary, supply_summary, headlines, reports_md,
                    dram_summary, community_summary, composite_summary, overheat_summary, futures_summary,
                )

                st.session_state["ai_analysis"] = analysis
                st.session_state["ai_analysis_time"] = time_label
                st.session_state["ai_analysis_headlines"] = headlines
                st.session_state["ai_analysis_reports"] = reports_df
            except Exception as e:
                st.error(f"AI 분석 생성에 실패했습니다: {e}")

        if "ai_analysis" in st.session_state:
            st.caption(f"기준 시각: {st.session_state['ai_analysis_time']}")
            st.markdown(st.session_state["ai_analysis"])

            with st.expander("분석에 사용된 원본 데이터 보기"):
                st.write("**뉴스 헤드라인**")
                st.write(st.session_state["ai_analysis_headlines"])
                st.write("**애널리스트 리포트**")
                st.dataframe(st.session_state["ai_analysis_reports"], use_container_width=True, hide_index=True)
        else:
            st.info("버튼을 눌러 AI 분석을 생성해주세요.")

