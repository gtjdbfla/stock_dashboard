import time as _time
_IMPORT_T0 = _time.monotonic()   # import 비용을 재려고 제일 먼저 잡는다

import csv
import datetime as dt
import json
import os
import re
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import ai_analysis
import analyst_digest
import analyst_targets
import disclosure
import financial_digest
import fnguide
import over_market as om
from bs4 import BeautifulSoup
from google import genai
from plotly.subplots import make_subplots
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from wordcloud import WordCloud

import ai_inputs
import ai_report
from ai_inputs import (
    ADR_HOST_TICKER,
    ADR_SHARE_RATIO,
    BIGTECH_CIKS,
    CONSENSUS_LOG,
    DC_GALLERY_ID,
    DC_GALLERY_VIEW_URL,
    DC_HEADERS,
    DECLINE_DRAWDOWN_THRESHOLD,
    DECLINE_HORIZON,
    DECLINE_PATTERN_VOL_WINDOW,
    DECLINE_PATTERN_WINDOW,
    DEFAULT_COMMUNITY_POST_COUNT,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_TICKER,
    INVESTOR_COLUMNS,
    NEGATIVE_KEYWORDS,
    OVERHEAT_DEFAULT_HORIZON,
    OVERHEAT_DEFAULT_MA_WINDOW,
    OVERHEAT_DEFAULT_THRESHOLD,
    OVERHEAT_QUANTILES,
    POSITIVE_KEYWORDS,
    RALLY_DRAWDOWN_THRESHOLD,
    RALLY_HORIZON,
    RALLY_PATTERN_VOL_WINDOW,
    RALLY_PATTERN_WINDOW,
    YAHOO_CHART_URL,
    build_community_summary,
    calc_slope,
    classify_sentiment,
    fetch_adr_baseline,
    fetch_adr_quote,
    fetch_analyst_reports,
    fetch_backtest_history,
    fetch_backtest_history_live,
    fetch_bigtech_capex,
    fetch_community_posts,
    fetch_current_price,
    fetch_daily_ohlcv,
    fetch_dc_gallery_posts,
    fetch_dram_chip_prices,
    fetch_dram_module_prices,
    fetch_fnguide_page,
    fetch_foreign_desk,
    fetch_intraday_price,
    fetch_investor_netbuy,
    fetch_latest_bars,
    fetch_market_flow,
    fetch_short_balance,
    fetch_stock_snapshot,
    fetch_target_price_history,
    forward_max_drawdown,
    forward_max_gain,
    load_over_market_ticks,
    run_boolean_pattern_backtest,
    run_overheat_backtest,
    two_proportion_ztest,
)
# 밑줄로 시작하는 이름은 from ... import *가 가져오지 않는다. 화면이 쓰는 것만 따로.
from ai_inputs import (  # noqa: F401
    _TAB_SUMMARY_UNSET,
    _call_gemini,
    _consensus_log,
    _fetch_adr_bars,
    _fetch_dram_soup,
    _fetch_frgn_page,
    _korea_session_now,
    _rolling_slope,
    _signed_pct,
    _to_number,
)

# 장중 화면에 걸린 셋(intraday 10초 · market_flow 20초 · foreign_desk 20초)은
# 프래그먼트가 5초마다 다시 그리는 것과 짝을 이룬다. ttl이 그리기 주기보다 길면
# 화면만 다시 그려지고 값은 묵은 채로 남는다. 5초까지 더 낮추지 않은 이유는 이 출처들이
# 실제로 1~2분마다 갱신돼서, 더 자주 찔러도 새 값은 안 오고 네이버 요청 수만 는다.
# ai_inputs는 캐시를 걸지 않는다 — 수집기에서는 캐시가 의미가 없고, streamlit을
# import하지도 않기 때문이다. 화면 쪽 반응성은 여기서 같은 ttl로 다시 씌워 되살린다.
# 이 표의 ttl은 예전에 각 함수 위에 붙어 있던 데코레이터를 그대로 옮긴 것이다.
_CACHED = {
    '_fetch_board_page': dict(ttl=1800, show_spinner="불러오는 중..."),
    '_fetch_dram_soup': dict(ttl=3600, show_spinner="불러오는 중..."),
    # 시세·그래프 두 곳이 부르는 야후 분봉. 캐시가 없으면 5초 화면 조각이 돌 때마다
    # 새로 받았다. 90초면 프리/애프터장 움직임을 따라가기에 충분하다.
    '_fetch_adr_bars': dict(ttl=90, show_spinner=False),
    'fetch_adr_baseline': dict(ttl=3600, show_spinner=False),
    # 위 _fetch_adr_bars가 캐시되므로 이건 값 조립만 한다. 120초로 늘려도 체감 차이가 없다.
    'fetch_adr_quote': dict(ttl=120, show_spinner=False),
    'fetch_analyst_reports': dict(ttl=900, show_spinner="불러오는 중..."),
    'fetch_backtest_history': dict(ttl=24 * 3600, show_spinner="불러오는 중..."),
    'fetch_bigtech_capex': dict(ttl=24 * 3600, show_spinner="불러오는 중..."),
    'fetch_community_posts': dict(ttl=1800, show_spinner="불러오는 중..."),
    'fetch_current_price': dict(ttl=5, show_spinner="불러오는 중..."),
    'fetch_daily_ohlcv': dict(ttl=600, show_spinner=False),
    'fetch_dc_gallery_posts': dict(ttl=1800, show_spinner="불러오는 중..."),
    'fetch_disclosures': dict(ttl=1800, show_spinner="공시 수집 중..."),
    'fetch_dram_chip_prices': dict(ttl=3600, show_spinner="불러오는 중..."),
    'fetch_dram_module_prices': dict(ttl=3600, show_spinner="불러오는 중..."),
    'fetch_earnings_calendar': dict(ttl=6 * 3600, show_spinner=False),
    'fetch_fnguide_page': dict(ttl=6 * 3600, show_spinner="재무 데이터를 가져오는 중..."),
    'fetch_foreign_desk': dict(ttl=20, show_spinner=False),
    'fetch_foreign_hold_ratio': dict(ttl=600, show_spinner=False),
    'fetch_intraday_price': dict(ttl=10, show_spinner="불러오는 중..."),
    'fetch_investor_netbuy': dict(ttl=3600, show_spinner="불러오는 중..."),
    'fetch_latest_bars': dict(ttl=60, show_spinner="불러오는 중..."),
    'fetch_macro_summary': dict(ttl=1800, show_spinner=False),
    'fetch_market_flow': dict(ttl=20, show_spinner=False),
    'fetch_news_with_summary': dict(ttl=1800, show_spinner="불러오는 중..."),
    'fetch_sector_news': dict(ttl=1800, show_spinner="업종·매크로 뉴스 수집 중..."),
    'fetch_stock_snapshot': dict(ttl=1800, show_spinner="불러오는 중..."),
    'fetch_trendforce_news': dict(ttl=6 * 3600, show_spinner="불러오는 중..."),
}
for _fn_name, _fn_kwargs in _CACHED.items():
    globals()[_fn_name] = st.cache_data(**_fn_kwargs)(getattr(ai_inputs, _fn_name))

# fetch_adr_quote는 ai_inputs 안에서 _fetch_adr_bars를 부른다 — 모듈 내부 참조라
# 위에서 씌운 캐시본이 아니라 원본을 탄다. 이 하나만 모듈 네임스페이스에 되써서
# 시세·그래프가 같은 캐시된 응답을 쓰게 한다(이 프로세스에만 적용, 수집기와 무관).
ai_inputs._fetch_adr_bars = globals()["_fetch_adr_bars"]


# ── 콜드 로드 프로파일 ──────────────────────────────────────────────────────
# 전체 스크립트가 처음부터 다시 도는 리런(첫 접속·사이드바 변경·자동 새로고침)마다
# 이 모듈이 위에서부터 재실행되므로, 여기서 잡는 시각이 그 리런의 시작점이다.
# BOOT_PROFILE=1 일 때만 단계별 경과를 stdout(=docker logs)에 남긴다.
_BOOT_T0 = _IMPORT_T0
_BOOT_PROFILE = os.environ.get("BOOT_PROFILE") == "1"


def _boot_lap(label: str) -> None:
    if _BOOT_PROFILE:
        print(f"[boot] +{time.monotonic() - _BOOT_T0:6.2f}s  {label}", flush=True)


_boot_lap(f"import 완료 (import에 {time.monotonic() - _IMPORT_T0:.2f}s)")


def _streamlit_pool(max_workers: int) -> ThreadPoolExecutor:
    """워커 스레드에도 Streamlit 실행 컨텍스트를 붙인 스레드풀.

    워커 안에서 @st.cache_data 함수를 부를 때 컨텍스트가 없으면 호출마다
    'missing ScriptRunContext' 경고가 쏟아지고 캐시 동작도 보장되지 않는다.
    순수 requests/pandas 작업만 던질 때는 굳이 필요 없다.
    """
    ctx = get_script_run_ctx()

    def _attach() -> None:
        add_script_run_ctx(threading.current_thread(), ctx)

    return ThreadPoolExecutor(max_workers=max_workers, initializer=_attach)

# 실시간 갱신 주기. 셋 다 5초다 — 새 값이 생기면 5초 안에 화면에 오른다.
# 셋을 각각 다른 프래그먼트로 둔 것은 주기를 다르게 하려는 게 아니라 **서로 기다리지
# 않게** 하려는 것이다. 한 덩어리였을 때는 차트 요청이 느린 순간 현재가 숫자까지 같이
# 멈췄고, 셋이 한꺼번에 리런되면서 화면 전체가 동시에 흐려졌다.
REFRESH_SEC = 5                 # 현재가 지표
INTRADAY_REFRESH_SEC = 5        # 장중 차트
MARKET_FLOW_REFRESH_SEC = 5     # 코스피 전체 수급
DEFAULT_STOCK_NAME = "SK하이닉스"
NAVER_SEARCH_URL = "https://ac.stock.naver.com/ac"
YAHOO_SYMBOLS = {
    "SOX": "%5ESOX",
    "DXY": "DX-Y.NYB",
}
MEMORY_SEMICONDUCTOR_TICKERS = {"000660": "SK하이닉스", "005930": "삼성전자"}

ALL_TAB_LABELS = [
    "매매 신호",
    "수급 현황",
    "가격 과열도",
    "선물 경보",
    "통합 신호",
    "하락 조기신호",
    "상승 조기신호",
    "DRAM 시세",
    "빅테크 Capex",
    "재무 데이터",
    "공시",
    "애널리스트",
    "커뮤니티",
    "AI 분석",
]

KOREAN_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
]
KOREAN_STOPWORDS = {
    "그리고", "그런데", "그래서", "근데", "이거", "저거", "그거", "여기", "저기", "거기",
    "진짜", "완전", "이제", "오늘", "내일", "어제", "우리", "너네", "자기", "이번", "저번",
    "다음", "하고", "해서", "해도", "하지만", "그냥", "조금", "엄청", "너무", "정말",
    "이런", "저런", "그런", "이렇게", "저렇게", "그렇게", "때문", "지금", "아니", "근대",
    "뭐임", "뭐냐", "그럼", "이제는", "습니다", "합니다", "됩니다",
}


DRAM_HISTORY_FILE = os.environ.get("DRAM_HISTORY_FILE", "data/dram_spot_history.csv")



















def _md_safe(text: str) -> str:
    """st.markdown에 넘겨도 글자가 안 깨지게 만든다.

    st.markdown은 $...$ 사이를 LaTeX 수식으로 해석한다. DRAM 현물가를 '$227.500' 형식으로
    넘겨주다 보니 AI가 그대로 인용했고, 한 줄에 달러 기호가 두 번 나오는 순간
    '$227.500 (+1.11%), DDR5 RDIMM 32GB $' 가 통째로 수식으로 바뀌면서 글꼴이 달라지고
    달러 기호까지 사라졌다. 달러 기호를 escape 해서 그냥 글자로 남긴다.
    """
    return re.sub(r"(?<!\\)\$", r"\\$", text)


def _md_stream_safe(partial: str) -> str:
    """스트리밍 도중의 조각용. 아직 안 닫힌 코드 표시가 뒷글을 통째로 먹지 않게 한다."""
    out = _md_safe(partial)
    if out.count("`") % 2:
        i = out.rfind("`")
        out = out[:i] + out[i + 1:]
    return out











# ── 매매 신호 파라미터 ────────────────────────────────────────────────────────
# SK하이닉스 2016-01 – 2026-08 (2,599거래일)로 후보 지표 19종 x 예측기간 3종을 검증해 고른 값이다.
# 기관 순매수만 살아남았다. SOX는 학습구간과 검증구간에서 상관 부호가 뒤집혔고(-0.10 -> +0.25),
# DXY와 이동평균 괴리율은 중첩 보정(블록 부트스트랩)을 하면 유의성이 사라졌다.
# 되돌아보기 창은 매년 과거 데이터만 보고 다시 고르게 해도 항상 20일이 선택됐다.
FLOW_SIGNAL_VOL_WINDOW = 20    # 기관 순매수를 나눠줄 평균 거래량 창 (종목 규모 효과 제거)
FLOW_SIGNAL_WINDOW = 20        # 정규화된 순매수를 누적할 창
# 기관 순매수는 장 마감 후에 공시되므로 t일 신호로는 t일 종가에 살 수 없다.
# t+1일 종가 체결을 가정해 2일 밀어서 성과를 계산한다 (보수적).
FLOW_SIGNAL_EXEC_LAG = 2
FLOW_BACKTEST_DAYS = 1200


def _subheader_with_help(title: str, help_text: str, key: str) -> None:
    """제목 바로 옆에 물음표 버튼을 두고, 클릭하면 설명이 열리고 바깥을 누르면 닫히게 한다.
    st.subheader(help=...)의 기본 툴팁은 마우스를 올려야만 열려서 모바일에서 쓰기 불편하다.
    두 칸의 실제 너비는 CSS(st-key-help_row_)에서 내용 크기에 맞게 다시 잡는다."""
    with st.container(key=f"help_row_{key}"):
        title_col, help_col = st.columns([0.9, 0.1], vertical_alignment="center")
        title_col.subheader(title)
        with help_col.popover("", icon=":material/help:"):
            st.markdown(help_text)


def _bold_label_with_help(label: str, help_text: str, key: str) -> None:
    """_subheader_with_help와 같은 구조를 굵은 소제목(st.markdown)에 적용한다."""
    with st.container(key=f"help_row_{key}"):
        title_col, help_col = st.columns([0.9, 0.1], vertical_alignment="center")
        title_col.markdown(f"**{label}**")
        with help_col.popover("", icon=":material/help:"):
            st.markdown(help_text)


def _metric_with_help(label: str, value, help_text: str, key: str, **metric_kwargs) -> None:
    """물음표를 눌러야 설명이 열리는 지표.

    st.metric(help=...)의 기본 물음표는 마우스를 올려야만 열린다. 화면 안에 클릭형(팝오버)과
    hover형이 섞여 있으면 어느 쪽인지 매번 헷갈리고, 모바일에서는 hover 자체가 안 된다.
    그래서 지표에도 팝오버를 붙이고, 위치는 CSS로 지표 칸 오른쪽 위에 고정한다
    (라벨 길이가 제각각이라 라벨 바로 옆에 붙이면 줄이 흔들린다).
    """
    with st.container(key=f"metric_help_{key}"):
        # 라벨과 물음표를 한 줄에 붙인다 ('장중 주가 추이'와 같은 방식).
        # 지표 칸 오른쪽 끝에 고정했더니 라벨이 짧을수록 멀리 떨어져 보였다.
        _bold_label_with_help(label, help_text, key=f"metric_{key}")
        # 라벨은 위에서 직접 그렸으므로 지표 자체의 라벨은 접는다
        st.metric(label, value, label_visibility="collapsed", **metric_kwargs)


# Plotly는 숨겨진 탭(display:none) 안에서 그려지면 컨테이너 폭을 못 재고 기본 700px로 그린다.
# 그리고 탭이 보이게 돼도 스스로 다시 계산하지 않는다. 데스크톱에서는 폭이 700 근처라 티가 안 났지만,
# 모바일 375px에서는 차트 절반이 잘려 나갔다(SVG width=700, 오른쪽 끝 716px).
# responsive를 켜면 컨테이너 크기 변화에 맞춰 다시 레이아웃한다.
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


def _style_chart_mobile(fig, title: str | None = None, show_legend: bool = True) -> None:
    """모바일 화면에서 확대/축소 등 모드바 아이콘이 제목과 겹치지 않도록 제목을 왼쪽 정렬하고
    상단 여백을 확보하며, 범례를 그래프 위쪽 가로 방향으로 옮긴다. 모든 차트에 공통 적용한다."""
    layout_kwargs = dict(margin=dict(t=80 if show_legend else 50))
    if title is not None:
        layout_kwargs["title"] = dict(text=title, x=0.01, xanchor="left", y=0.98, yanchor="top")
    if show_legend:
        layout_kwargs["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    else:
        layout_kwargs["showlegend"] = False
    fig.update_layout(**layout_kwargs)
    # 모바일에서 스크롤하다 그래프 위를 스치면 확대/축소로 인식되는 문제를 막기 위해 줌을 꺼둔다
    # (rangeslider가 있는 차트는 rangeslider 자체로 구간 조정이 가능하므로 영향 없음).
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)


@st.cache_data(ttl=3600, show_spinner="불러오는 중...")
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

st.markdown(
    """
    <style>
    div[data-testid="stSpinner"] {
        display: none !important;
    }
    /* 리런 중인 요소를 흐리게 만드는 기본 효과(opacity 1s ease-in 0.5s)를 끈다.
       이 화면은 5초마다 스스로 갱신하므로 그 표시가 멈추지 않고 되풀이돼서,
       값이 바뀌는 것보다 흐려졌다 진해지는 것이 먼저 눈에 들어왔다.
       진행 중이라는 신호는 오른쪽 위 'Running' 표시와, 오래 걸리는 작업의 진행 막대가
       따로 맡는다. 여기서는 값만 조용히 바뀌게 둔다. */
    .stElementContainer[data-stale="true"],
    div[data-stale="true"] {
        opacity: 1 !important;
        transition: none !important;
    }
    /* DRAM 추이 차트의 품목 선택 드롭다운(Plotly updatemenu). Streamlit의 plotly
       테마가 배경·글자를 밝게 덮어써서 다크 화면에서 글자가 안 보였다. SVG라 CSS로
       직접 색을 박는다. */
    .js-plotly-plot .updatemenu-item-rect,
    .js-plotly-plot .updatemenu-header {
        fill: #1e1e26 !important;
        stroke: #555 !important;
    }
    .js-plotly-plot .updatemenu-item-text {
        fill: #fafafa !important;
    }
    .js-plotly-plot .updatemenu-item-rect:hover {
        fill: #34343f !important;
    }
    div[class*="st-key-metric_small_"] [data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
    }
    div[class*="st-key-metric_small_"] [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
    }
    div[data-testid="stTable"] {
        overflow-x: auto !important;
    }
    /* 코스피 전체 수급: 순매수는 초록, 순매도는 빨강. 화면의 다른 상승/하락 색과 같은 톤이다.
       st.metric은 delta에만 색을 주므로, 숫자(값)까지 칠하려면 이렇게 직접 지정해야 한다. */
    div[class*="st-key-metric_small_flow_buy_"] [data-testid="stMetricValue"],
    div[class*="st-key-metric_small_flow_buy_"] [data-testid="stMetricDelta"],
    div[class*="st-key-metric_small_flow_buy_"] [data-testid="stMetricDelta"] svg {
        color: #1a9e5f !important;
        fill: #1a9e5f !important;
    }
    div[class*="st-key-metric_small_flow_sell_"] [data-testid="stMetricValue"],
    div[class*="st-key-metric_small_flow_sell_"] [data-testid="stMetricDelta"],
    div[class*="st-key-metric_small_flow_sell_"] [data-testid="stMetricDelta"] svg {
        color: #e04b4b !important;
        fill: #e04b4b !important;
    }
    /* 제목 + 물음표 버튼은 모바일에서도 한 줄에 붙어 있어야 한다 (기본은 세로로 쌓임).
       제목 칸을 글자 너비에 맞게 줄여서 물음표가 제목 바로 옆에 오도록 한다. */
    div[class*="st-key-help_row_"] div[data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important;
        align-items: center !important;
        gap: 0 !important;
    }
    div[class*="st-key-help_row_"] div[data-testid="stColumn"] {
        min-width: 0 !important;
    }
    div[class*="st-key-help_row_"] div[data-testid="stColumn"]:first-child {
        flex: 0 1 auto !important;
        width: auto !important;
    }
    div[class*="st-key-help_row_"] div[data-testid="stColumn"]:last-child {
        flex: 0 0 auto !important;
        width: auto !important;
    }
    div[class*="st-key-help_row_"] button {
        background: none !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 0 0 0.35rem !important;
        min-height: 0 !important;
        height: auto !important;
        opacity: 0.55;
    }
    div[class*="st-key-help_row_"] button:hover {
        opacity: 1;
    }
    /* 팝오버 기본 화살표(expand_more)는 물음표 아이콘만 남기기 위해 숨긴다 */
    div[class*="st-key-help_row_"] button div[aria-hidden="true"] {
        display: none !important;
    }
    /* 버튼에 라벨 없이 아이콘만 두므로, 아이콘 크기와 여백을 직접 잡아준다 */
    div[class*="st-key-help_row_"] button [data-testid="stIconMaterial"] {
        font-size: 1.05rem !important;
        width: 1.05rem !important;
        height: 1.05rem !important;
        margin: 0 !important;
    }
    div[class*="st-key-help_row_"] button > div {
        gap: 0 !important;
    }
    /* 지표에 붙는 물음표는 help_row_ 구조를 그대로 쓴다(라벨 바로 옆). 다만 지표 라벨은
       제목이 아니라 값의 설명이므로, 굵은 제목 톤 대신 Streamlit 기본 지표 라벨 톤으로 낮춘다. */
    div[class*="st-key-metric_help_"] div[class*="st-key-help_row_metric_"] p {
        font-size: 0.875rem !important;
        font-weight: 400 !important;
        opacity: 0.7;
        margin-bottom: 0 !important;
    }
    /* 라벨 줄과 값 사이가 벌어지지 않게. 물음표 없는 일반 지표는 이 간격이 0이라
       거기에 맞춘다 (팝오버 버튼이 라벨보다 키가 커서 줄 높이가 늘어난 만큼 당겨준다). */
    div[class*="st-key-metric_help_"] div[class*="st-key-help_row_metric_"] {
        margin-bottom: -0.85rem !important;
    }
    div[class*="st-key-metric_help_"] div[class*="st-key-help_row_metric_"]
        div[data-testid="stHorizontalBlock"] {
        min-height: 0 !important;
    }
    div[class*="st-key-metric_help_"] div[class*="st-key-help_row_metric_"] button {
        opacity: 0.45;
    }
    div[class*="st-key-metric_help_"] div[class*="st-key-help_row_metric_"] button:hover {
        opacity: 1;
    }
    div[class*="st-key-metric_help_"] div[class*="st-key-help_row_metric_"] button [data-testid="stIconMaterial"] {
        font-size: 0.95rem !important;
        width: 0.95rem !important;
        height: 0.95rem !important;
    }
    div[data-baseweb="tooltip"] {
        max-width: min(85vw, 320px) !important;
    }
    div[data-testid="stTooltipContent"] {
        max-width: min(85vw, 320px) !important;
        white-space: normal !important;
        word-wrap: break-word !important;
    }
    @media (max-width: 640px) {
        /* 현재가 주변의 지표 줄들은 좁은 화면에서 2열로 접는다.
           안 접으면 Streamlit 기본값대로 한 칸씩 전체 폭을 먹고 세로로 쌓여서,
           숫자 네 개에 화면 한 판을 다 쓰게 된다.
           컨테이너 이름을 하나씩 적는 대신 st-key-price_row_ 접두어로 한 번에 잡는다.
           예전에 개별 나열식이라, 새로 추가한 줄(코스피 전체 수급)이 목록에서 빠져
           혼자만 세로로 늘어지는 일이 있었다. */
        div[class*="st-key-price_row_"] div[data-testid="stHorizontalBlock"] {
            flex-direction: row !important;
            flex-wrap: wrap !important;
            gap: 0.25rem !important;
        }
        div[class*="st-key-price_row_"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
            flex: 1 1 47% !important;
            width: 47% !important;
            min-width: 47% !important;
        }
        /* 현재가 줄만 예외: 첫 칸(현재가 본체)은 한 줄을 다 쓴다. */
        div[class*="st-key-price_row_columns"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child {
            flex: 1 1 100% !important;
            width: 100% !important;
            min-width: 100% !important;
        }
        /* 위 47% 규칙은 price_row_ 안에 '중첩된' 칸까지 잡는다. 라벨+물음표 줄이 그 안에 있어서
           라벨 칸이 47%로 늘어나면 물음표가 멀리 밀린다. 라벨 줄은 글자 너비에 맞춰 되돌린다. */
        div[class*="st-key-help_row_"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child {
            flex: 0 1 auto !important;
            width: auto !important;
            min-width: 0 !important;
        }
        div[class*="st-key-help_row_"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:last-child {
            flex: 0 0 auto !important;
            width: auto !important;
            min-width: 0 !important;
        }
        /* 라벨이 길어(예: "80일선 괴리율 (현재가 기준)") 두 줄로 넘어가도 값이 밀리지 않게 */
        div[class*="st-key-metric_small_"] [data-testid="stMetricLabel"] {
            font-size: 0.68rem !important;
            line-height: 1.25 !important;
        }
        /* 물음표가 붙은 지표는 라벨을 직접 그리므로 그쪽도 같은 크기로 줄인다 */
        div[class*="st-key-metric_help_"] div[class*="st-key-help_row_metric_"] p {
            font-size: 0.68rem !important;
            line-height: 1.25 !important;
        }
        div[class*="st-key-metric_small_"] [data-testid="stMetricValue"] {
            font-size: 0.95rem !important;
        }
        div[class*="st-key-metric_small_"] [data-testid="stMetricDelta"] {
            font-size: 0.7rem !important;
        }
        /* 모바일에서 표가 화면을 넘칠 때 가로 스크롤 되게 (AI 분석 탭의 비교표 등) */
        div[data-testid="stTable"] table,
        div[data-testid="stDataFrame"] {
            font-size: 0.8rem !important;
        }
        /* 글자만 줄여서는 부족하다. TrendForce 표는 그래도 제 칸을 9px 넘겨서 오른쪽이 잘렸다.
           표를 칸 너비에 묶고 넘치는 만큼은 표 안에서 굴리게 한다. */
        div[data-testid="stDataFrame"] {
            width: 100% !important;
            max-width: 100% !important;
            overflow-x: auto !important;
        }
        /* AI 분석의 '원본 데이터 보기'는 st.text로 긴 줄을 그대로 뿌린다. <pre>는 줄바꿈이
           없어서 한 줄이 화면을 넘기면 블록마다 가로 스크롤이 생긴다. 접어서 보여준다. */
        div[data-testid="stText"], div[data-testid="stText"] pre {
            white-space: pre-wrap !important;
            word-break: break-word !important;
            font-size: 0.75rem !important;
        }
        /* 분석 본문의 표(강세/약세 비교 등)도 넘치면 가로로 굴린다 */
        div[data-testid="stMarkdown"] table {
            display: block !important;
            overflow-x: auto !important;
            width: 100% !important;
        }
        /* 물음표 설명은 글이 길다. 기본 팝오버 폭(약 400px)이 화면보다 넓어 오른쪽이 잘린다. */
        div[data-testid="stPopoverBody"] {
            max-width: 88vw !important;
            font-size: 0.85rem !important;
        }
        /* 탭 이름이 많아 한 줄을 넘칠 때 가로 스크롤 */
        div[data-testid="stTabs"] div[role="tablist"] {
            overflow-x: auto !important;
            scrollbar-width: none;
        }
        div[data-testid="stTabs"] div[role="tablist"]::-webkit-scrollbar {
            display: none;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
    # 매매 신호: 하이닉스 10년 백테스트는 강했지만 다른 20종목에서 재현되지 않아(예측력 평균 –0)
    # 기본으로 숨긴다. 참고 지표로 보고 싶을 때만 켜서 쓴다.
    # 커뮤니티 탭은 잘 안 보게 돼서 기본으로 숨긴다. 매 렌더마다 네이버 종목토론방(0.7초)과
    # 디시 갤러리(0.9초)를 훑어서 콜드 로딩의 4분의 1쯤을 차지했다.
    # 여론 자체는 AI 분석에 build_community_summary()로 계속 들어간다.
    DEFAULT_HIDDEN_TAB_LABELS = {"매매 신호", "선물 경보", "통합 신호",
                                 "하락 조기신호", "상승 조기신호", "커뮤니티"}
    with st.expander("표시할 탭 선택"):
        visible_tab_labels = [
            label for label in ALL_TAB_LABELS
            if st.checkbox(label, value=label not in DEFAULT_HIDDEN_TAB_LABELS, key=f"show_tab_{label}")
        ]

TICKER = st.session_state.ticker
STOCK_NAME = st.session_state.stock_name

st.title(STOCK_NAME)




























def _level_slope(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).apply(lambda w: calc_slope(pd.Series(w)), raw=False)




def run_overheat_threshold_strategy(
    df_with_deviation: pd.DataFrame, period_start: pd.Timestamp,
) -> dict:
    """괴리율이 0% 이상이면 보유하고, 0% 밑으로 내려가면 전량 매도했다가 다시 0% 이상으로
    올라오면 재매수하는 전략을 period_start 이후 구간에서 시뮬레이션하고 buy & hold와 비교한다.
    (= 주가가 이동평균선 위에 있을 때만 보유하는 추세추종 전략)
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
            if deviation >= 0:
                position = {"buy_date": date, "buy_price": price}
        else:
            mark = equity_base * (price / position["buy_price"])
            equity.append({"날짜": date, "자산가치": mark})
            if deviation < 0:
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


@st.cache_data(ttl=24 * 3600, show_spinner="불러오는 중...")
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


@st.cache_data(ttl=60, show_spinner="불러오는 중...")
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
    quantile: float = 0.2, drawdown_threshold: float = 0.07, gain_threshold: float | None = None,
) -> dict:
    """코스피200 선물 외국인 누적 순매수 기울기가 하위 quantile(강한 매도)일 때, 현재 종목의 향후 horizon일 내
    drawdown_threshold 이상 하락할 확률과 gain_threshold 이상 상승할 확률이 나머지 구간과 어떻게 다른지 함께 검증한다.
    gain_threshold를 안 주면 drawdown_threshold와 같은 크기를 쓴다."""
    gain_threshold = drawdown_threshold if gain_threshold is None else gain_threshold
    d = pd.DataFrame({"날짜": dates, "종가": price, "선물외국인": flow}).dropna(subset=["선물외국인"])
    d["slope"] = _rolling_slope(d["선물외국인"], window)
    d["drawdown"] = forward_max_drawdown(d["종가"], horizon)
    d["gain"] = forward_max_gain(d["종가"], horizon)
    valid = d.dropna(subset=["slope", "drawdown", "gain"])

    result = {
        "n": len(valid), "lo_n": 0, "rest_n": 0, "lo_rate": None, "rest_rate": None, "base_rate": None,
        "p_value": None, "lo_up_rate": None, "rest_up_rate": None, "base_up_rate": None, "up_p_value": None,
        "lo_cutoff": None, "current_slope": None, "current_regime": None,
    }
    slope_all = d["slope"].dropna()
    if len(slope_all) > 0:
        result["current_slope"] = float(slope_all.iloc[-1])

    if len(valid) < 30:
        return result

    valid = valid.assign(
        downtrend=(valid["drawdown"] <= -drawdown_threshold).astype(float),
        uptrend=(valid["gain"] >= gain_threshold).astype(float),
    )
    lo_cutoff = valid["slope"].quantile(quantile)
    lo_group = valid[valid["slope"] <= lo_cutoff]
    rest_group = valid[valid["slope"] > lo_cutoff]

    result["lo_n"] = len(lo_group)
    result["rest_n"] = len(rest_group)
    result["lo_rate"] = float(lo_group["downtrend"].mean()) if len(lo_group) else None
    result["rest_rate"] = float(rest_group["downtrend"].mean()) if len(rest_group) else None
    result["base_rate"] = float(valid["downtrend"].mean())
    result["lo_up_rate"] = float(lo_group["uptrend"].mean()) if len(lo_group) else None
    result["rest_up_rate"] = float(rest_group["uptrend"].mean()) if len(rest_group) else None
    result["base_up_rate"] = float(valid["uptrend"].mean())
    result["lo_cutoff"] = float(lo_cutoff)

    if result["lo_rate"] is not None and result["rest_rate"] is not None:
        result["p_value"] = two_proportion_ztest(
            lo_group["downtrend"].sum(), len(lo_group), rest_group["downtrend"].sum(), len(rest_group)
        )
    if result["lo_up_rate"] is not None and result["rest_up_rate"] is not None:
        result["up_p_value"] = two_proportion_ztest(
            lo_group["uptrend"].sum(), len(lo_group), rest_group["uptrend"].sum(), len(rest_group)
        )

    result["current_regime"] = (
        f"강한 매도 경고 (하위 {quantile:.0%})"
        if result["current_slope"] is not None and result["current_slope"] <= lo_cutoff
        else "평상시"
    )
    return result




@st.cache_data(ttl=24 * 3600, show_spinner="불러오는 중...")
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


def backtest_signal(
    df: pd.DataFrame, signal_col: str, horizon: int,
    drawdown_threshold: float = 0.07, gain_threshold: float | None = None,
) -> dict:
    gain_threshold = drawdown_threshold if gain_threshold is None else gain_threshold
    d = df.copy()
    d["fwd_return"] = d["종가"].shift(-horizon) / d["종가"] - 1
    d["drawdown"] = forward_max_drawdown(d["종가"], horizon)
    d["gain"] = forward_max_gain(d["종가"], horizon)
    valid = d.dropna(subset=[signal_col, "fwd_return", "drawdown", "gain"])

    result = {
        "signal": signal_col, "horizon": horizon,
        "n": len(valid), "pos_n": 0, "neg_n": 0,
        "pos_mean": None, "neg_mean": None, "p_value": None, "corr": None,
        "pos_down_rate": None, "neg_down_rate": None, "base_down_rate": None, "down_p_value": None,
        "pos_up_rate": None, "neg_up_rate": None, "base_up_rate": None, "up_p_value": None,
        "current_value": None, "current_regime": None,
    }
    # 현재 상태는 향후 수익률 계산 없이 전체 이력에서 바로 판단한다 (최근 horizon일은
    # fwd_return이 아직 계산 안 돼 valid에서 빠지므로, valid 기준으로 뽑으면 horizon일 지연된 값이 된다).
    signal_all = d[signal_col].dropna()
    if len(signal_all) > 0:
        current_value = float(signal_all.iloc[-1])
        result["current_value"] = current_value
        result["current_regime"] = "양수" if current_value > 0 else "음수"

    if len(valid) < 30:
        return result

    valid = valid.assign(
        down=(valid["drawdown"] <= -drawdown_threshold).astype(float),
        up=(valid["gain"] >= gain_threshold).astype(float),
    )
    pos = valid[valid[signal_col] > 0]
    neg = valid[valid[signal_col] < 0]
    result["pos_n"], result["neg_n"] = len(pos), len(neg)
    result["pos_mean"] = pos["fwd_return"].mean() if len(pos) else None
    result["neg_mean"] = neg["fwd_return"].mean() if len(neg) else None
    result["corr"] = float(valid[signal_col].corr(valid["fwd_return"]))

    result["pos_down_rate"] = float(pos["down"].mean()) if len(pos) else None
    result["neg_down_rate"] = float(neg["down"].mean()) if len(neg) else None
    result["base_down_rate"] = float(valid["down"].mean())
    result["pos_up_rate"] = float(pos["up"].mean()) if len(pos) else None
    result["neg_up_rate"] = float(neg["up"].mean()) if len(neg) else None
    result["base_up_rate"] = float(valid["up"].mean())

    if len(pos) >= 2 and len(neg) >= 2:
        from scipy import stats as scistats
        _, pval = scistats.ttest_ind(pos["fwd_return"], neg["fwd_return"], equal_var=False)
        result["p_value"] = float(pval)
        result["down_p_value"] = two_proportion_ztest(pos["down"].sum(), len(pos), neg["down"].sum(), len(neg))
        result["up_p_value"] = two_proportion_ztest(pos["up"].sum(), len(pos), neg["up"].sum(), len(neg))

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


def compute_flow_signal(df: pd.DataFrame) -> pd.Series:
    """기관 순매수(주)를 최근 평균 거래량으로 나눈 뒤 20일 누적한 값.

    거래량으로 나누는 이유는 절대 주식 수가 종목·시기마다 규모가 달라서다. 나눠주면
    '최근 하루 거래량의 몇 배만큼을 기관이 순매수했는가'라는 비교 가능한 단위가 된다.
    양수면 기관 순매수 우위, 음수면 순매도 우위.
    """
    if df.empty or "기관" not in df.columns or "거래량" not in df.columns:
        return pd.Series(dtype=float)
    volume = pd.to_numeric(df["거래량"], errors="coerce")
    inst = pd.to_numeric(df["기관"], errors="coerce")
    normalized = inst / volume.rolling(FLOW_SIGNAL_VOL_WINDOW).mean()
    return normalized.rolling(FLOW_SIGNAL_WINDOW).sum()


def backtest_flow_signal(df: pd.DataFrame, slippage: float = 0.0010) -> dict:
    """신호가 양수인 구간만 보유하는 전략을 비용까지 반영해 단순보유와 비교한다.

    체결 가정: 신호는 장 마감 후 확정되므로 FLOW_SIGNAL_EXEC_LAG일 뒤부터 수익에 반영한다.
    """
    cost_buy = 0.00015 + slippage
    cost_sell = 0.00015 + 0.0015 + slippage

    d = df.copy()
    d["신호"] = compute_flow_signal(d)
    d["수익률"] = pd.to_numeric(d["종가"], errors="coerce").pct_change()
    d = d.dropna(subset=["신호", "수익률"]).reset_index(drop=True)

    result = {"n": len(d), "ok": False}
    if len(d) < FLOW_SIGNAL_WINDOW * 3:
        return result

    position = (d["신호"].shift(FLOW_SIGNAL_EXEC_LAG) > 0).astype(float).fillna(0.0)
    change = position.diff().fillna(position.iloc[0])
    cost = change.clip(lower=0) * cost_buy + (-change).clip(lower=0) * cost_sell
    strategy_ret = position * d["수익률"] - cost

    def metrics(returns: pd.Series) -> dict:
        equity = (1 + returns).cumprod()
        years = len(returns) / 252
        vol = returns.std() * np.sqrt(252)
        downside = returns[returns < 0].std() * np.sqrt(252)
        return {
            "총수익": float(equity.iloc[-1] - 1),
            "CAGR": float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else float("nan"),
            "MDD": float((equity / equity.cummax() - 1).min()),
            "Sharpe": float(returns.mean() * 252 / vol) if vol > 0 else float("nan"),
            "Sortino": float(returns.mean() * 252 / downside) if downside and downside > 0 else float("nan"),
            "equity": equity,
        }

    result.update({
        "ok": True,
        "날짜": d["날짜"],
        "종가": d["종가"],
        "신호": d["신호"],
        "포지션": position,
        "전략": metrics(strategy_ret),
        "보유": metrics(d["수익률"]),
        "거래횟수": int((change != 0).sum()),
        "노출": float(position.mean()),
        "현재신호": float(d["신호"].iloc[-1]),
        "기간": (d["날짜"].iloc[0], d["날짜"].iloc[-1]),
    })
    return result




# 국내 증권사가 파는 '미국주식 주간거래(데이장)'는 Blue Ocean ATS의 오버나이트 세션
# (미국 동부 20:00–04:00 = 한국 09:00–17:00)이다. 브로커 앱에서는 이 시간에 SKHY 값이
# 움직이는데 대시보드는 '마감'으로 뜨니, 왜 다른지 화면에서 바로 알 수 있게 적어둔다.
# 2026-08-21 한국 10:24(세션 진행 중)에 확인: 야후 1분봉은 이 구간 봉이 0개이고,
# 네이버 해외주식도 marketStatus=CLOSE / overMarketPriceInfo=null 이었다.
ADR_DAY_SESSION_NOTE = (
    "**국내 증권사 '주간거래(데이장)'와 다를 수 있습니다.** 주간거래는 한국 09:00–17:00에 "
    "돌아가는 미국 오버나이트 세션(Blue Ocean ATS)인데, 그 체결가는 증권사 유료 시세라 "
    "무료 공개 경로(야후·네이버)에 나오지 않습니다. 이 화면은 미국 프리장–정규장–애프터장만 "
    "반영하므로, 한국 낮 시간에는 값이 멈춰 있는 게 맞습니다."
)








@st.cache_data(ttl=60, show_spinner=False)
def fetch_adr_intraday() -> pd.DataFrame:
    """SKHY의 가장 최근 하루치 1분봉을 프리장–애프터장 전 구간(04:00–20:00 ET) 가져온다.

    미국 거래시간을 한국시간으로 바꾸면 17:00 – 익일 09:00(서머타임 기준)이라 자정을 넘는다.
    그래도 '하루 흐름'으로 이어 보는 게 목적이므로 그대로 시계열로 둔다.

    range=1d로 받으면 안 되는 이유: 미국 애프터장이 끝나는 20:00 ET(한국 09:00)부터
    다음 프리장이 열리는 04:00 ET(한국 17:00)까지 야후가 '오늘'을 아직 시작 안 한 날로 잡아
    빈 응답을 준다. 그러면 한국 낮 시간 내내 ADR 그래프가 통째로 사라진다.
    2일치를 받아서 데이터가 있는 마지막 미국 거래일만 잘라 쓴다.
    """
    bars = _fetch_adr_bars()
    if bars.empty:
        return pd.DataFrame({"시각": pd.Series(dtype="datetime64[ns]"),
                             "가격": pd.Series(dtype="float64"),
                             "세션": pd.Series(dtype="object")})
    latest = bars[bars["거래일"] == bars["거래일"].max()]
    return latest.drop(columns="거래일").reset_index(drop=True)


















# ── 외국계 창구 추정 순매수 (장중) ──────────────────────────────────────────────
# 종목별 투자자 3분류(외국인/기관/개인)는 장중에 공개되지 않는다. 무료 경로뿐 아니라
# 증권사 공식 API도 '종목별 투자자매매동향(일별)'만 있고, 장중은 '회원사 실시간 매매동향(틱)'
# 즉 거래원 기준뿐이다. 거래소가 그렇게 공개하기 때문이지 출처를 못 찾아서가 아니다.
#
# 다만 네이버 종목 페이지의 거래원 표에는 '외국계추정합'이 있고 이건 장중에 갱신된다.
# 외국계 증권사 창구를 거친 거래만 합산한 추정치라 확정 수급과 다르고 외국인만 잡히지만,
# '오늘 외국인이 사는 쪽인가 파는 쪽인가'의 방향은 장중에 볼 수 있는 유일한 값이다.
FOREIGN_DESK_NOTE = (
    "외국계 증권사 창구를 거친 거래만 합산한 **추정치**입니다. 국내 증권사로 주문한 외국인은 "
    "빠지고, 외국계 창구를 쓴 내국인은 섞입니다. 마감 후 확정 수급과 다릅니다."
)




def _render_foreign_desk_line() -> None:
    """이 종목의 장중 외국계 추정 순매수를 한 줄로. 현재가 fragment 안이라 자동 갱신된다."""
    try:
        d = fetch_foreign_desk(TICKER)
    except Exception:
        return
    if not d:
        return
    net = d["순매수"]
    color = "green" if net > 0 else ("red" if net < 0 else "gray")
    word = "순매수" if net > 0 else ("순매도" if net < 0 else "보합")
    stamp = f" · {d['기준']} 조회" if d.get("기준") else ""
    _bold_label_with_help(
        f"외국계 창구 추정 {word} :{color}[{abs(net):,.0f}주]{stamp}",
        "종목별 외국인·기관·개인 수급은 장 마감 후에야 공개됩니다. 장중에 볼 수 있는 건 "
        f"거래원(증권사 창구) 기준인 이 값뿐입니다. {FOREIGN_DESK_NOTE}\n\n"
        "**기관·개인은 여기에 없습니다.** 갱신 간격은 45초~6분으로 일정하지 않고, "
        "표시된 시각은 페이지를 조회한 시각입니다.",
        key="foreign_desk",
    )
































@st.cache_data(ttl=1800, show_spinner="AI가 분류하는 중...")
def classify_sentiment_ai(titles: tuple[str, ...]) -> list[str]:
    """제목을 AI에 보내 긍정/부정/중립으로 분류한다. 키워드 매칭과 달리 반어법·비꼬는 말투·문맥을 고려할 수 있다.
    분류 결과는 제목 목록(titles) 기준으로 캐시하므로, 같은 게시글에 대해 재실행 시 API를 다시 호출하지 않는다."""
    numbered = "\n".join(f"{i}|{t}" for i, t in enumerate(titles))
    prompt = f"""다음은 주식 종목토론방 게시글 제목 목록입니다. 번호와 제목이 '|'로 구분되어 있습니다.
각 제목이 그 종목에 대해 긍정적(주가 상승 기대/호재)인지, 부정적(주가 하락 우려/악재)인지, 중립(감정이 뚜렷하지 않거나 판단하기 어려움)인지 분류해줘.
반어법이나 비꼬는 말투도 문맥을 보고 판단해줘 (예: "가즈아 220만원 ㅋㅋ 꿈 깨라"는 반어적 부정, "물렸다 ㅋㅋ 그래도 존버"는 부정).

{numbered}

아래 형식을 정확히 지켜서, 위 목록의 모든 번호에 대해 빠짐없이 한 줄씩 답변해줘 (다른 설명 없이):
번호|긍정 또는 부정 또는 중립"""
    raw_text, _ = _call_gemini(prompt)

    labels: dict[int, str] = {}
    for line in raw_text.strip().splitlines():
        m = re.match(r"\s*(\d+)\s*\|\s*(긍정|부정|중립)", line)
        if m:
            labels[int(m.group(1))] = m.group(2)
    return [labels.get(i, "중립") for i in range(len(titles))]




def _fetch_dc_post_content(post_no: str) -> str:
    resp = requests.get(DC_GALLERY_VIEW_URL, params={"id": DC_GALLERY_ID, "no": post_no}, headers=DC_HEADERS, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    content = soup.find("div", class_="write_div")
    return content.get_text(" ", strip=True) if content else ""


def _find_korean_font() -> str | None:
    for path in KOREAN_FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def extract_korean_word_freq(texts: list[str], min_len: int = 2) -> Counter:
    combined = re.sub(r"http\S+", " ", " ".join(texts))
    tokens = re.findall(r"[가-힣]{%d,}" % min_len, combined)
    return Counter(t for t in tokens if t not in KOREAN_STOPWORDS)


def _wordcloud_sentiment_color(word, font_size, position, orientation, random_state=None, **kwargs):
    if any(kw in word for kw in POSITIVE_KEYWORDS):
        return "green"
    if any(kw in word for kw in NEGATIVE_KEYWORDS):
        return "red"
    return "gray"


def render_wordcloud_image(word_freq: Counter, max_words: int = 80):
    """단어 크기는 언급 빈도에 비례하고(WordCloud 기본 동작), 호재 키워드가 포함된 단어는 초록색,
    악재 키워드가 포함된 단어는 빨간색, 나머지는 회색으로 칠한다."""
    font_path = _find_korean_font()
    if font_path is None or not word_freq:
        return None
    wc = WordCloud(
        font_path=font_path, width=900, height=450, background_color="white",
        max_words=max_words, color_func=_wordcloud_sentiment_color,
    ).generate_from_frequencies(word_freq)
    return wc.to_image()


def curate_good_dc_posts(posts_df: pd.DataFrame, stock_name: str, max_posts: int = 20) -> tuple[str, list[dict]]:
    """게시글 본문을 가져와 감정적 비방·잡담이 아닌 근거 기반 분석글을 AI가 추려낸다."""
    subset = posts_df.head(max_posts)
    items = []
    for _, row in subset.iterrows():
        try:
            content = _fetch_dc_post_content(row["번호"])
        except requests.RequestException:
            content = ""
        items.append(f"[글번호 {row['번호']}] 제목: {row['제목']}\n내용: {content[:400] if content else '(내용 없음)'}")
    prompt = f"""다음은 디시인사이드 주식갤러리(krstock)에서 '{stock_name}' 관련 검색으로 찾은 게시글 목록입니다.
이 갤러리는 욕설·비방·밈·단순 잡담 비중이 매우 높으니 그런 글은 반드시 제외하고,
실제 데이터나 근거를 바탕으로 {stock_name}에 대한 분석이나 의견을 제시하는 게시글만 최대 5개 골라줘.

{chr(10).join(f"{chr(10)}{item}" for item in items)}

아래 형식을 정확히 지켜서, 한 줄에 하나씩 답변해줘 (다른 설명 없이):
글번호|이유
해당하는 게시글이 하나도 없으면 "없음" 한 단어만 답해줘."""
    raw_text, _ = _call_gemini(prompt)

    picks = []
    for line in raw_text.strip().splitlines():
        m = re.match(r"\s*(\d+)\s*\|\s*(.+)", line)
        if not m:
            continue
        post_no, reason = m.group(1), m.group(2).strip()
        match_row = subset[subset["번호"] == post_no]
        if match_row.empty:
            continue
        row = match_row.iloc[0]
        picks.append({"제목": row["제목"], "url": row["url"], "이유": reason})
    return raw_text, picks














# 상승/하락 글자색. 밝은 테마와 다크 모드 양쪽에서 읽히는 톤으로 고른다.
_UP_COLOR = "#1a9e5f"
_DOWN_COLOR = "#e04b4b"


def _pct_text_color(value: object) -> str:
    """'+1.90%' / '-0.64%' 형태 문자열을 보고 글자색 CSS를 돌려준다 (0.00%·N/A는 기본색)."""
    if isinstance(value, str):
        if value.startswith("+"):
            return f"color: {_UP_COLOR}"
        if value.startswith("-"):
            return f"color: {_DOWN_COLOR}"
    return ""


def _stale_note(last_update: str | None, warn_days: int = 8) -> str:
    """사이트 기준일이 오래됐으면 그 사실을 한 마디로. 표의 'N일 전 대비'는 이 기준일
    시점에서 센 값이라, 기준일이 밀리면 라벨과 실제가 어긋난다(모듈은 원래 주 1회 갱신인데
    2026-08-24 이후 2주간 멈춰 있었다). 숫자만 보고 오늘 오른 것으로 읽지 않게 알린다."""
    if not last_update:
        return ""
    try:
        d = dt.datetime.strptime(str(last_update)[:10], "%Y-%m-%d").date()
    except Exception:
        return ""
    days = (dt.datetime.now(om.KST).date() - d).days
    if days < warn_days:
        return ""
    return (f"  ⚠️ {days}일째 갱신이 없습니다. 아래 변동률은 모두 이 기준일 시점에서 센 값이라 "
            "오늘까지의 변화가 아닙니다.")


def _render_dram_price_table(display_df: pd.DataFrame, pct_cols: list[str]) -> None:
    """변동률 열을 상승=초록 / 하락=빨강으로 칠해서 표를 그린다."""
    styler = display_df.style.map(_pct_text_color, subset=pct_cols)
    st.table(styler, width="stretch", hide_index=True)


def _sentiment_text_color(value: object) -> str:
    """'긍정'/'부정' 글자를 각각 초록/빨강으로 칠한다 ('중립'은 기본색)."""
    if value == "긍정":
        return f"color: {_UP_COLOR}"
    if value == "부정":
        return f"color: {_DOWN_COLOR}"
    return ""


def _period_change_pct(history: pd.DataFrame, item: str, days: int) -> float | None:
    """history에 쌓인 이력에서 해당 품목의 가장 최근 값과, days일 이전 시점에서 가장 가까운(그 이전) 값을
    비교한 변동률(%)을 계산한다. days일보다 오래된 기록이 없으면 None을 반환한다."""
    item_hist = history[history["품목"] == item].sort_values("날짜")
    if item_hist.empty:
        return None
    latest_row = item_hist.iloc[-1]
    cutoff = latest_row["날짜"] - pd.Timedelta(days=days)
    past_candidates = item_hist[item_hist["날짜"] <= cutoff]
    if past_candidates.empty:
        return None
    past_price = past_candidates.iloc[-1]["평균가(USD)"]
    if not past_price:
        return None
    return (latest_row["평균가(USD)"] / past_price - 1) * 100


def save_dram_snapshot(
    module_df: pd.DataFrame, module_last_update: str | None,
    chip_df: pd.DataFrame, chip_last_update: str | None,
) -> pd.DataFrame:
    """모듈가/칩가 표는 DRAMeXchange에서 서로 다른 시각에 갱신되므로, 각각 사이트에 표시된
    'Last Update' 시각을 '날짜'로 기록한다. 이 값이 이전 기록과 같으면(=사이트가 아직 안 바뀌었으면)
    drop_duplicates에서 그대로 덮어써져 이력에 새 행이 늘지 않으므로, 결과적으로 사이트의
    Last Update가 실제로 바뀔 때만 새 데이터 포인트가 쌓인다."""
    fallback = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    snapshots = []
    if not module_df.empty:
        snap = module_df.copy()
        snap.insert(0, "날짜", module_last_update or fallback)
        snapshots.append(snap)
    if not chip_df.empty:
        snap = chip_df.copy()
        snap.insert(0, "날짜", chip_last_update or fallback)
        snapshots.append(snap)

    if not snapshots:
        return pd.DataFrame(columns=["날짜", "품목", "평균가(USD)", "변동률(%)", "방향"])
    snapshot = pd.concat(snapshots, ignore_index=True)
    snapshot["날짜"] = pd.to_datetime(snapshot["날짜"])

    os.makedirs(os.path.dirname(DRAM_HISTORY_FILE) or ".", exist_ok=True)
    try:
        if os.path.exists(DRAM_HISTORY_FILE):
            history = pd.read_csv(DRAM_HISTORY_FILE, parse_dates=["날짜"])
            history = pd.concat([history, snapshot], ignore_index=True)
            history = history.drop_duplicates(subset=["날짜", "품목"], keep="last")
        else:
            history = snapshot
        history = history.sort_values("날짜")
        history.to_csv(DRAM_HISTORY_FILE, index=False)
        return history
    except OSError:
        return snapshot


# 회사별 고정 색상. px.bar에 color_discrete_map으로 그대로 넘겨 그래프 색을 고정하고,
# "표시 기업" 체크박스 라벨에도 같은 색의 사각형 이모지를 붙여 한눈에 매칭되게 한다.
CAPEX_COMPANY_COLORS = {
    "Alphabet(Google)": "#2ca02c",
    "Amazon": "#ff7f0e",
    "Meta": "#9467bd",
    "Microsoft": "#1f77b4",
}
CAPEX_COMPANY_EMOJI = {
    "Alphabet(Google)": "🟩",
    "Amazon": "🟧",
    "Meta": "🟪",
    "Microsoft": "🟦",
}




# ── FnGuide 재무 데이터 ────────────────────────────────────────────────────────
# 파싱은 fnguide.py에 있다. 수집기(collector.py)도 같은 코드를 써야 해서 밖으로 뺐다.
# 여기서는 캐시만 씌운다.




def fetch_financials(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        return fnguide.financial_frames(fetch_fnguide_page(ticker))
    except Exception:
        return pd.DataFrame(), pd.DataFrame()









# ── AI 분석에 넘길 추가 재료 ──────────────────────────────────────────────────


# Capex 그래프 보기 방식. 순서가 화면 라디오 버튼 순서이자 아래 분기 조건의 기준이다.
# '연도별 누적(YTD)'도 있었는데 뺐다. TTM과 같은 말(투자가 가속 중)을 할 뿐인데
# 1월이 되면 3개월치로 쪼그라들어 그때는 못 쓰고, 날짜 축에 그리면 해마다 0으로 떨어지는
# 톱니가 되어 '작년 같은 분기와 비교'라는 원래 목적도 눈으로 할 수 없었다.
CAPEX_VIEW_MODES = ("분기별", "최근 4분기 합(TTM)", "전체 누적")


def _capex_accumulate(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """기업별로 누적 방식을 적용한 표를 돌려준다.

    그냥 처음부터 더하기만 하는 '전체 누적'은 시작점이 데이터를 어디부터 받아왔는지로
    정해질 뿐이라 정보량이 적다. 추세를 보려면 TTM(최근 4분기 합)을 쓴다.
    """
    out = []
    for company, g in df.sort_values("분기말").groupby("기업"):
        g = g.copy()
        if mode == CAPEX_VIEW_MODES[1]:          # TTM
            g["capex_B"] = g["capex_B"].rolling(4).sum()
        else:                                    # 전체 누적
            g["capex_B"] = g["capex_B"].cumsum()
        out.append(g)
    if not out:
        return df
    return pd.concat(out, ignore_index=True).dropna(subset=["capex_B"])


















def record_consensus_snapshot(ticker: str, snapshot: dict) -> None:
    """하루 한 번 컨센서스를 파일에 남긴다. 같은 날 중복 기록은 하지 않는다."""
    target = _to_number(snapshot.get("목표주가"))
    opinion = _to_number(snapshot.get("투자의견"))
    if not target:
        return
    today = dt.datetime.now(om.KST).date().isoformat()
    try:
        os.makedirs(os.path.dirname(CONSENSUS_LOG) or ".", exist_ok=True)
        # '파일이 있나'가 아니라 '내용이 있나'로 판단한다. 빈 파일이 한 번 생기면
        # (쓰는 도중 프로세스가 죽으면 그렇게 된다) read_csv가 EmptyDataError를 내는데,
        # 예전 코드는 그 예외를 통째로 삼켜서 그 뒤로 영원히 기록이 안 됐다.
        has_rows = os.path.exists(CONSENSUS_LOG) and os.path.getsize(CONSENSUS_LOG) > 0
        if has_rows and not _consensus_log(ticker).empty:
            if today in set(_consensus_log(ticker)["날짜"].astype(str)):
                return
        with open(CONSENSUS_LOG, "a", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            if not has_rows:
                w.writerow(["날짜", "종목", "목표주가", "투자의견"])
            w.writerow([today, ticker, f"{target:.0f}", f"{opinion:.2f}" if opinion else ""])
    except Exception:
        pass          # 기록 실패로 분석이 멈추면 안 된다


















def missing_tab_summaries() -> list[str]:
    """AI가 못 받은 항목. 화면에서 사용자에게 알려주려고 쓴다.

    통합 신호·선물 경보는 애초에 프롬프트에 안 넣으므로 여기서도 세지 않는다.
    """
    pairs = ((overheat_summary, "가격 과열도"),)
    return [name for value, name in pairs
            if not value or any(k in value for k in _TAB_SUMMARY_UNSET)]




def _live_deviation(live_price: float, ma_window: int) -> tuple[float | None, float | None]:
    """장중 현재가를 시계열의 마지막 값으로 놓고 이동평균 대비 괴리율을 계산한다.
    가격 과열도 탭은 확정된 종가로 계산하므로, 장중에는 이 값이 그쪽보다 앞서 움직인다.
    과거 괴리율 분포에서의 백분위(상위 N%)도 함께 돌려준다."""
    # 250일이면 MA(최대 120)와 백분위 분포에 충분하다. 이 함수는 화면 상단 조각에서
    # 불리는데, 700일(네이버 ~35페이지, 5~7초)을 받으면 첫 화면이 그만큼 늦게 뜬다.
    # 700일 전체는 가격 과열도 탭이 따로 받는다(그쪽은 지연 렌더라 첫 화면을 안 막는다).
    hist = fetch_backtest_history_live(TICKER, target_days=250)
    if hist.empty or len(hist) < ma_window:
        return None, None
    closes = hist["종가"].astype(float).reset_index(drop=True)
    dates = pd.to_datetime(hist["날짜"]).reset_index(drop=True)
    # 오늘자 행이 이미 있으면 그 종가를 현재가로 갈아끼우고, 없으면(장 시작 직후 등) 뒤에 붙인다.
    if len(dates) > 0 and dates.iloc[-1].date() == dt.date.today():
        closes.iloc[-1] = live_price
    else:
        closes = pd.concat([closes, pd.Series([float(live_price)])], ignore_index=True)

    deviation = (closes / closes.rolling(ma_window).mean() - 1).dropna()
    if deviation.empty:
        return None, None
    current = float(deviation.iloc[-1])
    percentile = float((deviation >= current).mean())
    return current, percentile




def _note_optional_failure(what: str, exc: Exception) -> None:
    """현재가 화면의 '있으면 좋은' 부품이 실패했을 때 쓰는 처리.

    조용히 넘기면(except: pass) 화면에서 그 부분만 소리 없이 사라져서, 실제로
    장중 그래프가 통째로 없어진 걸 한참 뒤에야 알아챈 적이 두 번 있었다
    (빈 DataFrame dtype 문제, plotly 5.x가 모르는 속성 문제).
    나머지 화면은 그대로 두되 무엇이 왜 빠졌는지는 반드시 남긴다."""
    # 예외 종류와 메시지만으로는 어느 줄에서 났는지 못 찾는다. 실제로 '.dt accessor' 오류가
    # 났을 때 후보 줄을 하나씩 눌러보느라 시간을 버렸다. 트레이스백까지 남긴다.
    print(f"[render_current_price] {what} 실패: {type(exc).__name__}: {exc}", flush=True)
    print("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), flush=True)
    st.caption(f":gray[{what}를 표시하지 못했습니다 ({type(exc).__name__}).]")


# 시간외 시세 관련 로직은 over_market.py에 모아두고, 화면에서는 얇게 감싸서 쓴다.
# 실제 수집은 collector.py(별도 컨테이너)가 화면 접속과 무관하게 상시로 돌린다.
save_over_market_tick = om.save_tick
_parse_price_number = om.parse_price_number
_OVER_SESSION_LABELS = om.SESSION_LABELS
OVER_MARKET_COLLECT_TICKERS = om.COLLECT_TICKERS


@st.fragment(run_every=REFRESH_SEC)
def render_current_price():
    """현재가·시가·고가·저가·거래량과 프리장·ADR 지표. 5초마다 돈다.

    장중 차트와 코스피 수급은 여기 있었지만 갱신 주기가 달라 따로 뺐다.
    한 덩어리로 5초마다 다시 그리면 분봉 차트까지 매번 새로 그려서, 화면이
    쉬지 않고 흐려졌다 진해졌다 했다.
    """
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

        with st.container(key="price_row_columns"):
            # 시가총액은 폴링 응답에 marketValueFull(원 단위)로 이미 들어 있다.
            # 주가와 같은 주기로 갱신되므로 다른 지표와 같은 줄에 둔다.
            cap_raw = _to_number(data.get("marketValueFullRaw")) or _to_number(data.get("marketValueFull"))
            cap_txt = f"{cap_raw / 1e12:,.1f}조" if cap_raw else "-"

            (price_col, open_col, high_col,
             low_col, volume_col, cap_col) = st.columns(6)
            price_col.metric(
                label="현재가 (시세 지연)",
                value=f"{close_price:,}원",
                delta=f"{change:+,}원 ({change_pct:+.2f}%)",
                delta_color="normal",
            )
            for col, label, value, key in [
                (open_col, "시가", open_p, "open"),
                (high_col, "고가", high_p, "high"),
                (low_col, "저가", low_p, "low"),
                (volume_col, "거래량", volume, "volume"),
            ]:
                with col.container(key=f"metric_small_{key}"):
                    st.metric(label, value)
            with cap_col.container(key="metric_small_marketcap"):
                _metric_with_help(
                    "시가총액", cap_txt,
                    "상장예정주식수까지 포함한 시가총액입니다(보통주+우선주). "
                    "주가와 같은 주기로 갱신되므로 장중에는 계속 바뀝니다.",
                    key="marketcap",
                )

        # 프리장/애프터장(NXT) 실시간 시세. 정규장이 닫혀 있어도 이 구간에는 값이 움직인다.
        # 기준가는 '직전 정규장 종가'로 잡는다 — 프리장이면 전 거래일 종가, 애프터장이면 당일 종가라
        # 어느 쪽이든 '정규장 대비 지금 얼마나 움직였나'가 된다.
        over = data.get("overMarketPriceInfo") or {}
        over_price = _parse_price_number(over.get("overPrice"))
        if over.get("overMarketStatus") == "OPEN" and over_price is not None:
            session_label = _OVER_SESSION_LABELS.get(over.get("tradingSessionType"), "시간외")
            over_diff = over_price - close_price
            over_pct = (over_diff / close_price * 100) if close_price else 0.0
            over_volume = over.get("accumulatedTradingVolume", "-")
            try:
                over_at = pd.to_datetime(over.get("localTradedAt")).strftime("%H:%M:%S")
            except (ValueError, TypeError):
                over_at = None
            # 백그라운드 수집기가 기본 종목만 담당하므로, 다른 종목을 보고 있을 때는 화면에서도 기록한다
            if TICKER not in OVER_MARKET_COLLECT_TICKERS:
                save_over_market_tick(TICKER, over_price, over.get("localTradedAt"), session_label)

            with st.container(key="price_row_over"):
                over_col, over_vol_col = st.columns([2, 3])
                with over_col.container(key="metric_small_over_price"):
                    _metric_with_help(
                        f"{session_label} (NXT) 실시간",
                        f"{over_price:,.0f}원",
                        "정규장 종가 대비 변동입니다. 프리장은 전 거래일 종가, 애프터장은 당일 종가가 기준입니다.",
                        key="over_price",
                        delta=f"{over_diff:+,.0f}원 ({over_pct:+.2f}%)",
                        delta_color="normal",
                    )
                with over_vol_col.container(key="metric_small_over_volume"):
                    st.metric(f"{session_label} 거래량", over_volume)
            if over_at:
                st.caption(f"{session_label} 갱신시각: {over_at}")
        try:
            ma_window = int(st.session_state.get("overheat_ma_window", OVERHEAT_DEFAULT_MA_WINDOW))
            deviation, percentile = _live_deviation(close_price, ma_window)
            if deviation is not None:
                # 이 영역은 5초마다 다시 그려져서 팝오버를 열어둬도 곧 닫히므로, 설명은
                # 가격 과열도 탭의 ❓ 버튼에 모아두고 여기서는 라벨만으로 뜻이 통하게 둔다.
                with st.container(key="price_row_deviation"):
                    dev_col, adr_col, adr_krw_col, adr_gap_col = st.columns(4)
                    with dev_col.container(key="metric_small_deviation"):
                        st.metric(
                            f"{ma_window}일선 괴리율 (현재가 기준)",
                            f"{deviation:+.1%} (상위 {percentile:.0%})",
                        )
                    # 나스닥 상장 SK하이닉스(SKHY). 프리장·애프터장 체결까지 반영한다.
                    if TICKER == ADR_HOST_TICKER:
                        adr = fetch_adr_quote()
                        baseline = fetch_adr_baseline()
                        if adr:
                            adr_krw = adr["price"] * adr["fx"]
                            prev = adr.get("prev_close")
                            adr_delta = (
                                f"{(adr['price'] / prev - 1) * 100:+.2f}%" if prev else None
                            )
                            # 등락률이 무엇 대비인지 헷갈리지 않게 기준값을 그대로 적어준다
                            basis = (
                                "당일 정규장 종가" if adr["session"] == "애프터장" else "직전 거래일 종가"
                            )
                            basis_help = (
                                f"등락률은 {basis} ${prev:,.2f} 대비입니다."
                                if prev else "등락률 기준값을 구하지 못했습니다."
                            )
                            with adr_col.container(key="metric_small_adr"):
                                # 장이 닫혀 있으면 값이 안 움직이는 게 정상이라는 걸 라벨에서 바로 알 수 있게 한다
                                if adr.get("is_open"):
                                    adr_label = f"SKHY ({adr['session']})"
                                    adr_help = (
                                        f"나스닥 상장 SK하이닉스. 마지막 체결 {adr['time'] or '-'} KST "
                                        f"(환율 {adr['fx']:,.1f}원). 미국 프리장·애프터장 체결도 반영합니다.\n\n"
                                        f"{basis_help}"
                                    )
                                else:
                                    adr_label = "SKHY (미국장 마감)"
                                    adr_help = (
                                        f"미국장이 닫혀 있어 값이 멈춰 있는 게 정상입니다.\n\n"
                                        f"마지막 체결: {adr['time'] or '-'} KST ({adr['session']})\n\n"
                                        f"다음 프리장 개장: {adr['next_open'] or '-'} KST\n\n"
                                        f"미국 거래시간(KST): 프리장 17:00–22:30, 정규장 22:30–익일 05:00, "
                                        f"애프터장 –익일 09:00 (서머타임 기준)\n\n"
                                        f"{ADR_DAY_SESSION_NOTE}\n\n"
                                        f"{basis_help}"
                                    )
                                _metric_with_help(
                                    adr_label, f"${adr['price']:,.2f}", adr_help, key="adr",
                                    delta=adr_delta, delta_color="normal",
                                )
                            # ADR 1주는 본주 0.1주에 해당하므로, 본주 환산가로 되돌려 비교한다
                            adr_per_share = adr_krw / ADR_SHARE_RATIO
                            with adr_krw_col.container(key="metric_small_adr_krw"):
                                _metric_with_help(
                                    "SKHY 본주환산", f"{adr_per_share:,.0f}원",
                                    f"ADR ${adr['price']:,.2f} × 환율 {adr['fx']:,.1f} = {adr_krw:,.0f}원 "
                                    f"(ADR 1주). 공식 비율 1 ADR = 본주 {ADR_SHARE_RATIO}주로 나눠 "
                                    "본주 1주 기준으로 환산한 금액입니다.",
                                    key="adr_krw",
                                )
                            with adr_gap_col.container(key="metric_small_adr_gap"):
                                gap = (adr_per_share / close_price - 1) * 100
                                delta = None
                                if baseline:
                                    base_gap = (baseline / ADR_SHARE_RATIO - 1) * 100
                                    delta = f"{gap - base_gap:+.1f}%p vs 최근평균"
                                _metric_with_help(
                                    "ADR 괴리율", f"{gap:+.1f}%",
                                    f"공식 비율 1 ADR = 본주 {ADR_SHARE_RATIO}주 기준으로, ADR이 본주보다 "
                                    "얼마나 비싸게 거래되는지입니다.\n\n"
                                    "이 종목은 평소에도 30–40%대 프리미엄이 붙어 있어서, 절대값보다 "
                                    "'최근 평균 대비 얼마나 벌어졌나'(아래 숫자)가 더 의미 있습니다.\n\n"
                                    "한국 종가와 미국 시세는 최대 13시간 차이가 나므로, 이 값에는 "
                                    "그 사이의 시장 변화가 섞여 있습니다. 차익거래 기회가 아니라 "
                                    "미국 쪽 평가를 보는 선행 지표로 읽으세요.",
                                    key="adr_gap",
                                    delta=delta, delta_color="off",
                                )
        except Exception as exc:
            # 괴리율·ADR은 부가 정보라 현재가 표시는 그대로 두되, 사라진 이유는 남긴다.
            _note_optional_failure("괴리율·ADR", exc)

        st.caption(f"시장상태: {market_status} · 갱신시각: {updated_at}")
        # 장중에도 '종가'라고 적어 넘기는 바람에 AI가 진행 중인 가격을 확정 종가로 말했다.
        # 네이버가 필드 이름을 closePrice로 쓸 뿐 장중에는 현재가다. 상태에 맞는 말로 넘긴다.
        is_open = str(market_status).upper() == "OPEN"
        st.session_state["market_is_open"] = is_open
        st.session_state["market_status_raw"] = market_status
        st.session_state["price_updated_at"] = updated_at
        if is_open:
            _label, _state = "현재가", f"{_korea_session_now() or '정규장'} 진행 중 · {updated_at} 체결"
        else:
            _label, _state = "종가", f"정규장 마감 · {updated_at} 확정"
        st.session_state["current_price_summary"] = (
            f"{_label} {close_price:,}원, 전일대비 {change:+,}원 ({change_pct:+.2f}%) ({_state})")
        # AI 분석 탭에서 컨센서스 목표주가 대비 상승여력을 계산할 때 쓴다
        st.session_state["current_price_value"] = close_price
    except Exception as e:
        st.error(f"현재가 조회에 실패했습니다: {e}")
        st.session_state["current_price_summary"] = "현재가 데이터를 가져오지 못함"


@st.fragment(run_every=INTRADAY_REFRESH_SEC)
def render_intraday_chart():
    """장중 주가 추이. 분봉이라 1분에 한 번이면 충분하다.

    현재가 지표와 같은 5초로 돌리면 같은 그림을 12번 다시 그리게 되는데,
    이 차트가 화면에서 가장 큰 요소라 그 깜빡임이 제일 눈에 띄었다.
    """
    try:
        # 기준선을 그리는 데만 쓴다. fetch_current_price는 ttl=5로 캐시돼 있어서
        # 지표 쪽이 방금 받아 온 값을 그대로 재사용한다(추가 요청 없음).
        _quote = fetch_current_price(TICKER)
        close_price = int(_quote["closePriceRaw"])
        change = int(_quote["compareToPreviousClosePriceRaw"])
        intraday_df = fetch_intraday_price(TICKER)
        today_kst = dt.datetime.now(om.KST).date()

        # 본주 파트와 ADR 파트를 한 그림에 쌓고, 마지막에 버튼으로 묶는다.
        # 예전에는 분봉이 있을 때만 ADR을 붙여서, 분봉이 아직 없는 아침(프리장)에는
        # ADR 그래프가 통째로 사라졌다. 이제 어느 한쪽만 있어도 그 쪽을 보여준다.
        fig_intraday = go.Figure()
        host_traces = 0
        host_shapes: list[dict] = []
        host_title = None
        x_start = x_end = None
        has_over = False
        # 화면에 그린 본주 데이터가 '오늘 것'인지. 휴장일에는 직전 거래일 분봉이 실려서
        # 시간만 보면 장중인 줄 알게 되므로, 기본 화면을 고를 때 같이 본다.
        host_is_today = False
        help_lines: list[str] = []

        # 프리장(08:00–09:00)에는 당일 분봉이 아직 없어 intraday_df가 비어 있다.
        # 이때는 정규장 선 없이 시간외 기록만으로 그린다.
        if intraday_df.empty:
            # 네이버 분봉은 자정을 넘기면 빈 배열이 된다. 그래서 밤에는 오늘 기록도,
            # 어제 분봉도 없어서 본주 쪽이 통째로 사라지고 전환 버튼까지 없어졌다.
            # 다행히 수집기가 08:00–20:00을 20초 간격으로 찍어두므로, 오늘 기록이 없으면
            # 틱 파일에 남아 있는 가장 최근 거래일로 본주 화면을 그려준다.
            over_all = load_over_market_ticks(TICKER, today_kst - dt.timedelta(days=7))
            over_only = over_all[over_all["시각"].dt.date == today_kst]
            fallback_day = None
            if over_only.empty and not over_all.empty:
                fallback_day = over_all["시각"].max().date()
                over_only = over_all[over_all["시각"].dt.date == fallback_day]

            if not over_only.empty:
                day = fallback_day or today_kst
                open_t = dt.datetime.combine(day, dt.time(9, 0))
                close_t = dt.datetime.combine(day, dt.time(15, 30))
                # 네이버는 정규장 시간대의 NXT 체결도 'OVER_MARKET'으로 준다. 라벨을 그대로
                # 믿으면 한낮 체결이 회색으로 칠해지므로, 색은 라벨이 아니라 시각으로 나눈다.
                segments = [
                    ("프리장", over_only[over_only["시각"] < open_t], "#7f7f7f"),
                    ("정규장", over_only[(over_only["시각"] >= open_t)
                                       & (over_only["시각"] <= close_t)], "#d62728"),
                    ("애프터장", over_only[over_only["시각"] > close_t], "#7f7f7f"),
                ]
                for label, seg, color in segments:
                    if seg.empty:
                        continue
                    fig_intraday.add_trace(go.Scatter(
                        x=seg["시각"], y=seg["가격"],
                        # 개장 직후엔 점이 1–2개뿐이라 선만으로는 아무것도 안 보인다.
                        # 마커를 같이 찍어 초반에도 보이게 한다.
                        mode="lines+markers" if len(seg) < 10 else "lines",
                        line=dict(color=color), marker=dict(size=4), name=label,
                        hovertemplate="%{x|%H:%M}  %{y:,.0f}원<extra>" + label + "</extra>",
                    ))
                    host_traces += 1
                has_over = True
                host_is_today = fallback_day is None

                x_start = min(over_only["시각"].min().to_pydatetime(),
                              dt.datetime.combine(day, dt.time(8, 0)))
                x_end = max(over_only["시각"].max().to_pydatetime(),
                            dt.datetime.combine(day, dt.time(9, 0)))
                # 오늘 프리장만 그릴 때는 아직 정규장 전이라 기준선이 '직전 종가'(=close_price).
                # 지난 거래일을 되살려 그릴 때는 그 날의 종가가 close_price이므로 전일 종가로.
                base_price = close_price if host_is_today else close_price - change
                # add_hline(도형)이 아니라 트레이스로 그려야 ADR로 전환할 때 같이 숨는다.
                fig_intraday.add_trace(go.Scatter(
                    x=[x_start, x_end], y=[base_price, base_price], mode="lines",
                    line=dict(color="gray", dash="dash", width=1), opacity=0.6,
                    name="기준 종가", hoverinfo="skip", showlegend=False,
                ))
                host_traces += 1

                if host_is_today:
                    host_title = f"프리장 ({day})"
                    help_lines.append(
                        f"점선은 직전 정규장 종가({base_price:,}원) 기준선입니다. "
                        "09:00에 정규장이 열리면 본장 그래프에 이어붙습니다."
                    )
                else:
                    host_title = f"본주 ({day})"
                    # 정규장 시작·종료 세로선은 아래 공통 코드가 host_shapes로 그린다
                    for boundary in (open_t, close_t):
                        host_shapes.append(dict(
                            type="line", x0=boundary, x1=boundary, yref="paper", y0=0, y1=1,
                            line=dict(color="gray", dash="dot", width=1), opacity=0.35,
                        ))
                    help_lines.append(
                        f"네이버 분봉이 자정에 초기화돼서, 서버가 20초마다 직접 기록한 값으로 "
                        f"{day} 하루치를 그렸습니다(08:00–20:00). 점선은 전 거래일 종가"
                        f"({base_price:,}원) 기준선이고, 세로 점선은 정규장 시작·종료 시각입니다."
                    )
        else:
            trade_date = intraday_df["시각"].iloc[-1].date()
            x_start = dt.datetime.combine(trade_date, dt.time(9, 0))
            x_end = dt.datetime.combine(trade_date, dt.time(15, 30))
            prev_close = close_price - change

            # 직접 쌓아둔 시간외 체결가를 정규장 앞뒤에 이어붙인다.
            # 프리장 시간대에는 분봉이 아직 전 거래일 것이므로, 화면 기준일(chart_date)은
            # '분봉 날짜'와 '시간외 기록의 최신 날짜' 중 더 나중으로 잡는다.
            over_ticks = load_over_market_ticks(TICKER, trade_date)
            chart_date = trade_date
            if not over_ticks.empty:
                chart_date = max(trade_date, over_ticks["시각"].max().date())
            # 프리장 = 화면 기준일의 정규장 개장 전 / 애프터장 = 정규장 날짜의 폐장 후
            pre_ticks = over_ticks[
                (over_ticks["시각"].dt.date == chart_date)
                & (over_ticks["시각"] < dt.datetime.combine(chart_date, dt.time(9, 0)))
            ]
            post_ticks = over_ticks[
                (over_ticks["시각"].dt.date == trade_date) & (over_ticks["시각"] > x_end)
            ]
            if not pre_ticks.empty:
                x_start = min(x_start, pre_ticks["시각"].min().to_pydatetime())
            if not post_ticks.empty:
                x_end = max(x_end, post_ticks["시각"].max().to_pydatetime())
            if not pre_ticks.empty and chart_date > trade_date:
                # 전 거래일 정규장 + 어젯밤 애프터장 + 오늘 아침 프리장을 한 흐름으로 보여준다
                x_end = max(x_end, pre_ticks["시각"].max().to_pydatetime())

            # --- 본주 트레이스 (기본 표시) ---
            fig_intraday.add_trace(go.Scatter(
                x=intraday_df["시각"], y=intraday_df["현재가"],
                mode="lines", line=dict(color="#d62728"), name="정규장",
                hovertemplate="%{x|%H:%M}  %{y:,.0f}원<extra>정규장</extra>",
            ))
            host_traces = 1
            for ticks, label in ((pre_ticks, "프리장"), (post_ticks, "애프터장")):
                if not ticks.empty:
                    fig_intraday.add_trace(go.Scatter(
                        x=ticks["시각"], y=ticks["가격"],
                        mode="lines", line=dict(color="#7f7f7f"), name=label,
                        hovertemplate="%{x|%H:%M}  %{y:,.0f}원<extra>" + label + "</extra>",
                    ))
                    host_traces += 1
            has_over = not pre_ticks.empty or not post_ticks.empty

            # 전일 종가 기준선을 도형(shape)이 아니라 트레이스로 그린다.
            # 그래야 아래 버튼으로 본주/ADR을 바꿀 때 같이 숨겨진다.
            fig_intraday.add_trace(go.Scatter(
                x=[x_start, x_end], y=[prev_close, prev_close], mode="lines",
                line=dict(color="gray", dash="dash", width=1), opacity=0.6,
                name="전일 종가", hoverinfo="skip", showlegend=False,
            ))
            host_traces += 1

            host_is_today = chart_date == today_kst
            title_date = f"{trade_date}" if chart_date == trade_date else f"{trade_date} – {chart_date}"
            # 큰 제목은 그래프 위 Streamlit 헤더가 맡고, 그래프 안 제목은 '지금 무엇을 보는지'만 표시한다
            host_title = f"본주 ({title_date})"

            # 정규장 시작·종료 세로선. 본주 볼 때만 필요하므로 버튼에서 같이 켜고 끈다.
            if has_over:
                for boundary in (dt.datetime.combine(trade_date, dt.time(9, 0)),
                                 dt.datetime.combine(trade_date, dt.time(15, 30))):
                    host_shapes.append(dict(
                        type="line", x0=boundary, x1=boundary, yref="paper", y0=0, y1=1,
                        line=dict(color="gray", dash="dot", width=1), opacity=0.35,
                    ))
                help_lines.append(
                    "회색 선이 프리장(08:00부터)·애프터장(20:00까지) 구간이고, "
                    "세로 점선은 정규장 시작·종료 시각입니다.\n\n"
                    "네이버가 시간외 분봉을 제공하지 않아, 서버가 20초마다 직접 기록한 값입니다. "
                    "서버가 꺼져 있던 시간대는 비어 있습니다."
                )
            else:
                help_lines.append(
                    "점선은 전일 종가 기준선입니다. "
                    "장 마감 후에는 마지막 거래일의 09:00–15:30 데이터가 표시됩니다."
                )

        # --- ADR(SKHY) 트레이스: 미국 프리장–애프터장 전 구간 ---
        # 본주 파트가 비어 있어도 붙인다. 한국 분봉이 없는 아침에도 ADR은 볼 수 있어야 한다.
        host_available = host_traces > 0
        adr_df = fetch_adr_intraday() if TICKER == ADR_HOST_TICKER else pd.DataFrame()
        adr_quote = fetch_adr_quote() if not adr_df.empty else None
        adr_shapes: list[dict] = []
        adr_start = adr_end = adr_day = None

        # 어느 쪽을 먼저 보여줄지: 한국장이 실제로 돌아가는 시간이면 본주, 아니면 ADR.
        # 한국이 닫혀 있는 동안 움직이는 건 미국 쪽이라, 멈춘 본주 그래프를 띄워두는 것보다
        # 지금 값이 변하는 화면을 먼저 보여주는 게 맞다. 버튼으로 언제든 되돌릴 수 있다.
        korea_session = _korea_session_now()
        korea_live = korea_session is not None and host_is_today
        show_host_first = host_available and (korea_live or adr_df.empty)

        if not adr_df.empty:
                adr_visible = not show_host_first
                # 본주 그래프와 같은 색 규칙: 정규장 빨강, 프리장·애프터장 회색.
                # 구간이 끊겨 보이지 않게, 이어지는 지점 한 점씩 겹쳐서 선을 붙인다.
                for label, color in (("프리장", "#7f7f7f"), ("정규장", "#d62728"), ("애프터장", "#7f7f7f")):
                    seg = adr_df[adr_df["세션"] == label]
                    if seg.empty:
                        continue
                    idx = seg.index
                    lo = max(idx.min() - 1, 0)
                    hi = min(idx.max() + 2, len(adr_df))
                    seg = adr_df.iloc[lo:hi] if label != "프리장" else adr_df.iloc[idx.min():hi]
                    fig_intraday.add_trace(go.Scatter(
                        x=seg["시각"], y=seg["가격"], mode="lines",
                        line=dict(color=color), name=f"{label}(ADR)", visible=adr_visible,
                        hovertemplate="%{x|%H:%M}  $%{y:,.2f}<extra>" + label + "</extra>",
                    ))
                # 하루 전체를 그리는 그래프라 기준선은 세션과 무관하게 '직전 거래일 종가'다
                adr_prev = (adr_quote or {}).get("prev_day_close")
                if adr_prev:
                    fig_intraday.add_trace(go.Scatter(
                        x=[adr_df["시각"].min(), adr_df["시각"].max()], y=[adr_prev, adr_prev],
                        mode="lines", line=dict(color="gray", dash="dash", width=1), opacity=0.6,
                        name="전일 종가(ADR)", hoverinfo="skip", showlegend=False, visible=adr_visible,
                    ))
                # 미국 정규장 시작·종료(한국시간)에도 본주와 똑같이 세로 점선을 넣는다
                reg = adr_df[adr_df["세션"] == "정규장"]
                if not reg.empty:
                    for boundary in (reg["시각"].min().to_pydatetime(), reg["시각"].max().to_pydatetime()):
                        adr_shapes.append(dict(
                            type="line", x0=boundary, x1=boundary, yref="paper", y0=0, y1=1,
                            line=dict(color="gray", dash="dot", width=1), opacity=0.35,
                        ))
                adr_start = adr_df["시각"].min().to_pydatetime()
                adr_end = adr_df["시각"].max().to_pydatetime()
                adr_day = adr_df["시각"].max().date()

        # 어느 쪽을 먼저 보여주든, 반대쪽은 반드시 숨겨야 한다.
        # ADR 트레이스에만 visible을 주고 본주는 기본값(보임)으로 두면, 본주가 안 숨어서
        # 원(150만)과 달러(160) 두 선이 한 y축에 같이 그려진다.
        if host_available and not adr_df.empty:
            for i, trace in enumerate(fig_intraday.data):
                trace.visible = (i < host_traces) == show_host_first

        if not host_available and adr_df.empty:
            # 한국 분봉도 시간외 기록도 ADR도 없는 시간대(휴장일 새벽 등).
            # 예전에는 아무것도 그리지 않고 조용히 넘어가서, 그래프가 사라진 건지
            # 원래 데이터가 없는 건지 구분이 안 됐다.
            st.caption(":gray[장중 주가 추이: 아직 오늘 체결 기록이 없습니다.]")
        else:
            # 처음 보여줄 쪽 (show_host_first에서 이미 정해졌다)
            if show_host_first:
                view_title, view_range = host_title, [x_start, x_end]
                view_ytitle, view_shapes = "현재가(원)", host_shapes
            else:
                view_title, view_range = f"SKHY ({adr_day})", [adr_start, adr_end]
                view_ytitle, view_shapes = "SKHY($)", adr_shapes
            # ADR 화면은 정규장(빨강)/시간외(회색)를 색으로 구분하므로 범례가 있어야 읽힌다
            _style_chart_mobile(fig_intraday, title=view_title,
                                show_legend=has_over or not adr_df.empty)
            fig_intraday.update_xaxes(range=view_range, tickformat="%H:%M")
            fig_intraday.update_yaxes(title_text=view_ytitle)
            fig_intraday.update_layout(shapes=view_shapes)

            # 전환 버튼은 양쪽 다 있을 때만 의미가 있다
            if host_available and not adr_df.empty:
                total = len(fig_intraday.data)
                host_vis = [i < host_traces for i in range(total)]
                adr_vis = [i >= host_traces for i in range(total)]
                # updatemenus는 브라우저에서 바로 처리돼 Streamlit 재실행이 없다.
                # 그래서 전환이 끊기지 않고 부드럽게 이어진다.
                # 버튼은 제목과 같은 줄의 오른쪽 끝에 둔다.
                # 범례는 왼쪽(x=0, y=1.02)에 깔리므로, 오른쪽 위로 빼야 서로 안 가린다.
                # updatemenus는 xref/yref를 지원하지 않아(plotly 5.x) paper 좌표로만 잡는다.
                fig_intraday.update_layout(margin=dict(t=88))
                fig_intraday.update_layout(
                    updatemenus=[dict(
                        type="buttons", direction="right",
                        x=1.0, xanchor="right", y=1.28, yanchor="top",
                        # 박스 없이 글자만. 선택 표시(showactive)를 끄면 배경 하이라이트도 없어진다.
                        # 지금 어느 쪽을 보고 있는지는 제목이 바뀌어서 알 수 있다.
                        # (plotly 5.x의 Updatemenu에는 activecolor 속성이 없다)
                        showactive=False, pad=dict(t=0, b=0, l=0, r=0),
                        bgcolor="rgba(0,0,0,0)",
                        bordercolor="rgba(0,0,0,0)", borderwidth=0,
                        font=dict(size=12, color="#4a8ec2"),
                        buttons=[
                            dict(label="본주", method="update",
                                 args=[{"visible": host_vis},
                                       {"title.text": host_title,
                                        "xaxis.range": [x_start, x_end],
                                        "xaxis.tickformat": "%H:%M",
                                        "yaxis.title.text": "현재가(원)",
                                        "shapes": host_shapes,
                                        "transition": {"duration": 350, "easing": "cubic-in-out"}}]),
                            dict(label="ADR(SKHY)", method="update",
                                 args=[{"visible": adr_vis},
                                       # 제목이 길면 좁은 화면에서 전환 버튼과 겹친다.
                                       # 한국시간이라는 설명은 ? 도움말에 들어 있으므로 여기서는 뺀다.
                                       {"title.text": f"SKHY ({adr_day})",
                                        "xaxis.range": [adr_start, adr_end],
                                        "xaxis.tickformat": "%H:%M",
                                        "yaxis.title.text": "SKHY($)",
                                        "shapes": adr_shapes,
                                        "transition": {"duration": 350, "easing": "cubic-in-out"}}]),
                        ],
                    )],
                )

            # 긴 설명은 화면을 어지럽히므로 제목 옆 ? 버튼 안으로 넣는다
            # (본주 쪽 설명은 위에서 이미 help_lines에 담아뒀다)
            adr_range_txt = (
                f"(프리장 04:00 – 애프터장 20:00 ET, 한국시간 {adr_start:%H:%M}–{adr_end:%H:%M})"
                if adr_start else ""
            )
            if not adr_df.empty:
                if show_host_first:
                    help_lines.append(
                        f"**ADR(SKHY) 버튼**을 누르면 나스닥 상장분의 하루치가 같은 자리에 나옵니다 "
                        f"{adr_range_txt}.\n\n"
                        "ADR 화면도 본주와 같은 색 규칙입니다. 정규장은 빨간색, 프리장·애프터장은 회색.\n\n"
                        f"{ADR_DAY_SESSION_NOTE}"
                    )
                elif host_available:
                    # 한국장이 닫혀 있어 ADR을 먼저 띄운 경우
                    help_lines.append(
                        f"지금은 한국 시장(프리장 08:00–09:00 · 정규장 09:00–15:30 · 애프터장 15:40–20:00)이 "
                        f"열려 있지 않아, 값이 계속 움직이는 나스닥 상장분(SKHY) {adr_day} 하루치를 "
                        f"먼저 보여줍니다 {adr_range_txt}.\n\n"
                        "**본주 버튼**을 누르면 국내 그래프로 돌아갑니다. "
                        "한국장이 열리면 자동으로 본주가 기본 화면이 됩니다."
                    )
                else:
                    help_lines.append(
                        f"한국 분봉이 아직 없어 나스닥 상장분(SKHY) {adr_day} 하루치를 먼저 보여줍니다 "
                        f"{adr_range_txt}. 정규장은 빨간색, 프리장·애프터장은 회색입니다.\n\n"
                        "오늘 국내 체결이 쌓이면 **본주 / ADR 전환 버튼**이 생깁니다."
                    )
            _bold_label_with_help("장중 주가 추이", "\n\n".join(help_lines), key="intraday")
            st.plotly_chart(fig_intraday, width="stretch", key="chart_intraday_price", config=PLOTLY_CONFIG)
    except Exception as exc:
        _note_optional_failure("장중 주가 추이", exc)


@st.fragment(run_every=MARKET_FLOW_REFRESH_SEC)
def render_market_flow():
    """코스피 전체 수급. 거래소가 1~2분마다 올리므로 그 주기에 맞춘다."""
    try:
        flow = fetch_market_flow()
        if flow:
            live = flow["is_today"]
            flow_help = (
                "**이 종목이 아니라 코스피 시장 전체 수급입니다.**\n\n"
                "외국인·기관·개인 3분류를 종목별로 장중에 보는 방법은 없습니다. 거래소가 마감 후에만 "
                "공개하기 때문이고, 증권사 공식 API도 종목별은 일별만 제공합니다. "
                "바로 위의 '외국계 창구 추정'이 장중에 볼 수 있는 유일한 종목별 단서인데, "
                "그건 외국인만 잡히는 추정치입니다.\n\n"
                "시장 전체 잠정치는 장중 1~2분마다 갱신됩니다. 이 종목의 수급으로 읽지 말고, "
                "'오늘 시장에서 외국인이 사는 날인가 파는 날인가' 정도의 배경으로만 보세요.\n\n"
                "종목별 일별 확정 수급은 **수급 현황** 탭에 있습니다."
            )
            # 아래는 '시장 전체' 수급이라 이 종목 얘기가 아니다. 바로 위에 이 종목의
            # 장중 단서를 한 줄만 둔다(자세한 창구별 내역까지는 여기서 다루지 않는다).
            _render_foreign_desk_line()

            _bold_label_with_help(
                f"코스피 전체 수급 ({'장중 잠정' if live else flow['날짜'] + ' 확정'}, 억원)",
                flow_help, key="market_flow",
            )
            with st.container(key="price_row_market_flow"):
                cols = st.columns(4)
                items = [("개인", flow["개인"]), ("외국인", flow["외국인"]),
                         ("기관계", flow["기관계"]), ("프로그램 비차익", flow["비차익"])]
                for col, (label, value) in zip(cols, items):
                    # 순매수=초록 / 순매도=빨강. st.metric은 delta만 색을 입히고 값에는
                    # 못 입혀서, 컨테이너 key에 buy/sell을 넣고 CSS로 숫자를 칠한다.
                    side = "none" if not value else ("buy" if value > 0 else "sell")
                    with col.container(key=f"metric_small_flow_{side}_{label}"):
                        st.metric(
                            label,
                            f"{value:+,.0f}" if value is not None else "N/A",
                            delta={"buy": "순매수", "sell": "순매도"}.get(side),
                            delta_color="off",   # 색은 아래 CSS가 값·델타 양쪽에 같이 준다
                        )
    except Exception as exc:
        _note_optional_failure("코스피 전체 수급", exc)

_boot_lap("상단(사이드바·타이틀·CSS) 완료")

# 셋을 따로 부른다. 각자 자기 주기로 돌고, 하나가 갱신돼도 나머지는 그대로 있는다.
_lf_t = time.monotonic()
render_current_price()
if _BOOT_PROFILE:
    print(f"[boot]   현재가 지표 {time.monotonic() - _lf_t:.2f}s", flush=True)
_lf_t = time.monotonic()
render_intraday_chart()
if _BOOT_PROFILE:
    print(f"[boot]   장중 차트 {time.monotonic() - _lf_t:.2f}s", flush=True)
_lf_t = time.monotonic()
render_market_flow()
if _BOOT_PROFILE:
    print(f"[boot]   코스피 수급 {time.monotonic() - _lf_t:.2f}s", flush=True)

_boot_lap("현재가·장중차트·코스피수급 완료")

st.divider()

REFRESH_CHECK_INTERVAL_SEC = 60  # 예약된 시각이 지났는지 확인하는 주기

# 탭마다 데이터 성격이 달라 자동 새로고침 시각을 그룹별로 따로 둔다.
MARKET_DATA_REFRESH_HOURS = [16, 17, 18, 19]  # 수급현황·가격과열도·선물경보·통합신호·조기신호(종가/수급 기반)
DRAM_REFRESH_HOURS = [13, 16, 20]
BIGTECH_CAPEX_REFRESH_HOURS = [16]
# AI 분석: 장 전(8시) · 장중 매시(9~15) · 마감 후 확정판(16시). 하루 9번.
# 무료 등급이 모델당 하루 20회쯤이고 리포트·공시·재무 요약이 0~5번을 쓰므로 여유가 있다.
# AI 분석의 예약 생성은 수집기(ai_report.tick)가 맡는다. 화면은 저장된 것을 읽기만
# 한다. 양쪽이 다 만들면 브라우저를 열어 둔 날 하루 26회가 되어 무료 한도(모델당 20회)를
# 넘기고, 그 시점부터 폴백 모델로 떨어져 품질이 그날 통째로 내려앉는다.
# 예약 시각은 ai_report.REFRESH_HOURS에 있다.


# 선물·통합 신호 계열 탭이 쓰는 데이터. 코스피200 선물 이력은 페이지마다 다음 조회 날짜가
# 앞 응답에서 나와서 병렬로 못 받고, 700일을 채우는 데 8초 넘게 걸린다.
# 이 탭들은 기본으로 꺼져 있으므로, 켜져 있을 때만 미리 데워둔다.
_FUTURES_TAB_LABELS = {"선물 경보", "통합 신호", "하락 조기신호", "상승 조기신호", "매매 신호"}


def _refresh_market_data_caches() -> None:
    """수급현황·가격과열도·선물경보·통합신호·하락·상승 조기신호 탭이 공유하는 종가/수급 기반 캐시."""
    fetch_investor_netbuy.clear()
    fetch_backtest_history.clear()
    fetch_latest_bars.clear()
    fetch_yahoo_history.clear()
    fetch_investor_netbuy(TICKER, DEFAULT_LOOKBACK_DAYS)
    fetch_backtest_history(TICKER, target_days=700)
    fetch_latest_bars(TICKER)
    fetch_yahoo_history("SOX")
    fetch_yahoo_history("DXY")

    # 선물 이력은 화면에 쓰는 탭이 켜져 있을 때만. 꺼져 있는데 데워두면 아무도 안 보는
    # 데이터를 받느라 예약 새로고침이 8초 넘게 멈춘다.
    # (사이드바에서 만들어지는 visible_tab_labels를 쓴다. _visible_tab_labels는 탭을
    #  실제로 그리는 시점에야 생기는데, 이 함수는 그보다 먼저 불릴 수 있다.)
    if _FUTURES_TAB_LABELS & set(globals().get("visible_tab_labels") or ALL_TAB_LABELS):
        fetch_futures_foreign_history.clear()
        fetch_latest_futures_bars.clear()
        fetch_futures_foreign_history(target_days=700)
        fetch_latest_futures_bars()


def _refresh_dram_caches() -> None:
    _fetch_dram_soup.clear()
    fetch_dram_module_prices.clear()
    fetch_dram_chip_prices.clear()
    fetch_dram_module_prices()
    fetch_dram_chip_prices()


def _refresh_bigtech_capex_cache() -> None:
    fetch_bigtech_capex.clear()
    fetch_bigtech_capex()


def _mark_ai_analysis_due() -> None:
    """여기서 바로 만들지 않고 '만들어야 함' 표시만 남긴다.

    이 함수가 불리는 시점은 탭이 그려지기 **전**이다. 과열도·DRAM 요약은 그 탭이
    그려질 때 전역 변수에 채워지므로, 여기서 만들면 그 둘이 '계산되지 않음'으로 빠진다.
    실제 생성은 파일 맨 끝(탭을 다 그린 뒤)에서 한다.
    """
    st.session_state["_ai_refresh_pending"] = True


REFRESH_GROUPS = [
    ("market_data", "수급 현황 / 가격 과열도", MARKET_DATA_REFRESH_HOURS, _refresh_market_data_caches),
    ("dram", "DRAM 시세", DRAM_REFRESH_HOURS, _refresh_dram_caches),
    ("bigtech_capex", "빅테크 Capex", BIGTECH_CAPEX_REFRESH_HOURS, _refresh_bigtech_capex_cache),
]


def _refresh_all_indicator_caches(progress_bar=None) -> None:
    """'지표 새로고침' 버튼용: 예약된 자동 새로고침 대상 전체를 한 번에 갱신한다."""
    steps = [(label, fn) for _, label, _, fn in REFRESH_GROUPS]
    # 디시 갤러리는 커뮤니티 탭이 켜져 있을 때만. 꺼놓고도 매번 긁으면
    # 아무도 안 보는 데이터를 받느라 새로고침이 1초 가까이 길어진다.
    if "커뮤니티" in (globals().get("visible_tab_labels") or ALL_TAB_LABELS):
        fetch_dc_gallery_posts.clear()
        steps.append(("디시인사이드 주식갤러리",
                      lambda: fetch_dc_gallery_posts(STOCK_NAME, DEFAULT_COMMUNITY_POST_COUNT)))
    for i, (label, fetch_fn) in enumerate(steps):
        if progress_bar is not None:
            progress_bar.progress(i / len(steps), text=f"{label} 수집 중... ({i + 1}/{len(steps)})")
        try:
            fetch_fn()
        except Exception:
            pass  # 개별 항목이 실패해도 새로고침은 계속 진행하고, 각 탭에서 자체적으로 에러를 표시한다.
    if progress_bar is not None:
        progress_bar.progress(1.0, text="완료! 화면을 갱신합니다...")


def _last_passed_schedule_slot(hours: list[int], now: dt.datetime) -> dt.datetime:
    """hours(0–23) 중 now 이전에 지난 가장 최근 시각을 반환한다.
    오늘의 첫 시각도 아직 안 지났다면 어제의 마지막 시각을 반환한다."""
    today_slots = sorted(now.replace(hour=h, minute=0, second=0, microsecond=0) for h in hours)
    passed_today = [s for s in today_slots if s <= now]
    if passed_today:
        return passed_today[-1]
    return (now - dt.timedelta(days=1)).replace(hour=max(hours), minute=0, second=0, microsecond=0)


def _next_schedule_slot(hours: list[int], now: dt.datetime) -> dt.datetime:
    today_slots = sorted(now.replace(hour=h, minute=0, second=0, microsecond=0) for h in hours)
    upcoming_today = [s for s in today_slots if s > now]
    if upcoming_today:
        return upcoming_today[0]
    return (now + dt.timedelta(days=1)).replace(hour=min(hours), minute=0, second=0, microsecond=0)


@st.cache_resource(show_spinner="불러오는 중...")
def _get_global_refresh_state_store() -> dict:
    """세션마다 따로 있는 st.session_state와 달리, 서버 전체에서 공유되는 새로고침 상태 저장소.
    브라우저를 새로고침하거나 새 세션이 열려도 그룹별 '진짜 마지막 데이터 갱신 시각'을 그대로 유지한다.
    market_data는 티커별로 분리해야, 서로 다른 종목을 보는 세션들이 이 상태를 공유하면서 한쪽 종목이
    예약 새로고침을 가로채 다른 종목은 갱신되지 않는 문제가 생기지 않는다."""
    return {}


def _group_state_key(name: str) -> str:
    return f"{name}:{TICKER}" if name == "market_data" else name


def _get_group_refresh_state(group_key: str, hours: list[int]) -> dict:
    store = _get_global_refresh_state_store()
    if group_key not in store:
        now = dt.datetime.now()
        store[group_key] = {"last_slot": _last_passed_schedule_slot(hours, now), "last_time": now}
    return store[group_key]


# 장 마감 후 수급 확정치를 기다리는 구간. 고정 시각표(16·17·18·19시)만 쓰면
# 네이버가 그 사이에 올릴 때 최대 한 시간을 그냥 기다리게 된다.
# 이 구간에는 1페이지만 싸게 찔러보고(약 0.15초), 오늘 날짜가 뜨는 즉시 받아온다.
POST_CLOSE_WATCH_FROM = dt.time(15, 40)
POST_CLOSE_WATCH_TO = dt.time(20, 0)
POST_CLOSE_PROBE_SEC = 300


def _today_flow_published(ticker: str) -> bool:
    """오늘자 투자자 수급이 네이버에 올라왔는지 1페이지만 받아 확인한다."""
    try:
        page = _fetch_frgn_page(ticker, 1)
    except Exception:
        return False
    if page.empty:
        return False
    return page["날짜"].max().date() == dt.datetime.now(om.KST).date()


def _post_close_catch_up(now: dt.datetime, store: dict) -> bool:
    """마감 후 구간에서 오늘치가 올라왔으면 즉시 갱신한다. 하루 한 번만 돈다."""
    if now.weekday() >= 5 or not (POST_CLOSE_WATCH_FROM <= now.time() <= POST_CLOSE_WATCH_TO):
        return False
    today = now.date()
    done_key = f"post_close_done:{TICKER}"
    if store.get(done_key) == today:
        return False
    # 브라우저 세션마다 이 조각이 돌기 때문에, 탐침 간격은 서버 전체에서 공유되는
    # 저장소로 묶는다. 창을 여러 개 열어놔도 5분에 한 번만 찔러본다.
    probe_key = f"post_close_probe:{TICKER}"
    last_probe = store.get(probe_key)
    if last_probe and (now - last_probe).total_seconds() < POST_CLOSE_PROBE_SEC:
        return False
    store[probe_key] = now
    if not _today_flow_published(TICKER):
        return False
    _refresh_market_data_caches()
    store[done_key] = today
    return True


@st.fragment(run_every=REFRESH_CHECK_INTERVAL_SEC)
def _auto_refresh_indicators():
    now = dt.datetime.now()
    any_refreshed = False
    store = _get_global_refresh_state_store()
    for name, _label, hours, refresh_fn in REFRESH_GROUPS:
        state = _get_group_refresh_state(_group_state_key(name), hours)
        latest_slot = _last_passed_schedule_slot(hours, now)
        if latest_slot > state["last_slot"]:
            refresh_fn()
            state["last_slot"] = latest_slot
            state["last_time"] = now
            any_refreshed = True

    # 시각표와 별개로, 마감 후에는 공개되는 즉시 따라잡는다
    try:
        if _post_close_catch_up(now, store):
            state = _get_group_refresh_state(_group_state_key("market_data"), MARKET_DATA_REFRESH_HOURS)
            state["last_time"] = now
            any_refreshed = True
    except Exception:
        pass  # 탐침 실패가 나머지 새로고침을 막지 않게 한다

    if any_refreshed:
        st.rerun(scope="app")


_auto_refresh_indicators()
_boot_lap("자동 새로고침 체크 완료")

_now = dt.datetime.now()
_stalled_groups = []
_status_parts = []
for _name, _label, _hours, _ in REFRESH_GROUPS:
    _state = _get_group_refresh_state(_group_state_key(_name), _hours)
    _due_slot = _last_passed_schedule_slot(_hours, _now)
    _in_post_close_watch = (
        _name == "market_data" and _now.weekday() < 5
        and POST_CLOSE_WATCH_FROM <= _now.time() <= POST_CLOSE_WATCH_TO
    )
    if (not _in_post_close_watch and _due_slot > _state["last_slot"]
            and _now >= _due_slot + dt.timedelta(seconds=REFRESH_CHECK_INTERVAL_SEC * 3)):
        _stalled_groups.append(f"{_label}(**{_due_slot:%H:%M}**)")
    # 마감 후 감시 구간에는 시각표가 아니라 '올라오는 즉시'가 실제 동작이라 그렇게 적는다
    if (_name == "market_data" and _now.weekday() < 5
            and POST_CLOSE_WATCH_FROM <= _now.time() <= POST_CLOSE_WATCH_TO
            and _get_global_refresh_state_store().get(f"post_close_done:{TICKER}") != _now.date()):
        _status_parts.append(f"{_label} **공개되는 즉시** (5분마다 확인)")
    else:
        _status_parts.append(f"{_label} 다음 **{_next_schedule_slot(_hours, _now):%H:%M}**")

# 색을 직접 지정하지 않고 기본 본문 색을 그대로 쓴다. 밝은 테마에선 검정, 다크 모드에선 흰색으로
# 자동으로 바뀌어 시가·고가 같은 지표 글자와 같은 톤이 된다.
_REFRESH_LABEL_STYLE = "font-size:0.875rem;"
_REFRESH_DETAIL_STYLE = "font-size:0.875rem; opacity:0.65;"

if _stalled_groups:
    st.markdown(
        f'<span style="{_REFRESH_LABEL_STYLE}">⏳ 자동 새로고침 반영 대기 중</span> : '
        f'<span style="{_REFRESH_DETAIL_STYLE}">{", ".join(_stalled_groups)} 예정 시각이 지났지만 아직 반영 전입니다. '
        "대시보드를 열어둔 세션이 있으면 곧 반영되고, 아무도 열어두지 않았다면 다음 접속 시 바로 반영됩니다 "
        '(자동 새로고침은 브라우저 세션이 연결돼 있어야 동작합니다).</span>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<span style="{_REFRESH_LABEL_STYLE}">🟢 자동 새로고침 정상 작동 중</span> : '
        f'<span style="{_REFRESH_DETAIL_STYLE}">{" · ".join(_status_parts)}</span>',
        unsafe_allow_html=True,
    )

_last_time_parts = [
    f"{_label} **{_get_group_refresh_state(_group_state_key(_name), _hours)['last_time']:%m-%d %H:%M}**"
    for _name, _label, _hours, _ in REFRESH_GROUPS
]
st.markdown(
    f'<span style="{_REFRESH_LABEL_STYLE}">🕒 마지막 새로고침</span> : '
    f'<span style="{_REFRESH_DETAIL_STYLE}">{" · ".join(_last_time_parts)}</span>',
    unsafe_allow_html=True,
)

refresh_clicked = st.button("🔄 지표 새로고침")

if refresh_clicked:
    _click_now = dt.datetime.now()
    for _name, _label, _hours, _ in REFRESH_GROUPS:
        _state = _get_group_refresh_state(_group_state_key(_name), _hours)
        _state["last_slot"] = _last_passed_schedule_slot(_hours, _click_now)
        _state["last_time"] = _click_now
    progress_bar = st.progress(0, text="새로고침 준비 중...")
    _refresh_all_indicator_caches(progress_bar)
    # 버튼은 예약 시각과 달리 지금 보겠다는 뜻이므로 분석도 새로 만든다.
    _mark_ai_analysis_due()
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
    """가격대가 서로 다른 여러 품목을 한 그래프에 겹쳐 그리면 스케일 차이로 잘 안 보이므로,
    한 번에 한 품목만 보이게 하고 나머지는 숨긴다.

    품목 전환은 Plotly 드롭다운(updatemenus)으로 처리한다 — 클릭이 브라우저 안에서
    trace 가시성만 바꾸므로 서버로 왕복하지 않는다. 예전에는 st.radio라서 클릭 한 번에
    이 탭 프래그먼트가 통째로 다시 돌았고(CSV 재기록·표 2개·차트 2개 재렌더) 4초 가까이
    걸렸다. 이제는 사실상 즉시다.
    """
    n = len(items)
    fig = go.Figure()
    for i, item in enumerate(items):
        hist = history[history["품목"] == item].sort_values("날짜")
        fig.add_trace(go.Scatter(
            x=hist["날짜"], y=hist["평균가(USD)"], name=item,
            mode="lines", visible=(i == 0),
        ))

    if n > 1:
        # 각 버튼은 자기 trace만 보이게 한다. Plotly가 가시성 변화에 맞춰 y축을 알아서
        # 다시 잡으므로(품목마다 가격대가 다르다) 여기서 yaxis.autorange를 넘기지 않는다 —
        # fixedrange가 걸린 축에 relayout으로 autorange를 함께 주면 "axis scaling" 오류가 난다.
        buttons = [
            dict(label=item, method="update",
                 args=[{"visible": [j == i for j in range(n)]},
                       {"title.text": item}])
            for i, item in enumerate(items)
        ]
        fig.update_layout(updatemenus=[dict(
            type="dropdown", direction="down", active=0,
            buttons=buttons, showactive=True,
            x=1.0, xanchor="right", y=1.16, yanchor="top",
            pad=dict(t=2, r=2),
            # 다크 테마에 맞춘다. 기본값(흰 배경 + 밝은 글자)은 글자가 안 보였다.
            bgcolor="#1e1e26", bordercolor="#555", borderwidth=1,
            font=dict(color="#fafafa", size=12),
        )])

    _style_chart_mobile(fig, title=items[0] if items else None, show_legend=False)
    # _style_chart_mobile은 모든 축에 fixedrange=True를 건다. 그 상태로 품목을 바꾸면
    # (visible 토글) Plotly가 잠긴 y축을 autorange하려다 "axis scaling" 오류를 낸다.
    # y축만 풀어 준다 — 세로 줌이 열리지만, 스크롤 오작동의 원인인 가로 줌은 x축에
    # 그대로 잠겨 있고 x는 rangeslider로 조정한다.
    fig.update_yaxes(fixedrange=False, autorange=True)
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig, width="stretch", key=chart_key, config=PLOTLY_CONFIG)
    st.caption("오른쪽 위에서 품목을 고르면 바로 바뀝니다. 차트 하단 슬라이더로 기간을 좁힐 수 있습니다.")


DOWNTREND_WINDOW = 20


_visible_tab_labels = [label for label in ALL_TAB_LABELS if label in visible_tab_labels] or ALL_TAB_LABELS
tabs = st.tabs(_visible_tab_labels)
_tab_map = dict(zip(_visible_tab_labels, tabs))
_boot_lap("탭 컨테이너 생성 완료 (렌더 직전)")

# Plotly는 숨겨진 탭(display:none) 안에서 그려질 때 컨테이너 폭을 못 재고 기본 700px로 그린다.
# 탭이 보이게 돼도 스스로 다시 재지 않아서, 모바일 375px에서는 차트 오른쪽 절반이 잘려 나갔다
# (SVG width=700, 오른쪽 끝 716px). config의 responsive는 window resize를 듣고 다시 그리므로,
# 탭을 누른 뒤 resize를 한 번 쏴 주면 제 폭을 찾는다(실측 700 -> 343).
# 리런마다 이 iframe이 새로 생기므로, 부모 창에 표시를 남겨 리스너가 겹쳐 붙지 않게 한다.
st.iframe(
    """
    <script>
    (function () {
      const w = window.parent, d = w.document;
      if (w.__plotlyResizeHooked) return;
      w.__plotlyResizeHooked = true;
      const kick = () => setTimeout(() => w.dispatchEvent(new Event('resize')), 150);
      d.addEventListener('click', (e) => {
        if (e.target.closest && e.target.closest('[role="tab"]')) kick();
      }, true);
      kick();
    })();
    </script>
    """,
    height=1,   # st.iframe은 0을 안 받는다(양의 정수만). 1px은 눈에 띄지 않는다.
)

# 각 탭의 렌더링 코드는 아래에서 함수로 정의되고, 파일 맨 끝의 디스패치 루프에서
# 사용자가 사이드바에서 선택한(숨기지 않은) 탭만 실제로 호출된다.

# AI 분석 탭에서 참고하는 다른 탭들의 요약값. 해당 탭이 숨겨져 호출되지 않는 경우에도
# NameError가 나지 않도록 기본값을 미리 준비해둔다.
investor_df = pd.DataFrame()
overheat_summary = "가격 과열도 백테스트 미실행"
futures_summary = "코스피200 선물 하락 신호 백테스트 미실행"
composite_summary = "통합 신호 미실행"
dram_summary = "해당 없음 (메모리 반도체 관련주가 아니라 DRAM 시세를 표시하지 않음)"
community_summary = "커뮤니티 심리 데이터를 가져오지 못함"

# 탭 하나를 프래그먼트로 둔다. 안에 있는 위젯(표시 품목·기간 슬라이더 등)을 건드리면
# 이 탭만 다시 그린다. 예전에는 전부 모듈 수준이라 위젯 하나에 스크립트 전체가 다시
# 돌았고, 보이는 탭 8개가 통째로 재렌더됐다(실측: DRAM 표시 품목 전환에 13.7초).
# 안쪽에 또 프래그먼트를 두면 안 된다 — 중첩은 Streamlit이 막는다.
@st.fragment
def _render_tab_supply():
    global investor_df

    investor_df = pd.DataFrame()

    lookback_days = st.session_state.get("lookback_days_slider", DEFAULT_LOOKBACK_DAYS)
    _subheader_with_help(
        f"최근 {lookback_days}일 투자자별 순매수 거래량",
        "개인·기관·외국인의 일별 순매수와 그 누적 추세입니다. 표의 기울기는 하루 평균이고 "
        "양수면 매수 우위입니다.\n\n"
        "개인 순매수는 네이버가 안 줘서 기관·외국인 합산의 잔차로 추정한 값이라 오차가 섞일 수 있습니다.",
        key="supply",
    )
    lookback_days = st.slider(
        "수급 분석 기간(일)", min_value=10, max_value=180, value=DEFAULT_LOOKBACK_DAYS, step=10,
        key="lookback_days_slider",
    )

    try:
        df = fetch_investor_netbuy(TICKER, lookback_days)
        investor_df = df
        if df.empty:
            st.warning("투자자 수급 데이터를 가져오지 못했습니다. 잠시 후 다시 시도해 주세요.")
        else:
            flows = df[list(INVESTOR_COLUMNS)]
            df_long = flows.reset_index().melt(id_vars="날짜", var_name="투자자", value_name="순매수")
            fig_bar = px.bar(df_long, x="날짜", y="순매수", color="투자자", barmode="group")
            _style_chart_mobile(fig_bar, title="일별 순매수 거래량(주)")
            st.plotly_chart(fig_bar, width="stretch", key="chart_investor_bar", config=PLOTLY_CONFIG)

            # 절대 거래량. 순매수는 '누가 샀나'만 말해줄 뿐 그 날 얼마나 활발했는지는 안 보인다.
            # 같은 순매수라도 거래량이 평소의 3배인 날과 절반인 날은 의미가 다르다.
            #
            # 거래량은 수급표(frgn.naver)가 아니라 일봉 경로에서 따로 받는다. 같은 네이버인데도
            # 공개 시점이 다르다 - 15:51에 재보니 일봉은 이미 당일치(종가 1,730,000 /
            # 거래량 4,247,406)가 있는데 수급 페이지는 아직 전 거래일까지였다.
            # 한 소스에 묶어두면 이미 나와 있는 오늘 거래량을 몇 시간씩 못 보게 된다.
            try:
                ohlcv = fetch_daily_ohlcv(TICKER, lookback_days)
            except Exception:
                ohlcv = pd.DataFrame()
            vol_src = ohlcv if not ohlcv.empty else df
            if "거래량" in vol_src.columns and vol_src["거래량"].notna().any():
                vol = vol_src["거래량"].astype(float)
                # 상승 마감이면 초록, 하락이면 빨강 (화면의 다른 색 규칙과 동일)
                direction = vol_src["종가"].astype(float).diff()
                colors = [_DOWN_COLOR if d < 0 else _UP_COLOR for d in direction.fillna(0)]
                avg_window = min(20, max(len(vol) // 3, 2))
                vol_avg = vol.rolling(avg_window, min_periods=1).mean()

                fig_vol = go.Figure()
                fig_vol.add_trace(go.Bar(
                    x=vol_src.index, y=vol, marker_color=colors, name="거래량",
                    hovertemplate="%{x|%m-%d}  %{y:,.0f}주<extra>거래량</extra>",
                ))
                fig_vol.add_trace(go.Scatter(
                    x=vol_src.index, y=vol_avg, mode="lines", name=f"{avg_window}일 평균",
                    line=dict(color="#7f7f7f", width=1.5),
                    hovertemplate="%{x|%m-%d}  %{y:,.0f}주<extra>" + f"{avg_window}일 평균</extra>",
                ))
                # 설명은 제목 옆 ? 안으로. 그래프 아래 캡션으로 길게 깔면 화면이 어수선해진다.
                _bold_label_with_help(
                    "일별 거래량(주)",
                    f"막대 색은 그 날 등락, 회색 선은 {avg_window}일 이동평균입니다.\n\n"
                    "거래량은 마감 직후 바로 들어오지만 위 순매수는 18시가 넘어야 채워져서, "
                    "마감 직후에는 하루 앞서 있을 수 있습니다. 거래량 자체에 방향 예측력은 없었습니다.",
                    key="volume",
                )
                _style_chart_mobile(fig_vol)
                fig_vol.update_yaxes(title_text="거래량(주)")
                st.plotly_chart(fig_vol, width="stretch", key="chart_volume",
                                config=PLOTLY_CONFIG)

                latest_vol = float(vol.iloc[-1])
                base_vol = float(vol_avg.iloc[-1])
                vcol1, vcol2, vcol3 = st.columns(3)
                with vcol1.container(key="metric_small_vol_last"):
                    # 이 표는 장 마감 후 확정되는 값이라 오늘치가 아니다. 화면 위쪽 현재가의
                    # 실시간 거래량과 다른 날짜라서, 날짜를 라벨에 박아 헷갈리지 않게 한다.
                    st.metric(f"{vol.index[-1]:%m-%d} 거래량", f"{latest_vol:,.0f}주",
                              delta=f"{avg_window}일 평균 대비 {latest_vol / base_vol - 1:+.0%}"
                              if base_vol else None, delta_color="off")
                with vcol2.container(key="metric_small_vol_avg"):
                    st.metric(f"{avg_window}일 평균 거래량", f"{base_vol:,.0f}주")
                with vcol3.container(key="metric_small_vol_max"):
                    peak_day = vol.idxmax()
                    st.metric("기간 내 최대", f"{vol.max():,.0f}주",
                              delta=f"{peak_day:%m-%d}", delta_color="off")

            df_cum = flows.cumsum()
            df_cum_long = df_cum.reset_index().melt(id_vars="날짜", var_name="투자자", value_name="누적 순매수")
            fig_line = px.line(df_cum_long, x="날짜", y="누적 순매수", color="투자자")
            _style_chart_mobile(fig_line, title="누적 순매수 추세")
            st.plotly_chart(fig_line, width="stretch", key="chart_investor_line", config=PLOTLY_CONFIG)

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
            st.table(slope_df, width="stretch", hide_index=True)
    except Exception as e:
        st.error(f"투자자 수급 데이터 조회에 실패했습니다: {e}")

# 탭 하나를 프래그먼트로 둔다. 안에 있는 위젯(표시 품목·기간 슬라이더 등)을 건드리면
# 이 탭만 다시 그린다. 예전에는 전부 모듈 수준이라 위젯 하나에 스크립트 전체가 다시
# 돌았고, 보이는 탭 8개가 통째로 재렌더됐다(실측: DRAM 표시 품목 전환에 13.7초).
# 안쪽에 또 프래그먼트를 두면 안 된다 — 중첩은 Streamlit이 막는다.
@st.fragment
def _render_tab_overheat():
    global overheat_summary

    # 분위 구간은 ai_inputs에서 가져온다. 화면과 AI 재료가 같은 구간을 봐야
    # 탭에 적힌 확률과 분석에 인용된 확률이 어긋나지 않는다.
    overheat_summary = "가격 과열도 백테스트 미실행"

    _subheader_with_help(
        "가격 과열도 백테스트",
        "이동평균 대비 괴리율 구간별로 향후 하락 확률이 어떻게 달라지는지 보는 통계입니다. "
        "기본값은 168개 조합을 백테스트해 고른 값입니다.\n\n"
        "상승 확률은 괴리율과 관계가 없어(U자형) 표시하지 않습니다. 매매 신호가 아닙니다.",
        key="overheat",
    )

    with st.expander("백테스트 조건 조정"):
        param_col1, param_col2, param_col3 = st.columns(3)
        OVERHEAT_MA_WINDOW = param_col1.slider(
            "이동평균 기간 (일)", min_value=10, max_value=120, value=OVERHEAT_DEFAULT_MA_WINDOW, step=5,
            key="overheat_ma_window",
        )
        OVERHEAT_HORIZON = param_col2.slider(
            "예측 기간 (거래일)", min_value=5, max_value=40, value=OVERHEAT_DEFAULT_HORIZON, step=1,
            key="overheat_horizon",
        )
        OVERHEAT_DRAWDOWN_THRESHOLD = param_col3.slider(
            "하락 기준 (%)", min_value=3, max_value=20, value=int(OVERHEAT_DEFAULT_THRESHOLD * 100), step=1,
            key="overheat_threshold",
        ) / 100

    try:
        _ov_t = time.monotonic()
        overheat_hist = fetch_backtest_history_live(TICKER, target_days=700)
        if _BOOT_PROFILE:
            print(f"[boot]     과열도: 700일 이력 fetch {time.monotonic() - _ov_t:.2f}s", flush=True)
        if len(overheat_hist) < 80:
            st.warning("백테스트에 충분한 과거 데이터가 없습니다.")
        else:
            overheat_results = {
                q: run_overheat_backtest(
                    overheat_hist, "종가", OVERHEAT_MA_WINDOW, OVERHEAT_HORIZON,
                    quantile=q, drawdown_threshold=OVERHEAT_DRAWDOWN_THRESHOLD, side="high",
                )
                for q in OVERHEAT_QUANTILES
            }
            overheat_results_low = {
                q: run_overheat_backtest(
                    overheat_hist, "종가", OVERHEAT_MA_WINDOW, OVERHEAT_HORIZON,
                    quantile=q, drawdown_threshold=OVERHEAT_DRAWDOWN_THRESHOLD, side="low",
                )
                for q in OVERHEAT_QUANTILES
            }
            if _BOOT_PROFILE:
                print(f"[boot]     과열도: 백테스트 8회 {time.monotonic() - _ov_t:.2f}s (fetch 포함 누적)", flush=True)
            overheat_result = overheat_results[OVERHEAT_QUANTILES[0]]

            if overheat_result["n"] < 30:
                st.warning("백테스트에 충분한 표본이 없습니다.")
            else:
                current_deviation = overheat_result["current_deviation"]
                matched_quantile = None
                for q in sorted(OVERHEAT_QUANTILES):
                    cutoff = overheat_results[q]["hi_cutoff"]
                    if cutoff is not None and current_deviation is not None and current_deviation >= cutoff:
                        matched_quantile = q
                        break

                matched_quantile_low = None
                for q in sorted(OVERHEAT_QUANTILES):
                    cutoff = overheat_results_low[q]["hi_cutoff"]
                    if cutoff is not None and current_deviation is not None and current_deviation <= cutoff:
                        matched_quantile_low = q
                        break

                if matched_quantile is not None:
                    current_regime_label = f"상위 {matched_quantile:.0%} 구간(과열)"
                    current_down_rate = overheat_results[matched_quantile]["hi_rate"]
                elif matched_quantile_low is not None:
                    current_regime_label = f"하위 {matched_quantile_low:.0%} 구간(침체)"
                    current_down_rate = overheat_results_low[matched_quantile_low]["hi_rate"]
                else:
                    current_regime_label = "평상시"
                    widest_q = max(OVERHEAT_QUANTILES)
                    current_down_rate = overheat_results[widest_q]["rest_rate"]

                deviation_series = (overheat_hist["종가"] / overheat_hist["종가"].rolling(OVERHEAT_MA_WINDOW).mean() - 1).dropna()
                current_percentile = (
                    float((deviation_series >= current_deviation).mean())
                    if len(deviation_series) > 0 and current_deviation is not None
                    else None
                )

                metric_col1, metric_col2 = st.columns(2)
                metric_col1.metric(
                    "현재 괴리율",
                    f"{current_deviation:+.1%} (상위 {current_percentile:.0%})"
                    if current_deviation is not None and current_percentile is not None else "N/A",
                )
                metric_col2.metric(
                    f"현재 구간의 {OVERHEAT_HORIZON}거래일 내 하락 확률",
                    f"{current_down_rate:.1%}" if current_down_rate is not None else "N/A",
                )
                st.caption(
                    f"현재 상태: {current_regime_label} · 표본 {overheat_result['n']}일 · "
                    f"전체 기간 기저 하락 확률 {overheat_result['base_rate']:.1%}"
                )

                breakdown_rows = [
                    {
                        "과열 구간": f"하위 {q:.0%}",
                        "괴리율": overheat_results_low[q]["hi_cutoff"],
                        "표본 수": overheat_results_low[q]["hi_n"],
                        "하락 확률": overheat_results_low[q]["hi_rate"],
                        "p-value(하락)": overheat_results_low[q]["p_value"],
                    }
                    for q in sorted(OVERHEAT_QUANTILES)
                ] + [
                    {
                        "과열 구간": f"상위 {q:.0%}",
                        "괴리율": overheat_results[q]["hi_cutoff"],
                        "표본 수": overheat_results[q]["hi_n"],
                        "하락 확률": overheat_results[q]["hi_rate"],
                        "p-value(하락)": overheat_results[q]["p_value"],
                    }
                    for q in sorted(OVERHEAT_QUANTILES, reverse=True)
                ]
                breakdown_df = pd.DataFrame(breakdown_rows).dropna(subset=["괴리율"]).sort_values("괴리율").reset_index(drop=True)

                fig_breakdown = go.Figure()
                fig_breakdown.add_trace(go.Scatter(
                    x=breakdown_df["괴리율"], y=breakdown_df["하락 확률"], name="하락 확률",
                    mode="lines+markers", line=dict(color="#d62728"), marker=dict(size=8),
                    customdata=breakdown_df[["과열 구간", "표본 수", "p-value(하락)"]],
                    hovertemplate="%{customdata[0]} (괴리율 %{x:.1%})<br>하락 확률 %{y:.1%}<br>표본 %{customdata[1]}일 · p-value %{customdata[2]:.4f}<extra></extra>",
                ))
                fig_breakdown.add_hline(
                    y=overheat_result["base_rate"], line_dash="dash", line_color="#d62728", opacity=0.5,
                    annotation_text="기저 하락 확률", annotation_position="bottom right",
                )
                if current_deviation is not None:
                    fig_breakdown.add_vline(
                        x=current_deviation, line_dash="dot", line_color="gray",
                        annotation_text="현재 괴리율", annotation_position="bottom",
                    )
                fig_breakdown.update_xaxes(title_text="괴리율", tickformat=".0%")
                fig_breakdown.update_yaxes(title_text=f"{OVERHEAT_HORIZON}거래일 내 확률", tickformat=".0%")
                _style_chart_mobile(fig_breakdown)
                st.plotly_chart(fig_breakdown, width="stretch", key="chart_overheat_breakdown", config=PLOTLY_CONFIG)

                overheat_hist_ma = overheat_hist.copy()
                overheat_hist_ma["MA"] = overheat_hist_ma["종가"].rolling(OVERHEAT_MA_WINDOW).mean()
                overheat_hist_ma["괴리율"] = overheat_hist_ma["종가"] / overheat_hist_ma["MA"] - 1
                overheat_chart_df = overheat_hist_ma.dropna(subset=["괴리율"])

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
                if len(overheat_chart_df) > 0:
                    band_cutoffs = {
                        q: float(overheat_chart_df["괴리율"].quantile(1 - q)) for q in OVERHEAT_QUANTILES
                    }
                    for q in OVERHEAT_QUANTILES:
                        fig_overheat.add_hline(
                            y=band_cutoffs[q], line_dash="dot", line_color="orange", secondary_y=False,
                            annotation_text=f"상위 {q:.0%}", annotation_position="top right",
                        )
                _style_chart_mobile(fig_overheat, title=f"종가/{OVERHEAT_MA_WINDOW}일 이동평균 괴리율 vs 주가")
                fig_overheat.update_yaxes(title_text="괴리율", tickformat=".0%", secondary_y=False)
                fig_overheat.update_yaxes(title_text="종가(원)", secondary_y=True)
                fig_overheat.update_xaxes(rangeslider_visible=True)
                st.plotly_chart(fig_overheat, width="stretch", key="chart_overheat", config=PLOTLY_CONFIG)
                st.caption("차트 하단 슬라이더를 드래그하면 보고 싶은 기간만 확대해서 볼 수 있습니다.")

                down_summary_parts = [
                    f"{row['과열 구간']}: {row['하락 확률']:.1%}" if pd.notna(row["하락 확률"]) else f"{row['과열 구간']}: N/A"
                    for row in breakdown_rows
                ]
                overheat_summary = (
                    f"가격 과열도 백테스트({OVERHEAT_MA_WINDOW}일선 괴리율, 표본 {overheat_result['n']}일, "
                    f"기저 하락 확률 {overheat_result['base_rate']:.1%}): "
                    f"괴리율 구간별 {OVERHEAT_HORIZON}거래일 내 {OVERHEAT_DRAWDOWN_THRESHOLD:.0%} 이상 하락 확률"
                    f"({', '.join(down_summary_parts)})은 과열이 심할수록 높아짐. "
                    f"현재 상태: {current_regime_label} (괴리율 {current_deviation:+.1%}). (통계 참고용, 매매 신호 아님)"
                )

                st.divider()
                _subheader_with_help(
                    "이동평균선 추세추종 전략 백테스트 (참고용)",
                    f"괴리율이 0% 이상(= 주가가 {OVERHEAT_MA_WINDOW}일 이동평균선 위)이면 보유하고, "
                    "0% 밑으로 내려가면 전량 매도했다가 다시 0% 이상으로 올라오면 재매수하는 전략입니다. "
                    "거래비용·세금·슬리피지는 반영되지 않아 실제 수익률은 이보다 낮습니다. 매매 신호가 아닙니다.",
                    key="strategy",
                )
                strategy_months = st.slider(
                    "백테스트 기간 (최근 N개월)", min_value=1, max_value=24, value=6, step=1,
                    key="overheat_strategy_months",
                )
                period_start = overheat_hist_ma["날짜"].max() - pd.Timedelta(days=strategy_months * 30.44)
                _strat_t = time.monotonic()
                strategy_result = run_overheat_threshold_strategy(overheat_hist_ma, period_start)
                if _BOOT_PROFILE:
                    print(f"[boot]     과열도: 추세추종 백테스트 {time.monotonic() - _strat_t:.2f}s", flush=True)

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
                        f"기간: {strategy_result['period_start'].date()} – {strategy_result['period_end'].date()}"
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
                        st.table(trades_df_display, width="stretch", hide_index=True)

                    equity_curve = strategy_result["equity_curve"]
                    if equity_curve is not None and not equity_curve.empty:
                        buy_hold_curve = overheat_hist_ma[
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
                        _style_chart_mobile(fig_strategy, title="전략 vs buy & hold 누적 수익률 (%)")
                        fig_strategy.update_yaxes(title_text="누적 수익률 (%)")
                        st.plotly_chart(fig_strategy, width="stretch", key="chart_overheat_strategy", config=PLOTLY_CONFIG)
    except Exception as e:
        st.error(f"가격 과열도 백테스트에 실패했습니다: {e}")

# 탭 하나를 프래그먼트로 둔다. 안에 있는 위젯(표시 품목·기간 슬라이더 등)을 건드리면
# 이 탭만 다시 그린다. 예전에는 전부 모듈 수준이라 위젯 하나에 스크립트 전체가 다시
# 돌았고, 보이는 탭 8개가 통째로 재렌더됐다(실측: DRAM 표시 품목 전환에 13.7초).
# 안쪽에 또 프래그먼트를 두면 안 된다 — 중첩은 Streamlit이 막는다.
@st.fragment
def _render_tab_futures():
    global futures_summary

    FUTURES_WINDOW = 15
    FUTURES_HORIZON = 10
    FUTURES_DRAWDOWN_THRESHOLD = 0.07
    FUTURES_QUANTILES = [0.20, 0.15, 0.10, 0.05]

    futures_summary = "코스피200 선물 하락 신호 백테스트 미실행"

    _subheader_with_help(
        "코스피200 선물 외국인 순매도 백테스트",
        f"코스피200 선물 외국인 {FUTURES_WINDOW}일 누적 순매수 기울기(매도 강도)별로 향후 {FUTURES_HORIZON}거래일 내 "
        f"{FUTURES_DRAWDOWN_THRESHOLD:.0%} 이상 하락·상승 확률을 비교한 참고용 통계이며, 매도가 강할수록 방향성보다 "
        "변동성 확대 경보로 해석하는 게 적절합니다 (매매 신호 아님).",
        key="futures",
    )
    try:
        futures_hynix_hist = fetch_backtest_history_live(TICKER, target_days=700)
        futures_hist = fetch_futures_foreign_history_live(target_days=700)
        if len(futures_hynix_hist) < 80 or len(futures_hist) < 80:
            st.warning("백테스트에 충분한 과거 데이터가 없습니다.")
        else:
            futures_flow_aligned = futures_hist.set_index("날짜")["선물외국인"].reindex(futures_hynix_hist["날짜"]).reset_index(drop=True)
            futures_results = {
                q: run_futures_decline_backtest(
                    futures_hynix_hist["종가"], futures_hynix_hist["날짜"], futures_flow_aligned,
                    FUTURES_WINDOW, FUTURES_HORIZON, quantile=q, drawdown_threshold=FUTURES_DRAWDOWN_THRESHOLD,
                )
                for q in FUTURES_QUANTILES
            }
            futures_result = futures_results[FUTURES_QUANTILES[0]]

            if futures_result["n"] < 30:
                st.warning("백테스트에 충분한 표본이 없습니다.")
            else:
                current_slope = futures_result["current_slope"]
                matched_quantile = None
                for q in sorted(FUTURES_QUANTILES):
                    cutoff = futures_results[q]["lo_cutoff"]
                    if cutoff is not None and current_slope is not None and current_slope <= cutoff:
                        matched_quantile = q
                        break

                if matched_quantile is not None:
                    current_regime_label = f"강한 매도 경고 (하위 {matched_quantile:.0%})"
                    current_down_rate = futures_results[matched_quantile]["lo_rate"]
                    current_up_rate = futures_results[matched_quantile]["lo_up_rate"]
                else:
                    current_regime_label = "평상시"
                    widest_q = max(FUTURES_QUANTILES)
                    current_down_rate = futures_results[widest_q]["rest_rate"]
                    current_up_rate = futures_results[widest_q]["rest_up_rate"]

                slope_series = _rolling_slope(futures_flow_aligned, FUTURES_WINDOW).dropna()
                current_percentile = (
                    float((slope_series <= current_slope).mean())
                    if len(slope_series) > 0 and current_slope is not None
                    else None
                )

                metric_col1, metric_col2, metric_col3 = st.columns(3)
                metric_col1.metric(
                    "현재 매도 강도",
                    f"기울기 {current_slope:,.0f} (하위 {current_percentile:.0%})"
                    if current_slope is not None and current_percentile is not None else "N/A",
                )
                metric_col2.metric(
                    f"현재 구간의 {FUTURES_HORIZON}거래일 내 하락 확률",
                    f"{current_down_rate:.1%}" if current_down_rate is not None else "N/A",
                )
                metric_col3.metric(
                    f"현재 구간의 {FUTURES_HORIZON}거래일 내 상승 확률",
                    f"{current_up_rate:.1%}" if current_up_rate is not None else "N/A",
                )
                st.caption(
                    f"현재 상태: {current_regime_label} · 표본 {futures_result['n']}일 · "
                    f"전체 기간 기저 하락 확률 {futures_result['base_rate']:.1%} "
                    f"· 기저 상승 확률 {futures_result['base_up_rate']:.1%}"
                )

                futures_breakdown_rows = [
                    {
                        "매도 강도": f"하위 {q:.0%}",
                        "기울기": futures_results[q]["lo_cutoff"],
                        "표본 수": futures_results[q]["lo_n"],
                        "하락 확률": futures_results[q]["lo_rate"],
                        "p-value(하락)": futures_results[q]["p_value"],
                        "상승 확률": futures_results[q]["lo_up_rate"],
                        "p-value(상승)": futures_results[q]["up_p_value"],
                    }
                    for q in sorted(FUTURES_QUANTILES)
                ]
                futures_breakdown_df = pd.DataFrame(futures_breakdown_rows).dropna(subset=["기울기"]).sort_values("기울기").reset_index(drop=True)

                fig_futures_breakdown = go.Figure()
                fig_futures_breakdown.add_trace(go.Scatter(
                    x=futures_breakdown_df["기울기"], y=futures_breakdown_df["하락 확률"], name="하락 확률",
                    mode="lines+markers", line=dict(color="#d62728"), marker=dict(size=8),
                    customdata=futures_breakdown_df[["매도 강도", "표본 수", "p-value(하락)"]],
                    hovertemplate="%{customdata[0]} (기울기 %{x:,.0f})<br>하락 확률 %{y:.1%}<br>표본 %{customdata[1]}일 · p-value %{customdata[2]:.4f}<extra></extra>",
                ))
                fig_futures_breakdown.add_trace(go.Scatter(
                    x=futures_breakdown_df["기울기"], y=futures_breakdown_df["상승 확률"], name="상승 확률",
                    mode="lines+markers", line=dict(color="#1f77b4"), marker=dict(size=8),
                    customdata=futures_breakdown_df[["매도 강도", "표본 수", "p-value(상승)"]],
                    hovertemplate="%{customdata[0]} (기울기 %{x:,.0f})<br>상승 확률 %{y:.1%}<br>표본 %{customdata[1]}일 · p-value %{customdata[2]:.4f}<extra></extra>",
                ))
                fig_futures_breakdown.add_hline(
                    y=futures_result["base_rate"], line_dash="dash", line_color="#d62728", opacity=0.5,
                    annotation_text="기저 하락 확률", annotation_position="bottom right",
                )
                fig_futures_breakdown.add_hline(
                    y=futures_result["base_up_rate"], line_dash="dash", line_color="#1f77b4", opacity=0.5,
                    annotation_text="기저 상승 확률", annotation_position="top right",
                )
                if current_slope is not None:
                    fig_futures_breakdown.add_vline(
                        x=current_slope, line_dash="dot", line_color="gray",
                        annotation_text="현재 기울기", annotation_position="bottom",
                    )
                fig_futures_breakdown.update_xaxes(title_text="기울기")
                fig_futures_breakdown.update_yaxes(title_text=f"{FUTURES_HORIZON}거래일 내 확률", tickformat=".0%")
                _style_chart_mobile(fig_futures_breakdown)
                st.plotly_chart(fig_futures_breakdown, width="stretch", key="chart_futures_breakdown", config=PLOTLY_CONFIG)

                futures_chart_df = pd.DataFrame(
                    {
                        "날짜": futures_hynix_hist["날짜"],
                        "기울기": _rolling_slope(futures_flow_aligned, FUTURES_WINDOW),
                        "종가": futures_hynix_hist["종가"],
                    }
                ).dropna(subset=["기울기"])

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
                for q in FUTURES_QUANTILES:
                    if futures_results[q]["lo_cutoff"] is not None:
                        fig_futures.add_hline(
                            y=futures_results[q]["lo_cutoff"], line_dash="dot", line_color="orange", secondary_y=False,
                            annotation_text=f"하위 {q:.0%}", annotation_position="bottom right",
                        )
                _style_chart_mobile(fig_futures, title=f"코스피200 선물 외국인 {FUTURES_WINDOW}일 누적 순매수 기울기 vs 주가")
                fig_futures.update_yaxes(title_text="기울기", secondary_y=False)
                fig_futures.update_yaxes(title_text="종가(원)", secondary_y=True)
                fig_futures.update_xaxes(rangeslider_visible=True)
                st.plotly_chart(fig_futures, width="stretch", key="chart_futures", config=PLOTLY_CONFIG)
                st.caption("차트 하단 슬라이더를 드래그하면 보고 싶은 기간만 확대해서 볼 수 있습니다.")

                futures_down_parts = [
                    f"하위 {row['매도 강도'][3:]}: {row['하락 확률']:.1%}" if pd.notna(row["하락 확률"]) else f"{row['매도 강도']}: N/A"
                    for row in futures_breakdown_rows
                ]
                futures_up_parts = [
                    f"하위 {row['매도 강도'][3:]}: {row['상승 확률']:.1%}" if pd.notna(row["상승 확률"]) else f"{row['매도 강도']}: N/A"
                    for row in futures_breakdown_rows
                ]
                futures_summary = (
                    f"코스피200 선물 외국인 순매도 백테스트(표본 {futures_result['n']}일, "
                    f"기저 하락 확률 {futures_result['base_rate']:.1%}, 기저 상승 확률 {futures_result['base_up_rate']:.1%}): "
                    f"매도 강도별 {FUTURES_HORIZON}거래일 내 {FUTURES_DRAWDOWN_THRESHOLD:.0%} 이상 하락 확률"
                    f"({', '.join(futures_down_parts)}), 상승 확률({', '.join(futures_up_parts)})은 "
                    "매도가 강할수록 하락과 상승 확률이 함께 높아지는(변동성 확대) 경향을 보임. "
                    f"현재 상태: {current_regime_label} (기울기 {current_slope:,.0f}). (경보 참고용, 매매 신호 아님)"
                )
    except Exception as e:
        st.error(f"코스피200 선물 하락 신호 백테스트에 실패했습니다: {e}")

# 탭 하나를 프래그먼트로 둔다. 안에 있는 위젯(표시 품목·기간 슬라이더 등)을 건드리면
# 이 탭만 다시 그린다. 예전에는 전부 모듈 수준이라 위젯 하나에 스크립트 전체가 다시
# 돌았고, 보이는 탭 8개가 통째로 재렌더됐다(실측: DRAM 표시 품목 전환에 13.7초).
# 안쪽에 또 프래그먼트를 두면 안 된다 — 중첩은 Streamlit이 막는다.
@st.fragment
def _render_tab_composite():
    global composite_summary

    COMPOSITE_HORIZON = 10
    COMPOSITE_SIGNAL_LABELS = {
        "기관_기울기": "기관 수급",
        "SOX_기울기": "미국 반도체지수(SOX)",
        "DXY_기울기": "달러인덱스(DXY)",
    }

    composite_summary = "통합 신호 미실행"

    _subheader_with_help(
        "통합 매수/매도 신호 (실험적)",
        "기관 수급·SOX·달러인덱스를 유의성에 따라 가중합산한 실험적 신호입니다.\n\n"
        "**하락 경보로는 근거가 있지만 상승 예측으로는 못 씁니다.** 검증 결과 하락 확률 차이만 "
        "유의했고(p<0.01) 상승은 아니었습니다(p=0.14–0.95). 매매 신호가 아닙니다.",
        key="composite",
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
        st.table(pd.DataFrame(rows), width="stretch", hide_index=True)

        st.markdown("**종합 신호 (통계적 유의성 기반 가중합산, 자체 백테스트 검증)**")

        regime_label = "매수 우위" if (composite_result["current_value"] or 0) > 0 else "매도 우위"
        if regime_label == "매수 우위":
            current_down_rate = composite_result["pos_down_rate"]
            current_up_rate = composite_result["pos_up_rate"]
        else:
            current_down_rate = composite_result["neg_down_rate"]
            current_up_rate = composite_result["neg_up_rate"]

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric(
            "종합 점수", f"{composite_result['current_value']:+.2f}" if composite_result["current_value"] is not None else "N/A"
        )
        metric_col2.metric(
            f"현재 방향({regime_label})의 {COMPOSITE_HORIZON}거래일 내 하락 확률",
            f"{current_down_rate:.1%}" if current_down_rate is not None else "N/A",
        )
        metric_col3.metric(
            f"현재 방향({regime_label})의 {COMPOSITE_HORIZON}거래일 내 상승 확률",
            f"{current_up_rate:.1%}" if current_up_rate is not None else "N/A",
        )
        st.caption(
            f"표본 {composite_result['n']}일 · 전체 기간 기저 하락 확률 {composite_result['base_down_rate']:.1%} "
            f"· 기저 상승 확률 {composite_result['base_up_rate']:.1%}"
        )

        composite_breakdown_rows = [
            {
                "종합 신호 방향": "매도 우위 (음수)",
                "표본 수": composite_result["neg_n"],
                "하락 확률": composite_result["neg_down_rate"],
                "p-value(하락)": composite_result["down_p_value"],
                "상승 확률": composite_result["neg_up_rate"],
                "p-value(상승)": composite_result["up_p_value"],
            },
            {
                "종합 신호 방향": "매수 우위 (양수)",
                "표본 수": composite_result["pos_n"],
                "하락 확률": composite_result["pos_down_rate"],
                "p-value(하락)": composite_result["down_p_value"],
                "상승 확률": composite_result["pos_up_rate"],
                "p-value(상승)": composite_result["up_p_value"],
            },
        ]
        composite_breakdown_df = pd.DataFrame(composite_breakdown_rows)
        composite_breakdown_display = composite_breakdown_df.copy()
        composite_breakdown_display["표본 수"] = composite_breakdown_display["표본 수"].map(lambda v: f"{v}일")
        for col in ["하락 확률", "상승 확률"]:
            composite_breakdown_display[col] = composite_breakdown_display[col].map(
                lambda v: f"{v:.1%}" if pd.notna(v) else "N/A"
            )
        for col in ["p-value(하락)", "p-value(상승)"]:
            composite_breakdown_display[col] = composite_breakdown_display[col].map(
                lambda v: f"{v:.4f}" if pd.notna(v) else "N/A"
            )
        st.table(composite_breakdown_display, width="stretch", hide_index=True)

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
            _style_chart_mobile(fig_composite, title="종합 매수/매도 신호 vs 주가")
            fig_composite.update_yaxes(title_text="종합 신호", secondary_y=False)
            fig_composite.update_yaxes(title_text="종가(원)", secondary_y=True)
            fig_composite.update_xaxes(rangeslider_visible=True)
            st.plotly_chart(fig_composite, width="stretch", key="chart_composite", config=PLOTLY_CONFIG)
            composite_caption = "차트 하단 슬라이더를 드래그하면 보고 싶은 기간만 확대해서 볼 수 있습니다."
            if show_downtrend_composite:
                composite_caption += f" 빨간 음영 구간은 주가가 최근 {DOWNTREND_WINDOW}일간 {downtrend_pct_composite}% 이상 하락한 하락장 구간입니다."
            st.caption(composite_caption)

        if composite_result["p_value"] is not None:
            composite_summary = (
                f"통합 신호(기관 수급+미국 반도체지수 SOX, 유의성 가중합산, 표본 {composite_result['n']}일, "
                f"기저 하락 확률 {composite_result['base_down_rate']:.1%}, 기저 상승 확률 {composite_result['base_up_rate']:.1%}): "
                f"매도 우위일 때 하락 확률 {composite_result['neg_down_rate']:.1%}·상승 확률 {composite_result['neg_up_rate']:.1%}, "
                f"매수 우위일 때 하락 확률 {composite_result['pos_down_rate']:.1%}·상승 확률 {composite_result['pos_up_rate']:.1%}. "
                "단, 검증 구간을 나눠 확인하면 하락 예측력은 유지되나 상승 예측력은 크게 약해지므로 하락 경보 위주로 참고. "
                f"현재 상태: {regime_label} (종합 점수 {composite_result['current_value']:+.2f}). (인-샘플 결과, 매매 신호 아님)"
            )
        else:
            composite_summary = "통합 신호 표본 부족으로 계산되지 않음"
    except Exception as e:
        st.error(f"통합 신호 백테스트에 실패했습니다: {e}")


# 탭 하나를 프래그먼트로 둔다. 안에 있는 위젯(표시 품목·기간 슬라이더 등)을 건드리면
# 이 탭만 다시 그린다. 예전에는 전부 모듈 수준이라 위젯 하나에 스크립트 전체가 다시
# 돌았고, 보이는 탭 8개가 통째로 재렌더됐다(실측: DRAM 표시 품목 전환에 13.7초).
# 안쪽에 또 프래그먼트를 두면 안 된다 — 중첩은 Streamlit이 막는다.
@st.fragment
def _render_tab_signal():
    _subheader_with_help(
        "매매 신호 (기관 수급 기반)",
        "기관 순매수를 거래량으로 나눠 20일 누적한 값이 양수면 '보유', 음수면 '현금'입니다. "
        "후보 지표 19종 중 10년 검증을 통과한 유일한 지표로, 실은 글로벌 반도체 업황의 선행 지표입니다.\n\n"
        "백테스트일 뿐입니다. 아래 '검증 상세와 한계'를 읽으세요.",
        key="flow_signal",
    )

    try:
        hist = fetch_backtest_history_live(TICKER, target_days=FLOW_BACKTEST_DAYS)
        if hist.empty or "기관" not in hist.columns:
            st.warning("수급 이력을 가져오지 못해 신호를 계산할 수 없습니다.")
            return

        slippage = st.select_slider(
            "슬리피지 가정 (편도)", options=[0.0005, 0.0010, 0.0020, 0.0030, 0.0050],
            value=0.0010, format_func=lambda v: f"{v:.2%}", key="flow_slippage",
        )
        result = backtest_flow_signal(hist, slippage=slippage)
        if not result["ok"]:
            st.warning(f"신호를 계산하기에 이력이 부족합니다 (현재 {result['n']}일).")
            return

        signal_series = result["신호"]
        current = result["현재신호"]
        is_buy = current > 0

        # 현재 상태가 며칠째 이어지고 있는지 (부호가 바뀔 때마다 새 구간으로 세고, 마지막 구간의 길이를 센다)
        run_id = ((signal_series > 0) != (signal_series > 0).shift()).cumsum()
        streak = int((run_id == run_id.iloc[-1]).sum())

        col1, col2, col3 = st.columns(3)
        col1.metric("현재 판정", "매수 · 보유" if is_buy else "매도 · 현금")
        with col2.container(key="metric_small_signal_value"):
            _metric_with_help(
                "신호값", f"{current:+.2f}",
                "0보다 크면 기관 순매수 우위. 0에서 멀수록 강한 신호입니다.",
                key="signal_value",
            )
        col3.metric("현재 판정 지속", f"{streak}거래일")

        if is_buy:
            st.success("기관이 최근 20거래일 동안 순매수 우위입니다. 백테스트 기준으로는 보유 구간입니다.")
        else:
            st.warning("기관이 최근 20거래일 동안 순매도 우위입니다. 백테스트 기준으로는 현금 구간입니다.")
        st.caption(
            "기관 순매수는 장 마감 후 공시되므로, 오늘 신호는 다음 거래일부터 실행 가능한 것으로 계산했습니다."
        )

        st.divider()

        start, end = result["기간"]
        st.markdown(f"**이 종목 최근 구간 백테스트** ({start.date()} – {end.date()}, {result['n']}거래일)")
        strat, hold = result["전략"], result["보유"]
        compare_df = pd.DataFrame([
            {
                "구분": "이 신호대로 매매", "누적수익": f"{strat['총수익']:+.1%}", "연환산(CAGR)": f"{strat['CAGR']:+.1%}",
                "최대낙폭(MDD)": f"{strat['MDD']:.1%}", "Sharpe": f"{strat['Sharpe']:.2f}",
                "시장 노출": f"{result['노출']:.0%}", "매매 횟수": f"{result['거래횟수']}회",
            },
            {
                "구분": "그냥 계속 보유", "누적수익": f"{hold['총수익']:+.1%}", "연환산(CAGR)": f"{hold['CAGR']:+.1%}",
                "최대낙폭(MDD)": f"{hold['MDD']:.1%}", "Sharpe": f"{hold['Sharpe']:.2f}",
                "시장 노출": "100%", "매매 횟수": "1회",
            },
        ])
        st.table(compare_df, width="stretch", hide_index=True)
        st.caption(
            f"비용 가정: 매수 {0.00015 + slippage:.3%} / 매도 {0.00015 + 0.0015 + slippage:.3%}"
            " (위탁수수료 + 증권거래세·농특세 0.15% + 슬리피지). 배당은 양쪽 모두 제외했습니다."
        )

        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(
            x=result["날짜"], y=strat["equity"], name="이 신호대로 매매", line=dict(color=_UP_COLOR),
        ))
        fig_equity.add_trace(go.Scatter(
            x=result["날짜"], y=hold["equity"], name="그냥 계속 보유", line=dict(color="#888888"),
        ))
        _style_chart_mobile(fig_equity, title="원금 1로 놓았을 때의 자산 곡선")
        fig_equity.update_yaxes(title_text="자산 배수", type="log")
        st.plotly_chart(fig_equity, width="stretch", key="chart_flow_equity", config=PLOTLY_CONFIG)
        st.caption("세로축은 로그 눈금입니다. 같은 간격이 같은 배수를 뜻합니다.")

        fig_signal = make_subplots(specs=[[{"secondary_y": True}]])
        fig_signal.add_trace(
            go.Scatter(x=result["날짜"], y=signal_series, name="기관 수급 신호", line=dict(color="#1f77b4")),
            secondary_y=False,
        )
        fig_signal.add_trace(
            go.Scatter(x=result["날짜"], y=result["종가"], name="종가", line=dict(color="#d62728")),
            secondary_y=True,
        )
        fig_signal.add_hline(y=0, line_dash="dash", line_color="gray", secondary_y=False)
        _style_chart_mobile(fig_signal, title="기관 수급 신호 vs 주가")
        fig_signal.update_yaxes(title_text="신호값", secondary_y=False)
        fig_signal.update_yaxes(title_text="종가(원)", secondary_y=True)
        fig_signal.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig_signal, width="stretch", key="chart_flow_signal", config=PLOTLY_CONFIG)
        st.caption("파란 선이 0 위로 올라오면 보유, 아래로 내려가면 현금 구간입니다.")

        with st.expander("검증 상세와 한계 (실제 매매 전에 꼭 읽어주세요)"):
            st.markdown(
                "#### 이 지표를 어떻게 골랐나\n"
                "SK하이닉스 2016-01 – 2026-08 (2,599거래일)에서 후보 지표 19종 × 예측기간 3종을 검증했습니다. "
                "지표 선택은 앞 70% 구간만 보고 했고, 뒤 30%는 마지막에 한 번만 확인했습니다.\n\n"
                "#### 통과한 검증\n"
                "- **공시 지연 반영**: 기관 수급은 장 마감 후 공시되므로 신호를 2일 밀어 계산해도 예측력 유지 (IC +0.21)\n"
                "- **중첩 표본 보정**: 겹치지 않는 표본 p=0.006, 블록 부트스트랩 p=0.0035\n"
                "- **다중검정 보정**: 19개 지표를 뒤진 대가를 지불해도 family-wise p=0.016(10일)/0.043(20일)\n"
                "- **시기별 안정성**: 2016–2017부터 2026까지 2년 단위 6개 구간 모두 상관계수 양수\n"
                "- **모멘텀과 구별됨**: 모멘텀 상/중/하 어느 구간에서도 이 신호의 상·하위 향후 20일 수익 차이가 "
                "+5.5–6.4%p (모두 p<0.001). 단순히 '오른 주식 사기'가 아닙니다\n"
                "- **운이 아님**: 같은 노출·매매횟수로 시점만 무작위로 고른 1,000회 대조군의 CAGR 중앙값 18.9%, "
                "95분위 33.6% (전략 66.1%, p<0.001)\n"
                "- **되돌아보기 창**: 매년 과거 데이터만 보고 다시 골라도 항상 20일이 선택됨\n"
                "- **비용 내성**: 편도 슬리피지를 0.5%까지 올려도 단순보유를 앞섬\n\n"
                "- **데이터 문제 아님**: 네이버 원본 종가 대신 야후의 배당·분할 조정가로 다시 돌려도 "
                "결과가 그대로였습니다 (CAGR 68.2% vs 단순보유 48.4%, Sharpe 1.66 vs 1.08)\n\n"
                "#### 이 신호가 진짜인 결정적 근거 — 해외에서도 통한다\n"
                "하이닉스 기관 신호로 **해외 종목**의 향후 20일 수익을 예측해봤습니다 "
                "(신호 2일 지연, 중첩 보정 블록 부트스트랩).\n\n"
                "| 그룹 | 예측력(IC) 평균 | 결과 |\n"
                "|---|---|---|\n"
                "| 메모리 6종목 (마이크론·난야·윈본드·매크로닉스·WDC·씨게이트) | **+0.200** | 6/6 양수, 전부 유의 |\n"
                "| 반도체 장비·파운드리 6종목 (TSMC·도쿄일렉트론·어드반테스트·ASML·AMAT·엔비디아) | **+0.203** | 6/6 양수 |\n"
                "| 무관 대조군 5종목 (코카콜라·J&J·엑슨모빌·JP모건·도요타) | **+0.049** | 유의한 종목 없음, 0/5 단순보유에 패 |\n\n"
                "반도체 그룹과 무관 대조군의 차이는 통계적으로 유의합니다(p=0.0022). "
                "만약 이 신호가 그냥 '시장 전체 타이밍'을 맞히는 것이었다면 JP모건이나 도요타도 맞혔어야 합니다. "
                "그러지 못했다는 점이 **반도체 업황에 한정된 진짜 정보**라는 근거입니다.\n\n"
                "마이크론에서는 예측력이 +0.241로 하이닉스 자신(+0.207)보다도 높았습니다. "
                "하이닉스 주가 흐름에 과최적화된 결과라면 다른 나라 종목에서 이런 값이 나올 수 없습니다.\n\n"
                "섹터 모멘텀이 아닌 것도 확인했습니다. SOX 모멘텀을 제거한 편상관에서도 "
                "하이닉스 신호 → 마이크론은 +0.247(p<0.0001)로 유지된 반면, "
                "신호를 제거한 SOX 모멘텀 → 마이크론은 -0.023(p=0.26)으로 사라졌습니다.\n\n"
                "#### 통과하지 못한 검증\n"
                "'**각 종목의 자기 기관 수급으로 그 종목을 예측**'하는 방식은 통하지 않습니다. "
                "KOSPI 대형주 12종목·반도체 8종목에 각자의 기관 수급을 적용하면 예측력 평균이 0이었습니다. "
                "즉 정보를 가진 것은 **하이닉스에 들어오는 기관 자금**이지, 아무 종목의 기관 수급이 아닙니다.\n\n"
                "#### 실전에서 견뎌야 하는 것 (10.3년 백테스트 기준)\n"
                "- 최대 낙폭 -27.3% (2020년 코로나 급락). 고점을 회복하기까지 8.2개월\n"
                "- 매매 74건, 승률 62%, 연속으로 잃은 최장 기록 4건\n"
                "- 1년 단위로 끊어보면 손실인 구간이 전체의 2% (단순보유는 25%)\n"
                "- 다만 수익의 상당 부분이 길게 끌고 간 소수의 큰 상승장에서 나왔습니다. "
                "큰 추세가 없는 장에서는 잦은 매매 비용만 나갈 수 있습니다\n\n"
                "#### 그래도 남아 있는 위험\n"
                "- **반도체 종목들은 서로 강하게 같이 움직입니다.** 12종목이 이겼다고 해서 독립적인 검증 12번은 "
                "아닙니다. 실질적으로는 그보다 훨씬 적은 수의 베팅이라고 봐야 합니다\n"
                "- **'45번 뽑기의 최댓값'일 가능성이 남아 있습니다.** 한국 45종목에 각자의 기관 수급을 적용해 "
                "예측력 분포를 만들어보면 하이닉스가 1위(+0.299)인데, 이는 평균에서 +2.66 표준편차입니다. "
                "45번 뽑았을 때 우연히 기대되는 최댓값이 +2.02 표준편차이므로, 특별하긴 하지만 압도적이지는 "
                "않습니다\n"
                "- **작동 원리를 찾지 못했습니다.** '기관이 거래를 많이 차지하는 종목일수록 잘 통한다'면 "
                "메커니즘이 있는 것인데, 실제로는 기관 거래 비중과 예측력이 무관했습니다(ρ=-0.159, p=0.30). "
                "왜 하필 하이닉스인지를 설명하지 못한다는 뜻입니다\n"
                "- **대만 기관은 같은 일을 못 합니다.** 대만 증권거래소의 일별 법인 매매로 대만 메모리 4사를 "
                "예측해보면 예측력이 +0.02로 사실상 0입니다. 같은 기간 같은 종목을 하이닉스 신호로 예측하면 "
                "+0.18로 4/4 모두 앞섭니다. 즉 '메모리주의 기관 수급'이라는 일반 현상이 아니라 "
                "'한국 기관의 하이닉스 수급'에만 있는 현상이라, 오히려 설명이 더 어려워졌습니다\n"
                "- **최근 4.6년만 떼어 보면 개별 종목에서는 통계적으로 유의하지 않습니다**(p=0.06–0.32). "
                "평균은 살아 있지만 검정력이 부족합니다\n"
                "- **최근 구간이 가장 약합니다.** 마이크론 기준 2년 단위 예측력이 2016–2017 +0.36에서 "
                "2026년 +0.05까지 내려왔습니다. 알려진 신호는 닳습니다\n"
                "- **해외주식은 세금이 다릅니다.** 양도소득세 22%(연 250만원 공제)를 자주 실현하면, "
                "매매를 미루는 단순보유 대비 불리합니다. 백테스트에는 이 세금이 빠져 있습니다\n"
                "- 환율 변동(원/달러)도 백테스트에 없습니다\n"
                "- 10.3년 백테스트에서 매매는 148회(연 14회), 보유 구간의 절반이 6거래일 이하로 짧습니다\n\n"
                "#### 결론\n"
                "처음 검증에서는 '하이닉스에서만 통하는 우연'으로 보였지만, 해외까지 넓혀 보니 "
                "**글로벌 반도체 업황을 앞서 반영하는 신호**에 가깝다는 쪽으로 근거가 기울었습니다. "
                "반도체 12종목은 맞히고 무관한 5종목은 못 맞힌다는 구별이 가장 강한 근거입니다.\n\n"
                "다만 왜 하필 하이닉스인지는 설명하지 못했고, 45종목 중 1위라는 사실은 우연으로도 "
                "일부 설명됩니다. 근거가 기울었을 뿐 증명된 것은 아닙니다.\n\n"
                "이건 여전히 과거 데이터에 대한 백테스트입니다. 미래를 보장하지 않고, 위 위험들이 남아 있습니다. "
                "전 재산을 거는 기계적 매매 규칙이 아니라, **비중을 조절하는 근거 중 하나**로 쓰는 걸 권합니다."
            )
    except Exception as e:
        st.error(f"매매 신호 계산에 실패했습니다: {e}")


# 탭 하나를 프래그먼트로 둔다. 안에 있는 위젯(표시 품목·기간 슬라이더 등)을 건드리면
# 이 탭만 다시 그린다. 예전에는 전부 모듈 수준이라 위젯 하나에 스크립트 전체가 다시
# 돌았고, 보이는 탭 8개가 통째로 재렌더됐다(실측: DRAM 표시 품목 전환에 13.7초).
# 안쪽에 또 프래그먼트를 두면 안 된다 — 중첩은 Streamlit이 막는다.
@st.fragment
def _render_tab_decline():
    _subheader_with_help(
        "큰폭 하락 조기 신호 (SK하이닉스 전용, 참고용)",
        "과거 큰폭 하락 8건에서 뽑은 조건(외국인 순매도 + 거래량 증가)이 "
        f"향후 {DECLINE_HORIZON}거래일 내 {DECLINE_DRAWDOWN_THRESHOLD:.0%} 하락 확률을 높이는지 본 백테스트입니다"
        " (38.0% vs 24.2%, p=0.001).\n\n"
        "앞/뒤 절반 검증에서도 방향은 일관됐습니다. 다른 종목에는 일반화되지 않습니다. 매매 신호가 아닙니다.",
        key="decline",
    )
    if TICKER != DEFAULT_TICKER:
        st.caption("이 지표는 SK하이닉스에서만 검증되어 SK하이닉스에서만 표시됩니다 (다른 종목에는 일반화되지 않는 것으로 자체 검증됨).")
    else:
        try:
            decline_hist = fetch_backtest_history_live(TICKER, target_days=700)
            if len(decline_hist) < 40:
                st.warning("데이터가 부족합니다.")
            else:
                foreign_slope = _rolling_slope(decline_hist["외국인"], DECLINE_PATTERN_WINDOW)
                volume_avg = decline_hist["거래량"].rolling(DECLINE_PATTERN_VOL_WINDOW).mean()
                volume_ratio = decline_hist["거래량"] / volume_avg
                decline_pattern = (foreign_slope < 0) & (volume_ratio > 1.0)

                decline_backtest = run_boolean_pattern_backtest(
                    decline_hist["종가"], decline_hist["날짜"], decline_pattern,
                    DECLINE_HORIZON, drawdown_threshold=DECLINE_DRAWDOWN_THRESHOLD,
                )

                current_foreign_slope = float(foreign_slope.dropna().iloc[-1])
                current_volume_ratio = float(volume_ratio.dropna().iloc[-1])
                foreign_selling = current_foreign_slope < 0
                volume_spike = current_volume_ratio > 1.0
                pattern_now = bool(decline_backtest["current_match"])
                if pattern_now:
                    current_down_rate = decline_backtest["match_down_rate"]
                    current_up_rate = decline_backtest["match_up_rate"]
                else:
                    current_down_rate = decline_backtest["rest_down_rate"]
                    current_up_rate = decline_backtest["rest_up_rate"]

                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                metric_col1.metric(
                    f"외국인 {DECLINE_PATTERN_WINDOW}일 누적 순매수 기울기",
                    f"{current_foreign_slope:,.0f}",
                    delta="순매도 우위" if foreign_selling else "순매수 우위",
                    delta_color="inverse" if foreign_selling else "normal",
                )
                metric_col2.metric(
                    f"거래량 / 직전 {DECLINE_PATTERN_VOL_WINDOW}일 평균",
                    f"{current_volume_ratio:.2f}배",
                    delta="평균 이상" if volume_spike else "평균 이하",
                    delta_color="inverse" if volume_spike else "normal",
                )
                metric_col3.metric(
                    f"현재 상태의 {DECLINE_HORIZON}거래일 내 하락 확률",
                    f"{current_down_rate:.1%}" if current_down_rate is not None else "N/A",
                )
                metric_col4.metric(
                    f"현재 상태의 {DECLINE_HORIZON}거래일 내 상승 확률",
                    f"{current_up_rate:.1%}" if current_up_rate is not None else "N/A",
                )

                if pattern_now:
                    st.warning(
                        "현재 외국인 순매도 + 거래량 증가가 동시에 나타나고 있습니다. "
                        "과거 큰폭 하락 초기와 유사한 패턴이지만, 확정적 신호가 아니라 참고용입니다."
                    )
                else:
                    st.info("현재는 과거 큰폭 하락 초기 패턴(외국인 순매도 + 거래량 증가 동시 발생)과 일치하지 않습니다. (참고용)")

                decline_breakdown_rows = [
                    {
                        "조건": "패턴 일치 (순매도+거래량 급증)",
                        "표본 수": decline_backtest["match_n"],
                        "하락 확률": decline_backtest["match_down_rate"],
                        "p-value(하락)": decline_backtest["down_p_value"],
                        "상승 확률": decline_backtest["match_up_rate"],
                        "p-value(상승)": decline_backtest["up_p_value"],
                    },
                    {
                        "조건": "패턴 불일치",
                        "표본 수": decline_backtest["rest_n"],
                        "하락 확률": decline_backtest["rest_down_rate"],
                        "p-value(하락)": decline_backtest["down_p_value"],
                        "상승 확률": decline_backtest["rest_up_rate"],
                        "p-value(상승)": decline_backtest["up_p_value"],
                    },
                ]
                decline_breakdown_df = pd.DataFrame(decline_breakdown_rows)
                decline_breakdown_display = decline_breakdown_df.copy()
                decline_breakdown_display["표본 수"] = decline_breakdown_display["표본 수"].map(lambda v: f"{v}일")
                for col in ["하락 확률", "상승 확률"]:
                    decline_breakdown_display[col] = decline_breakdown_display[col].map(
                        lambda v: f"{v:.1%}" if pd.notna(v) else "N/A"
                    )
                for col in ["p-value(하락)", "p-value(상승)"]:
                    decline_breakdown_display[col] = decline_breakdown_display[col].map(
                        lambda v: f"{v:.4f}" if pd.notna(v) else "N/A"
                    )
                st.table(decline_breakdown_display, width="stretch", hide_index=True)

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
                _style_chart_mobile(fig_decline_foreign, title=f"외국인 {DECLINE_PATTERN_WINDOW}일 누적 순매수 기울기 vs 주가")
                fig_decline_foreign.update_yaxes(title_text="기울기", secondary_y=False)
                fig_decline_foreign.update_yaxes(title_text="종가(원)", secondary_y=True)
                fig_decline_foreign.update_xaxes(rangeslider_visible=True)
                st.plotly_chart(fig_decline_foreign, width="stretch", key="chart_decline_foreign", config=PLOTLY_CONFIG)

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
                _style_chart_mobile(fig_decline_volume, title=f"거래량 / 직전 {DECLINE_PATTERN_VOL_WINDOW}일 평균 vs 주가")
                fig_decline_volume.update_yaxes(title_text="거래량비율(배)", secondary_y=False)
                fig_decline_volume.update_yaxes(title_text="종가(원)", secondary_y=True)
                fig_decline_volume.update_xaxes(rangeslider_visible=True)
                st.plotly_chart(fig_decline_volume, width="stretch", key="chart_decline_volume", config=PLOTLY_CONFIG)
                st.caption("차트 하단 슬라이더를 드래그하면 보고 싶은 기간만 확대해서 볼 수 있습니다.")
        except Exception as e:
            st.error(f"조기 신호 조회에 실패했습니다: {e}")

# 탭 하나를 프래그먼트로 둔다. 안에 있는 위젯(표시 품목·기간 슬라이더 등)을 건드리면
# 이 탭만 다시 그린다. 예전에는 전부 모듈 수준이라 위젯 하나에 스크립트 전체가 다시
# 돌았고, 보이는 탭 8개가 통째로 재렌더됐다(실측: DRAM 표시 품목 전환에 13.7초).
# 안쪽에 또 프래그먼트를 두면 안 된다 — 중첩은 Streamlit이 막는다.
@st.fragment
def _render_tab_rally():
    _subheader_with_help(
        "큰폭 상승 조기 신호 (SK하이닉스 전용, 참고용)",
        "과거 큰폭 상승 16건에서 뽑은 조건(기관 순매수 + 개인 순매도 + 거래량 증가)이 "
        f"향후 {RALLY_HORIZON}거래일 확률을 바꾸는지 본 백테스트입니다.\n\n"
        "**신뢰도가 낮습니다.** 앞/뒤 절반으로 나누면 뒷반기에서만 나타나 시기에 치우친 결과일 수 있고, "
        "다른 종목에는 일반화되지 않습니다. 매매 신호가 아닙니다.",
        key="rally",
    )
    if TICKER != DEFAULT_TICKER:
        st.caption("이 지표는 SK하이닉스에서만 검증되어 SK하이닉스에서만 표시됩니다 (다른 종목에는 일반화되지 않는 것으로 자체 검증됨).")
    else:
        try:
            rally_hist = fetch_backtest_history_live(TICKER, target_days=700)
            if len(rally_hist) < 40:
                st.warning("데이터가 부족합니다.")
            else:
                inst_slope = _rolling_slope(rally_hist["기관"], RALLY_PATTERN_WINDOW)
                retail_slope = _rolling_slope(rally_hist["개인"], RALLY_PATTERN_WINDOW)
                volume_avg = rally_hist["거래량"].rolling(RALLY_PATTERN_VOL_WINDOW).mean()
                volume_ratio = rally_hist["거래량"] / volume_avg
                rally_pattern = (inst_slope > 0) & (retail_slope < 0) & (volume_ratio > 1.0)

                rally_backtest = run_boolean_pattern_backtest(
                    rally_hist["종가"], rally_hist["날짜"], rally_pattern,
                    RALLY_HORIZON, drawdown_threshold=RALLY_DRAWDOWN_THRESHOLD,
                )

                current_inst_slope = float(inst_slope.dropna().iloc[-1])
                current_retail_slope = float(retail_slope.dropna().iloc[-1])
                current_volume_ratio = float(volume_ratio.dropna().iloc[-1])
                inst_buying = current_inst_slope > 0
                retail_selling = current_retail_slope < 0
                volume_spike = current_volume_ratio > 1.0
                rally_pattern_now = bool(rally_backtest["current_match"])
                if rally_pattern_now:
                    rally_current_down_rate = rally_backtest["match_down_rate"]
                    rally_current_up_rate = rally_backtest["match_up_rate"]
                else:
                    rally_current_down_rate = rally_backtest["rest_down_rate"]
                    rally_current_up_rate = rally_backtest["rest_up_rate"]

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
                metric_col4, metric_col5 = st.columns(2)
                metric_col4.metric(
                    f"현재 상태의 {RALLY_HORIZON}거래일 내 하락 확률",
                    f"{rally_current_down_rate:.1%}" if rally_current_down_rate is not None else "N/A",
                )
                metric_col5.metric(
                    f"현재 상태의 {RALLY_HORIZON}거래일 내 상승 확률",
                    f"{rally_current_up_rate:.1%}" if rally_current_up_rate is not None else "N/A",
                )

                if rally_pattern_now:
                    st.success(
                        "현재 기관 순매수 + 개인 순매도 + 거래량 증가가 동시에 나타나고 있습니다. "
                        "과거 큰폭 상승 초기와 유사한 패턴이지만, 확정적 신호가 아니라 참고용입니다."
                    )
                else:
                    st.info("현재는 과거 큰폭 상승 초기 패턴(기관 순매수 + 개인 순매도 + 거래량 증가 동시 발생)과 일치하지 않습니다. (참고용)")

                rally_breakdown_rows = [
                    {
                        "조건": "패턴 일치 (기관매수+개인매도+거래량 급증)",
                        "표본 수": rally_backtest["match_n"],
                        "하락 확률": rally_backtest["match_down_rate"],
                        "p-value(하락)": rally_backtest["down_p_value"],
                        "상승 확률": rally_backtest["match_up_rate"],
                        "p-value(상승)": rally_backtest["up_p_value"],
                    },
                    {
                        "조건": "패턴 불일치",
                        "표본 수": rally_backtest["rest_n"],
                        "하락 확률": rally_backtest["rest_down_rate"],
                        "p-value(하락)": rally_backtest["down_p_value"],
                        "상승 확률": rally_backtest["rest_up_rate"],
                        "p-value(상승)": rally_backtest["up_p_value"],
                    },
                ]
                rally_breakdown_df = pd.DataFrame(rally_breakdown_rows)
                rally_breakdown_display = rally_breakdown_df.copy()
                rally_breakdown_display["표본 수"] = rally_breakdown_display["표본 수"].map(lambda v: f"{v}일")
                for col in ["하락 확률", "상승 확률"]:
                    rally_breakdown_display[col] = rally_breakdown_display[col].map(
                        lambda v: f"{v:.1%}" if pd.notna(v) else "N/A"
                    )
                for col in ["p-value(하락)", "p-value(상승)"]:
                    rally_breakdown_display[col] = rally_breakdown_display[col].map(
                        lambda v: f"{v:.4f}" if pd.notna(v) else "N/A"
                    )
                st.table(rally_breakdown_display, width="stretch", hide_index=True)

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
                _style_chart_mobile(fig_rally_flow, title=f"기관/개인 {RALLY_PATTERN_WINDOW}일 누적 순매수 기울기 vs 주가")
                fig_rally_flow.update_yaxes(title_text="기울기", secondary_y=False)
                fig_rally_flow.update_yaxes(title_text="종가(원)", secondary_y=True)
                fig_rally_flow.update_xaxes(rangeslider_visible=True)
                st.plotly_chart(fig_rally_flow, width="stretch", key="chart_rally_flow", config=PLOTLY_CONFIG)

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
                _style_chart_mobile(fig_rally_volume, title=f"거래량 / 직전 {RALLY_PATTERN_VOL_WINDOW}일 평균 vs 주가")
                fig_rally_volume.update_yaxes(title_text="거래량비율(배)", secondary_y=False)
                fig_rally_volume.update_yaxes(title_text="종가(원)", secondary_y=True)
                fig_rally_volume.update_xaxes(rangeslider_visible=True)
                st.plotly_chart(fig_rally_volume, width="stretch", key="chart_rally_volume", config=PLOTLY_CONFIG)
                st.caption("차트 하단 슬라이더를 드래그하면 보고 싶은 기간만 확대해서 볼 수 있습니다.")
        except Exception as e:
            st.error(f"조기 신호 조회에 실패했습니다: {e}")

# 탭 하나를 프래그먼트로 둔다. 안에 있는 위젯(표시 품목·기간 슬라이더 등)을 건드리면
# 이 탭만 다시 그린다. 예전에는 전부 모듈 수준이라 위젯 하나에 스크립트 전체가 다시
# 돌았고, 보이는 탭 8개가 통째로 재렌더됐다(실측: DRAM 표시 품목 전환에 13.7초).
# 안쪽에 또 프래그먼트를 두면 안 된다 — 중첩은 Streamlit이 막는다.
@st.fragment
def _render_tab_dram():
    global dram_summary

    dram_summary = "해당 없음 (메모리 반도체 관련주가 아니라 DRAM 시세를 표시하지 않음)"

    _subheader_with_help(
        "DRAM 현물가 (모듈 + 칩)",
        "DRAMeXchange에 공시되는 DRAM 현물가입니다. 칩(반도체 단품)과 모듈(칩을 붙인 완제품) 가격을 각각 보여줍니다. "
        "현물가는 기업 간 장기 계약가(고정가)보다 먼저 움직이는 편이라, 메모리 업황의 선행 지표로 참고합니다.\n\n"
        "'변동률(%)'은 사이트의 직전 갱신 대비, 'N일 전 대비'는 이 대시보드가 쌓은 이력과 비교한 값입니다. "
        "사이트의 Last Update가 바뀔 때만 새 기록이 쌓입니다.",
        key="dram",
    )
    if TICKER not in MEMORY_SEMICONDUCTOR_TICKERS:
        st.caption("이 지표는 메모리 반도체 관련주(SK하이닉스, 삼성전자)에서만 제공됩니다.")
    else:
        try:
            module_df, module_last_update = fetch_dram_module_prices()
            chip_df, chip_last_update = fetch_dram_chip_prices()
            combined_df = pd.concat([module_df, chip_df], ignore_index=True)

            if combined_df.empty:
                st.warning("DRAM 현물가 데이터를 가져오지 못했습니다.")
            else:
                dram_summary = "\n".join(
                    f"- {row['품목']}: ${row['평균가(USD)']:,.3f} ({_signed_pct(row)})" for _, row in combined_df.iterrows()
                )
                # 기준일을 안 붙이면 AI가 며칠 전 시세를 '오늘 올랐다'로 옮겨 적는다.
                _asof = " · ".join(x for x in (
                    f"모듈 {module_last_update}" if module_last_update else "",
                    f"칩 {chip_last_update}" if chip_last_update else "") if x)
                if _asof:
                    dram_summary += (f"\n- (TrendForce 기준일: {_asof}. 매일 갱신되지는 않으므로"
                                     " 이 날짜를 확인하고 인용해라)")
                history = save_dram_snapshot(module_df, module_last_update, chip_df, chip_last_update)

                chip_hist = history[history["품목"].isin(chip_df["품목"])]
                module_hist = history[history["품목"].isin(module_df["품목"])]

                def _max_compare_days(item_hist: pd.DataFrame) -> int:
                    if item_hist.empty:
                        return 1
                    return max((item_hist["날짜"].max() - item_hist["날짜"].min()).days, 1)

                chip_max_days = _max_compare_days(chip_hist)
                module_max_days = _max_compare_days(module_hist)

                _bold_label_with_help(
                    "칩 현물가 (DDR5 16Gb, DDR4 16Gb)",
                    "칩(반도체 단품) 자체의 현물가입니다. 모듈보다 유통 단계가 적어 메모리 시황 변화가 먼저 반영되는 편입니다.",
                    key="dram_chip_label",
                )
                st.caption(
                    (f"사이트 기준 업데이트: {chip_last_update} (GMT+8)"
                     + _stale_note(chip_last_update)) if chip_last_update
                    else "사이트의 업데이트 시각을 확인하지 못해 조회 시각으로 기록했습니다."
                )
                if chip_max_days >= 2:
                    compare_days_chip = st.slider(
                        "변동률 비교 기간 (일 전)", min_value=1, max_value=chip_max_days,
                        value=min(7, chip_max_days), step=1, key="dram_chip_compare_days",
                    )
                else:
                    compare_days_chip = 1
                    st.caption("이력이 더 쌓이면 비교 기간을 선택할 수 있습니다.")
                chip_display = chip_df.drop(columns=["방향"]).copy()
                chip_display["평균가(USD)"] = chip_display["평균가(USD)"].map(lambda v: f"${v:,.3f}")
                chip_display["변동률(%)"] = chip_df.apply(_signed_pct, axis=1)
                chip_display[f"{compare_days_chip}일 전 대비"] = [
                    "N/A" if (p := _period_change_pct(history, item, compare_days_chip)) is None else f"{p:+.2f}%"
                    for item in chip_df["품목"]
                ]
                _render_dram_price_table(chip_display, ["변동률(%)", f"{compare_days_chip}일 전 대비"])
                if chip_hist["날짜"].nunique() >= 2:
                    st.markdown("**칩 현물가 추이 (누적 기록)**")
                    _render_dram_trend_chart(history, list(chip_df["품목"]), "dram_chip_toggle", "chart_dram_chip")
                else:
                    st.caption("아직 사이트 업데이트가 한 번만 기록돼 있어서, 다음 업데이트부터 추이 그래프가 표시됩니다.")

                st.divider()

                _bold_label_with_help(
                    "모듈 현물가 (DDR5 UDIMM/RDIMM)",
                    "칩을 기판에 조립해 PC/서버에 바로 장착할 수 있게 만든 완제품(RAM 카드) 현물가입니다. "
                    "칩 가격에 조립·유통 마진이 더해집니다.",
                    key="dram_module_label",
                )
                st.caption(
                    (f"사이트 기준 업데이트: {module_last_update} (GMT+8)"
                     + _stale_note(module_last_update)) if module_last_update
                    else "사이트의 업데이트 시각을 확인하지 못해 조회 시각으로 기록했습니다."
                )
                if module_max_days >= 2:
                    compare_days_module = st.slider(
                        "변동률 비교 기간 (일 전)", min_value=1, max_value=module_max_days,
                        value=min(7, module_max_days), step=1, key="dram_module_compare_days",
                    )
                else:
                    compare_days_module = 1
                    st.caption("이력이 더 쌓이면 비교 기간을 선택할 수 있습니다.")
                module_display = module_df.drop(columns=["방향"]).copy()
                module_display["평균가(USD)"] = module_display["평균가(USD)"].map(lambda v: f"${v:,.2f}")
                module_display["변동률(%)"] = module_df.apply(_signed_pct, axis=1)
                module_display[f"{compare_days_module}일 전 대비"] = [
                    "N/A" if (p := _period_change_pct(history, item, compare_days_module)) is None else f"{p:+.2f}%"
                    for item in module_df["품목"]
                ]
                _render_dram_price_table(module_display, ["변동률(%)", f"{compare_days_module}일 전 대비"])
                if module_hist["날짜"].nunique() >= 2:
                    st.markdown("**모듈 현물가 추이 (누적 기록)**")
                    _render_dram_trend_chart(history, list(module_df["품목"]), "dram_module_toggle", "chart_dram_module")
                else:
                    st.caption("아직 사이트 업데이트가 한 번만 기록돼 있어서, 다음 업데이트부터 추이 그래프가 표시됩니다.")
        except Exception as e:
            st.error(f"DRAM 현물가 조회에 실패했습니다: {e}")

# 탭 하나를 프래그먼트로 둔다. 안에 있는 위젯(표시 품목·기간 슬라이더 등)을 건드리면
# 이 탭만 다시 그린다. 예전에는 전부 모듈 수준이라 위젯 하나에 스크립트 전체가 다시
# 돌았고, 보이는 탭 8개가 통째로 재렌더됐다(실측: DRAM 표시 품목 전환에 13.7초).
# 안쪽에 또 프래그먼트를 두면 안 된다 — 중첩은 Streamlit이 막는다.
@st.fragment
def _render_tab_capex():
    _subheader_with_help(
        "빅테크 분기별 Capex",
        "마이크로소프트·구글·아마존·메타의 분기별 Capex(SEC 공시 기준)입니다. AI/데이터센터 투자가 HBM·DRAM 수요의 "
        "핵심 동력이라 참고용으로 제공하며, 실적 발표 지연으로 최신 분기 수치가 며칠–몇 주 늦어질 수 있습니다.",
        key="capex",
    )
    try:
        capex_df_all = fetch_bigtech_capex()
        if capex_df_all.empty:
            st.warning("Capex 데이터를 가져오지 못했습니다.")
        else:
            capex_df_all = capex_df_all.copy()
            capex_df_all["capex_B"] = capex_df_all["capex_USD"] / 1e9

            company_list = list(BIGTECH_CIKS.keys())
            selected_companies = [c for c in company_list if st.session_state.get(f"capex_company_{c}", True)]

            if not selected_companies:
                st.info("표시할 기업을 하나 이상 선택해주세요.")
            else:
                capex_df = capex_df_all[capex_df_all["기업"].isin(selected_companies)]

                # 예전 세션에 남아 있는 값(빠진 '연도별 누적' 등)이 그대로 있으면 st.radio가 터진다
                if st.session_state.get("capex_view_mode") not in CAPEX_VIEW_MODES:
                    st.session_state.pop("capex_view_mode", None)
                _mode = st.radio(
                    "보기 방식", CAPEX_VIEW_MODES, horizontal=True, key="capex_view_mode",
                    help=(
                        "**분기별** — 그 분기에 쓴 돈. 기본값입니다.\n\n"
                        "**최근 4분기 합(TTM)** — 분기 들쭉날쭉함을 걷어낸 실제 투자 속도입니다. "
                        "추세를 보려면 이걸 보세요.\n\n"
                        "**전체 누적** — 첫 분기부터 계속 더합니다. 항상 우상향이라 보기엔 좋지만, "
                        "시작점이 데이터를 어디부터 받아왔는지에 따라 정해질 뿐이라 "
                        "읽어낼 수 있는 정보는 가장 적습니다."
                    ),
                )

                totals = capex_df.groupby("분기말")["capex_B"].sum().reset_index().sort_values("분기말")

                if _mode == CAPEX_VIEW_MODES[0]:            # 분기별 (기존 그래프)
                    fig_capex = px.bar(
                        capex_df, x="분기말", y="capex_B", color="기업", barmode="stack",
                        labels={"capex_B": "Capex (10억달러)", "분기말": "분기"},
                        color_discrete_map=CAPEX_COMPANY_COLORS,
                    )
                    totals["qoq_pct"] = totals["capex_B"].pct_change() * 100
                    fig_capex.add_trace(
                        go.Scatter(x=totals["분기말"], y=totals["capex_B"], name="합계",
                                   mode="lines+markers", line=dict(color="gray", dash="dot")),
                    )
                    fig_capex.add_trace(
                        go.Scatter(
                            x=totals["분기말"], y=totals["qoq_pct"], name="증감률",
                            mode="lines+markers", line=dict(color="#d62728"), yaxis="y2",
                        ),
                    )
                    fig_capex.update_layout(
                        yaxis=dict(title="Capex (10억달러)"),
                        yaxis2=dict(title="증감률(%)", overlaying="y", side="right", showgrid=False),
                    )
                    _capex_note = "차트 하단 슬라이더를 드래그하면 보고 싶은 기간만 확대해서 볼 수 있습니다."
                else:
                    acc = _capex_accumulate(capex_df, _mode)
                    fig_capex = px.line(
                        acc, x="분기말", y="capex_B", color="기업", markers=True,
                        labels={"capex_B": "Capex (10억달러)", "분기말": "분기"},
                        color_discrete_map=CAPEX_COMPANY_COLORS,
                    )
                    tot_acc = acc.groupby("분기말")["capex_B"].sum().reset_index().sort_values("분기말")
                    fig_capex.add_trace(
                        go.Scatter(x=tot_acc["분기말"], y=tot_acc["capex_B"], name="4사 합계",
                                   mode="lines+markers", line=dict(color="gray", dash="dot")),
                    )
                    fig_capex.update_layout(yaxis=dict(title="Capex (10억달러)"))
                    if _mode == CAPEX_VIEW_MODES[1]:
                        last, prev = tot_acc["capex_B"].iloc[-1], (
                            tot_acc["capex_B"].iloc[-5] if len(tot_acc) >= 5 else None)
                        _capex_note = (f"최근 4분기 합계 {last:,.0f}B 달러"
                                       + (f" · 1년 전 같은 시점 {prev:,.0f}B 대비 {last/prev-1:+.0%}"
                                          if prev else ""))
                    else:
                        _capex_note = ("첫 분기부터 계속 더한 값입니다. 시작점이 데이터 수집 시작일일 뿐이라"
                                       " 기울기(=투자 속도)만 의미가 있습니다.")

                _style_chart_mobile(fig_capex)
                fig_capex.update_layout(legend=dict(title=dict(text="")))
                fig_capex.update_xaxes(rangeslider_visible=True)
                st.plotly_chart(fig_capex, width="stretch", key="chart_bigtech_capex", config=PLOTLY_CONFIG)
                st.caption(_capex_note)

                company_cols = st.columns(len(company_list))
                for col, company in zip(company_cols, company_list):
                    with col:
                        emoji = CAPEX_COMPANY_EMOJI.get(company, "⬜")
                        st.checkbox(f"{emoji} {company}", value=True, key=f"capex_company_{company}")

                company_count = capex_df.groupby("분기말")["기업"].nunique()
                n_companies = len(selected_companies)
                complete_qs = sorted(company_count[company_count == n_companies].index)
                incomplete_latest = capex_df["분기말"].max() not in complete_qs
                if incomplete_latest:
                    missing = set(selected_companies) - set(
                        capex_df[capex_df["분기말"] == capex_df["분기말"].max()]["기업"]
                    )
                    st.caption(f"⚠️ 최근 분기는 {', '.join(missing)}의 실적 발표 전이라 그래프의 마지막 막대는 아직 미완성입니다.")

                with st.expander("분기별 상세 수치 보기"):
                    pivot = capex_df.pivot(index="분기말", columns="기업", values="capex_B").sort_index(ascending=False)
                    pivot["합계"] = pivot.sum(axis=1)
                    qoq_pivot = pivot.pct_change(periods=-1) * 100
                    pivot.index = pivot.index.strftime("%Y-%m")
                    qoq_pivot.index = qoq_pivot.index.strftime("%Y-%m")

                    st.markdown("**Capex (10억달러)**")
                    st.table(pivot.round(1), width="stretch")

                    st.markdown("**전분기 대비 증감률(%)**")
                    st.table(qoq_pivot.round(1), width="stretch")
    except Exception as e:
        st.error(f"빅테크 Capex 조회에 실패했습니다: {e}")

def _render_financial_digest() -> None:
    """재무 탭 맨 위 AI 요약. **여기서 만들지 않는다.**

    예전에는 렌더 안에서 생성했는데, Streamlit이 보이는 탭을 매 리런마다 다시 그리는 탓에
    생성이 필요한 순간 페이지 전체가 20~60초 멈췄다. 게다가 생성이 실패하면 저장할 게 없어
    다음 리런에 또 시도했고, 모델이 전부 막힌 날은 클릭할 때마다 100초씩 멈췄다.
    이제 생성은 수집기가 아침에 한 번 하고, 화면은 저장된 결과를 읽기만 한다.
    """
    saved = financial_digest.load()
    col_a, col_b = st.columns([0.75, 0.25], vertical_alignment="center")
    if saved.get("text"):
        col_a.caption(f"요약 기준: {saved.get('date', '-')} · 매일 아침 값이 바뀌었을 때만 갱신")
    else:
        col_a.caption("아직 요약이 없습니다. 내일 아침 수집기가 만들거나, 지금 바로 만들 수 있습니다.")

    if col_b.button("지금 갱신", key="fin_refresh",
                    help="수집기를 기다리지 않고 지금 다시 만듭니다. 20~60초 걸립니다."):
        with st.spinner("재무 데이터를 읽는 중..."):
            try:
                saved, _ = financial_digest.refresh(TICKER, f"{STOCK_NAME}({TICKER})", force=True)
            except Exception as e:
                st.error(f"요약 생성에 실패했습니다: {e}")

    if saved.get("text"):
        if saved.get("note"):
            st.warning(saved["note"])
        st.markdown(_md_safe(saved["text"]))
        st.divider()


def _fin_style(df: pd.DataFrame) -> pd.DataFrame:
    """표에 넣을 수 있게 숫자를 사람이 읽는 형태로 바꾼다."""
    def fmt(name, v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "-"
        if "억원" in name:
            return f"{v:,.0f}"
        if "원)" in name:            # EPS·BPS
            return f"{v:,.0f}"
        return f"{v:,.2f}"
    # 숫자 표에 문자열을 덮어쓰면 pandas가 dtype 경고를 낸다. 새 표를 만들어 돌려준다.
    return pd.DataFrame(
        [[fmt(idx, v) for v in df.loc[idx]] for idx in df.index],
        index=df.index, columns=df.columns,
    )


# 탭 하나를 프래그먼트로 둔다. 안에 있는 위젯(표시 품목·기간 슬라이더 등)을 건드리면
# 이 탭만 다시 그린다. 예전에는 전부 모듈 수준이라 위젯 하나에 스크립트 전체가 다시
# 돌았고, 보이는 탭 8개가 통째로 재렌더됐다(실측: DRAM 표시 품목 전환에 13.7초).
# 안쪽에 또 프래그먼트를 두면 안 된다 — 중첩은 Streamlit이 막는다.
@st.fragment
def _render_tab_financials():
    _subheader_with_help(
        "재무 데이터",
        "FnGuide(Company Guide)의 Financial Highlight를 그대로 가져옵니다. IFRS 연결 기준이며, "
        "단위는 매출·이익이 억원, EPS·BPS가 원, 나머지는 %/배입니다.\n\n"
        "**(E)가 붙은 열은 확정 실적이 아니라 증권사 추정치 평균**입니다. 확정치와 섞어서 "
        "추세를 읽지 마세요.\n\n"
        "순이익률·ROE·BPS는 지배주주 기준입니다.",
        key="financials",
    )
    try:
        annual, quarter = fetch_financials(TICKER)
    except Exception as e:
        st.error(f"재무 데이터를 가져오지 못했습니다: {e}")
        return
    if annual.empty and quarter.empty:
        st.warning("재무 데이터를 가져오지 못했습니다.")
        return

    # 표보다 먼저 '이 숫자들이 무슨 뜻인지'를 읽히게 한다
    _render_financial_digest()

    if not annual.empty:
        st.markdown("**연간 (IFRS 연결)**")
        st.table(_fin_style(annual), width="stretch")

        # 매출·이익 추이와 이익률을 한 그림에 둔다. 규모와 효율은 같이 봐야 뜻이 생긴다.
        rev = annual.loc[[i for i in annual.index if i.startswith("매출액")]]
        op = annual.loc[[i for i in annual.index if i.startswith("영업이익 ")]]
        npf = annual.loc[[i for i in annual.index if i.startswith("당기순이익")]]
        opm = annual.loc[[i for i in annual.index if i.startswith("영업이익률")]]
        if not rev.empty:
            periods = list(annual.columns)
            fig = go.Figure()
            for frame, name, color in ((rev, "매출액", "#4c78a8"),
                                       (op, "영업이익", "#54a24b"),
                                       (npf, "당기순이익", "#e45756")):
                if not frame.empty:
                    fig.add_trace(go.Bar(x=periods, y=list(frame.iloc[0]), name=name,
                                         marker_color=color))
            if not opm.empty:
                fig.add_trace(go.Scatter(x=periods, y=list(opm.iloc[0]), name="영업이익률(%)",
                                         mode="lines+markers", yaxis="y2",
                                         line=dict(color="#f58518")))
            _style_chart_mobile(fig)
            fig.update_layout(
                barmode="group",
                yaxis=dict(title="억원"),
                yaxis2=dict(title="영업이익률(%)", overlaying="y", side="right", showgrid=False),
                legend=dict(title=dict(text="")),
            )
            st.plotly_chart(fig, width="stretch", key="chart_fin_annual",
                            config=PLOTLY_CONFIG)

    if not quarter.empty:
        st.markdown("**분기 (IFRS 연결)**")
        st.table(_fin_style(quarter), width="stretch")

    st.divider()
    _subheader_with_help(
        "밸류에이션과 시장 평가",
        "왼쪽은 증권사 컨센서스 목표주가가 어떻게 움직여왔는지, 오른쪽은 차입공매도 비중입니다. "
        "둘 다 FnGuide 주간 데이터라 최근 1년치만 있습니다.\n\n"
        "목표주가는 **수준보다 방향**이 중요합니다. 올라가는 중이면 실적 기대가 상향되고 있다는 뜻입니다.\n\n"
        "차입공매도 비중은 전체 거래에서 공매도가 차지하는 비율입니다. 높다고 곧 하락은 아니고, "
        "되레 숏커버링이 나오면 상승 재료가 되기도 합니다.",
        key="valuation",
    )
    vcol1, vcol2 = st.columns(2)
    try:
        tp = fetch_target_price_history(TICKER)
        with vcol1:
            if tp.empty:
                st.caption("목표주가 추이를 가져오지 못했습니다.")
            else:
                fig_t = go.Figure()
                fig_t.add_trace(go.Scatter(x=tp["일자"], y=tp["목표주가"], name="목표주가",
                                           mode="lines", line=dict(color="#4c78a8")))
                fig_t.add_trace(go.Scatter(x=tp["일자"], y=tp["주가"], name="주가",
                                           mode="lines", line=dict(color="#888", dash="dot")))
                _style_chart_mobile(fig_t)
                fig_t.update_layout(yaxis=dict(title="원"), legend=dict(title=dict(text="")))
                st.plotly_chart(fig_t, width="stretch", key="chart_target_price",
                                config=PLOTLY_CONFIG)
                first, last = tp.iloc[0], tp.iloc[-1]
                st.caption(f"{first['일자']:%Y-%m-%d} {first['목표주가']:,.0f}원 → "
                           f"{last['일자']:%Y-%m-%d} {last['목표주가']:,.0f}원 "
                           f"({last['목표주가'] / first['목표주가'] - 1:+.1%})")
    except Exception as e:
        vcol1.caption(f"목표주가 추이 실패: {type(e).__name__}")

    try:
        sb = fetch_short_balance(TICKER)
        with vcol2:
            if sb.empty:
                st.caption("차입공매도 비중을 가져오지 못했습니다.")
            else:
                fig_s = go.Figure()
                fig_s.add_trace(go.Scatter(x=sb["일자"], y=sb["차입공매도비중"],
                                           name="차입공매도비중(%)", mode="lines",
                                           line=dict(color="#e45756")))
                fig_s.add_trace(go.Scatter(x=sb["일자"], y=sb["수정주가"], name="주가",
                                           mode="lines", yaxis="y2",
                                           line=dict(color="#888", dash="dot")))
                _style_chart_mobile(fig_s)
                fig_s.update_layout(
                    yaxis=dict(title="공매도 비중(%)"),
                    yaxis2=dict(title="주가", overlaying="y", side="right", showgrid=False),
                    legend=dict(title=dict(text="")),
                )
                st.plotly_chart(fig_s, width="stretch", key="chart_short_balance",
                                config=PLOTLY_CONFIG)
                st.caption(f"최근 {sb.iloc[-1]['일자']:%Y-%m-%d} 기준 "
                           f"{sb.iloc[-1]['차입공매도비중']:.2f}% · "
                           f"1년 평균 {sb['차입공매도비중'].mean():.2f}%")
    except Exception as e:
        vcol2.caption(f"차입공매도 비중 실패: {type(e).__name__}")


# ── 애널리스트 리포트 ──────────────────────────────────────────────────────────
# 목록 조회·지문·요약 생성은 analyst_digest.py에 있다. 수집기(collector.py)가 아침에
# 같은 코드로 갱신하기 때문에, 여기서 따로 구현하면 두 곳이 어긋난다.


def _render_broker_targets():
    """증권사별 목표주가와, 그걸로 직접 계산한 컨센서스.

    FnGuide 컨센서스는 주 1회만 갱신돼서 목표주가 변경을 며칠 늦게 반영한다.
    같은 방식(최근 3개월·증권사별 최신 1건)으로 직접 세면 실측 오차 0.3%였고,
    새 리포트가 올라온 당일에 바로 반영된다.
    """
    _subheader_with_help(
        "증권사별 목표주가",
        "각 리포트 상세 페이지에서 목표주가·투자의견 숫자만 뽑아 모은 것입니다. "
        "AI가 개입하지 않은 원자료라 숫자를 그대로 믿어도 됩니다.\n\n"
        "**컨센서스 계산 방식** — 기간 안에서 증권사별로 가장 최근 리포트 한 건씩만 씁니다. "
        "리포트를 자주 내는 증권사가 평균을 좌우하지 않게 하려는 것입니다.\n\n"
        "한 곳이 크게 다른 값을 내면 평균이 끌려가므로 중앙값도 같이 봅니다. "
        "네이버에 리포트를 싣지 않는 증권사(예: LS증권)는 여기 잡히지 않습니다.",
        key="broker_targets",
    )
    months = st.slider("집계 기간 (개월)", min_value=1, max_value=12,
                       value=analyst_targets.CONSENSUS_MONTHS, key="target_months",
                       help="짧게 잡으면 최신 시각만, 길게 잡으면 표본이 늘지만 과거 목표가가 섞입니다.")
    try:
        best = analyst_targets.latest_by_broker(months=months)
        con = analyst_targets.consensus(months=months)
    except Exception as e:
        st.warning(f"목표주가 집계를 읽지 못했습니다: {e}")
        return
    if not con:
        st.info("아직 모인 목표주가가 없습니다. 수집기가 채우면 표시됩니다.")
        return

    cur = _to_number(st.session_state.get("current_price_value"))
    row = st.container(key="price_row_broker_target")
    c1, c2, c3, c4 = row.columns(4)
    with c1.container(key="metric_small_bt_mean"):
        _metric_with_help(
            "직접 집계 (네이버 게재분)", f"{con['평균']:,}원",
            f"최근 {months}개월 안에 목표주가를 낸 {con['기관수']}곳의 평균입니다. "
            f"가장 최근 리포트는 {con['최신일']}자입니다.\n\n"
            "AI 분석 탭의 '컨센서스 목표주가(FnGuide)'와 값이 다른 것이 정상입니다. "
            "FnGuide는 네이버에 리포트를 싣지 않는 증권사까지 포함하고, 이 값은 "
            "아래 표에서 증권사별 내역을 직접 확인할 수 있는 대신 네이버 게재분만 셉니다. "
            "아래 캡션에 두 값의 차이를 적어 둡니다.",
            key="bt_mean",
            delta=(f"{(con['평균'] / cur - 1) * 100:+.0f}% 여력" if cur else None),
            delta_color="normal",
        )
    with c2.container(key="metric_small_bt_median"):
        _metric_with_help(
            "중앙값", f"{con['중앙값']:,}원",
            "가운데 값입니다. 평균과 크게 벌어져 있으면 한쪽에 치우친 목표가가 섞여 있다는 뜻입니다.",
            key="bt_median",
        )
    with c3.container(key="metric_small_bt_range"):
        _metric_with_help(
            "최고 / 최저", f"{con['최고'] / 10000:,.0f} / {con['최저'] / 10000:,.0f}만원",
            "증권가 시각이 얼마나 갈리는지 보여줍니다. 폭이 넓을수록 전망이 엇갈린다는 뜻입니다.",
            key="bt_range",
        )
    with c4.container(key="metric_small_bt_count"):
        _metric_with_help(
            "집계 기관 수", f"{con['기관수']}곳",
            "이 기간에 목표주가를 제시한 증권사 수입니다. 적을수록 평균이 흔들립니다.",
            key="bt_count",
        )

    rows = []
    for brk, r in sorted(best.items(), key=lambda x: -x[1]["목표주가"]):
        rows.append({
            "증권사": brk,
            "목표주가": r["목표주가"],
            # 백분율로 미리 바꿔 넣는다. "percent" 프리셋은 소수점 두 자리가 붙어(138.95%)
            # 목표가처럼 큰 수에서는 자릿수만 늘린다.
            "상승여력": (r["목표주가"] / cur - 1) * 100 if cur else None,
            "투자의견": r.get("투자의견") or "-",
            "작성일": r.get("작성일"),
            "제목": r.get("제목"),
        })
    st.dataframe(
        pd.DataFrame(rows), width="stretch", hide_index=True,
        column_config={
            # printf의 "%,d"는 없는 서식이고, 비율을 "%.0f%%"로 주면 0.15가 "0%"로 찍힌다.
            # 스트림릿 프리셋을 쓴다("percent"가 100을 곱해준다).
            "목표주가": st.column_config.NumberColumn("목표주가(원)", format="localized"),
            "상승여력": st.column_config.NumberColumn("상승여력", format="%+.0f%%"),
        },
    )

    # AI 분석 탭에 뜨는 FnGuide 값과 나란히 보여준다. 두 탭에 같은 이름의 다른 숫자가
    # 떠 있으면 어느 쪽이 맞는지 알 수 없어서, 차이와 그 이유를 여기서 못박는다.
    # 차이의 주된 원인은 갱신 시차가 아니라 모집단이다. 우리는 네이버 게재분만 세고,
    # FnGuide는 LS증권처럼 네이버에 안 실리는 곳까지 넣는다. 실측 차이는 1~2% 안쪽이었다.
    try:
        snap_fn = fetch_stock_snapshot(TICKER) or {}
        fn = _to_number(snap_fn.get("목표주가"))
    except Exception:
        snap_fn, fn = {}, None         # 비교용일 뿐이라 못 받아도 표는 그대로 보여준다
    if fn:
        gap = con["평균"] / fn - 1
        st.caption(
            f"AI 분석 탭의 **컨센서스 목표주가(FnGuide)** {fn:,.0f}원"
            f"(기준일 {snap_fn.get('컨센서스일자') or '-'}) 대비 {gap * 100:+.1f}%. "
            "두 값이 다른 건 어느 한쪽이 틀려서가 아니라 세는 대상이 달라서입니다 — "
            "FnGuide는 네이버에 리포트를 싣지 않는 증권사까지 넣고, 위 표는 네이버 게재분만 "
            "대신 증권사별로 내역을 보여줍니다."
            + (" 차이가 큰 편이니 위 표의 작성일을 함께 보세요." if abs(gap) >= 0.05 else ""))


# 탭 하나를 프래그먼트로 둔다. 안에 있는 위젯(표시 품목·기간 슬라이더 등)을 건드리면
# 이 탭만 다시 그린다. 예전에는 전부 모듈 수준이라 위젯 하나에 스크립트 전체가 다시
# 돌았고, 보이는 탭 8개가 통째로 재렌더됐다(실측: DRAM 표시 품목 전환에 13.7초).
# 안쪽에 또 프래그먼트를 두면 안 된다 — 중첩은 Streamlit이 막는다.
@st.fragment
def _render_tab_disclosure():
    _subheader_with_help(
        "공시",
        "거래소·금감원에 접수된 전자공시입니다(네이버가 중계하는 KOSCOM 자료). "
        "뉴스보다 빠르고 숫자가 확정적이라, 주가를 움직인 원인이 여기 한 줄인 경우가 많습니다.\n\n"
        "맨 위 요약은 최근 공시 15건을 AI가 읽고 정리한 것으로, 그중 10건은 **본문까지** "
        "읽혀 금액·주식수·기간 같은 숫자가 들어갑니다.\n\n"
        "요약은 **새 공시가 떴을 때만** 다시 만듭니다(무료 AI 한도가 하루 20회라 매번 만들지 않습니다). "
        "아래 표에서 각 공시를 펼치면 본문 원문을 그대로 볼 수 있습니다.\n\n"
        "'주식선물·주식옵션 가격제한폭 확대요건 도달'처럼 거래소가 기계적으로 내는 공시도 "
        "섞여 있습니다. 회사의 의사결정이 아닙니다.",
        key="disclosure",
    )
    try:
        df = disclosure.fetch_list(TICKER)
    except Exception as e:
        st.error(f"공시 목록을 가져오지 못했습니다: {e}")
        return
    if df.empty:
        st.info("수집된 공시가 없습니다.")
        return

    saved = disclosure.load()
    fp = disclosure.fingerprint(df)
    new_items = bool(saved.get("fingerprint")) and saved["fingerprint"] != fp

    col_a, col_b = st.columns([0.75, 0.25], vertical_alignment="center")
    if saved.get("text"):
        col_a.caption(
            f"요약 기준: {saved.get('date', '-')} · 공시 {saved.get('count', '-')}건"
            f"(본문 {saved.get('body_count', '-')}건)"
            + (" · 새 공시가 있어 아직 반영 전입니다" if new_items else ""))
    else:
        col_a.caption("아직 요약이 없습니다. 수집기가 곧 만들거나, 지금 바로 만들 수 있습니다.")

    # 생성은 수집기가 맡는다. 화면에서 자동으로 만들면 그동안 페이지가 통째로 멈춘다.
    if col_b.button("요약 갱신", key="disc_refresh",
                    help="새 공시가 없어도 강제로 다시 정리합니다. 20~60초 걸립니다."):
        if not os.environ.get("GEMINI_API_KEY"):
            st.info("요약을 만들려면 GEMINI_API_KEY 환경변수가 필요합니다.")
        else:
            with st.spinner("공시를 읽는 중..."):
                try:
                    saved, _ = disclosure.refresh(
                        TICKER, f"{STOCK_NAME}({TICKER})", df=df, force=True)
                except Exception as e:
                    st.error(f"요약 생성에 실패했습니다: {e}")

    if saved.get("text"):
        st.markdown(_md_safe(saved["text"]))
    elif not os.environ.get("GEMINI_API_KEY"):
        st.info("요약을 보려면 GEMINI_API_KEY 환경변수를 설정해주세요.")

    st.divider()
    st.markdown(f"**공시 목록 ({len(df)}건)**")
    st.dataframe(df[["일시", "제목", "출처"]], width="stretch", hide_index=True)

    # 본문은 이미 받아 둔 것만 보여준다. 여기서 새로 받으면 탭을 열 때마다 수십 번
    # 요청이 나간다(펼치지 않아도 expander 안은 매번 실행된다).
    bodies = disclosure._load_bodies()
    shown = [(r["일시"], r["제목"], bodies.get(str(r["공시ID"])))
             for _, r in df.head(disclosure.BODY_COUNT).iterrows()]
    shown = [x for x in shown if x[2]]
    if shown:
        st.markdown("**공시 본문 원문**")
        for when, title, body in shown:
            with st.expander(f"{when}  {title}"):
                st.text(body)
    else:
        st.caption("본문은 요약을 한 번 만든 뒤에 여기 쌓입니다.")


# 탭 하나를 프래그먼트로 둔다. 안에 있는 위젯(표시 품목·기간 슬라이더 등)을 건드리면
# 이 탭만 다시 그린다. 예전에는 전부 모듈 수준이라 위젯 하나에 스크립트 전체가 다시
# 돌았고, 보이는 탭 8개가 통째로 재렌더됐다(실측: DRAM 표시 품목 전환에 13.7초).
# 안쪽에 또 프래그먼트를 두면 안 된다 — 중첩은 Streamlit이 막는다.
@st.fragment
def _render_tab_analyst():
    _subheader_with_help(
        "애널리스트 리포트",
        "네이버 금융이 모아주는 증권사 리포트 목록입니다. 맨 위 요약은 제목·증권사·날짜에 더해 "
        "최근 몇 건은 **PDF 본문까지** 읽혀 정리한 것이라, 목표주가와 추정 실적 숫자가 들어갑니다.\n\n"
        "그 아래 **증권사별 목표주가**는 리포트 상세 페이지에서 숫자만 따로 모아 누적한 것으로, "
        "AI를 거치지 않은 원자료입니다. 컨센서스도 이 값으로 직접 계산합니다.\n\n"
        "요약은 하루에 한 번, 그리고 **새 리포트가 올라왔을 때만** 다시 만듭니다. "
        "무료 AI 호출 한도가 하루 20회라 매번 새로 만들지 않습니다.\n\n"
        "리포트 제목은 홍보성으로 붙는 경우가 많아 단정적으로 읽지 마세요.",
        key="analyst",
    )
    try:
        df = fetch_analyst_reports(TICKER, count=40)
    except Exception as e:
        st.error(f"리포트 목록을 가져오지 못했습니다: {e}")
        return
    if df.empty:
        st.info("수집된 리포트가 없습니다.")
        return

    fp = analyst_digest.fingerprint(df)
    saved = analyst_digest.load()

    # 새 리포트가 올라왔으면 알아서 다시 정리한다. 날짜만 바뀐 경우에는 다시 하지 않는다
    # (내용이 그대로인데 매일 새로 부르면 하루 20회뿐인 무료 한도를 그냥 태운다).
    # 평일 아침에는 수집기가 먼저 갱신해두므로, 보통 여기서는 파일을 읽기만 한다.
    new_reports = bool(saved.get("fingerprint")) and saved["fingerprint"] != fp
    first_time = not saved.get("text")

    col_a, col_b = st.columns([0.75, 0.25], vertical_alignment="center")
    if saved.get("text"):
        col_a.caption(f"요약 기준: {saved.get('date', '-')} · 리포트 {saved.get('count', '-')}건"
                      + (" · 새 리포트를 반영해 다시 정리했습니다" if new_reports else ""))
    rerun = col_b.button("요약 갱신", key="analyst_refresh",
                         help="내용이 그대로여도 강제로 다시 정리합니다.")

    # 생성은 수집기가 아침에 맡는다. 화면에서 자동으로 만들면 그동안 페이지가 멈추고,
    # 실패했을 때 리런마다 다시 시도해서 클릭할 때마다 화면이 굳는다.
    if rerun and os.environ.get("GEMINI_API_KEY"):
        with st.spinner("리포트를 읽는 중..."):
            try:
                saved, _ = analyst_digest.refresh(
                    TICKER, f"{STOCK_NAME}({TICKER})", df=df, force=True)
            except Exception as e:
                st.error(f"요약 생성에 실패했습니다: {e}")

    if saved.get("text"):
        if saved.get("note"):
            st.warning(saved["note"])
        st.markdown(_md_safe(saved["text"]))
    elif not os.environ.get("GEMINI_API_KEY"):
        st.info("요약을 보려면 GEMINI_API_KEY 환경변수를 설정해주세요.")

    st.divider()
    _render_broker_targets()

    st.divider()
    st.markdown(f"**리포트 목록 ({len(df)}건)**")
    show = df.copy()
    if "url" in show.columns:
        st.dataframe(
            show, width="stretch", hide_index=True,
            column_config={"url": st.column_config.LinkColumn("원문", display_text="PDF")},
        )
    else:
        st.dataframe(show, width="stretch", hide_index=True)

    if "증권사" in df.columns:
        counts = df["증권사"].value_counts().head(8)
        st.caption("리포트를 많이 낸 곳 — " + " · ".join(f"{k} {v}건" for k, v in counts.items()))


# 탭 하나를 프래그먼트로 둔다. 안에 있는 위젯(표시 품목·기간 슬라이더 등)을 건드리면
# 이 탭만 다시 그린다. 예전에는 전부 모듈 수준이라 위젯 하나에 스크립트 전체가 다시
# 돌았고, 보이는 탭 8개가 통째로 재렌더됐다(실측: DRAM 표시 품목 전환에 13.7초).
# 안쪽에 또 프래그먼트를 두면 안 된다 — 중첩은 Streamlit이 막는다.
@st.fragment
def _render_tab_community():
    global community_summary

    community_summary = "커뮤니티 심리 데이터를 가져오지 못함"

    community_post_count = st.slider(
        "게시글 조회 개수 (네이버·디시인사이드 공통)", min_value=20, max_value=600, value=DEFAULT_COMMUNITY_POST_COUNT, step=20,
    )

    _subheader_with_help(
        "커뮤니티 심리 (네이버 종목토론방)",
        f"네이버 종목토론방의 최근 게시글 {community_post_count}건을 긍정/부정/중립으로 분류한 결과입니다. "
        "기본은 제목에 담긴 키워드로 분류하며, 반어법이나 문맥은 잡지 못합니다. "
        "아래 'AI로 더 정확하게 분류' 토글을 켜면 Gemini가 제목의 문맥까지 보고 다시 분류합니다(GEMINI_API_KEY 필요, "
        "시간이 다소 걸릴 수 있음).\n\n"
        "익명 게시판의 여론일 뿐 사실이 아니며, 매매 신호로 쓰지 마세요.",
        key="community",
    )
    try:
        posts_df = fetch_community_posts(TICKER, community_post_count)
        if posts_df.empty:
            st.warning("커뮤니티 게시글을 가져오지 못했습니다.")
        else:
            use_ai_sentiment = st.toggle(
                "AI로 더 정확하게 분류 (키워드 매칭 대신 Gemini 사용)", key="use_ai_sentiment",
            )
            if not use_ai_sentiment:
                sentiment_df = classify_sentiment(posts_df)
                sentiment_method_caption = "키워드 기반 단순 분류"
            elif not os.environ.get("GEMINI_API_KEY"):
                st.info("AI 분류를 사용하려면 GEMINI_API_KEY 환경변수를 설정해주세요. 키워드 분류로 표시합니다.")
                sentiment_df = classify_sentiment(posts_df)
                sentiment_method_caption = "키워드 기반 단순 분류"
            else:
                ai_labels = classify_sentiment_ai(tuple(posts_df["제목"]))
                sentiment_df = posts_df[["날짜", "제목"]].copy()
                sentiment_df["심리"] = ai_labels
                sentiment_method_caption = "Gemini AI 분류"

            st.caption(f"현재 분류 방식: {sentiment_method_caption}")

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
                    color_discrete_map={"긍정": _UP_COLOR, "부정": _DOWN_COLOR, "중립": "#7f7f7f"},
                )
                _style_chart_mobile(fig_sentiment)
                st.plotly_chart(fig_sentiment, width="stretch", key="chart_sentiment", config=PLOTLY_CONFIG)
            else:
                st.caption(f"조회된 게시글이 전부 {sentiment_df['날짜'].iloc[0]} 하루에 몰려 있어 일자별 비교는 아직 어렵습니다 (조회 개수를 늘려보세요).")

            with st.expander(f"게시글 {total}건 상세 보기 (날짜별)"):
                sentiment_table = sentiment_df.sort_values("날짜", ascending=False)
                st.table(
                    sentiment_table.style.map(_sentiment_text_color, subset=["심리"]),
                    width="stretch", hide_index=True,
                )

            community_summary = (
                f"최근 게시글 {total}건({num_days}일치) 중 긍정 {pos_n}건({pos_n / total:.0%}), "
                f"부정 {neg_n}건({neg_n / total:.0%}), 중립 {neu_n}건({neu_n / total:.0%}). "
                f"({sentiment_method_caption}이며 여론 참고용)"
            )
    except Exception as e:
        st.error(f"커뮤니티 심리 분석에 실패했습니다: {e}")

    st.divider()

    _subheader_with_help(
        "디시인사이드 주식갤러리 (krstock)",
        "특정 종목 전용 갤러리가 아니라 국내 주식 전반을 다루는 갤러리라, 거래량·관심도가 낮은 종목은 "
        "검색 결과가 적거나 없을 수 있습니다. 제목·본문에 종목명이 포함된 게시글만 모았습니다.\n\n"
        "워드클라우드는 제목에 자주 등장한 단어를 크기로 나타낸 것이고, 초록색은 호재, 빨간색은 악재 키워드가 "
        "포함된 단어입니다. 익명 게시판의 여론일 뿐 사실이 아니며, 매매 신호로 쓰지 마세요.",
        key="dcinside",
    )
    try:
        dc_posts_df = fetch_dc_gallery_posts(STOCK_NAME, community_post_count)
        if dc_posts_df.empty:
            st.warning(f"'{STOCK_NAME}' 관련 게시글을 찾지 못했습니다.")
        else:
            st.metric("검색된 게시글", f"{len(dc_posts_df)}건")

            word_freq = extract_korean_word_freq(dc_posts_df["제목"].tolist())
            wc_image = render_wordcloud_image(word_freq)
            if wc_image is not None:
                st.markdown("**워드클라우드 (게시글 제목 기반)**")
                st.caption("단어가 클수록 자주 언급된 것이고, 초록색은 호재 키워드, 빨간색은 악재 키워드가 포함된 단어입니다 (키워드 기반 단순 분류).")
                st.image(wc_image, width="stretch")
            else:
                st.caption("한글 폰트를 찾지 못해 워드클라우드를 표시할 수 없습니다 (서버에 한글 폰트 설치가 필요합니다).")

            with st.expander(f"게시글 {len(dc_posts_df)}건 목록 보기"):
                dc_sentiment = classify_sentiment(dc_posts_df)["심리"].values
                dc_display_df = dc_posts_df[["날짜", "제목", "조회수", "추천", "url"]].copy()
                dc_display_df["심리"] = dc_sentiment
                st.caption("심리는 제목의 키워드로 분류한 것입니다 (키워드 기반 단순 분류).")
                st.dataframe(
                    dc_display_df[["날짜", "제목", "심리", "조회수", "추천", "url"]]
                    .sort_values("날짜", ascending=False)
                    .style.map(_sentiment_text_color, subset=["심리"]),
                    width="stretch", hide_index=True,
                    column_config={"url": st.column_config.LinkColumn("링크")},
                )

            st.markdown("**AI로 우수 분석글 찾기**")
            st.caption(
                "최근 게시글 중 최대 20개의 본문을 가져와 AI가 잡담·비방을 걸러내고 근거 있는 분석글만 추려줍니다. "
                "시간이 다소 걸릴 수 있습니다."
            )
            if not os.environ.get("GEMINI_API_KEY"):
                st.info("이 기능을 사용하려면 GEMINI_API_KEY 환경변수를 설정해주세요.")
            else:
                if st.button("우수 분석글 추리기", key="dc_curate_button"):
                    with st.spinner("게시글 본문을 확인하고 분석글을 추리는 중..."):
                        try:
                            raw_text, picks = curate_good_dc_posts(dc_posts_df, STOCK_NAME)
                            st.session_state["dc_curation_raw"] = raw_text
                            st.session_state["dc_curation_picks"] = picks
                        except Exception as e:
                            st.error(f"AI 분석글 추리기에 실패했습니다: {e}")

                if "dc_curation_picks" in st.session_state:
                    picks = st.session_state["dc_curation_picks"]
                    if picks:
                        for i, pick in enumerate(picks, 1):
                            st.markdown(_md_safe(f"{i}. [{pick['제목']}]({pick['url']}) — {pick['이유']}"))
                    else:
                        st.info(st.session_state.get("dc_curation_raw", "").strip() or "조건에 맞는 분석글을 찾지 못했습니다.")
    except Exception as e:
        st.error(f"디시인사이드 주식갤러리 조회에 실패했습니다: {e}")

# 탭 하나를 프래그먼트로 둔다. 안에 있는 위젯(표시 품목·기간 슬라이더 등)을 건드리면
# 이 탭만 다시 그린다. 예전에는 전부 모듈 수준이라 위젯 하나에 스크립트 전체가 다시
# 돌았고, 보이는 탭 8개가 통째로 재렌더됐다(실측: DRAM 표시 품목 전환에 13.7초).
# 안쪽에 또 프래그먼트를 두면 안 된다 — 중첩은 Streamlit이 막는다.
@st.fragment
def _render_tab_ai():
    _subheader_with_help(
        "AI 분석: 오늘의 주가 변동 요인",
        "공시·뉴스·리포트·DRAM·매크로·대시보드 지표에 장중 흐름과 시간외까지 넘겨서, "
        "강세/약세 근거를 갈래별로 정리하게 합니다. 커뮤니티 여론도 따로 짚습니다.\n\n"
        "버튼을 눌러야 실행됩니다. AI가 지어낼 수 있으니 참고용으로만 보세요.",
        key="ai",
    )

    # AI를 돌리지 않아도 판단 재료는 바로 보이게 한다 (컨센서스·밸류에이션은 다른 탭에 없던 정보다)
    snapshot = None
    try:
        snapshot = fetch_stock_snapshot(TICKER)
    except Exception:
        snapshot = None

    if snapshot:
        # 목표주가 추이는 아무도 안 주므로 직접 쌓는다. 같은 날 두 번은 안 쓴다.
        record_consensus_snapshot(TICKER, snapshot)
        target = _to_number(snapshot.get("목표주가"))
        cur = _to_number(st.session_state.get("current_price_value"))
        # price_row_ 접두어를 붙여야 좁은 화면에서 2열로 접힌다.
        # 안 붙이면 지표 넷이 세로로 쌓여 모바일에서 화면 한 판을 다 먹는다.
        snapshot_row = st.container(key="price_row_ai_snapshot")
        col1, col2, col3, col4 = snapshot_row.columns(4)
        with col1.container(key="metric_small_ai_target"):
            upside = f"{(target / cur - 1) * 100:+.0f}% 여력" if (target and cur) else None
            # 애널리스트 탭에는 우리가 직접 센 컨센서스가 따로 있다. 모집단이 달라서
            # 값이 조금 다른데(FnGuide는 네이버 미게재 증권사까지 포함), 두 탭에 같은
            # 이름의 다른 숫자가 떠 있으면 어느 쪽이 맞는지 알 수 없다. 출처를 이름에 박고
            # 도움말에서 다른 쪽 값을 함께 보여준다.
            try:
                _own = analyst_targets.consensus()
            except Exception:
                _own = {}
            _cmp = ""
            if _own and target:
                _gap = _own["평균"] / target - 1
                _cmp = (f"\n\n애널리스트 탭의 **직접 집계**는 {_own['평균']:,}원입니다"
                        f"(네이버 게재 {_own['기관수']}곳, 최신 {_own['최신일']}). "
                        f"이 값과 {_gap * 100:+.1f}% 차이인데, 둘 중 하나가 틀린 게 아니라 "
                        "모집단이 다릅니다 — FnGuide는 네이버에 리포트를 싣지 않는 증권사까지 "
                        "포함하고, 직접 집계는 증권사별 내역을 눈으로 확인할 수 있습니다.")
            _metric_with_help(
                "컨센서스 목표주가 (FnGuide)", f"{target:,.0f}원" if target else "N/A",
                f"증권사 평균 목표주가 (기준일 {snapshot.get('컨센서스일자') or '-'}). "
                "네이버·FnGuide가 제공하는 값으로 주 1회 갱신됩니다. 기대치일 뿐 보장이 아닙니다."
                + _cmp,
                key="ai_target", delta=upside, delta_color="normal",
            )
        with col2.container(key="metric_small_ai_recomm"):
            _metric_with_help(
                "투자의견 평균", snapshot.get("투자의견") or "N/A",
                "5점 만점에 가까울수록 매수 의견이 우세하다는 뜻입니다.",
                key="ai_recomm",
            )
        with col3.container(key="metric_small_ai_per"):
            _metric_with_help(
                "PER", snapshot.get("PER") or "N/A",
                f"EPS {snapshot.get('EPS') or '-'}", key="ai_per",
            )
        with col4.container(key="metric_small_ai_52w"):
            # 고가와 저가를 "1,758,000 / 890,000"처럼 한 줄에 붙이면 좁은 화면에서 잘린다.
            # 고가를 값으로, 저가를 아래 줄로 내려 두 줄에 나눠 담는다.
            st.metric(
                "52주 최고", f"{snapshot.get('52주최고') or '-'}",
                delta=f"최저 {snapshot.get('52주최저') or '-'}", delta_color="off",
            )

        peers = snapshot.get("동일업종") or []
        if peers:
            peer_txt = " · ".join(f"{p['종목']} {p['등락률']}%" for p in peers[:6] if p.get("등락률"))
            st.caption(f"같은 업종 오늘 등락률 — {peer_txt}")

    if not os.environ.get("GEMINI_API_KEY"):
        st.info("AI 분석을 사용하려면 GEMINI_API_KEY 환경변수를 설정해주세요.")
    else:
        # 토글은 지표가 아니라 겹쳐 띄울 자리가 없다. 제목 옆 물음표와 같은 방식으로
        # 토글 바로 오른쪽에 팝오버 버튼을 붙인다 (help_row_ CSS가 한 줄로 붙여준다).
        with st.container(key="help_row_ai_search"):
            _toggle_col, _toggle_help = st.columns([0.9, 0.1], vertical_alignment="center")
            use_search = _toggle_col.toggle(
                "업종·매크로 뉴스까지 넓게 수집",
                key="ai_use_search", value=True,
            )
            with _toggle_help.popover("", icon=":material/help:"):
                st.markdown(
                    "종목명으로만 뉴스를 모으면 '올랐다/내렸다'는 시황 기사만 쌓여서 원인을 못 짚습니다.\n\n"
                    "이 옵션을 켜면 HBM·메모리 업황·D램 가격·엔비디아·반도체 수출로도 각각 검색해서, "
                    "주가가 왜 움직였는지에 해당하는 재료를 같이 넘깁니다. 수집에 몇 초 더 걸립니다.\n\n"
                    "'투자의견' 검색도 함께 돕니다. 아래 리포트 목록은 네이버가 싣는 13개 증권사뿐이라, "
                    "그 밖의 증권사가 목표주가를 바꾼 건은 이 기사로만 잡힙니다."
                )
        # 꺼진 탭의 항목은 계산 자체가 안 돌아서 AI가 못 본다. 조용히 빠지면
        # "왜 이건 분석에 안 나오지?" 하게 되므로 미리 알려주고 켜는 법도 적어 둔다.
        _missing = missing_tab_summaries()
        if _missing:
            st.caption(
                f"참고 — {' · '.join(_missing)}은(는) 해당 탭이 꺼져 있어 이번 분석에서 빠집니다. "
                "왼쪽 사이드바 '표시할 탭 선택'에서 켜면 다음 분석부터 반영됩니다."
            )

        _render_saved_ai_analysis()




def _render_saved_ai_analysis() -> None:
    """저장된 분석을 읽어 보여준다. 여기서 만들지 않는다.

    만드는 건 자동 새로고침 시각과 '지표 새로고침' 버튼이 맡는다. 화면에서 만들면
    탭을 열 때마다 30~100초를 기다려야 한다(예전에 그랬다).
    """
    saved = ai_analysis.load(TICKER)
    if not saved.get("text"):
        st.info("아직 만들어 둔 분석이 없습니다. 위 **🔄 지표 새로고침**을 누르거나, "
                "예약된 자동 새로고침 시각이 되면 만들어집니다.")
        return

    age = ai_analysis.age_note(saved)
    st.caption(f"기준 시각: {saved.get('time')} ({age})" if age else f"기준 시각: {saved.get('time')}")
    # 장중에는 몇십 분만 지나도 이 안의 주가·수급 숫자가 지금과 다르다. 눈에 띄게 알린다.
    _mins = 0
    try:
        _made = dt.datetime.strptime(str(saved.get("time")), "%Y-%m-%d %H:%M")
        _mins = int((dt.datetime.now() - _made).total_seconds() // 60)
    except Exception:
        pass
    if _mins >= 60:
        st.warning(f"이 분석은 {age} 것입니다. 안에 적힌 현재가·수급 숫자는 그 시점 기준이라 "
                   "지금과 다를 수 있습니다. 최신으로 보려면 위 **🔄 지표 새로고침**을 누르세요.")

    note = saved.get("search_note")
    if note == "search_ok":
        st.success("구글 검색으로 최신 정보를 보강해 분석했습니다.")
    elif note:
        st.warning(note)
    st.markdown(_md_safe(saved["text"]))

    with st.expander("분석에 사용된 원본 데이터 보기"):
        for label, key in ai_analysis.SOURCE_FIELDS:
            value = saved.get(key)
            if value:
                st.write(f"**{label}**")
                st.text(value)
        st.write("**뉴스 헤드라인**")
        _heads = saved.get("headlines") or []
        st.markdown(_md_safe("\n".join(f"- {h}" for h in _heads)) if _heads
                    else "수집된 헤드라인 없음")


_TAB_RENDERERS = {
    "매매 신호": _render_tab_signal,
    "수급 현황": _render_tab_supply,
    "가격 과열도": _render_tab_overheat,
    "선물 경보": _render_tab_futures,
    "통합 신호": _render_tab_composite,
    "하락 조기신호": _render_tab_decline,
    "상승 조기신호": _render_tab_rally,
    "DRAM 시세": _render_tab_dram,
    "빅테크 Capex": _render_tab_capex,
    "재무 데이터": _render_tab_financials,
    "공시": _render_tab_disclosure,
    "애널리스트": _render_tab_analyst,
    "커뮤니티": _render_tab_community,
    "AI 분석": _render_tab_ai,
}

# 사이드바에서 숨기지 않은(선택된) 탭만 실제로 렌더링한다. 숨겨진 탭은 함수 자체가
# 호출되지 않으므로 데이터 조회도 일어나지 않는다.
#
# 지연 렌더: `st.tabs`는 보이는 탭 8개를 한 번에 다 그리고, 그 루프가 동기라서
# 처음 열 때 8개 데이터 조회가 순서대로 쌓여 20초가 걸렸다. 사용자는 한 번에 한
# 탭만 본다. 그래서 **새 세션의 첫 렌더에서는 처음 열려 있는 탭(=첫 탭)만** 그리고,
# 곧바로 st.rerun()을 한 번 걸어 나머지를 채운다. 그 사이에도 첫 탭은 이미 보이고
# 눌린다 — 첫 탭까지 걸리는 시간이 ~20초에서 ~6초로 줄고, 나머지는 읽는 동안 채워진다.
# (탭 전환은 그대로 즉시다. `st.tabs`의 CSS 토글이라 서버로 안 간다.)
_TABS_WARMED_KEY = "_all_tabs_rendered"
_tabs_warmed = st.session_state.get(_TABS_WARMED_KEY, False)

for _i, _label in enumerate(_visible_tab_labels):
    _tab_t0 = time.monotonic()
    with _tab_map[_label]:
        if _tabs_warmed or _i == 0:
            _TAB_RENDERERS[_label]()
        else:
            st.caption("불러오는 중…")
    if _BOOT_PROFILE:
        print(f"[boot]   탭 '{_label}' {time.monotonic() - _tab_t0:6.2f}s"
              + ("" if (_tabs_warmed or _i == 0) else " (지연)"), flush=True)

_boot_lap("탭 렌더 루프 완료")

if not _tabs_warmed:
    st.session_state[_TABS_WARMED_KEY] = True
    st.rerun()


# AI 분석 생성은 **여기서** 한다. 탭을 다 그린 뒤라야 과열도·DRAM 요약(그 탭이 그려질 때
# 전역에 채워지는 값)이 들어간다. 예약 시각이 지났거나 '지표 새로고침'을 누르면
# _mark_ai_analysis_due()가 표시를 남기고, 그 표시를 여기서 받아 만든다.
if st.session_state.pop("_ai_refresh_pending", False):
    if os.environ.get("GEMINI_API_KEY"):
        with st.spinner("AI 분석을 만드는 중... 30~100초 걸립니다 (다 만들면 화면에 남습니다)"):
            _made = ai_report.build_and_save(
                TICKER, STOCK_NAME,
                use_search=st.session_state.get("ai_use_search", True))
        if _made:
            st.rerun()          # 새로 만든 분석을 화면에 반영한다

