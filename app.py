import datetime as dt
import os
import re
import threading
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
import over_market as om
from bs4 import BeautifulSoup
from google import genai
from plotly.subplots import make_subplots
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from wordcloud import WordCloud


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

DEFAULT_TICKER = "000660"
REFRESH_SEC = 5
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
    "커뮤니티",
    "AI 분석",
]

DC_GALLERY_ID = "krstock"
DC_GALLERY_LIST_URL = "https://gall.dcinside.com/mgallery/board/lists/"
DC_GALLERY_VIEW_URL = "https://gall.dcinside.com/mgallery/board/view/"
DC_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Referer": f"https://gall.dcinside.com/mgallery/board/lists/?id={DC_GALLERY_ID}",
}
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

POSITIVE_KEYWORDS = [
    "상승", "오른다", "올랐", "급등", "떡상", "가즈아", "가보자", "존버", "매수",
    "저점매수", "반등", "호재", "강세", "상한가", "신고가", "돌파", "추매", "줍줍", "익절",
]
NEGATIVE_KEYWORDS = [
    "하락", "내린다", "내렸", "급락", "떡락", "손절", "물렸", "물림", "개미지옥", "지옥",
    "악재", "약세", "하한가", "신저가", "붕괴", "패닉", "팔아", "매도", "손실", "탈출",
    "폭락", "마이너스", "마이나스", "개미눈물",
]
# fetch_investor_netbuy가 돌려주는 순매수 열. 거래량·종가가 같은 표에 붙어 있어서,
# '투자자별'만 훑어야 하는 곳(그래프·누적추세·AI 요약)이 열 목록을 여기서 가져다 쓴다.
INVESTOR_COLUMNS = ("개인", "외국인", "기관")
DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_COMMUNITY_POST_COUNT = 60
# 가격 과열도 기본 파라미터. 이동평균 7종 x 예측기간 5종 x 임계폭 5종(168개 유효 조합)을 SK하이닉스
# 700거래일로 백테스트해, 괴리율 밴드가 커질수록 하락확률이 오르는 단조성(스피어만 ρ)이 가장 뚜렷하고
# 학습/검증 분할을 50–80%로 옮겨도 그 관계가 유지된 조합을 골랐다 (검증구간 평균 ρ 0.994).
# 이전 기본값(60일·10거래일·7%)은 검증구간 평균 ρ가 0.484로, 예측기간이 짧을수록 관계가 급격히 흐려진다.
OVERHEAT_DEFAULT_MA_WINDOW = 80
OVERHEAT_DEFAULT_HORIZON = 15
OVERHEAT_DEFAULT_THRESHOLD = 0.10
DRAM_HISTORY_FILE = os.environ.get("DRAM_HISTORY_FILE", "data/dram_spot_history.csv")
# 프리장/애프터장 시세는 over_market.py가 담당한다. 수집은 collector.py(별도 컨테이너)가 상시로 하고,
# 화면은 쌓인 기록을 읽어 그래프에 이어붙이기만 한다.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
# 무료 요금제는 모델마다 하루 요청 수가 따로 잡힌다(예: gemini-3.7-flash 는 20회).
# 기본 모델이 소진되면 아래 순서로 갈아타서 분석이 아예 안 되는 상황을 피한다.
GEMINI_FALLBACK_MODELS = [
    m.strip() for m in os.environ.get(
        "GEMINI_FALLBACK_MODELS", "gemini-3.6-flash,gemini-flash-lite-latest,gemini-3.5-flash-lite"
    ).split(",") if m.strip()
]


def _is_quota_error(exc: Exception) -> bool:
    return "RateLimit" in type(exc).__name__ or "429" in str(exc)


def _call_gemini(prompt: str, tools: list | None = None) -> tuple[str, str]:
    """모델 하나가 한도에 걸리면 다음 모델로 넘어가며 호출한다.

    반환: (응답 텍스트, 실제로 사용된 모델명)
    한도 외의 오류(잘못된 요청 등)는 모델을 바꿔도 소용없으므로 바로 올린다.
    """
    client = genai.Client()
    tried: list[str] = []
    last_exc: Exception | None = None
    for model in [GEMINI_MODEL] + [m for m in GEMINI_FALLBACK_MODELS if m != GEMINI_MODEL]:
        tried.append(model)
        try:
            kwargs = {"tools": tools} if tools else {}
            interaction = client.interactions.create(model=model, input=prompt, **kwargs)
            return (interaction.output_text or "", model)
        except Exception as exc:
            last_exc = exc
            if not _is_quota_error(exc):
                raise
    raise RuntimeError(
        f"사용 가능한 모델을 찾지 못했습니다. 시도한 모델: {', '.join(tried)}. "
        f"마지막 오류: {type(last_exc).__name__}"
    ) from last_exc

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
# 국내 개인 온라인 계좌 기준 비용. 매도 시 증권거래세/농특세 0.15%가 추가된다.
FLOW_COST_BUY = 0.00015 + 0.0010          # 위탁수수료 + 슬리피지
FLOW_COST_SELL = 0.00015 + 0.0015 + 0.0010  # 위탁수수료 + 거래세 + 슬리피지
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
        st.metric(label, value, **metric_kwargs)
        with st.popover("", icon=":material/help:"):
            st.markdown(help_text)


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
    /* 지표(st.metric)에 붙는 물음표. 라벨 길이가 제각각이라 라벨 옆이 아니라
       지표 칸 오른쪽 위에 띄워 고정한다. 클릭해야 열리는 팝오버라 hover 툴팁과 동작이 같다. */
    div[class*="st-key-metric_help_"] {
        position: relative !important;
    }
    /* 팝오버 컨테이너와 버튼은 기본값이 칸 전체 폭이라, 그대로 두면 물음표 버튼이
       라벨 글자 위를 통째로 덮어 라벨 아무 데나 눌러도 열린다. 아이콘 크기로 좁힌다. */
    div[class*="st-key-metric_help_"] div[data-testid="stPopover"] {
        position: absolute !important;
        top: 0 !important;
        right: 0 !important;
        width: auto !important;
        min-width: 0 !important;
        z-index: 1;
    }
    div[class*="st-key-metric_help_"] div[data-testid="stPopover"] > div,
    div[class*="st-key-metric_help_"] div[data-testid="stPopover"] button {
        width: auto !important;
        min-width: 0 !important;
    }
    div[class*="st-key-metric_help_"] div[data-testid="stPopover"] button {
        background: none !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
        min-height: 0 !important;
        height: auto !important;
        opacity: 0.45;
    }
    div[class*="st-key-metric_help_"] div[data-testid="stPopover"] button:hover {
        opacity: 1;
    }
    div[class*="st-key-metric_help_"] div[data-testid="stPopover"] button div[aria-hidden="true"] {
        display: none !important;
    }
    div[class*="st-key-metric_help_"] div[data-testid="stPopover"] button [data-testid="stIconMaterial"] {
        font-size: 0.95rem !important;
        width: 0.95rem !important;
        height: 0.95rem !important;
        margin: 0 !important;
    }
    div[class*="st-key-metric_help_"] div[data-testid="stPopover"] button > div {
        gap: 0 !important;
    }
    /* 물음표가 값 위에 겹치지 않도록 라벨 오른쪽에 자리를 비워둔다 */
    div[class*="st-key-metric_help_"] [data-testid="stMetricLabel"] {
        padding-right: 1.2rem !important;
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
        /* 라벨이 길어(예: "80일선 괴리율 (현재가 기준)") 두 줄로 넘어가도 값이 밀리지 않게 */
        div[class*="st-key-metric_small_"] [data-testid="stMetricLabel"] {
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


@st.cache_data(ttl=5, show_spinner="불러오는 중...")
def fetch_current_price(ticker: str) -> dict:
    return om.fetch_current_price_raw(ticker)


@st.cache_data(ttl=30, show_spinner="불러오는 중...")
def fetch_intraday_price(ticker: str) -> pd.DataFrame:
    """네이버 모바일 API에서 정규장 분봉을 가져온다.

    주의: 프리장(08:00–09:00)에는 당일 분봉이 아직 없어서 빈 배열 []이 내려온다. 이때
    df["localDateTime"]을 그대로 건드리면 KeyError가 나고, 호출부의 except에 먹혀서
    장중 그래프가 통째로 사라진다. 그래서 빈 응답은 여기서 타입 맞춘 빈 표로 돌려준다.
    """
    url = f"https://m.stock.naver.com/api/chart/domestic/item/{ticker}/minute"
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.naver.com/"}
    empty = pd.DataFrame({"시각": pd.Series(dtype="datetime64[ns]"), "현재가": pd.Series(dtype="float64")})
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    payload = resp.json()
    if not payload:
        return empty
    df = pd.DataFrame(payload)
    if "localDateTime" not in df.columns or "currentPrice" not in df.columns:
        return empty
    df["시각"] = pd.to_datetime(df["localDateTime"], format="%Y%m%d%H%M%S")
    df["현재가"] = df["currentPrice"].astype(float)
    return df[["시각", "현재가"]].sort_values("시각").reset_index(drop=True)


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


@st.cache_data(ttl=3600, show_spinner="불러오는 중...")
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
    # 순매수 세 열 뒤에 거래량·종가를 덧붙인다. 순매수는 '누가 샀나'만 알려줄 뿐,
    # 그 날 거래가 얼마나 활발했는지는 알 수 없어서 절대 거래량을 같이 본다.
    # 순매수 열만 골라 쓰는 곳이 있으므로 순서를 지켜 INVESTOR_COLUMNS를 앞에 둔다.
    df = df.set_index("날짜")[list(INVESTOR_COLUMNS) + ["거래량", "종가"]]
    return df


@st.cache_data(ttl=600, show_spinner=False)
def fetch_daily_ohlcv(ticker: str, days: int) -> pd.DataFrame:
    """일별 종가·거래량. 투자자 수급(frgn.naver)과 달리 마감 직후 바로 확정된다.

    같은 네이버인데도 공개 시점이 다르다. 15:51에 재보니 일봉 경로는 이미 당일치가
    올라와 있는 반면(종가 1,730,000 / 거래량 4,247,406), 수급 페이지는 아직 전 거래일까지였다.
    거래량 그래프까지 수급과 같은 소스에 묶어두면 볼 수 있는 값을 몇 시간씩 늦게 보게 된다.
    """
    end = dt.datetime.now(om.KST)
    start = end - dt.timedelta(days=days + 10)     # 휴장일을 감안해 여유를 둔다
    try:
        r = requests.get(f"https://api.stock.naver.com/chart/domestic/item/{ticker}/day",
                         params={"startDateTime": start.strftime("%Y%m%d0000"),
                                 "endDateTime": end.strftime("%Y%m%d2359")},
                         headers={"User-Agent": "Mozilla/5.0", "Referer": "https://m.stock.naver.com/"},
                         timeout=10)
        r.raise_for_status()
        rows = r.json()
    except Exception:
        return pd.DataFrame(columns=["거래량", "종가"])
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame(columns=["거래량", "종가"])
    df = pd.DataFrame([{
        "날짜": pd.to_datetime(str(x.get("localDate")), format="%Y%m%d"),
        "종가": float(x.get("closePrice") or 0),
        "거래량": float(x.get("accumulatedTradingVolume") or 0),
    } for x in rows if x.get("localDate")])
    cutoff = pd.Timestamp(dt.datetime.now(om.KST).date() - dt.timedelta(days=days))
    df = df[df["날짜"] >= cutoff].sort_values("날짜")
    return df.set_index("날짜")[["거래량", "종가"]]


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


@st.cache_data(ttl=24 * 3600, show_spinner="불러오는 중...")
def fetch_backtest_history(ticker: str, target_days: int = 500) -> pd.DataFrame:
    # 페이지끼리 의존이 없으므로(page=N은 그냥 N번째 묶음) 한 장씩 순서대로 기다릴 이유가 없다.
    # 700일치면 35페이지쯤인데, 순차로 받으면 왕복 지연만 5초가 넘는다.
    # 1페이지로 '한 장에 몇 줄인지'만 확인한 뒤 나머지를 한꺼번에 받는다.
    first = _fetch_frgn_page(ticker, 1)
    if first.empty:
        return pd.DataFrame(columns=["날짜", "종가", "기관", "외국인", "개인"])

    frames = [first]
    per_page = max(len(first), 1)
    max_pages = 60
    # 휴장일·중복으로 한두 장 모자랄 수 있어 여유분을 둔다
    need_pages = min(max_pages, -(-target_days // per_page) + 1)
    if need_pages > 1:
        def _safe_page(p: int) -> pd.DataFrame:
            # 상장 기간이 짧은 종목은 요청한 페이지가 아예 없을 수 있다. 순차 루프일 때는
            # 빈 페이지에서 멈추면 그만이었지만, 한꺼번에 받는 지금은 한 장이 실패해도
            # 나머지는 살려야 한다.
            try:
                return _fetch_frgn_page(ticker, p)
            except Exception:
                return pd.DataFrame()

        with ThreadPoolExecutor(max_workers=8) as pool:
            for page_df in pool.map(_safe_page, range(2, need_pages + 1)):
                if not page_df.empty:
                    frames.append(page_df)

    if not frames:
        return pd.DataFrame(columns=["날짜", "종가", "기관", "외국인", "개인"])

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="날짜").sort_values("날짜")
    df["개인"] = -(df["기관"] + df["외국인"])
    # 여유분으로 한 장 더 받으므로 순차 시절보다 며칠 더 딸려온다. 백테스트 구간이 조회
    # 시점에 따라 들쭉날쭉해지지 않게 최근 target_days개로 잘라 맞춘다.
    return df.tail(target_days).reset_index(drop=True)


@st.cache_data(ttl=60, show_spinner="불러오는 중...")
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


def forward_max_gain(price: pd.Series, horizon: int) -> pd.Series:
    """t 시점 가격 대비, 이후 horizon일 내 최고가까지의 상승률(최대 상승률, 양수)."""
    values = price.to_numpy()
    n = len(values)
    result = np.full(n, np.nan)
    for i in range(n - horizon):
        window = values[i + 1 : i + 1 + horizon]
        result[i] = window.max() / values[i] - 1
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
    quantile: float = 0.2, drawdown_threshold: float = 0.05, gain_threshold: float | None = None,
    side: str = "high",
) -> dict:
    """종가가 ma_window일 이동평균 대비 상위(side="high", 과열) 또는 하위(side="low", 침체) quantile
    구간일 때, 향후 horizon일 내 drawdown_threshold 이상 하락할 확률과 gain_threshold 이상 상승할
    확률이 나머지 구간과 어떻게 다른지 함께 검증한다. gain_threshold를 안 주면 drawdown_threshold와
    같은 크기를 쓴다."""
    gain_threshold = drawdown_threshold if gain_threshold is None else gain_threshold
    d = df.copy()
    d["ma"] = d[price_col].rolling(ma_window).mean()
    d["deviation"] = d[price_col] / d["ma"] - 1
    d["drawdown"] = forward_max_drawdown(d[price_col], horizon)
    d["gain"] = forward_max_gain(d[price_col], horizon)
    valid = d.dropna(subset=["deviation", "drawdown", "gain"])

    result = {
        "ma_window": ma_window, "horizon": horizon, "quantile": quantile, "side": side,
        "drawdown_threshold": drawdown_threshold, "gain_threshold": gain_threshold, "n": len(valid),
        "hi_n": 0, "rest_n": 0, "hi_rate": None, "rest_rate": None, "base_rate": None, "p_value": None,
        "hi_up_rate": None, "rest_up_rate": None, "base_up_rate": None, "up_p_value": None,
        "hi_cutoff": None, "current_deviation": None, "current_regime": None,
    }
    # 현재 상태는 향후 수익률 계산 없이 전체 이력에서 바로 판단한다 (최근 horizon일은 drawdown/gain이 아직 계산 안 돼 valid에서 빠짐).
    deviation_all = d["deviation"].dropna()
    if len(deviation_all) > 0:
        result["current_deviation"] = float(deviation_all.iloc[-1])

    if len(valid) < 30:
        return result

    valid = valid.assign(
        downtrend=(valid["drawdown"] <= -drawdown_threshold).astype(float),
        uptrend=(valid["gain"] >= gain_threshold).astype(float),
    )
    if side == "high":
        hi_cutoff = valid["deviation"].quantile(1 - quantile)
        hi_group = valid[valid["deviation"] >= hi_cutoff]
        rest_group = valid[valid["deviation"] < hi_cutoff]
        is_current_in_group = result["current_deviation"] is not None and result["current_deviation"] >= hi_cutoff
        regime_label = f"과열 (상위 {quantile:.0%})"
    else:
        hi_cutoff = valid["deviation"].quantile(quantile)
        hi_group = valid[valid["deviation"] <= hi_cutoff]
        rest_group = valid[valid["deviation"] > hi_cutoff]
        is_current_in_group = result["current_deviation"] is not None and result["current_deviation"] <= hi_cutoff
        regime_label = f"침체 (하위 {quantile:.0%})"

    result["hi_n"] = len(hi_group)
    result["rest_n"] = len(rest_group)
    result["hi_rate"] = float(hi_group["downtrend"].mean()) if len(hi_group) else None
    result["rest_rate"] = float(rest_group["downtrend"].mean()) if len(rest_group) else None
    result["base_rate"] = float(valid["downtrend"].mean())
    result["hi_up_rate"] = float(hi_group["uptrend"].mean()) if len(hi_group) else None
    result["rest_up_rate"] = float(rest_group["uptrend"].mean()) if len(rest_group) else None
    result["base_up_rate"] = float(valid["uptrend"].mean())
    result["hi_cutoff"] = float(hi_cutoff)

    if result["hi_rate"] is not None and result["rest_rate"] is not None:
        result["p_value"] = two_proportion_ztest(
            hi_group["downtrend"].sum(), len(hi_group), rest_group["downtrend"].sum(), len(rest_group)
        )
    if result["hi_up_rate"] is not None and result["rest_up_rate"] is not None:
        result["up_p_value"] = two_proportion_ztest(
            hi_group["uptrend"].sum(), len(hi_group), rest_group["uptrend"].sum(), len(rest_group)
        )

    result["current_regime"] = regime_label if is_current_in_group else "평상시"
    return result


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


def run_boolean_pattern_backtest(
    price: pd.Series, dates: pd.Series, pattern: pd.Series, horizon: int,
    drawdown_threshold: float = 0.07, gain_threshold: float | None = None,
) -> dict:
    """불리언 조건(예: 외국인 순매도 + 거래량 급증)이 참일 때와 거짓일 때, 향후 horizon일 내
    drawdown_threshold 이상 하락할 확률과 gain_threshold 이상 상승할 확률이 어떻게 다른지 검증한다."""
    gain_threshold = drawdown_threshold if gain_threshold is None else gain_threshold
    d = pd.DataFrame({"날짜": dates, "종가": price, "패턴": pattern})
    d["drawdown"] = forward_max_drawdown(d["종가"], horizon)
    d["gain"] = forward_max_gain(d["종가"], horizon)
    valid = d.dropna(subset=["패턴", "drawdown", "gain"])

    result = {
        "n": len(valid), "match_n": 0, "rest_n": 0,
        "match_down_rate": None, "rest_down_rate": None, "base_down_rate": None, "down_p_value": None,
        "match_up_rate": None, "rest_up_rate": None, "base_up_rate": None, "up_p_value": None,
        "current_match": None,
    }
    # 현재 상태는 향후 수익률 계산 없이 전체 이력에서 바로 판단한다 (최근 horizon일은
    # drawdown/gain이 아직 계산 안 돼 valid에서 빠지므로, valid 기준으로 뽑으면 horizon일 지연된 값이 된다).
    pattern_all = d["패턴"].dropna()
    if len(pattern_all) > 0:
        result["current_match"] = bool(pattern_all.iloc[-1])

    if len(valid) < 30:
        return result

    valid = valid.assign(
        down=(valid["drawdown"] <= -drawdown_threshold).astype(float),
        up=(valid["gain"] >= gain_threshold).astype(float),
    )
    match = valid[valid["패턴"]]
    rest = valid[~valid["패턴"]]

    result["match_n"] = len(match)
    result["rest_n"] = len(rest)
    result["match_down_rate"] = float(match["down"].mean()) if len(match) else None
    result["rest_down_rate"] = float(rest["down"].mean()) if len(rest) else None
    result["base_down_rate"] = float(valid["down"].mean())
    result["match_up_rate"] = float(match["up"].mean()) if len(match) else None
    result["rest_up_rate"] = float(rest["up"].mean()) if len(rest) else None
    result["base_up_rate"] = float(valid["up"].mean())

    if len(match) >= 2 and len(rest) >= 2:
        result["down_p_value"] = two_proportion_ztest(match["down"].sum(), len(match), rest["down"].sum(), len(rest))
        result["up_p_value"] = two_proportion_ztest(match["up"].sum(), len(match), rest["up"].sum(), len(rest))

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


# AI 분석에 넘길 거시경제 지표. YAHOO_SYMBOLS와 따로 두는 이유는, 그쪽에 넣으면
# build_composite_dataset이 통합 신호용 데이터셋에 열을 같이 병합해버리기 때문이다.
# (label, 야후 심볼, 변동을 %로 볼지 %p로 볼지)
MACRO_SYMBOLS = [
    ("필라델피아 반도체지수(SOX)", "%5ESOX", "pct"),
    ("나스닥", "%5EIXIC", "pct"),
    ("달러인덱스(DXY)", "DX-Y.NYB", "pct"),
    ("원/달러 환율", "KRW=X", "pct"),
    # 금리는 그 자체가 %라, 변동을 %가 아니라 %p로 봐야 말이 된다
    ("미국 10년물 금리", "%5ETNX", "pp"),
]


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_macro_summary() -> str:
    """반도체 업황을 좌우하는 거시 지표의 최근 움직임을 한 덩어리 텍스트로 만든다.

    이 종목은 수출·달러·미국 반도체 수요에 크게 붙어 있는데, 지금까지 AI 분석에는
    이런 매크로 정보가 아예 안 들어가서 '뉴스 요약'에 가까운 답이 나왔다.
    1일/5일/20일 변화를 같이 주면 '오늘만의 일'과 '추세'를 구분해서 쓸 수 있다.
    """
    headers = {"User-Agent": "Mozilla/5.0"}

    def one(item) -> str:
        label, symbol, kind = item
        try:
            r = requests.get(YAHOO_CHART_URL.format(symbol=symbol), headers=headers, timeout=15,
                             params={"range": "3mo", "interval": "1d"})
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            closes = pd.Series(res["indicators"]["quote"][0]["close"]).dropna().reset_index(drop=True)
            if len(closes) < 21:
                return ""
            last = float(closes.iloc[-1])
            parts = []
            for days, name in ((1, "1일"), (5, "5일"), (20, "20일")):
                past = float(closes.iloc[-1 - days])
                if kind == "pp":
                    parts.append(f"{name} {last - past:+.2f}%p")
                else:
                    parts.append(f"{name} {(last / past - 1) * 100:+.2f}%")
            fmt = f"{last:,.2f}%" if kind == "pp" else f"{last:,.2f}"
            return f"- {label}: {fmt} ({', '.join(parts)})"
        except Exception:
            return f"- {label}: (수집 실패)"

    # 지표끼리 서로 무관해서 순서대로 기다릴 이유가 없다 (5종 순차 2초 -> 병렬 0.5초 수준)
    with ThreadPoolExecutor(max_workers=len(MACRO_SYMBOLS)) as pool:
        lines = [ln for ln in pool.map(one, MACRO_SYMBOLS) if ln]
    return "\n".join(lines) if lines else "(수집 실패)"


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


@st.cache_data(ttl=1800, show_spinner="불러오는 중...")
def fetch_stock_snapshot(ticker: str) -> dict:
    """네이버 모바일 통합 API에서 AI 분석에 쓸 '판단 재료'를 모아온다.

    지금까지 AI에는 제목 목록만 넘겨서 뻔한 요약밖에 못 나왔다. 여기서 가져오는 것들:
      - 컨센서스 목표주가/투자의견 : 현재가가 증권가 기대 대비 어디인지
      - PER·EPS·52주 고저        : 밸류에이션 위치
      - 동일업종 등락률           : 오늘 움직임이 이 종목만의 일인지 업종 전체인지 가르는 핵심 근거
      - 최근 5일 투자자별 순매수   : 수급 방향과 외국인 보유율 변화
    """
    url = f"https://m.stock.naver.com/api/stock/{ticker}/integration"
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://m.stock.naver.com/"}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    js = resp.json()

    totals = {item.get("code"): item.get("value") for item in (js.get("totalInfos") or [])}
    consensus = js.get("consensusInfo") or {}
    peers = [
        {"종목": p.get("stockName"), "등락률": p.get("fluctuationsRatio"), "종가": p.get("closePrice")}
        for p in (js.get("industryCompareInfo") or [])
        if p.get("stockName")
    ]
    flows = [
        {
            "날짜": f.get("bizdate"), "종가": f.get("closePrice"),
            "개인": f.get("individualPureBuyQuant"), "외국인": f.get("foreignerPureBuyQuant"),
            "기관": f.get("organPureBuyQuant"), "외국인보유율": f.get("foreignerHoldRatio"),
        }
        for f in (js.get("dealTrendInfos") or [])
    ]
    return {
        "종목명": js.get("stockName"),
        "목표주가": consensus.get("priceTargetMean"),
        "투자의견": consensus.get("recommMean"),
        "컨센서스일자": consensus.get("createDate"),
        "PER": totals.get("per"),
        "EPS": totals.get("eps"),
        "52주최고": totals.get("highPriceOf52Weeks"),
        "52주최저": totals.get("lowPriceOf52Weeks"),
        "시가총액": totals.get("marketValue"),
        "외국인소진율": totals.get("foreignRate"),
        "동일업종": peers,
        "수급추이": flows,
    }


ADR_SYMBOL = os.environ.get("ADR_SYMBOL", "SKHY")      # 나스닥 상장 SK하이닉스
ADR_HOST_TICKER = "000660"                              # ADR 비교 대상 본주
# 공식 비율: 1 ADR = 본주 0.1주 (ADR : 본주 = 1 : 10)
ADR_SHARE_RATIO = float(os.environ.get("ADR_SHARE_RATIO", "0.1"))
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
ADR_BASELINE_DAYS = 20


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_adr_bars(days: int = 5) -> pd.DataFrame:
    """SKHY 1분봉을 여러 거래일치 받아 세션·거래일까지 붙여서 돌려준다.

    시세(fetch_adr_quote)와 그래프(fetch_adr_intraday)가 같은 응답을 쓰도록 한 곳에 모았다.
    예전에는 같은 URL을 각자 한 번씩 불러서 왕복이 두 번 났다.

    하루치(range=1d)로는 부족한 이유가 두 가지다.
      1) 애프터장이 끝나는 20:00 ET부터 다음 프리장 04:00 ET까지 야후가 빈 응답을 준다.
      2) 프리장 동안 meta.chartPreviousClose가 한 세션 뒤처진다. 창이 아직 전 거래일에
         걸려 있어서, 오늘 프리장 체결을 '이틀 전 종가'와 비교하게 된다.
         (실측: 8/19 프리장 $162.02를 8/17 종가 $171.38과 비교해 -5.46%로 표시. 실제는
          8/18 종가 $155.62 대비 +4.11%로, 부호까지 뒤집혔다.)
    그래서 여러 날을 받아 기준값을 데이터에서 직접 고른다.
    """
    empty = pd.DataFrame({"시각": pd.Series(dtype="datetime64[ns]"),
                          "가격": pd.Series(dtype="float64"),
                          "세션": pd.Series(dtype="object"),
                          "거래일": pd.Series(dtype="object")})
    try:
        r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ADR_SYMBOL}",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=15,
                         params={"range": f"{days}d", "interval": "1m", "includePrePost": "true"})
        r.raise_for_status()
        res = r.json()["chart"]["result"][0]
        stamps = res.get("timestamp") or []
        closes = res["indicators"]["quote"][0].get("close") or []
        et_tz = ZoneInfo("America/New_York")
        rows = []
        for t, c in zip(stamps, closes):
            if c is None:
                continue
            et_dt = dt.datetime.fromtimestamp(t, et_tz)
            et = et_dt.time()
            # 본주 그래프와 같은 규칙으로 칠하기 위해 세션을 나눠둔다
            if et < dt.time(9, 30):
                session = "프리장"
            elif et < dt.time(16, 0):
                session = "정규장"
            else:
                session = "애프터장"
            rows.append({
                "시각": dt.datetime.fromtimestamp(t, om.KST).replace(tzinfo=None),
                "가격": float(c), "세션": session,
                # 프리장 04:00 ~ 애프터장 20:00은 모두 같은 미국 날짜라 이 값으로 하루가 갈린다
                "거래일": et_dt.date(),
            })
        return pd.DataFrame(rows).sort_values("시각").reset_index(drop=True) if rows else empty
    except Exception:
        return empty


def _adr_baselines(bars: pd.DataFrame, last_day, last_session) -> tuple[float | None, float | None]:
    """(등락률 기준값, 전 거래일 종가)를 봉 데이터에서 직접 고른다.

    등락률 기준은 세션마다 다르다. 프리장·정규장은 '직전 거래일 정규장 종가'와 비교하고,
    애프터장은 '당일 정규장 종가'와 비교하는 게 통상 표기다. 본주(NXT) 쪽도 같은 규칙이다.
    전 거래일 종가는 하루치 그래프의 기준선용이라 세션과 무관하게 늘 직전 거래일 값이다.
    """
    reg = bars[bars["세션"] == "정규장"]
    if reg.empty:
        return None, None
    closes = reg.groupby("거래일")["가격"].last()          # 거래일별 정규장 종가
    earlier = [d for d in closes.index if d < last_day]
    prev_day_close = float(closes[max(earlier)]) if earlier else None
    if last_session == "애프터장" and last_day in closes.index:
        return float(closes[last_day]), prev_day_close
    return prev_day_close, prev_day_close


@st.cache_data(ttl=60, show_spinner=False)
def fetch_adr_quote() -> dict | None:
    """SKHY 최신 체결가를 프리장/애프터장까지 포함해서 가져온다.

    includePrePost=true 로 1분봉을 받으면 미국 정규장 밖(프리장 04:00 ET – 애프터 20:00 ET)
    체결도 들어온다. 한국 장이 열리기 전 미국 시간외 움직임을 보는 게 이 지표의 핵심이라
    마지막 유효 체결가를 그대로 쓴다.
    """
    try:
        # 봉과 환율은 서로 기다릴 이유가 없다. 이 함수는 5초짜리 화면 조각 안에서 불리므로
        # 왕복 한 번을 줄이는 것도 체감에 바로 들어온다.
        with _streamlit_pool(2) as pool:
            bars_f = pool.submit(_fetch_adr_bars)
            fx_f = pool.submit(
                requests.get, "https://query1.finance.yahoo.com/v8/finance/chart/KRW=X",
                headers={"User-Agent": "Mozilla/5.0"}, timeout=10,
                params={"range": "1d", "interval": "1d"})
        bars = bars_f.result()
        if bars.empty:
            return None
        last = bars.iloc[-1]
        last_price = float(last["가격"])
        # 세션은 '체결 시각' 기준으로 이미 갈라져 있다. meta의 currentTradingPeriod는 '지금'
        # 기준이라, 장 마감 후에 조회하면 애프터장 체결을 프리장으로 잘못 표시한다.
        session = str(last["세션"])
        prev_close, prev_day_close = _adr_baselines(bars, last["거래일"], session)
        last_ts = last["시각"].to_pydatetime()

        # 지금 미국이 거래 중인지(프리장 04:00 – 애프터 20:00 ET, 평일)를 따로 본다.
        # 이걸 구분하지 않으면 장이 닫힌 새벽에도 '애프터장 $166.60'이 떠서
        # 실시간 시세가 멈춘 것처럼 보인다.
        now_et = dt.datetime.now(ZoneInfo("America/New_York"))
        is_open = now_et.weekday() < 5 and dt.time(4, 0) <= now_et.time() < dt.time(20, 0)
        next_open = None
        if not is_open:
            nxt = now_et.replace(hour=4, minute=0, second=0, microsecond=0)
            if now_et.time() >= dt.time(4, 0):
                nxt += dt.timedelta(days=1)
            while nxt.weekday() >= 5:
                nxt += dt.timedelta(days=1)
            next_open = nxt.astimezone(om.KST).strftime("%m-%d %H:%M")

        fx = fx_f.result().json()
        fx_rate = fx["chart"]["result"][0]["meta"].get("regularMarketPrice")
        if not (last_price and fx_rate):
            return None
        return {
            "price": last_price,
            # 세션에 맞는 등락률 기준 (프리장·정규장=직전 거래일 종가 / 애프터장=당일 종가)
            "prev_close": prev_close,
            # 하루치 그래프의 점선 기준선용. 세션과 무관하게 늘 직전 거래일 종가다.
            "prev_day_close": prev_day_close,
            "session": session,
            "is_open": is_open,
            "next_open": next_open,
            "fx": float(fx_rate),
            "time": last_ts.strftime("%m-%d %H:%M"),
        }
    except Exception:
        return None


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


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_adr_baseline(days: int = ADR_BASELINE_DAYS) -> float | None:
    """'ADR 원화환산 / 본주' 배수의 최근 중앙값.

    괴리율 자체는 공식 비율(ADR_SHARE_RATIO)로 계산하지만, 이 ADR은 평소에도 30–40%대
    프리미엄이 붙어 거래된다. 그래서 절대 괴리율만 보면 늘 '고평가'로 보인다.
    최근 배수의 중앙값을 같이 구해서 '평소 대비 지금 얼마나 더/덜 벌어졌는지'를 보여준다.
    """
    base = "https://query1.finance.yahoo.com/v8/finance/chart/{s}"
    headers = {"User-Agent": "Mozilla/5.0"}

    def daily(sym):
        r = requests.get(base.format(s=sym), headers=headers, timeout=15,
                         params={"range": "3mo", "interval": "1d"})
        r.raise_for_status()
        res = r.json()["chart"]["result"][0]
        return pd.DataFrame({
            "날짜": pd.to_datetime(res["timestamp"], unit="s").normalize(),
            sym: res["indicators"]["quote"][0]["close"],
        }).dropna()

    try:
        # 세 종목의 일봉은 서로 무관하다. 순서대로 받으면 왕복 지연이 3번 쌓인다.
        with ThreadPoolExecutor(max_workers=3) as pool:
            adr_d, host_d, fx_d = pool.map(daily, (ADR_SYMBOL, "000660.KS", "KRW=X"))
        m = adr_d.merge(host_d, on="날짜").merge(fx_d, on="날짜")
        if m.empty:
            return None
        ratio = (m[ADR_SYMBOL] * m["KRW=X"]) / m["000660.KS"]
        ratio = ratio.tail(days)
        return float(ratio.median()) if len(ratio) >= 5 else None
    except Exception:
        return None


def _to_number(text: object) -> float | None:
    """'3,317,917' / '15.89배' / '50.96%' 처럼 단위가 붙은 문자열에서 숫자만 뽑는다."""
    if text is None:
        return None
    cleaned = re.sub(r"[^0-9.\-]", "", str(text))
    if cleaned in ("", "-", "."):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


@st.cache_data(ttl=1800, show_spinner="불러오는 중...")
def fetch_news_with_summary(query: str, count: int = 6) -> list[dict]:
    """뉴스 제목만이 아니라 본문 요약까지 같이 가져온다.
    제목만으로는 AI가 내용을 추측할 수밖에 없어서, 요약문을 붙여야 분석이 구체적으로 나온다."""
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(
        NAVER_NEWS_URL, params={"where": "news", "query": query, "sort": "1"}, headers=headers, timeout=10
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    items = []
    for head in soup.select("span.sds-comps-text-type-headline1")[:count]:
        title = head.get_text(strip=True)
        summary = ""
        node = head
        # 헤드라인에서 위로 올라가며 기사 카드를 찾고, 그 안에서 요약문을 뽑는다
        for _ in range(8):
            node = node.parent
            if node is None:
                break
            bodies = [b.get_text(strip=True) for b in node.select("span.sds-comps-text-type-body1")]
            bodies = [b for b in bodies if len(b) > 40 and b != title]
            if bodies:
                summary = bodies[0]
                break
        items.append({"제목": title, "요약": summary})
    return items


# 코스피 '전체' 투자자별·프로그램 매매. 종목별 장중 수급은 한국거래소가 장 마감 후에만
# 배포해서 무료로는 구할 수 없다(네이버·다음·KRX 모두 전 거래일까지만 준다. 2026-08-21 확인).
# 대신 시장 전체 잠정치는 장중 1~2분마다 갱신되므로, 종목별이 아니라는 점을 명시하고 참고로 쓴다.
NAVER_INVESTOR_TREND_URL = "https://finance.naver.com/sise/investorDealTrendDay.naver"
NAVER_PROGRAM_TREND_URL = "https://finance.naver.com/sise/programDealTrendDay.naver"


def _fetch_market_trend_row(url: str) -> dict | None:
    """네이버 시장 전체 매매동향 표에서 가장 최근(=맨 윗줄) 행을 뽑는다. 단위는 억원."""
    resp = requests.get(url, params={"bizdate": dt.datetime.now(om.KST).strftime("%Y%m%d"), "sosok": "01"},
                        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.naver.com/"},
                        timeout=10)
    resp.raise_for_status()
    resp.encoding = "euc-kr"
    table = pd.read_html(StringIO(resp.text))[0]
    # 헤더가 2단이라 MultiIndex로 잡힌다. '기관 금융투자'처럼 이어 붙여 단순한 이름으로 바꾼다.
    table.columns = [c[1] if c[0] == c[1] else f"{c[0]} {c[1]}" for c in table.columns]
    table = table.dropna(how="all")
    date_col = table.columns[0]
    table = table[table[date_col].astype(str).str.match(r"\d{2}\.\d{2}\.\d{2}")]
    if table.empty:
        return None
    row = table.iloc[0]
    out = {"날짜": str(row[date_col])}
    for col in table.columns[1:]:
        out[col] = float(row[col]) if pd.notna(row[col]) else None
    return out


@st.cache_data(ttl=60, show_spinner=False)
def fetch_market_flow() -> dict | None:
    """코스피 전체 투자자별 순매수 + 프로그램 매매(차익/비차익)를 한 번에.

    반환값의 is_today로 '장중 잠정치'인지 '직전 거래일 확정치'인지 구분한다.
    """
    try:
        with _streamlit_pool(2) as pool:
            inv_f = pool.submit(_fetch_market_trend_row, NAVER_INVESTOR_TREND_URL)
            prg_f = pool.submit(_fetch_market_trend_row, NAVER_PROGRAM_TREND_URL)
        inv, prg = inv_f.result(), prg_f.result()
    except Exception:
        return None
    if not inv:
        return None
    today_txt = dt.datetime.now(om.KST).strftime("%y.%m.%d")
    return {
        "날짜": inv["날짜"],
        "is_today": inv["날짜"] == today_txt,
        "개인": inv.get("개인"),
        "외국인": inv.get("외국인"),
        "기관계": inv.get("기관계"),
        "연기금등": inv.get("기관 연기금등"),
        "차익": (prg or {}).get("차익거래 순매수"),
        "비차익": (prg or {}).get("비차익거래 순매수"),
    }


def build_market_flow_summary() -> str:
    """AI 분석에 넘길 시장 전체 수급 요약."""
    flow = fetch_market_flow()
    if not flow:
        return ""
    when = "장중 잠정치" if flow["is_today"] else "직전 거래일 확정치"
    def fmt(v):
        return f"{v:+,.0f}억원" if v is not None else "N/A"
    lines = [
        f"- 코스피 전체 투자자별 순매수 ({flow['날짜']}, {when}): "
        f"개인 {fmt(flow['개인'])} / 외국인 {fmt(flow['외국인'])} / 기관계 {fmt(flow['기관계'])}"
        f" (연기금등 {fmt(flow['연기금등'])})",
    ]
    if flow["비차익"] is not None:
        lines.append(f"- 코스피 프로그램 매매: 차익 {fmt(flow['차익'])} / 비차익 {fmt(flow['비차익'])}"
                     " (비차익은 외국인·기관 바스켓 매매의 대용 지표)")
    lines.append("- 주의: 이 수치는 코스피 시장 전체이지 이 종목의 수급이 아니다. "
                 "종목별 장중 수급은 거래소가 마감 후에만 공개하므로, 방향의 참고로만 써라.")
    return "\n".join(lines)


NAVER_DISCLOSURE_URL = "https://m.stock.naver.com/api/stock/{code}/disclosure"


@st.cache_data(ttl=1800, show_spinner="공시 수집 중...")
def fetch_disclosures(ticker: str, count: int = 20, body_days: int = 3) -> str:
    """전자공시(KOSCOM/DART) 목록과, 최근 것들의 본문 요지를 마크다운으로 만든다.

    뉴스는 공시를 몇 시간~하루 늦게 따라간다. 정작 주가를 움직인 원인이 공시 한 줄인 경우가
    많은데(자기주식 취득·소각, 신규시설투자, 조회공시 답변 등) 지금까지 AI는 이걸 아예 못 봤다.
    제목만으로는 규모를 알 수 없어서, 최근 body_days일 안의 공시는 본문까지 받아 숫자를 넘긴다.
    """
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://m.stock.naver.com/"}
    try:
        r = requests.get(NAVER_DISCLOSURE_URL.format(code=ticker), headers=headers,
                         params={"pageSize": count}, timeout=10)
        r.raise_for_status()
        items = r.json() or []
    except Exception:
        return ""
    if not isinstance(items, list) or not items:
        return ""

    cutoff = (dt.datetime.now(om.KST).date() - dt.timedelta(days=body_days)).isoformat()
    recent = [d for d in items if str(d.get("datetime", ""))[:10] >= cutoff]

    def body(item) -> tuple[int, str]:
        did = item.get("disclosureId")
        try:
            rr = requests.get(f"{NAVER_DISCLOSURE_URL.format(code=ticker)}/{did}",
                              headers=headers, timeout=10)
            rr.raise_for_status()
            html = ((rr.json() or {}).get("disclosure") or {}).get("contents") or ""
        except Exception:
            return did, ""
        text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
        return did, re.sub(r"\s+", " ", text)[:700]

    bodies: dict[int, str] = {}
    if recent:
        with _streamlit_pool(min(len(recent), 6)) as pool:
            bodies = dict(pool.map(body, recent))

    lines = []
    for d in items:
        when = str(d.get("datetime", "")).replace("T", " ")[:16]
        lines.append(f"- {when}  {d.get('title')}")
        text = bodies.get(d.get("disclosureId"))
        if text:
            lines.append(f"    본문: {text}")
    return "\n".join(lines)


def build_community_summary(ticker: str, stock_name: str, titles_per_source: int = 35) -> str:
    """AI 분석에 넘길 커뮤니티 여론. 비율만이 아니라 '제목 원문'까지 같이 넘긴다.

    커뮤니티 탭은 기본으로 꺼져 있어서 탭 렌더링에 기대면 안 된다. 그래서 여기서 직접 모은다.
    비율(긍정 40% 부정 35%)만 주면 AI가 '의견이 갈린다'는 하나 마나 한 말밖에 못 한다.
    실제 제목을 읽혀야 무엇이 대세인지, 무엇을 걱정하는지 짚어낼 수 있다.
    분류는 키워드 방식을 쓴다. AI 분류는 호출을 하나 더 먹는데, 어차피 분석 모델이
    제목 원문을 직접 읽으므로 여기서 굳이 정확도를 살 이유가 없다.
    """
    blocks = []
    try:
        posts = fetch_community_posts(ticker, DEFAULT_COMMUNITY_POST_COUNT)
    except Exception:
        posts = pd.DataFrame()
    if not posts.empty:
        labeled = classify_sentiment(posts)
        counts = labeled["심리"].value_counts()
        total = len(labeled)
        pos, neg, neu = (int(counts.get(k, 0)) for k in ("긍정", "부정", "중립"))
        days = labeled["날짜"].nunique()
        blocks.append(
            f"[네이버 종목토론방] 최근 {total}건({days}일치) — "
            f"긍정 {pos}건({pos / total:.0%}) / 부정 {neg}건({neg / total:.0%}) / 중립 {neu}건({neu / total:.0%})"
            " (키워드 기반 대략치)\n"
            + "\n".join(f"  - [{r['심리']}] {r['날짜']} {r['제목']}"
                        for _, r in labeled.head(titles_per_source).iterrows())
        )
    try:
        dc = fetch_dc_gallery_posts(stock_name, DEFAULT_COMMUNITY_POST_COUNT)
    except Exception:
        dc = pd.DataFrame()
    if not dc.empty:
        blocks.append(
            f"[디시인사이드 주식갤러리] {len(dc)}건 (종목 전용 갤러리가 아니라 검색 결과)\n"
            + "\n".join(f"  - {r['날짜']} {r['제목']} (조회 {r['조회수']}, 추천 {r['추천']})"
                        for _, r in dc.head(titles_per_source).iterrows())
        )
    return "\n\n".join(blocks)


def build_over_market_summary(ticker: str, close_price: int | None) -> str:
    """프리장·애프터장(NXT)에서 오늘 실제로 무슨 일이 있었는지 정리한다.

    정규장이 끝난 뒤 공시 한 줄에 시간외에서 크게 되돌리는 날이 있는데(예: 마감 직후
    자사주 취득·소각 공시), 종가만 보면 그 사실이 통째로 빠진다. 화면에는 이미 그리고 있지만
    AI에는 안 넘어가고 있었다.
    """
    today = dt.datetime.now(om.KST).date()
    try:
        ticks = load_over_market_ticks(ticker, today)
        ticks = ticks[ticks["시각"].dt.date == today]
    except Exception:
        ticks = pd.DataFrame()
    if ticks.empty:
        return ""

    lines = []
    for label in ("프리장", "애프터장"):
        seg = ticks[ticks["세션"] == label]
        if seg.empty:
            continue
        first, last = float(seg["가격"].iloc[0]), float(seg["가격"].iloc[-1])
        line = (f"- {label} {seg['시각'].min():%H:%M}~{seg['시각'].max():%H:%M}: "
                f"{first:,.0f} → {last:,.0f}원 "
                f"(고가 {seg['가격'].max():,.0f} / 저가 {seg['가격'].min():,.0f})")
        # 애프터장은 '정규장 종가 대비'가 핵심이다. 마감 후 재료가 반영된 폭이 그대로 보인다.
        if label == "애프터장" and close_price:
            line += f", 정규장 종가({close_price:,}원) 대비 {(last / close_price - 1) * 100:+.2f}%"
        lines.append(line)
    return "\n".join(lines)


def build_intraday_summary(ticker: str, close_price: int | None) -> str:
    """오늘 장중에 어떻게 움직였는지(언제 밀렸는지/되돌렸는지)를 분봉에서 뽑는다.
    '종가 -9.75%'만 넘기면 하루 종일 흘러내린 건지 특정 시각에 급락한 건지 구분할 수 없다."""
    try:
        bars = fetch_intraday_price(ticker)
    except Exception:
        return ""
    if bars.empty or len(bars) < 10:
        return ""
    px = bars["현재가"].astype(float)
    open_p, high_p, low_p, last_p = float(px.iloc[0]), float(px.max()), float(px.min()), float(px.iloc[-1])
    lines = [
        f"- 장중 {bars['시각'].min():%H:%M}~{bars['시각'].max():%H:%M}: "
        f"시가 {open_p:,.0f} / 고가 {high_p:,.0f} / 저가 {low_p:,.0f} / 마지막 {last_p:,.0f}원",
        f"- 고점 대비 낙폭 {(low_p / high_p - 1) * 100:+.2f}%, 저점 대비 회복 {(last_p / low_p - 1) * 100:+.2f}%",
    ]
    # 30분 단위로 가장 크게 움직인 구간을 짚어준다
    step = max(len(px) // 13, 1)
    moves = []
    for i in range(0, len(px) - step, step):
        chg = px.iloc[i + step] / px.iloc[i] - 1
        moves.append((abs(chg), chg, bars["시각"].iloc[i], bars["시각"].iloc[i + step]))
    if moves:
        _, chg, t0, t1 = max(moves)
        lines.append(f"- 가장 급했던 구간: {t0:%H:%M}~{t1:%H:%M} {chg * 100:+.2f}%")
    return "\n".join(lines)


# 종목명만으로 뉴스를 뽑으면 '주가가 올랐다/내렸다'류 시황 기사만 모여서, AI가 원인을 짚지 못한다.
# 업황·전방수요·매크로 쪽 질의를 같이 던져서 '왜'에 해당하는 재료를 확보한다.
SECTOR_NEWS_QUERIES = [
    "{name} HBM",
    "메모리 반도체 업황",
    "D램 가격",
    "엔비디아 실적 AI 반도체",
    "반도체 수출 실적",
]


@st.cache_data(ttl=1800, show_spinner="업종·매크로 뉴스 수집 중...")
def fetch_sector_news(stock_name: str, per_query: int = 3) -> str:
    """업종·전방수요·매크로 관련 뉴스를 질의별로 모아 하나의 마크다운으로 만든다.

    구글 검색 grounding이 무료 요금제에서 막혀 있어(429), '검색으로 보강'은 실제로는
    한 번도 동작한 적이 없다. 대신 네이버 뉴스에 질의를 여러 개 던져서 같은 목적
    - 대시보드에 없는 바깥 소식을 채우는 것 - 을 실제로 달성한다.
    """
    queries = [t.format(name=stock_name) for t in SECTOR_NEWS_QUERIES]

    def one(query: str):
        try:
            return query, fetch_news_with_summary(query, count=per_query)
        except Exception:
            return query, []

    # 질의 5개를 순서대로 던지면 왕복 지연이 그대로 쌓인다. 한꺼번에 보내고 결과만 순서대로 정리한다.
    # 워커가 캐시된 fetch_news_with_summary를 부르므로 컨텍스트를 붙인 풀을 쓴다.
    with _streamlit_pool(len(queries)) as pool:
        fetched = list(pool.map(one, queries))

    blocks = []
    seen_titles: set[str] = set()
    for query, items in fetched:
        rows = []
        for it in items:
            if it["제목"] in seen_titles:      # 질의끼리 겹치는 기사는 한 번만
                continue
            seen_titles.add(it["제목"])
            rows.append(f"- {it['제목']}" + (f"\n  요약: {it['요약']}" if it["요약"] else ""))
        if rows:
            blocks.append(f"[검색어: {query}]\n" + "\n".join(rows))
    return "\n\n".join(blocks)


@st.cache_data(ttl=6 * 3600, show_spinner="불러오는 중...")
def fetch_news_headlines(query: str, count: int = 6) -> list[str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(
        NAVER_NEWS_URL, params={"where": "news", "query": query, "sort": "1"}, headers=headers, timeout=10
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    headlines = [el.get_text(strip=True) for el in soup.select("span.sds-comps-text-type-headline1")]
    return headlines[:count]


TRENDFORCE_SEMICONDUCTOR_URL = "https://www.trendforce.com/research/category/Semiconductors"


@st.cache_data(ttl=6 * 3600, show_spinner="불러오는 중...")
def fetch_trendforce_news(count: int = 5) -> pd.DataFrame:
    """TrendForce 반도체(Semiconductors) 카테고리 페이지에 실린 최신 무료 뉴스 기사 목록
    (유료 리서치 데이터시트가 아니라 /presscenter/news/ 기사만 대상)."""
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(TRENDFORCE_SEMICONDUCTOR_URL, headers=headers, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    rows = []
    seen_href = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/presscenter/news/" not in href:
            continue
        if href in seen_href:
            continue
        title = a.get_text(strip=True)
        if not title:
            continue
        seen_href.add(href)

        row_container = a.find_parent("div", class_="row")
        date_text = ""
        if row_container is not None:
            m = re.search(r"\d{4}/\d{2}/\d{2}", row_container.get_text(" ", strip=True))
            date_text = m.group() if m else ""

        rows.append({
            "제목": title,
            "날짜": date_text,
            "url": href if href.startswith("http") else f"https://www.trendforce.com{href}",
        })
    return pd.DataFrame(rows[:count])


@st.cache_data(ttl=6 * 3600, show_spinner="불러오는 중...")
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


@st.cache_data(ttl=1800, show_spinner="불러오는 중...")
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


@st.cache_data(ttl=1800, show_spinner="불러오는 중...")
def fetch_community_posts(ticker: str, count: int = 60) -> pd.DataFrame:
    all_posts: list[dict] = []
    max_pages = min(30, count // 20 + 2)
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


@st.cache_data(ttl=1800, show_spinner="불러오는 중...")
def fetch_dc_gallery_posts(keyword: str, count: int = 60) -> pd.DataFrame:
    """디시인사이드 주식갤러리(krstock)에서 keyword가 제목/본문에 포함된 게시글을 검색한다.
    krstock은 특정 종목 전용 갤러리가 아니라 국내 주식 전반을 다루는 갤러리라,
    거래량·관심도가 낮은 종목은 검색 결과가 적거나 없을 수 있다."""
    max_pages = min(30, count // 20 + 2)

    def one_page(page: int) -> list[dict]:
        params = {"id": DC_GALLERY_ID, "s_type": "search_subject_memo", "s_keyword": keyword, "page": str(page)}
        try:
            resp = requests.get(DC_GALLERY_LIST_URL, params=params, headers=DC_HEADERS, timeout=10)
            resp.raise_for_status()
        except requests.RequestException:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", class_="gall_list")
        if table is None:
            return []
        found = []
        for tr in table.find("tbody").find_all("tr", class_="us-post"):
            title_td = tr.find("td", class_="gall_tit")
            a = title_td.find("a") if title_td else None
            if a is None:
                continue
            m = re.search(r"no=(\d+)", a.get("href", ""))
            if not m:
                continue
            post_no = m.group(1)
            date_td = tr.find("td", class_="gall_date")
            date_text = (date_td.get("title") if date_td else None) or (date_td.get_text(strip=True) if date_td else "")
            count_td = tr.find("td", class_="gall_count")
            recommend_td = tr.find("td", class_="gall_recommend")
            found.append({
                "제목": a.get_text(strip=True),
                "번호": post_no,
                "날짜": date_text,
                "조회수": pd.to_numeric(count_td.get_text(strip=True), errors="coerce") if count_td else None,
                "추천": pd.to_numeric(recommend_td.get_text(strip=True), errors="coerce") if recommend_td else None,
                "url": f"{DC_GALLERY_VIEW_URL}?id={DC_GALLERY_ID}&no={post_no}",
            })
        return found

    # 검색 결과 페이지는 서로 독립이라 한꺼번에 받는다. 순차로 돌면 페이지 수만큼 왕복이 쌓인다.
    # 결과는 페이지 순서대로 이어붙여서 기존과 같은 정렬(최신순)을 유지한다.
    with ThreadPoolExecutor(max_workers=min(max_pages, 8)) as pool:
        pages = list(pool.map(one_page, range(1, max_pages + 1)))

    posts: list[dict] = []
    seen_no: set[str] = set()
    for page_posts in pages:
        for post in page_posts:
            if post["번호"] in seen_no:
                continue
            seen_no.add(post["번호"])
            posts.append(post)
    return pd.DataFrame(posts[:count])


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


def _parse_dram_last_update(soup: BeautifulSoup, category_label: str) -> str | None:
    """'Module Spot Price Last Update: Jul.20 2026  14:40 (GMT+8)'처럼 카테고리별 헤더 행에 있는
    'Last Update' 표시를 찾아 'YYYY-MM-DD HH:MM' 형태로 반환한다. 이 표시는 데이터가 들어있는 표와는
    별개의 헤더 테이블에 있어서, tbody의 DOM 조상이 아니라 카테고리명(예: "Module Spot Price")으로 찾아야 한다.
    모듈가/칩가 표는 서로 다른 시각에 갱신되므로 따로 확인한다."""
    for span in soup.find_all("span", class_="tab_time"):
        row = span.find_parent("tr")
        row_text = row.get_text(" ", strip=True) if row is not None else span.get_text(" ", strip=True)
        if category_label not in row_text:
            continue
        m = re.search(
            r"Last\s*Update:\s*([A-Za-z]{3})\.(\d{1,2})\s+(\d{4})\s+(\d{1,2}):(\d{2})",
            row_text,
        )
        break
    else:
        return None
    if not m:
        return None
    mon_str, day, year, hour, minute = m.groups()
    try:
        month = dt.datetime.strptime(mon_str, "%b").month
    except ValueError:
        return None
    return f"{int(year):04d}-{month:02d}-{int(day):02d} {int(hour):02d}:{int(minute):02d}"


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


@st.cache_data(ttl=3600, show_spinner="불러오는 중...")
def _fetch_dram_soup() -> BeautifulSoup:
    """모듈가·칩가가 같은 페이지에 있으므로, 페이지 요청 자체는 한 번만 캐시해서 공유한다."""
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(DRAMEXCHANGE_URL, headers=headers, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


@st.cache_data(ttl=3600, show_spinner="불러오는 중...")
def fetch_dram_module_prices() -> tuple[pd.DataFrame, str | None]:
    soup = _fetch_dram_soup()
    df = _parse_dram_table(
        soup, "tb_ModuleSpotPrice",
        item_filter={
            "DDR5 UDIMM 16GB 4800/5600",
            "DDR5 RDIMM 32GB 4800/5600",
        },
    )
    last_update = _parse_dram_last_update(soup, "Module Spot Price")
    return df, last_update


@st.cache_data(ttl=3600, show_spinner="불러오는 중...")
def fetch_dram_chip_prices() -> tuple[pd.DataFrame, str | None]:
    soup = _fetch_dram_soup()
    df = _parse_dram_table(
        soup, "tb_NationalDramSpotPrice",
        item_filter={
            "DDR5 16Gb (2Gx8) 4800/5600",
            "DDR5 16Gb (2Gx8) eTT",
            "DDR4 16Gb (2Gx8) 3200",
            "DDR4 16Gb (2Gx8) eTT",
        },
    )
    last_update = _parse_dram_last_update(soup, "DRAM Spot Price")
    return df, last_update


def _signed_pct(row: pd.Series) -> str:
    """부호는 방향(상승/하락 화살표)에서만 가져온다. 사이트가 변동률 텍스트에 '-'를 이미 포함해
    내려주는 경우가 있어, 값을 절댓값으로 바꾸지 않으면 '--0.64%'처럼 부호가 겹친다."""
    magnitude = abs(float(row["변동률(%)"]))
    if row["방향"] == "상승":
        return f"+{magnitude:.2f}%"
    if row["방향"] == "하락":
        return f"-{magnitude:.2f}%"
    return f"{magnitude:.2f}%"


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


BIGTECH_CIKS = {
    "Alphabet(Google)": "0001652044",
    "Amazon": "0001018724",
    "Meta": "0001326801",
    "Microsoft": "0000789019",
}
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


@st.cache_data(ttl=24 * 3600, show_spinner="불러오는 중...")
def fetch_bigtech_capex() -> pd.DataFrame:
    """빅테크(마이크로소프트/구글/아마존/메타)의 분기별 설비투자(capex) 실적을 SEC 공시(XBRL)에서 가져온다."""
    def one(item):
        name, cik = item
        try:
            return name, _fetch_company_standalone_capex_quarters(cik)
        except Exception:
            return name, []

    # 회사 4곳의 SEC 공시는 서로 무관하다. 한 곳이 느려도 나머지를 붙잡아두지 않게 동시에 받는다.
    with ThreadPoolExecutor(max_workers=len(BIGTECH_CIKS)) as pool:
        fetched = list(pool.map(one, BIGTECH_CIKS.items()))

    rows = []
    for name, quarters in fetched:
        for q in quarters:
            rows.append({"기업": name, "분기말": q["end"], "capex_USD": q["val"]})
    if not rows:
        return pd.DataFrame(columns=["기업", "분기말", "capex_USD"])
    df = pd.DataFrame(rows)
    df["분기말"] = pd.to_datetime(df["분기말"])
    df = df[df["분기말"] >= "2022-01-01"].sort_values(["분기말", "기업"])
    return df.reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner="불러오는 중...")
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
    trendforce_md: str = "",
    snapshot_md: str = "",
    news_md: str = "",
    use_search: bool = False,
    macro_md: str = "",
    sector_news_md: str = "",
    adr_md: str = "",
    disclosure_md: str = "",
    over_market_md: str = "",
    intraday_md: str = "",
    market_flow_md: str = "",
) -> tuple[str, str | None]:
    prompt = f"""오늘은 {time_label}입니다. 다음은 이 시점 기준 {stock_label} 관련 데이터입니다.

**중요**: 아래 데이터에 적힌 사실만 쓰세요. 학습 시점에 알고 있던 과거 뉴스나 날짜를 끌어오지 마세요.
(실제로 이 지시가 없으면 몇 년 전 사건을 '최근 뉴스'라고 답하는 일이 생깁니다.)

[오늘 주가]
{price_summary}

[오늘 장중 흐름]
{intraday_md if intraday_md else "(수집 실패)"}

[정규장 밖 움직임 — 프리장 · 애프터장(NXT)]
{over_market_md if over_market_md else "(오늘 시간외 기록 없음)"}

[전자공시 — 최근순, 최근 것은 본문 요지 포함]
{disclosure_md if disclosure_md else "(수집 실패)"}

[밸류에이션 · 컨센서스 · 동일업종 · 수급추이]
{snapshot_md if snapshot_md else "(수집 실패)"}

[최근 수급 동향 — 이 종목, 일별 확정치]
{supply_summary}

[코스피 시장 전체 수급 — 종목별 아님, 장중 잠정치]
{market_flow_md if market_flow_md else "(수집 실패)"}

[통합 매수/매도 신호 — 실험적 백테스트, 매매 신호 아님]
{composite_summary}

[가격 과열도 백테스트 — 과거 통계 참고용, 매매 신호 아님]
{overheat_summary}

[코스피200 선물 외국인 순매도 하락 경보 — 과거 통계 참고용, 매매 신호 아님]
{futures_summary}

[DRAM 현물가]
{dram_summary}

[관련 뉴스 (제목 + 본문 요약)]
{news_md if news_md else (chr(10).join(f"- {h}" for h in headlines) if headlines else "(수집된 뉴스 없음)")}

[최근 애널리스트 리포트]
{reports_md if reports_md else "(수집된 리포트 없음)"}

[업종 · 전방수요 · 매크로 관련 뉴스]
{sector_news_md if sector_news_md else "(수집 안 함)"}

[해외 반도체 산업 리서치 뉴스 — TrendForce]
{trendforce_md if trendforce_md else "(수집된 자료 없음)"}

[거시경제 지표]
{macro_md if macro_md else "(수집 실패)"}

[해외 상장분(ADR) 괴리율]
{adr_md if adr_md else "(해당 없음)"}

[투자자 커뮤니티 — 게시글 원문. 여론이지 사실이 아님]
{community_summary if community_summary else "(수집 실패)"}

아래 형식 그대로, 한국어로 작성해줘.

## 한 줄 요약
오늘 이 종목에서 가장 중요한 사실 한 문장.

## 오늘 이렇게 움직인 이유
정규장 등락을 '언제 무엇 때문에'로 설명해라. 장중 흐름에서 급했던 구간을 짚고,
그 시각 전후의 공시·뉴스와 연결해라. 연결할 근거가 없으면 "직접 연결되는 재료는 데이터에 없음"이라고 써라.
**정규장이 끝난 뒤 시간외에서 방향이 바뀌었다면 반드시 별도 문단으로 짚어라.**
종가만 보면 놓치는 부분이고, 공시가 마감 직후에 나오는 경우가 잦다.
시간외 등락률은 정규장 종가 대비 값을 쓰고, 전일 종가 대비와 헷갈리지 마라.

## 종목 이슈인가, 업종 전체인가
'동일업종' 등락률과 비교해서 판단해줘. 동종 종목들이 비슷하게 움직였으면 업종/매크로 요인이고,
이 종목만 튀었으면 개별 이슈다. 반드시 다른 종목 등락률 숫자를 인용해서 근거를 대줘.

## 강세 근거
## 약세 근거 / 리스크

이 두 항목이 이 분석의 핵심이다. 다음 규칙을 지켜라.

1) 아래 여섯 갈래를 **모두** 훑어보고, 각 갈래에서 나온 근거를 빠짐없이 배치해라.
   `[공시]`   전자공시 (규모·기간 같은 숫자를 본문에서 인용해라)
   `[뉴스]`   종목 뉴스 + 업종·전방수요·매크로 뉴스
   `[리포트]` 애널리스트 리포트
   `[산업]`   TrendForce 산업 리서치 + DRAM 현물가
   `[거시]`   거시경제 지표(SOX·나스닥·달러인덱스·환율·미국 금리) + ADR 괴리율
   `[대시보드]` 수급, 통합 신호, 가격 과열도, 선물 경보, 밸류에이션·컨센서스, 동일업종 등락률
2) 각 항목은 **`[갈래] 내용 (근거 숫자·출처)`** 형식으로 써라.
   말머리는 위 여섯 개(`[공시]` `[뉴스]` `[리포트]` `[산업]` `[거시]` `[대시보드]`)만 쓰고,
   다른 이름을 지어내지 마라.
   예: `[거시] SOX 20일 +6.37%로 반도체 업종 전반이 강세 (SOX 12,417.05)`
3) 어느 갈래에서 쓸 만한 근거가 안 나오면 그 갈래는 `[갈래] 이번엔 뚜렷한 신호 없음`이라고 한 줄로 적어라.
   억지로 만들어내지 마라.
4) 매크로 지표는 숫자만 옮기지 말고 **이 종목까지 어떻게 연결되는지** 한 마디로 붙여라
   (예: 원화 약세면 수출 채산성에 유리, 금리 급등이면 성장주 밸류에이션에 부담).
5) 강세만 길게 쓰고 약세를 형식적으로 채우지 마라. 양쪽을 같은 밀도로 써라.

## 근거의 무게
위 강세·약세 중 지금 더 무거운 쪽은 어디이고 왜인지 2–3문장.
'확인된 실적·가격 데이터'가 '기대·심리'보다 무겁다는 기준으로 판단해라.

## 커뮤니티 대세 반응
게시글 제목을 실제로 읽고, 지금 개인 투자자 사이에서 **우세한 반응이 무엇인지** 2–3문장으로 짚어라.
- 비율만 옮기지 마라("긍정 40% 부정 35%"는 그 자체로는 아무 말도 아니다).
  무엇을 기대하고 무엇을 걱정하는지, 반복해서 나오는 화제가 무엇인지를 써라.
- 근거로 실제 제목을 1–2개 짧게 인용해라.
- 대세와 다른 소수 의견이 눈에 띄면 한 줄로 덧붙여라.
- 여론이 위 강세·약세 근거와 어긋나면 그 점을 지적해라 (예: 데이터는 우호적인데 여론은 공포).
- 반어법·비꼬는 말투가 많은 곳이다. 표면 단어가 아니라 문맥으로 읽어라.
- 이건 사실이 아니라 여론이다. 여기서 나온 이야기를 사실 근거로 올려 쓰지 마라.

## 지금 위치
현재가가 컨센서스 목표주가, 52주 고저, PER 대비 어디에 있는지 숫자로 정리해줘.
컨센서스는 증권사들의 기대치일 뿐 보장이 아니라는 점도 한 줄로 덧붙여줘.

## 앞으로 확인할 것
막연한 말("시장을 주시") 말고, 구체적으로 무엇을 보면 판단이 갈리는지 2–3개.
가능하면 이 대시보드에서 볼 수 있는 지표(DRAM 현물가, 기관 순매수, 빅테크 Capex 등)로 짚어줘.

작성 규칙:
- 위 데이터에 없는 사실은 지어내지 마. 모르면 "데이터에 없음"이라고 써.
- 숫자를 인용할 때는 위 데이터의 값을 그대로 써.
- "긍정적 흐름이 예상된다" 같은 하나 마나 한 문장은 쓰지 마.
- 통합 신호·가격 과열도·선물 경보는 과거 통계를 돌린 참고 지표다. 근거로 써도 되지만,
  쓸 때는 "백테스트 기반 참고치" 같은 단서를 반드시 함께 붙여라. 단독 근거로 결론을 내지 마.
- 커뮤니티 심리는 사실이 아니라 여론이다. 사실 근거와 같은 급으로 취급하지 마.
- 매수/매도 추천이나 목표가 제시는 하지 마. 사실과 해석만.
- 날짜를 쓸 때는 위 데이터에 있는 날짜만 써. 기억에 의존한 날짜는 쓰지 마.
- 같은 내용을 여러 갈래에 중복해서 적지 마. 한 근거는 가장 잘 맞는 갈래 한 곳에만."""

    note = None
    # 구글 검색 grounding은 무료 요금제에서 막혀 있다(쿼터가 남은 모델로 시험해도 즉시 429).
    # 그래서 '검색으로 보강'은 실제로 한 번도 동작한 적이 없고, 매번 실패 후 되돌아오느라
    # 호출만 한 번 더 쓰고 있었다. 유료 키로 바꿀 때만 켜지도록 환경변수 뒤로 옮긴다.
    # 바깥 소식은 fetch_sector_news()가 네이버 뉴스 다중 질의로 실제로 채워온다.
    if use_search and os.environ.get("GEMINI_ENABLE_GROUNDING") == "1":
        try:
            text, used = _call_gemini(prompt, tools=[{"type": "google_search"}])
            return (text or "AI가 응답을 생성하지 못했습니다.", "search_ok")
        except Exception:
            note = "구글 검색 grounding에 실패해, 수집된 뉴스만으로 분석했습니다."

    text, used = _call_gemini(prompt)
    if used != GEMINI_MODEL:
        # 기본 모델이 일일 한도에 걸려 다른 모델로 넘어간 경우 그 사실을 알려준다
        switched = f"{GEMINI_MODEL}의 오늘 무료 한도가 소진되어 {used} 모델로 분석했습니다."
        note = f"{note} {switched}" if note else switched
    return (text or "AI가 응답을 생성하지 못했습니다.", note)


def _live_deviation(live_price: float, ma_window: int) -> tuple[float | None, float | None]:
    """장중 현재가를 시계열의 마지막 값으로 놓고 이동평균 대비 괴리율을 계산한다.
    가격 과열도 탭은 확정된 종가로 계산하므로, 장중에는 이 값이 그쪽보다 앞서 움직인다.
    과거 괴리율 분포에서의 백분위(상위 N%)도 함께 돌려준다."""
    hist = fetch_backtest_history_live(TICKER, target_days=700)
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


def _korea_session_now(now_kst: dt.datetime | None = None) -> str | None:
    """지금 한국 시장이 거래 중인 구간이면 그 이름을, 아니면 None.

    NXT 프리장 08:00–09:00 / KRX 정규장 09:00–15:30 / NXT 애프터장 15:40–20:00.
    15:30–15:40 공백은 따로 가르지 않는다. 10분 때문에 화면 기본값이 ADR로 튀었다가
    돌아오면 오히려 더 어수선하다.
    공휴일은 이 함수로 알 수 없으므로, 호출부에서 '오늘 실제 체결이 있었는지'와 같이 본다.
    """
    now = now_kst or dt.datetime.now(om.KST)
    if now.weekday() >= 5:
        return None
    t = now.time()
    if dt.time(8, 0) <= t < dt.time(9, 0):
        return "프리장"
    if dt.time(9, 0) <= t < dt.time(15, 40):
        return "정규장"
    if dt.time(15, 40) <= t <= dt.time(20, 0):
        return "애프터장"
    return None


def _note_optional_failure(what: str, exc: Exception) -> None:
    """현재가 화면의 '있으면 좋은' 부품이 실패했을 때 쓰는 처리.

    조용히 넘기면(except: pass) 화면에서 그 부분만 소리 없이 사라져서, 실제로
    장중 그래프가 통째로 없어진 걸 한참 뒤에야 알아챈 적이 두 번 있었다
    (빈 DataFrame dtype 문제, plotly 5.x가 모르는 속성 문제).
    나머지 화면은 그대로 두되 무엇이 왜 빠졌는지는 반드시 남긴다."""
    print(f"[render_current_price] {what} 실패: {type(exc).__name__}: {exc}", flush=True)
    st.caption(f":gray[{what}를 표시하지 못했습니다 ({type(exc).__name__}).]")


# 시간외 시세 관련 로직은 over_market.py에 모아두고, 화면에서는 얇게 감싸서 쓴다.
# 실제 수집은 collector.py(별도 컨테이너)가 화면 접속과 무관하게 상시로 돌린다.
save_over_market_tick = om.save_tick
load_over_market_ticks = om.load_ticks
_parse_price_number = om.parse_price_number
_OVER_SESSION_LABELS = om.SESSION_LABELS
OVER_MARKET_COLLECT_TICKERS = om.COLLECT_TICKERS


@st.fragment(run_every=REFRESH_SEC)
def render_current_price():
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
            price_col, open_col, high_col, low_col, volume_col = st.columns(5)
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
        st.session_state["current_price_summary"] = f"종가 {close_price:,}원, 전일대비 {change:+,}원 ({change_pct:+.2f}%)"
        # AI 분석 탭에서 컨센서스 목표주가 대비 상승여력을 계산할 때 쓴다
        st.session_state["current_price_value"] = close_price

        try:
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
                st.plotly_chart(fig_intraday, width="stretch", key="chart_intraday_price", config={"displayModeBar": False})
        except Exception as exc:
            _note_optional_failure("장중 주가 추이", exc)

        # 장중 그래프 아래. 종목 그래프를 먼저 보고, 그 다음에 시장 전체 배경을 보는 순서다.
        try:
            flow = fetch_market_flow()
            if flow:
                live = flow["is_today"]
                flow_help = (
                    "**이 종목이 아니라 코스피 시장 전체 수급입니다.**\n\n"
                    "종목별 장중 수급은 한국거래소가 장 마감 후에만 공개해서 무료로는 구할 수 없습니다. "
                    "증권사 HTS가 장중에 보여주는 종목별 수급은 거래소 유료 실시간 피드입니다.\n\n"
                    "시장 전체 잠정치는 장중 1~2분마다 갱신됩니다. 이 종목의 수급으로 읽지 말고, "
                    "'오늘 시장에서 외국인이 사는 날인가 파는 날인가' 정도의 배경으로만 보세요.\n\n"
                    "종목별 일별 확정 수급은 **수급 현황** 탭에 있습니다."
                )
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
    except Exception as e:
        st.error(f"현재가 조회에 실패했습니다: {e}")
        st.session_state["current_price_summary"] = "현재가 데이터를 가져오지 못함"


render_current_price()

st.divider()

REFRESH_CHECK_INTERVAL_SEC = 60  # 예약된 시각이 지났는지 확인하는 주기

# 탭마다 데이터 성격이 달라 자동 새로고침 시각을 그룹별로 따로 둔다.
MARKET_DATA_REFRESH_HOURS = [16, 17, 18, 19]  # 수급현황·가격과열도·선물경보·통합신호·조기신호(종가/수급 기반)
DRAM_REFRESH_HOURS = [13, 16, 20]
BIGTECH_CAPEX_REFRESH_HOURS = [16]


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
        # 기록이 쌓일수록 점이 빽빽해져 선이 안 보이므로 마커 없이 선만 그린다
        fig = px.line(hist, x="날짜", y="평균가(USD)")
        _style_chart_mobile(fig, title=selected, show_legend=False)
        fig.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig, width="stretch", key=chart_key, config={"displayModeBar": False})
        st.caption("차트 하단 슬라이더를 드래그하면 보고 싶은 기간만 확대해서 볼 수 있습니다.")


DOWNTREND_WINDOW = 20


_visible_tab_labels = [label for label in ALL_TAB_LABELS if label in visible_tab_labels] or ALL_TAB_LABELS
tabs = st.tabs(_visible_tab_labels)
_tab_map = dict(zip(_visible_tab_labels, tabs))

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

def _render_tab_supply():
    global investor_df

    investor_df = pd.DataFrame()

    lookback_days = st.session_state.get("lookback_days_slider", DEFAULT_LOOKBACK_DAYS)
    _subheader_with_help(
        f"최근 {lookback_days}일 투자자별 순매수 거래량",
        "개인·기관·외국인이 이 종목을 하루에 얼마나 순매수(매수-매도)했는지와, 그 누적 추세를 봅니다. "
        "아래 표의 기울기는 누적 순매수 추세선의 하루 평균 기울기로, 양수면 매수 우위입니다.\n\n"
        "개인 순매수는 네이버가 따로 제공하지 않아 기관·외국인 합산의 잔차로 추정한 값이라 "
        "기타법인 등의 소액 오차가 섞일 수 있습니다.\n\n"
        "**절대 거래량**도 같이 그립니다. 순매수는 '누가 샀나'만 알려줄 뿐이라, 같은 순매수라도 "
        "거래량이 평소의 3배인 날과 절반인 날은 의미가 다릅니다 (자세한 설명은 그 그래프 옆 ❓에).",
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
            st.plotly_chart(fig_bar, width="stretch", key="chart_investor_bar", config={"displayModeBar": False})

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
                    "막대 색은 그 날 종가가 전일 대비 올랐으면 초록, 내렸으면 빨강입니다. "
                    f"회색 선은 {avg_window}일 이동평균이고, 조회 기간에 맞춰 자동으로 조정됩니다.\n\n"
                    "거래량은 일봉 경로에서 따로 받아서, 마감 직후 오늘치가 바로 들어옵니다. "
                    "위 순매수 그래프는 투자자 수급이 공개된 뒤에야 오늘치가 채워지므로, "
                    "마감 직후에는 거래량 쪽이 하루 앞서 있을 수 있습니다.\n\n"
                    "거래량이 평소보다 크게 늘어난 날은 위 순매수 그래프에서 누가 움직였는지 같이 보세요. "
                    "다만 거래량 자체는 이 종목 과거 데이터에서 방향 예측력이 없었습니다 — "
                    "크기의 참고치로만 쓰세요.",
                    key="volume",
                )
                _style_chart_mobile(fig_vol)
                fig_vol.update_yaxes(title_text="거래량(주)")
                st.plotly_chart(fig_vol, width="stretch", key="chart_volume",
                                config={"displayModeBar": False})

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
            st.plotly_chart(fig_line, width="stretch", key="chart_investor_line", config={"displayModeBar": False})

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

def _render_tab_overheat():
    global overheat_summary

    OVERHEAT_QUANTILES = [0.20, 0.15, 0.10, 0.05]

    overheat_summary = "가격 과열도 백테스트 미실행"

    _subheader_with_help(
        "가격 과열도 백테스트",
        "이동평균 대비 괴리율(과열·침체) 구간별로, 향후 일정 기간 내 일정 폭 이상 하락할 확률이 "
        "어떻게 달라지는지 보는 참고용 통계입니다. 기본값(80일·15거래일·10%)은 파라미터 168개 조합을 "
        "백테스트해 괴리율과 하락확률의 단조 관계가 가장 뚜렷하고 검증구간에서도 유지된 조합으로 정했습니다. "
        "상승 확률은 괴리율과 뚜렷한 관계가 없어(양 극단에서 모두 높은 U자형) 표시하지 않습니다. "
        "매매 신호가 아닙니다.\n\n"
        "화면 맨 위의 '괴리율 (현재가 기준)'은 같은 값을 장중 현재가로 계산한 것이라, 확정 종가로 계산하는 "
        "이 탭보다 장중에 먼저 움직입니다. 이동평균 기간은 아래 설정을 함께 따릅니다.",
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
        overheat_hist = fetch_backtest_history_live(TICKER, target_days=700)
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
                st.plotly_chart(fig_breakdown, width="stretch", key="chart_overheat_breakdown", config={"displayModeBar": False})

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
                st.plotly_chart(fig_overheat, width="stretch", key="chart_overheat", config={"displayModeBar": False})
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
                strategy_result = run_overheat_threshold_strategy(overheat_hist_ma, period_start)

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
                        st.plotly_chart(fig_strategy, width="stretch", key="chart_overheat_strategy", config={"displayModeBar": False})
    except Exception as e:
        st.error(f"가격 과열도 백테스트에 실패했습니다: {e}")

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
                st.plotly_chart(fig_futures_breakdown, width="stretch", key="chart_futures_breakdown", config={"displayModeBar": False})

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
                st.plotly_chart(fig_futures, width="stretch", key="chart_futures", config={"displayModeBar": False})
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
        "기관 수급, 미국 반도체지수(SOX), 달러인덱스(DXY)를 통계적 유의성에 따라 가중합산한 실험적 종합 신호입니다 "
        "(다른 지표들은 백테스트 결과 상관관계가 낮거나 정보가 중복돼 제외했습니다). "
        f"종합 신호가 매수/매도 우위일 때 향후 {COMPOSITE_HORIZON}거래일 내 하락/상승 확률이 어떻게 달라지는지 봅니다.\n\n"
        "매매 신호가 아니라 백테스트 기반 참고 자료이며, 가중치를 같은 기간 데이터로 정한 인-샘플 결과입니다. "
        "학습 구간(초반)과 검증 구간(후반)을 60/70/80%로 나눠 각각 확인해보니, 하락 확률 차이는 세 분할 모두 "
        "통계적으로 유의했지만(p<0.01), 상승 확률 차이는 세 분할 모두 유의하지 않았습니다(p=0.14–0.95). "
        "즉 하락 경보로는 어느 정도 근거가 있지만, 상승 예측으로는 신뢰하기 어렵습니다.",
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
            st.plotly_chart(fig_composite, width="stretch", key="chart_composite", config={"displayModeBar": False})
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


def _render_tab_signal():
    _subheader_with_help(
        "매매 신호 (기관 수급 기반)",
        "기관 순매수를 최근 평균 거래량으로 나눠 20일 누적한 값이 양수면 '보유', 음수면 '현금'으로 판정합니다.\n\n"
        "대시보드가 쓰는 후보 지표 19종(이동평균 괴리율, 모멘텀, 외국인 수급, SOX, 달러인덱스, 거래량 등)을 "
        "SK하이닉스 10년치로 검증해 유일하게 살아남은 지표입니다. SOX는 학습구간과 검증구간에서 상관 부호가 "
        "뒤집혔고, 달러인덱스와 괴리율은 중첩 표본을 보정하면 유의성이 사라졌습니다.\n\n"
        "이 신호의 정체는 '기관이 사면 오른다'는 일반 법칙이 아니라, **글로벌 반도체 업황의 선행 지표**입니다. "
        "SK하이닉스에 대한 한국 기관의 순매수는 마이크론(+0.24)·대만 메모리 3사(+0.17–+0.23)·"
        "도쿄일렉트론(+0.25)·ASML(+0.20)까지 예측했지만, 반도체와 무관한 종목(코카콜라·JP모건·도요타 등)은 "
        "전혀 예측하지 못했습니다(평균 +0.05, 유의성 없음). 이 구별이 신호가 진짜라는 근거입니다.\n\n"
        "반대로 '각 종목의 자기 기관 수급으로 그 종목을 예측'하는 방식은 통하지 않습니다. "
        "정보를 가진 건 하이닉스에 들어오는 기관 자금이지, 아무 종목의 기관 수급이 아닙니다.\n\n"
        "아래 '검증 상세와 한계'를 반드시 읽으세요. 백테스트일 뿐이고, 남는 위험이 있습니다.",
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
        st.plotly_chart(fig_equity, width="stretch", key="chart_flow_equity", config={"displayModeBar": False})
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
        st.plotly_chart(fig_signal, width="stretch", key="chart_flow_signal", config={"displayModeBar": False})
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


def _render_tab_decline():
    DECLINE_PATTERN_WINDOW = 20
    DECLINE_PATTERN_VOL_WINDOW = 20
    DECLINE_HORIZON = 10
    DECLINE_DRAWDOWN_THRESHOLD = 0.07

    _subheader_with_help(
        "큰폭 하락 조기 신호 (SK하이닉스 전용, 참고용)",
        "과거 SK하이닉스의 큰폭 하락(고점 대비 15% 이상) 8건을 분석해서 만든 조건(외국인 순매도 + 거래량 증가)이 실제로 "
        f"향후 {DECLINE_HORIZON}거래일 내 {DECLINE_DRAWDOWN_THRESHOLD:.0%} 이상 하락할 확률을 높이는지 백테스트했습니다.\n\n"
        "전체 기간(38.0% vs 24.2%, p=0.001)과, 데이터를 앞/뒤 절반으로 나눠 따로 검증했을 때도 방향이 일관되게 유지됐습니다 "
        "(뒷반기 p=0.003, 앞반기는 표본이 적어 유의하진 않았지만 같은 방향). 코스피 시가총액 상위 10개 종목 전체로는 "
        "일반화되지 않는 것으로 확인되어, SK하이닉스 개별 참고용으로만 제공합니다. 매매 신호가 아닙니다.",
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
                st.plotly_chart(fig_decline_foreign, width="stretch", key="chart_decline_foreign", config={"displayModeBar": False})

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
                st.plotly_chart(fig_decline_volume, width="stretch", key="chart_decline_volume", config={"displayModeBar": False})
                st.caption("차트 하단 슬라이더를 드래그하면 보고 싶은 기간만 확대해서 볼 수 있습니다.")
        except Exception as e:
            st.error(f"조기 신호 조회에 실패했습니다: {e}")

def _render_tab_rally():
    RALLY_PATTERN_WINDOW = 20
    RALLY_PATTERN_VOL_WINDOW = 20
    RALLY_HORIZON = 10
    RALLY_DRAWDOWN_THRESHOLD = 0.07

    _subheader_with_help(
        "큰폭 상승 조기 신호 (SK하이닉스 전용, 참고용)",
        "과거 SK하이닉스의 큰폭 상승(저점 대비 15% 이상) 16건을 분석해서 만든 조건(기관 순매수 + 개인 순매도 + 거래량 증가)이 "
        f"실제로 향후 {RALLY_HORIZON}거래일 내 하락/상승 확률에 영향을 주는지 백테스트했습니다.\n\n"
        "전체 기간으로는 하락 확률 12.8% vs 29.0%(p=0.003), 상승 확률 65.4% vs 50.4%(p=0.013)로 유의했지만, "
        "데이터를 앞/뒤 절반으로 나눠 각각 확인하면 뒷반기에서만 강하게 나타나고 앞반기에서는 방향이 뒤집히거나 "
        "무의미했습니다. 즉 특정 시기에 치우친 결과일 가능성이 높아 다른 조건들보다 신뢰도가 낮습니다. "
        "코스피 시가총액 상위 10개 종목 전체로도 일반화되지 않는 것으로 확인되어"
        "(특히 거래량 조건은 다른 종목에서 오히려 반대로 나타남), SK하이닉스 개별 참고용으로만 제공합니다. "
        "매매 신호가 아닙니다.",
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
                st.plotly_chart(fig_rally_flow, width="stretch", key="chart_rally_flow", config={"displayModeBar": False})

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
                st.plotly_chart(fig_rally_volume, width="stretch", key="chart_rally_volume", config={"displayModeBar": False})
                st.caption("차트 하단 슬라이더를 드래그하면 보고 싶은 기간만 확대해서 볼 수 있습니다.")
        except Exception as e:
            st.error(f"조기 신호 조회에 실패했습니다: {e}")

def _render_tab_dram():
    global dram_summary

    dram_summary = "해당 없음 (메모리 반도체 관련주가 아니라 DRAM 시세를 표시하지 않음)"

    _subheader_with_help(
        "DRAM 현물가 (모듈 + 칩)",
        "DRAMeXchange에 공시되는 DRAM 현물가입니다. 칩(반도체 단품)과 모듈(칩을 붙인 완제품) 가격을 각각 보여줍니다. "
        "현물가는 기업 간 장기 계약가(고정가)보다 먼저 움직이는 편이라, 메모리 업황의 선행 지표로 참고합니다.\n\n"
        "'변동률(%)'은 사이트가 표시하는 직전 갱신 대비 값이고, 'N일 전 대비'는 이 대시보드가 쌓아온 이력에서 "
        "슬라이더로 고른 기간만큼 이전 값과 비교한 것입니다. 그 기간만큼 이력이 없으면 N/A로 표시됩니다. "
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
                    f"사이트 기준 업데이트: {chip_last_update} (GMT+8)" if chip_last_update
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
                    f"사이트 기준 업데이트: {module_last_update} (GMT+8)" if module_last_update
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

                fig_capex = px.bar(
                    capex_df, x="분기말", y="capex_B", color="기업", barmode="stack",
                    labels={"capex_B": "Capex (10억달러)", "분기말": "분기"},
                    color_discrete_map=CAPEX_COMPANY_COLORS,
                )
                totals = capex_df.groupby("분기말")["capex_B"].sum().reset_index().sort_values("분기말")
                totals["qoq_pct"] = totals["capex_B"].pct_change() * 100
                fig_capex.add_trace(
                    go.Scatter(x=totals["분기말"], y=totals["capex_B"], name="합계", mode="lines+markers", line=dict(color="gray", dash="dot")),
                )
                fig_capex.add_trace(
                    go.Scatter(
                        x=totals["분기말"], y=totals["qoq_pct"], name="증감률",
                        mode="lines+markers", line=dict(color="#d62728"), yaxis="y2",
                    ),
                )
                _style_chart_mobile(fig_capex)
                fig_capex.update_layout(
                    yaxis=dict(title="Capex (10억달러)"),
                    yaxis2=dict(title="증감률(%)", overlaying="y", side="right", showgrid=False),
                    legend=dict(title=dict(text="")),
                )
                fig_capex.update_xaxes(rangeslider_visible=True)
                st.plotly_chart(fig_capex, width="stretch", key="chart_bigtech_capex", config={"displayModeBar": False})
                st.caption("차트 하단 슬라이더를 드래그하면 보고 싶은 기간만 확대해서 볼 수 있습니다.")

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
                st.plotly_chart(fig_sentiment, width="stretch", key="chart_sentiment", config={"displayModeBar": False})
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
                            st.markdown(f"{i}. [{pick['제목']}]({pick['url']}) — {pick['이유']}")
                    else:
                        st.info(st.session_state.get("dc_curation_raw", "").strip() or "조건에 맞는 분석글을 찾지 못했습니다.")
    except Exception as e:
        st.error(f"디시인사이드 주식갤러리 조회에 실패했습니다: {e}")

def _render_tab_ai():
    _subheader_with_help(
        "AI 분석: 오늘의 주가 변동 요인",
        "강세 근거와 약세 근거를 여섯 갈래로 나눠서, 각 갈래를 빠짐없이 훑도록 시킵니다.\n\n"
        "① **전자공시**(제목만이 아니라 본문 요지까지) ② 뉴스(종목 + 업종·전방수요·매크로) "
        "③ 애널리스트 리포트 ④ TrendForce 산업 리서치 + DRAM 현물가 "
        "⑤ 거시경제 지표(SOX·나스닥·달러인덱스·환율·미국 10년물 금리)와 ADR 괴리율 "
        "⑥ 대시보드 지표(수급·통합 신호·가격 과열도·선물 경보·컨센서스·동일업종 등락률)\n\n"
        "여기에 **오늘 장중 흐름**(언제 급했는지)과 **정규장 밖 움직임**(프리장·애프터장)을 같이 넘겨서, "
        "'오늘 이렇게 움직인 이유'를 시각 단위로 설명하게 합니다. 마감 직후 공시가 떠서 시간외에서 "
        "방향이 바뀌는 날은 종가만 봐서는 알 수 없기 때문입니다.\n\n"
        "각 근거에는 `[갈래] 내용 (근거 숫자)` 형태로 출처를 달게 하고, 매크로 숫자는 이 종목까지 "
        "어떻게 연결되는지 설명하게 합니다. 마지막에 어느 쪽 근거가 더 무거운지도 짚습니다.\n\n"
        "**커뮤니티 대세 반응**도 따로 정리합니다. 비율만 넘기면 '의견이 갈린다'는 뻔한 답이 나와서, "
        "네이버 종목토론방·디시 갤러리 게시글 **제목 원문**을 그대로 읽혀서 무엇을 기대하고 무엇을 "
        "걱정하는지 짚게 했습니다. 반어법이 많은 곳이라 문맥으로 읽으라고 지시해뒀고, "
        "여론이 데이터와 어긋나면 그 점도 지적하게 합니다. 커뮤니티 탭을 꺼놔도 들어갑니다.\n\n"
        "동일업종 비교가 들어가면서 '오늘 움직임이 이 종목만의 이슈인지, 업종 전체가 같이 움직인 것인지'를 "
        "구분할 수 있게 됐습니다. 위쪽 지표는 AI를 돌리지 않아도 바로 보입니다.\n\n"
        "버튼을 누를 때만 실행됩니다(자동 갱신 없음). AI가 잘못 짚거나 지어낼 수 있으니, "
        "투자 조언이 아닌 참고용 정리로만 보세요. 컨센서스 목표주가는 증권사 기대치일 뿐 보장이 아닙니다.",
        key="ai",
    )

    # AI를 돌리지 않아도 판단 재료는 바로 보이게 한다 (컨센서스·밸류에이션은 다른 탭에 없던 정보다)
    snapshot = None
    try:
        snapshot = fetch_stock_snapshot(TICKER)
    except Exception:
        snapshot = None

    if snapshot:
        target = _to_number(snapshot.get("목표주가"))
        cur = _to_number(st.session_state.get("current_price_value"))
        col1, col2, col3, col4 = st.columns(4)
        with col1.container(key="metric_small_ai_target"):
            upside = f"{(target / cur - 1) * 100:+.0f}% 여력" if (target and cur) else None
            _metric_with_help(
                "컨센서스 목표주가", f"{target:,.0f}원" if target else "N/A",
                f"증권사 평균 목표주가 (기준일 {snapshot.get('컨센서스일자') or '-'}). "
                "기대치일 뿐 보장이 아닙니다.",
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
            st.metric("52주 고/저", f"{snapshot.get('52주최고') or '-'} / {snapshot.get('52주최저') or '-'}")

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
                    "주가가 왜 움직였는지에 해당하는 재료를 같이 넘깁니다. 수집에 몇 초 더 걸립니다."
                )
        if st.button("지금 바로 분석하기"):
            try:
                price_summary = st.session_state.get("current_price_summary", "현재가 데이터를 가져오지 못함")

                if not investor_df.empty:
                    recent = investor_df.tail(5)
                    # 순매수 열만 돈다. 같은 표에 거래량·종가가 붙어 있어서 전체 열을 돌면
                    # '거래량 순매수 합계' 같은 말이 안 되는 줄이 생긴다.
                    lines = [
                        f"- 최근 {len(recent)}거래일 {col} 순매수 합계: {recent[col].sum():+,.0f}주"
                        for col in INVESTOR_COLUMNS if col in recent.columns
                    ]
                    # 거래량은 수급보다 먼저 확정되므로 빠른 경로를 우선 쓴다
                    try:
                        _ohlcv = fetch_daily_ohlcv(TICKER, DEFAULT_LOOKBACK_DAYS)
                    except Exception:
                        _ohlcv = pd.DataFrame()
                    _vol_src = _ohlcv if not _ohlcv.empty else investor_df
                    if "거래량" in _vol_src.columns and _vol_src["거래량"].notna().any():
                        vols = _vol_src["거래량"].astype(float)
                        lines.append(
                            f"- 절대 거래량: 최근 {vols.iloc[-1]:,.0f}주, "
                            f"기간 평균 {vols.mean():,.0f}주 대비 {vols.iloc[-1] / vols.mean() - 1:+.0%} "
                            f"(기간 최대 {vols.max():,.0f}주)"
                        )
                    supply_summary = "\n".join(lines)
                else:
                    supply_summary = "수급 데이터를 가져오지 못함"

                news_items = fetch_news_with_summary(STOCK_NAME)
                headlines = [n["제목"] for n in news_items]
                news_md = "\n".join(
                    f"- {n['제목']}" + (f"\n  요약: {n['요약']}" if n["요약"] else "") for n in news_items
                )

                snap = snapshot or {}
                snapshot_lines = [
                    f"- 컨센서스 목표주가: {snap.get('목표주가') or 'N/A'}원 "
                    f"(투자의견 평균 {snap.get('투자의견') or 'N/A'}/5, 기준일 {snap.get('컨센서스일자') or '-'})",
                    f"- PER {snap.get('PER') or 'N/A'} · EPS {snap.get('EPS') or 'N/A'}",
                    f"- 52주 최고 {snap.get('52주최고') or 'N/A'} / 최저 {snap.get('52주최저') or 'N/A'}",
                    f"- 시가총액 {snap.get('시가총액') or 'N/A'} · 외국인소진율 {snap.get('외국인소진율') or 'N/A'}",
                ]
                if snap.get("동일업종"):
                    snapshot_lines.append("- 동일업종 오늘 등락률: " + ", ".join(
                        f"{p['종목']} {p['등락률']}%" for p in snap["동일업종"][:6] if p.get("등락률")
                    ))
                if snap.get("수급추이"):
                    snapshot_lines.append("- 최근 투자자별 순매수(주) / 외국인 보유율:")
                    for f in snap["수급추이"][:5]:
                        snapshot_lines.append(
                            f"    {f['날짜']} 종가 {f['종가']} | 개인 {f['개인']} · 외국인 {f['외국인']} · 기관 {f['기관']}"
                            f" | 외국인보유율 {f['외국인보유율']}"
                        )
                snapshot_md = "\n".join(snapshot_lines)

                reports_df = fetch_analyst_reports(TICKER)
                reports_md = "\n".join(
                    f"- [{row['증권사']}] {row['제목']} ({row['작성일']})" for _, row in reports_df.iterrows()
                )
                trendforce_df = fetch_trendforce_news()
                trendforce_md = "\n".join(
                    f"- {row['제목']} ({row['날짜']})" for _, row in trendforce_df.iterrows()
                )

                # 매크로와 업종 뉴스는 '왜 움직였나'를 짚는 데 필요한 재료라, 실패해도 분석은 계속한다
                try:
                    macro_md = fetch_macro_summary()
                except Exception:
                    macro_md = ""
                sector_news_md = fetch_sector_news(STOCK_NAME) if use_search else ""

                # 공시 · 장중 흐름 · 시간외. 종가 한 줄로는 안 보이는 것들이라 개별로 실패를 감싼다.
                try:
                    disclosure_md = fetch_disclosures(TICKER)
                except Exception:
                    disclosure_md = ""
                cur_close = _to_number(st.session_state.get("current_price_value"))
                cur_close = int(cur_close) if cur_close else None
                try:
                    intraday_md = build_intraday_summary(TICKER, cur_close)
                except Exception:
                    intraday_md = ""
                try:
                    over_market_md = build_over_market_summary(TICKER, cur_close)
                except Exception:
                    over_market_md = ""
                try:
                    market_flow_md = build_market_flow_summary()
                except Exception:
                    market_flow_md = ""
                # 커뮤니티 탭은 기본으로 꺼져 있어서 탭이 채워주는 전역값에 기댈 수 없다.
                # 여기서 직접 모아 제목 원문까지 넘긴다.
                try:
                    community_md = build_community_summary(TICKER, STOCK_NAME)
                except Exception:
                    community_md = ""

                adr_md = ""
                if TICKER == ADR_HOST_TICKER:
                    try:
                        adr_q, adr_base = fetch_adr_quote(), fetch_adr_baseline()
                        cur_price = _to_number(st.session_state.get("current_price_value"))
                        if adr_q and cur_price:
                            per_share = adr_q["price"] * adr_q["fx"] / ADR_SHARE_RATIO
                            gap = (per_share / cur_price - 1) * 100
                            adr_md = (
                                f"- 나스닥 SKHY ${adr_q['price']:,.2f} ({adr_q['session']}), "
                                f"본주 환산 {per_share:,.0f}원 → 괴리율 {gap:+.1f}%"
                            )
                            if adr_base:
                                base_gap = (adr_base / ADR_SHARE_RATIO - 1) * 100
                                adr_md += (
                                    f"\n- 최근 20일 평균 괴리율 {base_gap:+.1f}% 대비 {gap - base_gap:+.1f}%p."
                                    " 이 종목은 평소에도 30–40% 프리미엄이 붙으므로 절대값이 아니라"
                                    " 평균 대비 벌어진 정도로 읽어야 한다."
                                )
                    except Exception:
                        adr_md = ""

                time_label = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
                analysis, search_note = generate_ai_analysis(
                    f"{STOCK_NAME}({TICKER})",
                    time_label, price_summary, supply_summary, headlines, reports_md,
                    dram_summary, community_md, composite_summary, overheat_summary, futures_summary,
                    trendforce_md, snapshot_md, news_md, use_search,
                    macro_md, sector_news_md, adr_md,
                    disclosure_md, over_market_md, intraday_md, market_flow_md,
                )

                st.session_state["ai_search_note"] = search_note
                st.session_state["ai_analysis"] = analysis
                st.session_state["ai_analysis_time"] = time_label
                st.session_state["ai_analysis_headlines"] = headlines
                st.session_state["ai_analysis_reports"] = reports_df
                st.session_state["ai_analysis_trendforce"] = trendforce_df
                st.session_state["ai_analysis_macro"] = macro_md
                st.session_state["ai_analysis_sector_news"] = sector_news_md
                st.session_state["ai_analysis_adr"] = adr_md
                st.session_state["ai_analysis_disclosure"] = disclosure_md
                st.session_state["ai_analysis_over_market"] = over_market_md
                st.session_state["ai_analysis_intraday"] = intraday_md
                st.session_state["ai_analysis_market_flow"] = market_flow_md
                st.session_state["ai_analysis_community"] = community_md
            except Exception as e:
                st.error(f"AI 분석 생성에 실패했습니다: {e}")

        if "ai_analysis" in st.session_state:
            st.caption(f"기준 시각: {st.session_state['ai_analysis_time']}")
            note = st.session_state.get("ai_search_note")
            if note == "search_ok":
                st.success("구글 검색으로 최신 정보를 보강해 분석했습니다.")
            elif note:
                st.warning(note)
            st.markdown(st.session_state["ai_analysis"])

            with st.expander("분석에 사용된 원본 데이터 보기"):
                for label, key in (("오늘 장중 흐름", "ai_analysis_intraday"),
                                   ("정규장 밖 움직임 (프리장·애프터장)", "ai_analysis_over_market"),
                                   ("코스피 시장 전체 수급", "ai_analysis_market_flow"),
                                   ("커뮤니티 게시글", "ai_analysis_community"),
                                   ("전자공시", "ai_analysis_disclosure")):
                    value = st.session_state.get(key)
                    if value:
                        st.write(f"**{label}**")
                        st.text(value)
                st.write("**뉴스 헤드라인**")
                st.write(st.session_state["ai_analysis_headlines"])
                sector_news = st.session_state.get("ai_analysis_sector_news")
                if sector_news:
                    st.write("**업종 · 전방수요 · 매크로 뉴스**")
                    st.text(sector_news)
                st.write("**거시경제 지표**")
                st.text(st.session_state.get("ai_analysis_macro") or "수집 실패")
                adr_used = st.session_state.get("ai_analysis_adr")
                if adr_used:
                    st.write("**해외 상장분(ADR) 괴리율**")
                    st.text(adr_used)
                st.write("**애널리스트 리포트**")
                st.table(st.session_state["ai_analysis_reports"], width="stretch", hide_index=True)
                st.write("**해외 반도체 산업 리서치 뉴스 (TrendForce)**")
                trendforce_hist_df = st.session_state.get("ai_analysis_trendforce", pd.DataFrame())
                if trendforce_hist_df.empty:
                    st.caption("수집된 자료 없음")
                else:
                    st.dataframe(
                        trendforce_hist_df, width="stretch", hide_index=True,
                        column_config={"url": st.column_config.LinkColumn("링크")},
                    )
        else:
            st.info("버튼을 눌러 AI 분석을 생성해주세요.")


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
    "커뮤니티": _render_tab_community,
    "AI 분석": _render_tab_ai,
}

# 사이드바에서 숨기지 않은(선택된) 탭만 실제로 렌더링한다. 숨겨진 탭은 함수 자체가
# 호출되지 않으므로 데이터 조회도 일어나지 않는다.
for _label in _visible_tab_labels:
    with _tab_map[_label]:
        _TAB_RENDERERS[_label]()

