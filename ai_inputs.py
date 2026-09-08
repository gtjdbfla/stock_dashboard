"""AI 분석에 넣을 재료를 만드는 계층. **streamlit을 import하지 않는다.**

대시보드(app.py)와 수집기(collector.py)가 같이 쓴다. 원래는 전부 app.py 안에 있었고,
그래서 AI 분석은 브라우저 세션이 열려 있을 때만 만들어졌다. Streamlit은 화면이 붙어야
스크립트를 돌리기 때문이다. 아무도 대시보드를 안 열어 둔 시각은 통째로 건너뛰었다
(실측: 2026-09-07 21:32 다음 생성이 11시간 뒤였다).

이제 24시간 도는 수집기가 정해진 시각에 만들고, 화면은 저장된 결과를 읽기만 한다.
리포트·공시·재무 요약이 쓰는 방식과 같다.

app.py 쪽 사정 두 가지를 여기서 흡수한다:
  - `@st.cache_data`는 화면의 반응성을 위한 것이라 여기엔 없다. app.py가 import한 뒤
    같은 ttl로 다시 씌운다(app.py의 _CACHED 표를 볼 것). 수집기는 캐시 없이 그냥 부른다.
  - 화면이 session_state에 남기던 장 상태·현재가는 build_price_context()가 대신 만든다.
"""
import csv
import datetime as dt
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from google import genai

import analyst_digest
import analyst_targets
import disclosure
import financial_digest
import fnguide
import over_market as om


DEFAULT_TICKER = "000660"


NAVER_FRGN_URL = "https://finance.naver.com/item/frgn.naver"


NAVER_NEWS_URL = "https://search.naver.com/search.naver"


NAVER_RESEARCH_URL = "https://finance.naver.com/research/company_list.naver"


NAVER_BOARD_URL = "https://finance.naver.com/item/board.naver"


YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


DRAMEXCHANGE_URL = "https://www.dramexchange.com/"


DC_GALLERY_ID = "krstock"


DC_GALLERY_LIST_URL = "https://gall.dcinside.com/mgallery/board/lists/"


DC_GALLERY_VIEW_URL = "https://gall.dcinside.com/mgallery/board/view/"


DC_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Referer": f"https://gall.dcinside.com/mgallery/board/lists/?id={DC_GALLERY_ID}",
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


INVESTOR_COLUMNS = ("개인", "외국인", "기관")
DEFAULT_LOOKBACK_DAYS = 30


DECLINE_PATTERN_WINDOW = 20


DECLINE_PATTERN_VOL_WINDOW = 20


DECLINE_HORIZON = 10


DECLINE_DRAWDOWN_THRESHOLD = 0.07


RALLY_PATTERN_WINDOW = 20


RALLY_PATTERN_VOL_WINDOW = 20


RALLY_HORIZON = 10


RALLY_DRAWDOWN_THRESHOLD = 0.07


DEFAULT_COMMUNITY_POST_COUNT = 60


GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")


# 모델마다 무료 한도가 따로 잡힌다. 그래서 체인에 온전한 모델을 하나 더 두면 기본
# 모델이 소진돼도 품질을 지킨 채로 넘어갈 수 있다 — 예전 체인은 3.6이 떨어지는 즉시
# flash-lite로 내려갔고, 실제로 2026-09-08 13시에 그 일이 일어났다.
# gemini-3.7-flash는 실제 프롬프트(18,547자)로 재봤다: 첫 글자 13.4초, 완료 23.8초,
# 2,757자, 10개 섹션·7개 출처 갈래를 모두 채우고 근거 인용 14건.
# gemini-3.8-flash는 넣지 않는다. 짧은 프롬프트에는 답하지만 실제 프롬프트로는
# 두 번 다 "currently experiencing high demand"만 돌려줬다.
GEMINI_FALLBACK_MODELS = [
    m.strip() for m in os.environ.get(
        "GEMINI_FALLBACK_MODELS",
        "gemini-3.7-flash,gemini-3.5-flash,gemini-3.5-flash-lite,gemini-flash-lite-latest"
    ).split(",") if m.strip()
]


GEMINI_PRIMARY_RETRIES = int(os.environ.get("GEMINI_PRIMARY_RETRIES", "1"))


def _is_quota_error(exc: Exception) -> bool:
    return "RateLimit" in type(exc).__name__ or "429" in str(exc)


def _gemini_models_to_try() -> list[str]:
    order = ([GEMINI_MODEL] * (1 + GEMINI_PRIMARY_RETRIES)
             + [m for m in GEMINI_FALLBACK_MODELS if m != GEMINI_MODEL])
    today = dt.datetime.now(om.KST).date()
    fresh = [m for m in order if _GEMINI_EXHAUSTED.get(m) != today]
    # 전부 소진으로 기록돼 있으면 그래도 한 번씩은 시도한다 (한도가 이미 풀렸을 수 있다)
    return fresh or order


def _mark_exhausted(model: str) -> None:
    _GEMINI_EXHAUSTED[model] = dt.datetime.now(om.KST).date()


GEMINI_THINKING_LEVEL = os.environ.get("GEMINI_THINKING_LEVEL", "high").strip()


def _gemini_gen_config() -> dict:
    return {"generation_config": {"thinking_level": GEMINI_THINKING_LEVEL}} if GEMINI_THINKING_LEVEL else {}


GEMINI_MIN_USABLE = int(os.environ.get("GEMINI_MIN_USABLE", "1000"))


def _looks_truncated(text: str) -> bool:
    """답이 중간에 끊겼는지. 마지막 섹션이 없으면 끊긴 것으로 본다.

    모델이 스트림을 예고 없이 끝내는 일이 있다(실제로 "52주 최고가 대비 -44.1"에서 끊기고
    마지막 섹션이 통째로 사라졌다). 잘린 걸 멀쩡한 답처럼 보여주면 안 되니 화면에 알린다.
    """
    if not text.strip():
        return False        # 아예 빈 응답은 다른 경로에서 처리한다
    return "앞으로 확인할 것" not in text


def _is_thinking_unsupported(exc: Exception) -> bool:
    """thinking_level을 못 받는 모델이 낸 400인지.

    기본 모델 이름이 gemini-flash-latest 라는 '별칭'이라, 구글이 가리키는 대상을 바꾸면
    이 필드를 안 받는 모델이 될 수 있다. 그때 분석이 통째로 죽지 않게 한 번은 빼고 재시도한다.
    """
    return "thinking_level" in str(exc)


GEMINI_STALL_SEC = float(os.environ.get("GEMINI_STALL_SEC", "120"))


STREAM_PAINT_SEC = float(os.environ.get("STREAM_PAINT_SEC", "0.4"))


def _is_timeout(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    return "timeout" in name or "timeout" in str(exc).lower()


def _stream_gemini(prompt: str, used: dict):
    """모델을 순서대로 시도하며 응답 조각을 흘려준다.

    한 번에 다 받으면 그동안 화면이 멈춘 것처럼 보인다. 흘려보내면 첫 문장이 곧바로 뜬다.
    thinking_level까지 낮춘 뒤 실측(프롬프트 12,400자): 첫 글자 5.6초 / 완료 12.2초.
    실제로 답한 모델명은 used["model"]에 넣어 호출부에 알린다.
    """
    client = genai.Client()
    tried: list[str] = []
    last_exc: Exception | None = None
    order = _gemini_models_to_try()
    # 오늘 이미 한도에 걸린 걸로 기억해서 아예 부르지도 않은 모델들. 이유를 함께 남겨야
    # "왜 기본 모델을 안 썼나"를 화면에서 온전히 설명할 수 있다.
    for skipped_model in [GEMINI_MODEL] + GEMINI_FALLBACK_MODELS:
        if skipped_model not in order:
            used.setdefault("skipped", []).append((skipped_model, "오늘 무료 한도 소진"))
    for pos, model in enumerate(order):
        tried.append(model)
        produced = False
        got: list[str] = []            # 이 모델이 뱉은 것만 따로 모아 품질을 본다
        for cfg in (_gemini_gen_config(), {}):
            try:
                stream_error = None
                for event in client.interactions.create(
                        model=model, input=prompt, stream=True,
                        timeout=GEMINI_STALL_SEC, **cfg):
                    # 스트림은 예외 대신 ErrorEvent 하나를 흘리고 조용히 끝나기도 한다.
                    # 그 메시지를 안 읽으면 한도 소진과 일시 과부하가 똑같이 '빈 응답'으로
                    # 기록돼서, 기다리면 되는 건지 모델을 바꿔야 하는 건지 알 수 없다.
                    err = getattr(event, "error", None)
                    if err is not None:
                        stream_error = getattr(err, "message", None) or str(err)
                    delta = getattr(event, "delta", None)
                    text = getattr(delta, "text", None) if delta is not None else None
                    if text:
                        produced = True
                        used["model"] = model
                        got.append(text)
                        yield text
                if produced:
                    whole = "".join(got)
                    # 짧은 데다 마지막 섹션까지 없으면 쓸 수 없는 답이다. 남은 모델이 있으면
                    # 그 답을 버리고 다시 시작한다(_RESTART로 화면에 흘린 글자도 지운다).
                    if (_looks_truncated(whole) and len(whole) < GEMINI_MIN_USABLE
                            and pos + 1 < len(order)):
                        used.setdefault("skipped", []).append(
                            (model, f"답이 {len(whole)}자에서 잘려 버림"))
                        used.pop("model", None)
                        yield _RESTART
                        break
                    return
                # 예외 없이 조각을 하나도 안 준 경우다. 이것도 건너뛴 이유로 남겨야 한다.
                # 안 남기면 화면에 "(사용 불가)"라고만 떠서 왜 다른 모델을 썼는지 알 수 없다.
                if stream_error and "quota" in stream_error.lower():
                    _mark_exhausted(model)
                    used.setdefault("skipped", []).append((model, "오늘 무료 한도 소진"))
                else:
                    used.setdefault("skipped", []).append(
                        (model, stream_error[:80] if stream_error else "빈 응답"))
                break
            except Exception as exc:
                last_exc = exc
                # 이미 글자가 나간 뒤 끊기면 다른 모델로 다시 시작할 수 없다. 그대로 올린다.
                if produced:
                    raise
                # thinking_level을 못 받는 모델이면 그 옵션만 빼고 같은 모델로 한 번 더
                if cfg and _is_thinking_unsupported(exc):
                    continue
                if _is_quota_error(exc):
                    _mark_exhausted(model)
                    used.setdefault("skipped", []).append((model, "오늘 무료 한도 소진"))
                    break
                # 말문을 못 떼고 붙잡고 있는 모델은 버리고 다음 모델로 간다.
                # 오늘 소진으로 기록하지는 않는다. 한도가 아니라 그때그때의 지연이다.
                if _is_timeout(exc):
                    used.setdefault("skipped", []).append(
                        (model, f"{GEMINI_STALL_SEC:.0f}초 동안 응답 없음"))
                    break
                raise
    raise RuntimeError(
        f"사용 가능한 모델을 찾지 못했습니다. 시도한 모델: {', '.join(tried)}. "
        f"마지막 오류: {type(last_exc).__name__ if last_exc else '응답 없음'}"
    )


def _call_gemini(prompt: str, tools: list | None = None) -> tuple[str, str]:
    """모델 하나가 한도에 걸리면 다음 모델로 넘어가며 호출한다.

    반환: (응답 텍스트, 실제로 사용된 모델명)
    한도 외의 오류(잘못된 요청 등)는 모델을 바꿔도 소용없으므로 바로 올린다.
    """
    client = genai.Client()
    tried: list[str] = []
    last_exc: Exception | None = None
    for model in _gemini_models_to_try():
        tried.append(model)
        for cfg in (_gemini_gen_config(), {}):
            try:
                kwargs = {"tools": tools} if tools else {}
                kwargs.update(cfg)
                # 이쪽은 스트리밍이 아니라 답을 다 만들 때까지 기다려야 한다.
                # 그래서 상한을 넉넉히 잡는다(정상 생성이 30초대까지 걸린다).
                interaction = client.interactions.create(
                    model=model, input=prompt, timeout=GEMINI_STALL_SEC * 4, **kwargs)
                return (interaction.output_text or "", model)
            except Exception as exc:
                last_exc = exc
                if cfg and _is_thinking_unsupported(exc):
                    continue
                if _is_quota_error(exc):
                    _mark_exhausted(model)
                    break
                if _is_timeout(exc):    # 붙잡고 있는 모델은 버리고 다음 모델로
                    break
                raise
    raise RuntimeError(
        f"사용 가능한 모델을 찾지 못했습니다. 시도한 모델: {', '.join(tried)}. "
        f"마지막 오류: {type(last_exc).__name__}"
    ) from last_exc


def fetch_current_price(ticker: str) -> dict:
    return om.fetch_current_price_raw(ticker)


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


MACRO_SYMBOLS = [
    ("필라델피아 반도체지수(SOX)", "%5ESOX", "pct"),
    ("나스닥", "%5EIXIC", "pct"),
    ("달러인덱스(DXY)", "DX-Y.NYB", "pct"),
    ("원/달러 환율", "KRW=X", "pct"),
    # 금리는 그 자체가 %라, 변동을 %가 아니라 %p로 봐야 말이 된다
    ("미국 10년물 금리", "%5ETNX", "pp"),
]


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
ADR_SHARE_RATIO = float(os.environ.get("ADR_SHARE_RATIO", "0.1"))


ADR_BASELINE_DAYS = 20


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


def fetch_adr_quote() -> dict | None:
    """SKHY 최신 체결가를 프리장/애프터장까지 포함해서 가져온다.

    includePrePost=true 로 1분봉을 받으면 미국 정규장 밖(프리장 04:00 ET – 애프터 20:00 ET)
    체결도 들어온다. 한국 장이 열리기 전 미국 시간외 움직임을 보는 게 이 지표의 핵심이라
    마지막 유효 체결가를 그대로 쓴다.
    """
    try:
        # 봉과 환율은 서로 기다릴 이유가 없다. 이 함수는 5초짜리 화면 조각 안에서 불리므로
        # 왕복 한 번을 줄이는 것도 체감에 바로 들어온다.
        with ThreadPoolExecutor(max_workers=2) as pool:
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


def _clean_text(text: str) -> str:
    """<mark> 때문에 생기는 이중 공백만 정리한다. 낱말 사이 한 칸은 남긴다."""
    return re.sub(r"\s{2,}", " ", text).strip()


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
        # 네이버는 검색어에 <mark>를 씌운다. strip=True를 주면 태그 안팎 조각을 각각
        # 다듬은 뒤 공백 없이 붙여서 "반도체 수출209% 증가"가 된다(검색어는 늘 강조되므로
        # 사실상 모든 제목이 해당된다). 구분자를 주면 이번엔 "반도체 주"처럼 없던 공백이
        # 생긴다. strip 없이 뽑아야 태그 사이의 원문 공백이 그대로 남는다.
        title = _clean_text(head.get_text())
        summary = ""
        when = ""
        node = head
        # 헤드라인에서 위로 올라가며 요약문과 게재 시점을 뽑는다.
        # 둘이 서로 다른 높이에 있어서, 요약을 찾자마자 멈추면 시점을 놓친다(실제로 0건이 나왔다).
        # 둘 다 채워질 때까지 계속 올라간다.
        for _ in range(8):
            node = node.parent
            if node is None:
                break
            if not when:
                for sp in node.select("span"):
                    t = sp.get_text(strip=True)
                    if _NEWS_WHEN_PAT.match(t):
                        when = t
                        break
            if not summary:
                bodies = [_clean_text(b.get_text())
                          for b in node.select("span.sds-comps-text-type-body1")]
                bodies = [b for b in bodies if len(b) > 40 and b != title]
                if bodies:
                    summary = bodies[0]
            if when and summary:
                break
        items.append({"제목": title, "요약": summary, "시점": when})
    return items


NAVER_INVESTOR_TREND_URL = "https://finance.naver.com/sise/investorDealTrendDay.naver"


NAVER_PROGRAM_TREND_URL = "https://finance.naver.com/sise/programDealTrendDay.naver"


def fetch_market_flow() -> dict | None:
    """코스피 전체 투자자별 순매수 + 프로그램 매매(차익/비차익)를 한 번에.

    반환값의 is_today로 '장중 잠정치'인지 '직전 거래일 확정치'인지 구분한다.
    """
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
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


def fetch_foreign_desk(ticker: str) -> dict:
    """거래원 표에서 외국계추정합과 매도·매수 상위 증권사를 뽑는다."""
    r = requests.get(NAVER_FRGN_URL, params={"code": ticker},
                     headers={"User-Agent": "Mozilla/5.0",
                              "Referer": "https://finance.naver.com/"}, timeout=10)
    r.raise_for_status()
    r.encoding = "euc-kr"
    table = None
    for tb in pd.read_html(StringIO(r.text)):
        if list(tb.columns)[:2] == ["매도상위", "거래량"]:
            table = tb
            break
    if table is None:
        return {}

    hit = table[table["매도상위"].astype(str).str.contains("외국계추정합", na=False)]
    if hit.empty:
        return {}
    row = hit.iloc[0]
    sell, buy = _to_number(row["거래량"]), _to_number(row["거래량.1"])
    if sell is None or buy is None:
        return {}

    # 상위 증권사 목록. '외국계추정합' 줄과 빈 줄을 걷어낸다.
    brokers = table[~table["매도상위"].astype(str).str.contains("외국계추정합", na=False)]
    brokers = brokers.dropna(subset=["매도상위", "매수상위"])

    # 페이지에 찍힌 시각. 다만 이건 '시세' 블록의 시각이고, 거래원 표에는 자체 시각 표기가 없다.
    # 거래원 값이 정확히 언제 것인지는 네이버가 밝히지 않으므로 '조회 시각'으로만 쓴다.
    stamp = None
    m = re.search(r"(\d{2})시\s*(\d{2})분\s*기준", re.sub(r"<[^>]+>", " ", r.text))
    if m:
        stamp = f"{m.group(1)}:{m.group(2)}"
    return {"매도": sell, "매수": buy, "순매수": buy - sell,
            "기준": stamp, "상위": brokers[["매도상위", "거래량", "매수상위", "거래량.1"]]}


def build_foreign_desk_summary(ticker: str) -> str:
    """AI 분석에 넘길 한 덩어리. 추정치라는 점을 반드시 함께 넘긴다."""
    try:
        d = fetch_foreign_desk(ticker)
    except Exception:
        return ""
    if not d:
        return ""
    lines = [
        f"- 외국계 창구 추정 순매수 {d['순매수']:+,.0f}주"
        f" (매수 {d['매수']:,.0f}주 / 매도 {d['매도']:,.0f}주"
        + (f", {d['기준']} 조회" if d.get("기준") else "") + ")",
        "- 이 값은 외국계 증권사 창구를 거친 거래만 합산한 추정치다. 국내 증권사로 주문한"
        " 외국인은 빠지고 외국계 창구를 쓴 내국인은 섞이므로, 마감 후 확정 수급과 다르다.",
        "- 기관·개인은 장중에 종목별로 공개되지 않는다. 이 값으로 기관이나 개인의 매매를 추측하지 마라.",
    ]
    top = d.get("상위")
    if top is not None and not top.empty:
        sell_top = " · ".join(f"{r['매도상위']} {r['거래량']:,.0f}" for _, r in top.head(3).iterrows())
        buy_top = " · ".join(f"{r['매수상위']} {r['거래량.1']:,.0f}" for _, r in top.head(3).iterrows())
        lines.append(f"- 매도 상위 창구: {sell_top}")
        lines.append(f"- 매수 상위 창구: {buy_top}")
    return "\n".join(lines)


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
        with ThreadPoolExecutor(max_workers=min(len(recent), 6)) as pool:
            bodies = dict(pool.map(body, recent))

    lines = []
    for d in items:
        when = str(d.get("datetime", "")).replace("T", " ")[:16]
        lines.append(f"- {when}  {d.get('title')}")
        text = bodies.get(d.get("disclosureId"))
        if text:
            lines.append(f"    본문: {text}")
    return "\n".join(lines)


def build_community_summary(ticker: str, stock_name: str, titles_per_source: int = 20) -> str:
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


SECTOR_NEWS_QUERIES = [
    "{name} HBM",
    "메모리 반도체 업황",
    "D램 가격",
    "엔비디아 실적 AI 반도체",
    "반도체 수출 실적",
    # 네이버 리서치는 13개 증권사만 싣는다. 여기 없는 증권사(예: LS증권)가 목표주가를
    # 크게 바꿔도 리포트 목록에는 안 잡히는데, 기사로는 당일 나온다. 그 구멍을 이 질의가 메운다.
    # 문구는 재보고 골랐다. 2026-08-31 LS증권 하향(330만->240만) 기준으로
    # '투자의견'은 관련 기사 3건을 제목에서 바로 잡았고, '목표주가'/'증권가 리포트'는
    # 0건이었다(네이버 뉴스가 관련도순이 아니라 최신순이라 시황 기사에 밀린다).
    "{name} 투자의견",
]


def fetch_sector_news(stock_name: str, per_query: int = 3) -> str:
    """업종·전방수요·매크로 관련 뉴스를 질의별로 모아 하나의 마크다운으로 만든다.

    구글 검색 grounding이 무료 요금제에서 막혀 있어(429), '검색으로 보강'은 실제로는
    한 번도 동작한 적이 없다. 대신 네이버 뉴스에 질의를 여러 개 던져서 같은 목적
    - 대시보드에 없는 바깥 소식을 채우는 것 - 을 실제로 달성한다.
    """
    queries = [t.format(name=stock_name) for t in SECTOR_NEWS_QUERIES]

    def one(query: str):
        # 투자의견 질의는 한 건 더 받는다. 목표주가를 바꾼 날은 관련 기사가 여러 개 쏟아지는데,
        # 상위 3건이 시황 기사로 채워지면 정작 숫자가 든 제목('330만->240만')이 밀려난다.
        n = per_query + 1 if query.endswith("투자의견") else per_query
        try:
            return query, fetch_news_with_summary(query, count=n)
        except Exception:
            return query, []

    # 질의를 순서대로 던지면 왕복 지연이 그대로 쌓인다. 한꺼번에 보내고 결과만 순서대로 정리한다.
    # 워커가 캐시된 fetch_news_with_summary를 부르므로 컨텍스트를 붙인 풀을 쓴다.
    with ThreadPoolExecutor(max_workers=len(queries)) as pool:
        fetched = list(pool.map(one, queries))

    blocks = []
    seen_titles: set[str] = set()
    for query, items in fetched:
        rows = []
        for it in items:
            if it["제목"] in seen_titles:      # 질의끼리 겹치는 기사는 한 번만
                continue
            seen_titles.add(it["제목"])
            # 업종 뉴스는 질의 5개 x 3건이라 요약까지 넣으면 프롬프트가 2,700자 불어난다.
            # 프롬프트가 길수록 글자가 나오는 속도가 급격히 느려져서(실측 16,600자에서 34자/초,
            # 14,700자에서 130자/초) 여기는 제목만 넘긴다. 종목 뉴스는 요약을 그대로 둔다.
            rows.append(f"- {it['제목']}"
                        + (f" [{it['시점']}]" if it.get("시점") else " [게재 시점 미확인]"))
        if rows:
            blocks.append(f"[검색어: {query}]\n" + "\n".join(rows))
    return "\n\n".join(blocks)


TRENDFORCE_SEMICONDUCTOR_URL = "https://www.trendforce.com/research/category/Semiconductors"


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


def fetch_analyst_reports(ticker: str, count: int = 5) -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(
        NAVER_RESEARCH_URL, params={"searchType": "itemCode", "itemCode": ticker}, headers=headers, timeout=10
    )
    resp.encoding = "euc-kr"
    tables = pd.read_html(StringIO(resp.text))
    df = tables[0].dropna(subset=["제목"]).copy()
    df["작성일"] = pd.to_datetime(df["작성일"], format="%y.%m.%d").dt.strftime("%Y-%m-%d")
    cols = ["제목", "증권사", "작성일"]
    # 원문 PDF 링크. 표에는 없어서 같은 페이지의 행을 직접 훑어 제목과 짝지어 붙인다.
    # 링크 개수만 세어 순서대로 붙이면 첨부가 없는 리포트에서 한 칸씩 밀린다(실제로 30행 28링크였다).
    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        link_by_title = {}
        for tr in soup.select("tr"):
            title_el = tr.select_one("td.file + td, td a[href*='company_read']") or tr.select_one("a")
            pdf = tr.select_one("a[href$='.pdf']")
            title = (title_el.get_text(strip=True) if title_el else "")
            if title and pdf:
                link_by_title[title] = pdf["href"]
        if link_by_title:
            df["url"] = df["제목"].map(lambda t: link_by_title.get(str(t).strip()))
            cols.append("url")
    except Exception:
        pass
    if "조회수" in df.columns:
        cols.append("조회수")
    return df[cols].head(count)


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


def _fetch_dram_soup() -> BeautifulSoup:
    """모듈가·칩가가 같은 페이지에 있으므로, 페이지 요청 자체는 한 번만 캐시해서 공유한다."""
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(DRAMEXCHANGE_URL, headers=headers, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


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


BIGTECH_CIKS = {
    "Alphabet(Google)": "0001652044",
    "Amazon": "0001018724",
    "Meta": "0001326801",
    "Microsoft": "0000789019",
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


def fetch_fnguide_page(ticker: str) -> str:
    return fnguide.fetch_page(ticker)


def fetch_short_balance(ticker: str) -> pd.DataFrame:
    """차입공매도 비중 주간 추이.

    예전에 KRX·KOFIA·세이브로를 다 뒤지고 '무료로는 못 가져온다'고 결론냈는데,
    재무 데이터를 붙이다가 같은 fnguide 페이지 안에서 발견했다. 주간 단위이고
    기간도 1년치뿐이라 KRX 원본만큼 상세하지는 않다.
    """
    try:
        return fnguide.short_balance(fetch_fnguide_page(ticker))
    except Exception:
        return pd.DataFrame()


def fetch_target_price_history(ticker: str) -> pd.DataFrame:
    """컨센서스 목표주가 주간 추이(약 1년)."""
    try:
        return fnguide.target_price_history(fetch_fnguide_page(ticker))
    except Exception:
        return pd.DataFrame()


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


def build_market_state_summary(market_open: bool = False) -> str:
    """지금이 장중인지 마감 뒤인지, 그래서 어떤 숫자가 아직 안 굳었는지 못 박는다.

    AI가 장중인데도 "종가"라고 쓰는 일이 있었다. 원인은 모델이 아니라 우리가 준 재료였고
    (현재가 문자열이 늘 '종가'로 시작했다), 그걸 고친 뒤에도 시각만 주면 모델이 알아서
    장 상태를 추측한다. 추측하지 않게 상태와 그 결과를 문장으로 적어 준다.
    """
    now = dt.datetime.now(om.KST)
    session = _korea_session_now(now)
    is_open = bool(market_open)
    lines = [f"- 지금 시각: {now:%Y-%m-%d %H:%M} (한국시간, 요일 {'월화수목금토일'[now.weekday()]})"]

    if is_open and session:
        lines.append(f"- 지금은 **{session} 진행 중**이다. 오늘 가격은 아직 확정되지 않았다.")
        lines.append("- 따라서 오늘 값을 '종가'라고 부르지 마라. '현재가' 또는 "
                     "'{}시 기준'처럼 진행 중임이 드러나게 써라.".format(now.strftime("%H")))
        lines.append("- 오늘의 시가·고가·저가·거래량도 장이 끝나기 전까지는 더 바뀔 수 있다.")
    elif session and not is_open:
        lines.append(f"- 시간상 {session} 구간이지만 거래소는 체결이 없다고 알린다"
                     " (공휴일이거나 아직 첫 체결 전일 수 있다).")
    else:
        lines.append("- 지금은 **정규장 시간이 아니다**. 오늘 정규장 가격은 종가로 확정됐다.")
        if now.weekday() >= 5:
            lines.append("- 주말이다. 마지막 거래일 기준 값이다.")

    # 수급은 마감 후 한참 뒤에야 올라온다. 이걸 모르면 '오늘 외국인이 샀다'고 지어낸다.
    lines.append("- 투자자별 수급(개인·외국인·기관) 확정치는 장 마감 후 18:20~18:30에야 공개된다."
                 " 그 전에는 아래 수급 숫자의 마지막 날짜가 오늘이 아니라 전 거래일이다."
                 " 날짜를 확인하고, 오늘 수급인 것처럼 쓰지 마라.")
    lines.append("- '코스피 시장 전체 수급'은 장중 잠정치라 마감 후 확정치와 달라질 수 있다.")
    return "\n".join(lines)


def build_capex_summary() -> str:
    """빅테크 4사 분기 Capex. HBM 수요의 선행지표라 '왜 오르나'의 배경이 된다."""
    df = fetch_bigtech_capex()
    if df.empty:
        return ""
    lines = []
    for company, g in df.groupby("기업"):
        g = g.sort_values("분기말")
        last = g.iloc[-1]
        cur = float(last["capex_USD"]) / 1e9
        line = f"- {company}: {last['분기말']:%Y-%m} 분기 ${cur:,.1f}B"
        if len(g) >= 5:      # 1년 전 같은 분기와 비교해야 계절성에 안 속는다
            yoy = float(g.iloc[-5]["capex_USD"]) / 1e9
            if yoy:
                line += f" (전년 동기 ${yoy:,.1f}B 대비 {cur / yoy - 1:+.0%})"
        lines.append(line)
    total = df[df["분기말"] == df["분기말"].max()]["capex_USD"].sum() / 1e9
    lines.append(f"- 4사 합계(최근 분기): ${total:,.1f}B")
    return "\n".join(lines)


def build_early_signal_summary(ticker: str) -> str:
    """하락·상승 조기신호. 탭에는 있는데 프롬프트에는 빠져 있던 값이다.

    두 탭과 같은 함수·같은 상수를 써서 화면과 AI가 다른 숫자를 보는 일이 없게 한다.
    """
    if ticker != DEFAULT_TICKER:      # 이 두 지표는 SK하이닉스에서만 검증됐다
        return ""
    try:
        hist = fetch_backtest_history_live(ticker, target_days=700)
    except Exception:
        return ""
    if len(hist) < 40:
        return ""

    lines = []
    try:
        foreign_slope = _rolling_slope(hist["외국인"], DECLINE_PATTERN_WINDOW)
        vol_ratio = hist["거래량"] / hist["거래량"].rolling(DECLINE_PATTERN_VOL_WINDOW).mean()
        bt = run_boolean_pattern_backtest(
            hist["종가"], hist["날짜"], (foreign_slope < 0) & (vol_ratio > 1.0),
            DECLINE_HORIZON, drawdown_threshold=DECLINE_DRAWDOWN_THRESHOLD)
        matched = bool(bt["current_match"])
        down = bt["match_down_rate"] if matched else bt["rest_down_rate"]
        lines.append(
            f"- 하락 조기신호(외국인 순매도 + 거래량 증가): 현재 "
            f"{'조건 충족' if matched else '조건 미충족'} "
            f"(외국인 {DECLINE_PATTERN_WINDOW}일 기울기 {float(foreign_slope.dropna().iloc[-1]):,.0f}, "
            f"거래량 {float(vol_ratio.dropna().iloc[-1]):.2f}배). "
            f"이 상태의 과거 {DECLINE_HORIZON}거래일 내 "
            f"{DECLINE_DRAWDOWN_THRESHOLD:.0%} 하락 확률 "
            + (f"{down:.1%}" if down is not None else "N/A"))
    except Exception:
        pass

    try:
        inst_slope = _rolling_slope(hist["기관"], RALLY_PATTERN_WINDOW)
        retail_slope = _rolling_slope(hist["개인"], RALLY_PATTERN_WINDOW)
        vol_ratio = hist["거래량"] / hist["거래량"].rolling(RALLY_PATTERN_VOL_WINDOW).mean()
        bt = run_boolean_pattern_backtest(
            hist["종가"], hist["날짜"],
            (inst_slope > 0) & (retail_slope < 0) & (vol_ratio > 1.0),
            RALLY_HORIZON, drawdown_threshold=RALLY_DRAWDOWN_THRESHOLD)
        matched = bool(bt["current_match"])
        up = bt["match_up_rate"] if matched else bt["rest_up_rate"]
        lines.append(
            f"- 상승 조기신호(기관 순매수 + 개인 순매도 + 거래량 증가): 현재 "
            f"{'조건 충족' if matched else '조건 미충족'}. "
            f"이 상태의 과거 {RALLY_HORIZON}거래일 내 상승 확률 "
            + (f"{up:.1%}" if up is not None else "N/A")
            + " (앞반기에서는 방향이 뒤집혀 신뢰도가 낮은 지표다)")
    except Exception:
        pass
    return "\n".join(lines)


def fetch_foreign_hold_ratio(ticker: str) -> pd.DataFrame:
    """외국인 보유율 일별 추이. 스냅샷에는 '오늘 값' 하나만 있어서 방향을 알 수 없었다."""
    r = requests.get(f"https://m.stock.naver.com/api/stock/{ticker}/trend",
                     headers={"User-Agent": "Mozilla/5.0", "Referer": "https://m.stock.naver.com/"},
                     timeout=10)
    r.raise_for_status()
    js = r.json()
    rows = js if isinstance(js, list) else ((js or {}).get("dealTrendInfos") or [])
    if not rows:
        return pd.DataFrame(columns=["날짜", "외국인보유율"])
    out = pd.DataFrame([{
        "날짜": dt.datetime.strptime(str(x["bizdate"]), "%Y%m%d").date(),
        "외국인보유율": _to_number(x.get("foreignerHoldRatio")),
    } for x in rows if x.get("bizdate")])
    return out.dropna().sort_values("날짜").reset_index(drop=True)


def build_recent_price_summary(ticker: str, days: int = 10,
                               market_open: bool = False) -> str:
    """최근 며칠의 종가·등락률·거래량과 외국인 보유율 추이.

    지금까지는 '오늘 하루'와 '5일 순매수 합계'만 넘겨서, 오늘의 움직임이
    연속 하락 끝의 반등인지 상승 5일차인지를 AI가 구분할 수 없었다.
    """
    try:
        ohlcv = fetch_daily_ohlcv(ticker, days + 15)
    except Exception:
        return ""
    if ohlcv.empty or "종가" not in ohlcv.columns:
        return ""
    tail = ohlcv.tail(days + 1).copy()
    tail["등락률"] = tail["종가"].astype(float).pct_change() * 100
    tail = tail.tail(days)

    ratio_by_date = {}
    try:
        fh = fetch_foreign_hold_ratio(ticker)
        ratio_by_date = dict(zip(fh["날짜"], fh["외국인보유율"]))
    except Exception:
        pass

    # 장중에는 일봉 마지막 줄이 '오늘 진행 중인 값'이다. 그대로 두면 확정 종가로 읽힌다.
    today = dt.datetime.now(om.KST).date()
    market_open = bool(market_open)

    lines = []
    for idx, row in tail.iterrows():
        day = idx.date() if hasattr(idx, "date") else idx
        label = "현재가(장중, 미확정)" if (market_open and day == today) else "종가"
        line = (f"    {day} {label} {float(row['종가']):,.0f} "
                f"({float(row['등락률']):+.2f}%) 거래량 {float(row['거래량']):,.0f}")
        held = ratio_by_date.get(day)
        if held is not None:
            line += f" 외국인보유율 {held:.2f}%"
        lines.append(line)
    if not lines:
        return ""

    head = "- 최근 일별 종가 · 등락률 · 거래량:"
    closes = tail["종가"].astype(float)
    streak, direction = 0, None
    for chg in reversed(tail["등락률"].tolist()):
        cur = "상승" if chg > 0 else ("하락" if chg < 0 else None)
        if cur is None or (direction and cur != direction):
            break
        direction, streak = cur, streak + 1
    tail_note = ""
    if direction and streak >= 2:
        tail_note = f"\n- 최근 흐름: {streak}거래일 연속 {direction}"
    if len(closes) >= 2:
        tail_note += (f"\n- 이 구간 누적 등락률: "
                      f"{(closes.iloc[-1] / closes.iloc[0] - 1) * 100:+.2f}%")
    return head + "\n" + "\n".join(lines) + tail_note


EARNINGS_WATCH = {
    "NVDA": "엔비디아 (HBM 최대 수요처)",
    "MU": "마이크론 (직접 경쟁사)",
    "AMD": "AMD",
    "AVGO": "브로드컴",
    "TSM": "TSMC",
    "INTC": "인텔",
    "WDC": "웨스턴디지털 (낸드)",
    "STX": "씨게이트 (낸드)",
}


EARNINGS_LOOKAHEAD_DAYS = 10


def _nasdaq_earnings_on(day: dt.date) -> list[tuple[str, str]]:
    r = requests.get("https://api.nasdaq.com/api/calendar/earnings",
                     params={"date": day.isoformat()},
                     headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
                     timeout=10)
    r.raise_for_status()
    rows = ((r.json() or {}).get("data") or {}).get("rows") or []
    return [(x["symbol"], x.get("time") or "") for x in rows
            if x.get("symbol") in EARNINGS_WATCH]


def fetch_earnings_calendar() -> str:
    """앞으로 열흘 안에 실적을 발표하는 반도체 관련 기업."""
    today = dt.datetime.now(om.KST).date()
    days = [today + dt.timedelta(days=i) for i in range(EARNINGS_LOOKAHEAD_DAYS)]

    def one(day: dt.date) -> list[tuple[str, str]]:
        try:                       # 한 날짜가 실패해도 나머지 날짜는 살린다
            return _nasdaq_earnings_on(day)
        except Exception:
            return []

    # 날짜별로 독립된 조회라 동시에 받는다
    found: list[str] = []
    with ThreadPoolExecutor(max_workers=min(5, len(days))) as pool:
        for day, res in zip(days, pool.map(one, days)):
            for symbol, when in res:
                d_left = (day - today).days
                label = "오늘" if d_left == 0 else ("내일" if d_left == 1 else f"{d_left}일 뒤")
                slot = {"time-after-hours": "장 마감 후",
                        "time-pre-market": "개장 전"}.get(when, "")
                found.append(f"- {day} ({label}) {EARNINGS_WATCH[symbol]}"
                             + (f" — 미국장 {slot}" if slot else ""))
    return "\n".join(found)


CONSENSUS_LOG = os.path.join("data", "consensus_log.csv")


def _consensus_log(ticker: str) -> pd.DataFrame:
    """쌓아둔 컨센서스 기록에서 이 종목 것만. 읽을 수 없으면 빈 표를 준다.

    종목코드는 반드시 문자열로 읽어야 한다. 그냥 두면 pandas가 000660을 정수 660으로
    바꿔버려서 종목 비교가 영원히 거짓이 된다.
    """
    try:
        log = pd.read_csv(CONSENSUS_LOG, dtype={"종목": str})
    except Exception:
        return pd.DataFrame(columns=["날짜", "종목", "목표주가", "투자의견"])
    if "종목" not in log.columns:
        return pd.DataFrame(columns=["날짜", "종목", "목표주가", "투자의견"])
    log["종목"] = log["종목"].astype(str).str.zfill(6)
    return log[log["종목"] == ticker].sort_values("날짜").reset_index(drop=True)


def build_consensus_trend_summary(ticker: str, snapshot: dict) -> str:
    """쌓인 기록으로 목표주가가 상향인지 하향인지 알려준다."""
    target = _to_number(snapshot.get("목표주가"))
    if not target:
        return ""
    lines = [f"- 현재 컨센서스 목표주가 {target:,.0f}원"
             f" (투자의견 {snapshot.get('투자의견') or 'N/A'}/5,"
             f" 기준일 {snapshot.get('컨센서스일자') or '-'})"]

    # FnGuide에 주간 1년치가 이미 있다. 직접 쌓은 기록(consensus_log)보다 훨씬 길어서 이쪽을 먼저 쓴다.
    try:
        hist = fetch_target_price_history(ticker)
    except Exception:
        hist = pd.DataFrame()
    if len(hist) >= 2:
        now = dt.datetime.now(om.KST)
        for label, weeks in (("1개월 전", 4), ("3개월 전", 13), ("1년 전", 52)):
            cutoff = pd.Timestamp(now.date() - dt.timedelta(weeks=weeks))
            past = hist[hist["일자"] <= cutoff]
            if past.empty:
                continue
            old = float(past.iloc[-1]["목표주가"])
            if old:
                lines.append(
                    f"- {label}({past.iloc[-1]['일자']:%Y-%m-%d}) 목표주가 {old:,.0f}원 대비"
                    f" {target / old - 1:+.1%}"
                    f" ({'상향' if target > old else '하향' if target < old else '변화 없음'})")
        lines.append("- (목표주가 추이는 FnGuide 주간 데이터. 목표주가는 수준보다 방향이 중요하다)")
        return "\n".join(lines)

    log = _consensus_log(ticker)
    if len(log) < 2:
        lines.append("- 목표주가 변화: 비교할 과거 기록이 아직 없음"
                     " (오늘부터 매일 한 줄씩 쌓는 중)")
        return "\n".join(lines)
    for label, back in (("1주 전", 7), ("1개월 전", 30)):
        cutoff = (dt.datetime.now(om.KST).date() - dt.timedelta(days=back)).isoformat()
        past = log[log["날짜"] <= cutoff]
        if past.empty:
            continue
        old = float(past.iloc[-1]["목표주가"])
        if old:
            lines.append(f"- {label}({past.iloc[-1]['날짜']}) 목표주가 {old:,.0f}원 대비"
                         f" {target / old - 1:+.1%} ({'상향' if target > old else '하향' if target < old else '변화 없음'})")
    return "\n".join(lines)


def build_short_sale_summary(ticker: str) -> str:
    """차입공매도 비중 추이.

    "숏커버링이었나"를 판단할 재료가 대시보드에 통째로 없었다. KRX·KOFIA·세이브로가 다 막혀
    포기했었는데 FnGuide 페이지 안에 주간 1년치가 있었다. 주간이라 일별 급등락은 못 짚는다.
    """
    try:
        df = fetch_short_balance(ticker)
    except Exception:
        return ""
    if df.empty:
        return ""
    last = df.iloc[-1]
    avg = float(df["차입공매도비중"].mean())
    cur = float(last["차입공매도비중"])
    lines = [f"- 최근 {last['일자']:%Y-%m-%d} 기준 차입공매도 비중 {cur:.2f}%"
             f" (최근 1년 평균 {avg:.2f}%, 최고 {df['차입공매도비중'].max():.2f}%,"
             f" 최저 {df['차입공매도비중'].min():.2f}%)"]
    if len(df) >= 5:
        prev = float(df.iloc[-5]["차입공매도비중"])
        lines.append(f"- 4주 전 {prev:.2f}% 대비 {cur - prev:+.2f}%p"
                     f" ({'증가' if cur > prev else '감소' if cur < prev else '변화 없음'})")
    lines.append("- 주간 단위 값이라 특정 날짜의 급등락은 설명하지 못한다."
                 " 비중이 높다고 곧 하락이 아니고, 되레 숏커버링이 상승 재료가 되기도 한다.")
    return "\n".join(lines)


def _price_summary_fallback(ticker: str) -> str:
    """세션에 현재가 요약이 없을 때(탭이 그려지기 전에 분석을 만드는 경우) 직접 만든다."""
    try:
        q = fetch_current_price(ticker) or {}
        close = q.get("closePrice")
        chg = q.get("compareToPreviousClosePrice")
        pct = q.get("fluctuationsRatio")
        state = (q.get("marketStatus") or "").upper()
        if close:
            return f"종가 {close}원, 전일대비 {chg}원 ({pct}%) ({state})"
    except Exception:
        pass
    return "현재가 데이터를 가져오지 못함"


def _dated_digest(payload: dict) -> str:
    """미리 만들어 둔 요약에 '언제 만든 것인지'를 붙인다.

    이 요약들은 새 재료가 있을 때만 다시 만든다. 그래서 며칠 전 것이 그대로 쓰이는 게
    정상인데, 날짜를 안 알려주면 모델은 그걸 '오늘의 증권가 시각'으로 읽는다.
    리포트가 일주일째 안 나온 날과, 오늘 아침에 갱신된 날을 구분할 수 있어야 한다.
    """
    text = (payload or {}).get("text") or ""
    if not text:
        return ""
    when = (payload or {}).get("date") or "시점 미상"
    return f"(이 정리는 {when} 기준이다. 그 뒤로 새 자료가 없어서 그대로 쓰고 있다)\n{text}"


def generate_ai_analysis(
    stock_label: str,
    time_label: str,
    price_summary: str,
    supply_summary: str,
    headlines: list[str],
    reports_md: str,
    dram_summary: str,
    community_summary: str,
    overheat_summary: str,
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
    capex_md: str = "",
    early_signal_md: str = "",
    recent_price_md: str = "",
    earnings_md: str = "",
    consensus_md: str = "",
    market_state_md: str = "",
    short_sale_md: str = "",
    foreign_desk_md: str = "",
    analyst_view_md: str = "",
    broker_targets_md: str = "",
    disclosure_view_md: str = "",
    financial_view_md: str = "",
    # 밑줄로 시작하는 인자는 st.cache_data가 해시(=캐시 키)에서 빼준다.
    # 콜백 함수는 해시가 안 되고, 캐시 키에 들어가서도 안 된다.
    _stream_to=None,
) -> tuple[str, str | None]:
    prompt = f"""오늘은 {time_label}입니다. 다음은 이 시점 기준 {stock_label} 관련 데이터입니다.

**중요**: 아래 데이터에 적힌 사실만 쓰세요. 학습 시점에 알고 있던 과거 뉴스나 날짜를 끌어오지 마세요.
(실제로 이 지시가 없으면 몇 년 전 사건을 '최근 뉴스'라고 답하는 일이 생깁니다.)

[지금 시장 상태 — 다른 무엇보다 먼저 읽어라]
{market_state_md if market_state_md else "(상태 확인 실패 - 장중 여부를 단정하지 마라)"}

[오늘 주가]
{price_summary}

[오늘 장중 흐름]
{intraday_md if intraday_md else "(수집 실패)"}

[정규장 밖 움직임 — 프리장 · 애프터장(NXT)]
{over_market_md if over_market_md else "(오늘 시간외 기록 없음)"}

[전자공시 — 최근순 목록. 오늘 것은 본문 요지 포함]
{disclosure_md if disclosure_md else "(수집 실패)"}

[공시 정리 — 최근 공시 15건을 본문까지 읽고 미리 정리해 둔 것. 금액·주식수·기간이 들어 있다]
{disclosure_view_md if disclosure_view_md else "(정리된 요약 없음)"}

[밸류에이션 · 컨센서스 · 동일업종 · 수급추이]
{snapshot_md if snapshot_md else "(수집 실패)"}

[최근 일별 주가 흐름 — 오늘을 그 앞 며칠과 이어서 봐라]
{recent_price_md if recent_price_md else "(수집 실패)"}

[최근 수급 동향 — 이 종목, 일별 확정치]
{supply_summary}

[코스피 시장 전체 수급 — 종목별 아님, 장중 잠정치]
{market_flow_md if market_flow_md else "(수집 실패)"}

[이 종목의 외국계 창구 추정 순매수 — 장중, 추정치. 확정 수급이 아니다]
{foreign_desk_md if foreign_desk_md else "(수집 실패)"}

[가격 과열도 백테스트 — 과거 통계 참고용, 매매 신호 아님]
{_tab_summary(overheat_summary, "가격 과열도")}

[하락 · 상승 조기신호 — 과거 통계 참고용, 매매 신호 아님]
{early_signal_md if early_signal_md else "(해당 없음)"}

[DRAM 현물가 — 현물가만 있고 고정거래가(계약가)는 없음. 실적에 직결되는 건 고정가이므로 현물가만으로 단정하지 마라]
{dram_summary}

[빅테크 분기 설비투자(Capex) — HBM 수요의 선행지표]
{capex_md if capex_md else "(수집 실패)"}

[다가오는 반도체 관련 실적 발표]
{earnings_md if earnings_md else "(앞으로 열흘 내 예정 없음 또는 수집 실패)"}

[관련 뉴스 (제목 + 본문 요약)]
{news_md if news_md else (chr(10).join(f"- {h}" for h in headlines) if headlines else "(수집된 뉴스 없음)")}

[최근 애널리스트 리포트 — 제목 목록]
{reports_md if reports_md else "(수집된 리포트 없음)"}

[증권가 시각 정리 — 리포트 PDF 본문까지 읽고 미리 정리해 둔 것. 증권사별 목표주가와 추정 실적이 들어 있다]
{analyst_view_md if analyst_view_md else "(정리된 요약 없음)"}

[재무 정리 — 재무제표(FnGuide)를 읽고 미리 정리해 둔 것. 매출·영업이익 추이와 추정치]
{financial_view_md if financial_view_md else "(정리된 요약 없음)"}

[업종 · 전방수요 · 매크로 뉴스 + 증권사 투자의견 변경 기사]
※ 위 리포트 목록은 네이버가 싣는 13개 증권사뿐이다. 그 밖의 증권사가 목표주가를
   바꾼 건은 여기 '투자의견' 검색어 기사에만 나오니, 있으면 [리포트] 갈래로 함께 다뤄라.
{sector_news_md if sector_news_md else "(수집 안 함)"}

[해외 반도체 산업 리서치 뉴스 — TrendForce]
{trendforce_md if trendforce_md else "(수집된 자료 없음)"}

[거시경제 지표 — 모두 '직전 미국장 종가' 기준이다. 한국이 장중이면 이 값은 어젯밤 것이고,
오늘 한국 장중 움직임의 배경은 될 수 있어도 '오늘 미국 증시'라고 쓰면 틀린다]
{macro_md if macro_md else "(수집 실패)"}

[해외 상장분(ADR) 괴리율]
{adr_md if adr_md else "(해당 없음)"}

[증권사별 목표주가 — 리포트 상세에서 숫자만 뽑아 모은 원자료. '직접 집계' 컨센서스]
{broker_targets_md if broker_targets_md else "(집계 없음)"}

[컨센서스 목표주가와 그 변화 — FnGuide 제공]
{consensus_md if consensus_md else "(수집 실패)"}

※ 위 두 컨센서스는 값이 다르다. 어느 쪽이 틀린 게 아니라 세는 대상이 다르다.
   FnGuide는 네이버에 리포트를 싣지 않는 증권사까지 포함하고, 직접 집계는 네이버
   게재분만 대신 증권사별 내역이 있다. 목표주가를 인용할 때는 **어느 쪽 값인지 반드시
   밝혀라**(예: "컨센서스 3,279,565원(FnGuide)"). 두 값을 평균 내거나 섞지 마라.

[차입공매도 비중 — 주간, 최근 1년]
{short_sale_md if short_sale_md else "(수집 실패)"}

[투자자 커뮤니티 — 게시글 원문. 여론이지 사실이 아님]
{community_summary if community_summary else "(수집 실패)"}

아래 형식 그대로, 한국어로 작성해줘.

## 한 줄 요약
오늘 가장 중요한 사실 한 문장.

## 오늘 이렇게 움직인 이유
먼저 '최근 일별 주가 흐름'으로 오늘을 앞 며칠과 이어서 한 문장(연속 하락 끝 반등인지, 상승 며칠째인지).
그다음 장중에서 급했던 구간을 짚고 그 시각의 공시·뉴스와 연결해라. 없으면 "직접 연결되는 재료는 데이터에 없음".
시간외에서 방향이 바뀌었으면 별도 문단으로. 시간외 등락률은 정규장 종가 대비다.

## 종목 이슈인가, 업종 전체인가
'동일업종' 등락률 숫자를 인용해 개별 이슈인지 업종 전체인지 판단해라.

## 강세 근거
## 약세 근거 / 리스크
이 둘이 핵심이다.
1) 여섯 갈래를 모두 훑고, 한 갈래에 다른 이야기가 여럿이면 항목을 나눠 적어라
   (특히 `[산업]`의 DRAM 현물가/Capex, `[대시보드]`의 수급/과열도/조기신호는 각각 따로).
   `[공시]` 전자공시(숫자 인용) · `[뉴스]` 종목+업종·매크로 뉴스 · `[리포트]` 애널리스트(증권가 시각 정리의 목표주가·추정 실적 숫자 인용)
   `[산업]` TrendForce+DRAM 현물가+빅테크 Capex+실적 발표 일정
   `[거시]` SOX·나스닥·달러인덱스·환율·금리+ADR 괴리율
   `[대시보드]` 수급·과열도·조기신호·컨센서스·동일업종·외국인 보유율·차입공매도
   `[재무]` 재무 정리의 매출·영업이익·이익률 추이와 추정치(숫자 인용)
2) 형식은 `[갈래] 내용 (근거 숫자)`. 말머리는 위 일곱 개만.
3) 쓸 근거가 없는 갈래는 `[갈래] 이번엔 뚜렷한 신호 없음` 한 줄.
4) 매크로 숫자는 이 종목까지 어떻게 연결되는지 한 마디 붙여라.
5) 강세·약세를 같은 밀도로.

## 근거의 무게
어느 쪽이 더 무거운지 2문장. '확인된 데이터'가 '기대·심리'보다 무겁다.

## 커뮤니티 대세 반응
제목을 읽고 우세한 반응을 2문장. 비율만 옮기지 말고 무엇을 기대·걱정하는지 써라.
실제 제목 1–2개 짧게 인용, 눈에 띄는 소수 의견 한 줄, 여론이 데이터와 어긋나면 지적.
반어법이 많으니 문맥으로 읽고, 여론을 사실 근거로 올려 쓰지 마라.

## 수요 배경과 지표 점검
아래 다섯 줄을 빠짐없이. 데이터가 없으면 "데이터에 없음".
- 빅테크 Capex: 최근 분기와 전년 동기 대비 방향 + HBM 수요 의미 (중장기 배경이지 오늘 등락 원인 아님)
- 하락·상승 조기신호: 충족 여부와 확률 ("백테스트 기반 참고치" 필수)
- 다가오는 실적 발표: 며칠 뒤 무엇 (결과 예측 금지)
- 목표주가 방향: 상향/하향 (기록 없으면 없다고)
- 차입공매도 비중: 현재 수준과 1년 평균 대비 위치, 최근 방향

## 지금 위치
현재가가 목표주가·52주 고저·PER 대비 어디인지 숫자로. 목표주가는 어느 집계인지 괄호로
밝히고(FnGuide / 직접 집계), 두 값이 다르면 그 사실도 한 줄로. 컨센서스는 기대치일 뿐이라는 단서 한 줄.
**실적이 어디까지 왔는지도 한 줄** — 재무 정리의 매출·영업이익 추이와 추정치를 숫자로.
지금 밸류에이션이 그 실적 대비 어디인지 짚어라.

## 앞으로 확인할 것
무엇을 보면 판단이 갈리는지 구체적으로 2–3개. 이 대시보드에서 볼 수 있는 지표로.

작성 규칙:
- **짧게.** 초당 40자로 나오므로 길어진 만큼 기다린다. 전체 2,000자 안쪽. 항목을 빼지 말고 문장을 눌러 담아라.
- 각 근거는 한 줄. 서론·맺음말·재요약 금지.
- 데이터에 없는 사실·날짜·숫자를 지어내지 마. 모르면 "데이터에 없음".
- 과열도·조기신호는 "백테스트 기반 참고치" 단서를 붙이고 단독 근거로 쓰지 마.
- 실적 발표가 3일 이내면 관망세 배경으로 짚되 결과는 예측 금지.
- 목표주가는 수준보다 방향(상향/하향)이 중요. 기록 없으면 없다고 써라.
- '외국계 창구 추정 순매수'는 거래원 기반 추정치다. '추정'임을 밝히고, 이 값으로 기관·개인을 추측하지 마라.
- 차입공매도 비중은 주간 값이다. 특정 날짜의 원인으로 쓰지 말고 평균 대비 수준·방향으로만.
  높다고 하락을 예상하지 마라(숏커버링이 상승 재료가 되기도 한다).
- 커뮤니티는 여론이지 사실이 아니다. 매수/매도 추천·목표가 제시 금지.
- 장중에는 오늘 값을 '종가'·'마감'으로 쓰지 마. [지금 시장 상태]대로 진행 중임이 드러나게.
- 수급 숫자는 날짜를 확인해라. 장중이면 오늘 수급은 아직 없다.
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

    skipped: list[tuple[str, str]] = []
    if _stream_to is None:
        text, used = _call_gemini(prompt)
    else:
        # 스트리밍: 받는 대로 화면에 흘려보낸다. 총 시간은 같지만 첫 글자가 곧바로 보인다.
        used_holder: dict = {}
        chunks: list[str] = []
        last_paint = 0.0
        for piece in _stream_gemini(prompt, used_holder):
            if piece is _RESTART:
                # 앞 모델의 답을 버리고 다음 모델 답으로 갈아끼운다.
                # 안 지우면 두 모델의 글이 화면에 이어붙는다.
                chunks.clear()
                last_paint = 0.0
                _stream_to("")
                continue
            chunks.append(piece)
            # 조각마다 그리면 브라우저가 글 전체를 60번 넘게 다시 파싱한다. 조각은 약 900ms
            # 간격으로 오므로 0.4초로 묶어도 끊겨 보이지 않으면서 렌더 횟수는 절반이 된다.
            now = time.monotonic()
            if now - last_paint >= STREAM_PAINT_SEC:
                last_paint = now
                _stream_to("".join(chunks))
        text = "".join(chunks)
        _stream_to(text)          # 마지막 묶음이 화면에 빠지지 않게 한 번 더
        used = used_holder.get("model", GEMINI_MODEL)
        skipped = used_holder.get("skipped") or []

    if _looks_truncated(text):
        cut = "AI 응답이 중간에 끊겼습니다(마지막 항목이 빠졌습니다). 다시 눌러 보세요."
        note = f"{note} {cut}" if note else cut

    if used != GEMINI_MODEL:
        # 기본 모델을 건너뛴 이유를 그대로 알려준다. 예전에는 이유를 묻지 않고 늘
        # "한도가 소진되어"라고 적었는데, 응답이 느려서 넘어간 경우까지 한도 탓으로 말해버렸다.
        why = ", ".join(f"{m}: {reason}" for m, reason in skipped) if skipped else "사용 불가"
        switched = f"{used} 모델로 분석했습니다 ({why})."
        note = f"{note} {switched}" if note else switched
    return (text or "AI가 응답을 생성하지 못했습니다.", note)


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


def build_price_context(ticker: str) -> dict:
    """화면의 render_current_price()가 session_state에 남기던 값을 헤드리스로 만든다.

    _price_summary_fallback()은 장중에도 늘 '종가'라고 적는다. 그러면 CONTEXT §5가
    경고하는 실수(진행 중인 가격을 확정 종가로 말하기)를 재료 단계에서 저지르게 된다.
    수집기에서 만들 때는 이쪽을 써서 화면과 같은 문장을 넘긴다.

    반환: {"market_open": bool, "summary": str, "value": int | None}
    """
    ctx = {"market_open": False, "summary": _price_summary_fallback(ticker), "value": None}
    try:
        d = fetch_current_price(ticker) or {}
        close = int(d["closePriceRaw"])
        chg = int(d["compareToPreviousClosePriceRaw"])
        pct = float(d["fluctuationsRatioRaw"])
        try:
            updated = pd.to_datetime(d["localTradedAt"]).strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            updated = d.get("localTradedAt") or "-"
        # 네이버는 필드 이름을 closePrice로 쓸 뿐 장중에는 현재가다. 상태에 맞는 말로 넘긴다.
        is_open = str(d.get("marketStatus") or "").upper() == "OPEN"
        if is_open:
            label = "현재가"
            state = f"{_korea_session_now() or '정규장'} 진행 중 · {updated} 체결"
        else:
            label, state = "종가", f"정규장 마감 · {updated} 확정"
        ctx = {
            "market_open": is_open,
            "value": close,
            "summary": f"{label} {close:,}원, 전일대비 {chg:+,}원 ({pct:+.2f}%) ({state})",
        }
    except Exception:
        pass
    return ctx


# ── 탭 없이 만드는 요약 ──────────────────────────────────────────────────────
# 과열도·DRAM 요약은 원래 그 탭의 렌더 함수가 전역 변수에 채웠다. 그래서 탭이 그려질
# 때만 값이 있었고, 화면 없이 도는 수집기에서는 늘 '계산되지 않음'으로 빠졌다
# (CONTEXT §5의 "탭 렌더에 묶인 요약을 조심할 것"이 바로 이 이야기다).
# 여기서는 탭과 무관하게 데이터만으로 만든다.

OVERHEAT_DEFAULT_MA_WINDOW = 80
OVERHEAT_DEFAULT_HORIZON = 15
OVERHEAT_DEFAULT_THRESHOLD = 0.10
OVERHEAT_QUANTILES = [0.20, 0.15, 0.10, 0.05]


def build_overheat_summary(ticker: str) -> str:
    """가격 과열도 백테스트 요약.

    화면에는 이 조건을 바꾸는 슬라이더가 있지만, 그건 사람이 이것저것 눌러 보라고 둔
    것이다. AI에 넘기는 재료는 기준이 매번 달라지면 안 되므로 기본값만 쓴다.
    """
    ma, horizon = OVERHEAT_DEFAULT_MA_WINDOW, OVERHEAT_DEFAULT_HORIZON
    threshold = OVERHEAT_DEFAULT_THRESHOLD
    hist = fetch_backtest_history_live(ticker, target_days=700)
    if len(hist) < 80:
        return "가격 과열도 백테스트 미실행 (과거 데이터 부족)"

    high = {q: run_overheat_backtest(hist, "종가", ma, horizon, quantile=q,
                                     drawdown_threshold=threshold, side="high")
            for q in OVERHEAT_QUANTILES}
    low = {q: run_overheat_backtest(hist, "종가", ma, horizon, quantile=q,
                                    drawdown_threshold=threshold, side="low")
           for q in OVERHEAT_QUANTILES}
    result = high[OVERHEAT_QUANTILES[0]]
    if result["n"] < 30:
        return "가격 과열도 백테스트 미실행 (표본 부족)"

    deviation = result["current_deviation"]
    # 현재 괴리율이 어느 구간에 드는지. 위(과열)부터 보고, 아니면 아래(침체)를 본다.
    label = "평상시"
    for q in sorted(OVERHEAT_QUANTILES):
        cutoff = high[q]["hi_cutoff"]
        if cutoff is not None and deviation is not None and deviation >= cutoff:
            label = f"상위 {q:.0%} 구간(과열)"
            break
    else:
        for q in sorted(OVERHEAT_QUANTILES):
            cutoff = low[q]["hi_cutoff"]
            if cutoff is not None and deviation is not None and deviation <= cutoff:
                label = f"하위 {q:.0%} 구간(침체)"
                break

    # 화면 표와 같은 차례로 적는다: 하위는 낮은 분위부터, 상위는 넓은 분위부터.
    # 침체 쪽을 빼면 지금처럼 하위 구간에 들어와 있을 때 근거가 통째로 사라진다.
    parts = []
    for q in sorted(OVERHEAT_QUANTILES):
        rate = low[q]["hi_rate"]
        parts.append(f"하위 {q:.0%}: " + (f"{rate:.1%}" if rate is not None else "N/A"))
    for q in sorted(OVERHEAT_QUANTILES, reverse=True):
        rate = high[q]["hi_rate"]
        parts.append(f"상위 {q:.0%}: " + (f"{rate:.1%}" if rate is not None else "N/A"))

    return (
        f"가격 과열도 백테스트({ma}일선 괴리율, 표본 {result['n']}일, "
        f"기저 하락 확률 {result['base_rate']:.1%}): "
        f"괴리율 구간별 {horizon}거래일 내 {threshold:.0%} 이상 하락 확률"
        f"({', '.join(parts)})은 과열이 심할수록 높아짐. "
        f"현재 상태: {label}"
        + (f" (괴리율 {deviation:+.1%})" if deviation is not None else "")
        + ". (통계 참고용, 매매 신호 아님)"
    )


def build_dram_summary() -> str:
    """DRAM 현물가 요약. 기준일을 반드시 붙인다 —
    안 붙이면 며칠 전 시세를 AI가 '오늘 올랐다'로 옮겨 적는다."""
    module_df, module_asof = fetch_dram_module_prices()
    chip_df, chip_asof = fetch_dram_chip_prices()
    combined = pd.concat([module_df, chip_df], ignore_index=True)
    if combined.empty:
        return "DRAM 현물가를 가져오지 못함"

    out = "\n".join(f"- {row['품목']}: ${row['평균가(USD)']:,.3f} ({_signed_pct(row)})"
                    for _, row in combined.iterrows())
    asof = " · ".join(x for x in (f"모듈 {module_asof}" if module_asof else "",
                                  f"칩 {chip_asof}" if chip_asof else "") if x)
    if asof:
        out += (f"\n- (TrendForce 기준일: {asof}. 매일 갱신되지는 않으므로"
                " 이 날짜를 확인하고 인용해라)")
    return out


_TAB_SUMMARY_UNSET = ("미실행",)


def _tab_summary(summary: str, label: str) -> str:
    """계산이 안 된 항목을 '없다'고 못박아 넘긴다.

    빈 문자열을 그냥 넣으면 모델이 그 자리를 자기가 아는 것으로 메운다. 근거로 쓰지
    말라고 분명히 적어야 한다. 예전에는 이 상황이 '탭이 꺼져 있다' 하나뿐이었지만,
    이제는 수집기가 만들다 실패한 경우도 있어서 원인을 단정하지 않는다.
    """
    if not summary or any(k in summary for k in _TAB_SUMMARY_UNSET):
        return f"(계산되지 않음 — {label}. 이 항목은 근거로 쓰지도, 언급하지도 마라)"
    return summary


# ── 위 함수들이 쓰는 모듈 수준 값 ─────────────────────────────────────────────
# 오늘 한도에 걸린 모델을 기억한다. 한도는 하루 단위라 한 번 막히면 그날은 계속 막히는데,
# 매번 첫 모델부터 부르면 확정된 429를 받으려고 5초씩 버리게 된다(실측 5.2초).
_GEMINI_EXHAUSTED: dict[str, dt.date] = {}


# 스트리밍 도중 "지금까지 받은 건 버리고 다시 시작한다"는 신호.
# 잘린 답을 뱉은 모델을 버리고 다음 모델로 넘어갈 때, 화면에 이미 흘러간 글자를
# 지우지 않으면 두 모델의 답이 이어붙어 보인다.
_RESTART = object()


# 뉴스 카드에 붙는 게재 시점("3분 전", "2026.08.25."). 날짜 없이 제목만 넘기면 AI가
# 일주일 전 기사를 '오늘 뉴스'로 옮겨 적는다. 실제로 그런 문장이 나왔다.
_NEWS_WHEN_PAT = re.compile(r"^(\d+\s*(?:분|시간|일|주|개월)\s*전|\d{4}\.\d{2}\.\d{2}\.?)$")


load_over_market_ticks = om.load_ticks


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
