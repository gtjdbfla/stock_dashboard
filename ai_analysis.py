"""AI 분석 결과를 파일에 남겨 두는 곳.

예전에는 '지금 바로 분석하기'를 누를 때마다 만들었다. 그런데 생성에 30~100초가 걸리고
(대부분 모델이 생각하는 시간이다. 붐비는 날은 더 걸린다) 캐시도 사실상 안 먹혔다 —
캐시 키에 분 단위 시각이 들어가서 1분만 지나도 새로 만들었기 때문이다.
그래서 화면을 열 때마다 매번 기다려야 했다.

이제는 자동 새로고침 시각에 미리 만들어 여기 저장하고, 화면은 읽기만 한다.
리포트·공시·재무 요약이 쓰는 방식과 같다. 다만 저 셋과 달리 이 분석은 시의성이 있어서
(주가·수급이 계속 바뀐다) 화면에 **기준 시각을 반드시 같이 보여준다.**

이 파일은 streamlit도 app도 import하지 않는다. app.py가 이걸 읽어야 하는데
여기서 app을 부르면 순환 import가 된다. 만드는 쪽은 app.py가 맡는다.
"""
import datetime as dt
import json
import os

KST = dt.timezone(dt.timedelta(hours=9))
STORE = os.environ.get("AI_ANALYSIS_FILE", "data/ai_analysis.json")

# 원본 데이터 보기에 뿌릴 항목. (라벨, payload 키) — app.py의 표시 순서와 맞춘다.
SOURCE_FIELDS = [
    ("지금 시장 상태", "market_state"),
    ("최근 일별 주가 흐름", "recent_price"),
    ("오늘 장중 흐름", "intraday"),
    ("정규장 밖 움직임 (프리장·애프터장)", "over_market"),
    ("코스피 시장 전체 수급", "market_flow"),
    ("하락·상승 조기신호", "early_signal"),
    ("가격 과열도 백테스트", "overheat"),
    ("DRAM 현물가", "dram"),
    ("빅테크 분기 Capex", "capex"),
    ("다가오는 실적 발표", "earnings"),
    ("컨센서스 목표주가 변화", "consensus"),
    ("차입공매도 비중", "short_sale"),
    ("외국계 창구 추정 순매수", "foreign_desk"),
    ("증권가 시각 정리", "analyst_view"),
    ("공시 정리", "disclosure_view"),
    ("재무 정리", "financial_view"),
    ("증권사별 목표주가", "broker_targets"),
    ("커뮤니티 게시글", "community"),
    ("전자공시", "disclosure"),
    ("거시경제 지표", "macro"),
    ("업종·매크로 뉴스", "sector_news"),
    ("해외 상장분(ADR) 괴리율", "adr"),
]


def load(ticker: str | None = None) -> dict:
    """저장된 분석. 종목이 다르면 빈 dict(다른 종목 분석을 보여주면 안 된다)."""
    try:
        with open(STORE, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {}
    if ticker and data.get("ticker") != ticker:
        return {}
    return data


def save(payload: dict) -> None:
    try:
        os.makedirs(os.path.dirname(STORE) or ".", exist_ok=True)
        tmp = STORE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        os.replace(tmp, STORE)          # 쓰다 만 파일을 화면이 읽지 않게 통째로 바꾼다
    except Exception:
        pass


def age_note(payload: dict, now: dt.datetime | None = None) -> str:
    """만든 지 얼마나 됐는지 한 마디. 장중에는 몇 분만 지나도 주가가 달라진다."""
    when = (payload or {}).get("time")
    if not when:
        return ""
    try:
        made = dt.datetime.strptime(str(when), "%Y-%m-%d %H:%M").replace(tzinfo=KST)
    except Exception:
        return ""
    mins = int(((now or dt.datetime.now(KST)) - made).total_seconds() // 60)
    if mins < 1:
        return "방금"
    if mins < 60:
        return f"{mins}분 전"
    if mins < 24 * 60:
        return f"{mins // 60}시간 {mins % 60}분 전"
    return f"{mins // (24 * 60)}일 전"
