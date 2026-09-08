"""AI 분석을 만들어 저장한다. **streamlit을 import하지 않는다.**

수집기(collector.py)와 대시보드(app.py)가 같이 쓴다. 예전에는 이 함수가 app.py 안에
있었고, 그래서 분석은 브라우저가 대시보드를 열어 둔 시각에만 만들어졌다. Streamlit은
화면이 붙어야 스크립트를 돌리기 때문이다(실측: 21:32 다음 생성이 11시간 뒤).

재료는 전부 ai_inputs가 만든다. 여기는 그것들을 한꺼번에 받아 프롬프트에 태우고
ai_analysis.save()로 남기는 일만 한다.
"""
import datetime as dt
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import ai_analysis
import analyst_digest
import analyst_targets
import disclosure
import financial_digest
from ai_inputs import (
    ADR_HOST_TICKER,
    ADR_SHARE_RATIO,
    DEFAULT_LOOKBACK_DAYS,
    INVESTOR_COLUMNS,
    _dated_digest,
    _to_number,
    build_capex_summary,
    build_community_summary,
    build_consensus_trend_summary,
    build_dram_summary,
    build_early_signal_summary,
    build_foreign_desk_summary,
    build_intraday_summary,
    build_market_flow_summary,
    build_market_state_summary,
    build_over_market_summary,
    build_overheat_summary,
    build_price_context,
    build_recent_price_summary,
    build_short_sale_summary,
    fetch_adr_baseline,
    fetch_adr_quote,
    fetch_analyst_reports,
    fetch_daily_ohlcv,
    fetch_disclosures,
    fetch_earnings_calendar,
    fetch_investor_netbuy,
    fetch_macro_summary,
    fetch_news_with_summary,
    fetch_sector_news,
    fetch_stock_snapshot,
    fetch_trendforce_news,
    generate_ai_analysis,
)

KST = dt.timezone(dt.timedelta(hours=9))


def _log_ai_analysis_error(exc: Exception) -> None:
    """분석 생성 실패는 화면을 막지 않는다. 저장된 이전 분석이 계속 보이면 되고,
    실패 사실만 남겨 둔다(연속 실패하면 화면의 기준 시각이 안 바뀌는 것으로 드러난다)."""
    try:
        print(f"[ai_analysis] 생성 실패: {type(exc).__name__}: {exc}", flush=True)
    except Exception:
        pass


def build_and_save(ticker: str, stock_name: str, use_search: bool = True,
                   stream_to=None) -> bool:
    """분석을 만들어 파일에 저장한다. 화면은 이 파일을 읽기만 한다.

    stock_name은 종목코드가 붙지 않은 이름("SK하이닉스")이다. 프롬프트에 넣을 이름표는
    여기서 f"{stock_name}({ticker})"로 만들기 때문에, 코드가 붙은 문자열을 넘기면
    "SK하이닉스(000660)(000660)"이 된다.

    자동 새로고침 시각과 '지표 새로고침' 버튼이 이 함수를 부른다. 예전처럼 화면에서
    직접 만들지 않는 이유는 ai_analysis.py 주석에 적어 뒀다(매번 30~100초를 기다려야 했다).
    반환값은 성공 여부.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        return False
    snapshot = None
    try:
        snapshot = fetch_stock_snapshot(ticker)
    except Exception:
        snapshot = {}
    investor_df = pd.DataFrame()
    try:
        investor_df = fetch_investor_netbuy(ticker, DEFAULT_LOOKBACK_DAYS)
    except Exception:
        pass
    # 화면이 session_state에 남기던 장 상태·현재가. 나머지 재료가 이 값을 참조하므로
    # 병렬 배치보다 먼저 잡는다(호출 한 번, 0.2초).
    _price = build_price_context(ticker)

    try:
        price_summary = _price["summary"]

        # 바깥에서 받아오는 재료는 전부 서로 무관하다. 하나씩 순서대로 받으면
        # 왕복 지연이 그대로 쌓인다(실측 순차 3.9초 + 병렬 3.7초 = 합계 7.6초).
        # 한 번에 몰아서 받으면 가장 느린 하나(커뮤니티·조기신호)만큼만 걸린다.
        def _quiet(fn):
            try:
                return fn()
            except Exception:
                return ""

        cur_close = _price["value"]
        cur_close = int(cur_close) if cur_close else None
        _snap = snapshot or {}
        _fetch_jobs = (
            lambda: fetch_news_with_summary(stock_name),
            lambda: fetch_analyst_reports(ticker),
            lambda: fetch_trendforce_news(),
            lambda: fetch_macro_summary(),
            lambda: (fetch_sector_news(stock_name) if use_search else ""),
            # 본문은 '오늘 것'만 받는다. 그 앞의 공시는 disclosure.py가 15건을
            # 본문까지 읽어 정리해 두므로(아래 disclosure_view_md), 3일치 본문을
            # 여기서 또 넣으면 같은 내용이 프롬프트에 두 번 들어간다.
            lambda: fetch_disclosures(ticker, count=10, body_days=1),
            lambda: build_intraday_summary(ticker, cur_close),
            lambda: build_over_market_summary(ticker, cur_close),
            lambda: build_market_flow_summary(),
            lambda: build_community_summary(ticker, stock_name),
            lambda: fetch_daily_ohlcv(ticker, DEFAULT_LOOKBACK_DAYS),
            lambda: build_capex_summary(),
            lambda: build_early_signal_summary(ticker),
            lambda: build_recent_price_summary(ticker, market_open=_price["market_open"]),
            lambda: fetch_earnings_calendar(),
            lambda: build_consensus_trend_summary(ticker, _snap),
            lambda: build_market_state_summary(_price["market_open"]),
            lambda: build_short_sale_summary(ticker),
            lambda: build_foreign_desk_summary(ticker),
            # 리포트 본문을 원문 그대로 넣으면 12,000자가 붙어 프롬프트가 두 배가 된다.
            # 이미 본문을 읽고 정리해 둔 요약(약 1,500자)을 대신 넘긴다.
            lambda: _dated_digest(analyst_digest.load()),
            # 증권사별 목표주가는 파일에서 읽기만 한다(수집기가 채운다). 비용 없음.
            lambda: analyst_targets.summary_md(),
            # 공시 요약도 파일에서 읽기만 한다. 수집기가 새 공시가 뜰 때만 만든다.
            lambda: _dated_digest(disclosure.load()),
            # 재무 요약은 여태 재무 탭에서만 쓰고 AI 분석에는 안 넣고 있었다.
            # 매출·영업이익 추이와 추정치는 분석의 바탕이라 같이 넘긴다.
            lambda: _dated_digest(financial_digest.load()),
            # 예전에는 이 둘이 탭 렌더가 채우는 전역이었다. 이제 탭과 무관하게 만든다.
            lambda: build_overheat_summary(ticker),
            lambda: build_dram_summary(),
        )
        with ThreadPoolExecutor(max_workers=len(_fetch_jobs)) as _pool:
            (news_items, reports_df, trendforce_df, macro_md, sector_news_md,
             disclosure_md, intraday_md, over_market_md, market_flow_md,
             community_md, _ohlcv, capex_md, early_signal_md, recent_price_md,
             earnings_md, consensus_md, market_state_md, short_sale_md,
             foreign_desk_md, analyst_view_md, broker_targets_md,
             disclosure_view_md, financial_view_md,
             _overheat, _dram) = list(_pool.map(_quiet, _fetch_jobs))

        # 실패한 자리에는 _quiet가 ""를 넣는다. 표를 기대하는 쪽은 빈 표로 되돌린다.
        news_items = news_items or []
        if not isinstance(reports_df, pd.DataFrame):
            reports_df = pd.DataFrame()
        if not isinstance(trendforce_df, pd.DataFrame):
            trendforce_df = pd.DataFrame()
        if not isinstance(_ohlcv, pd.DataFrame):
            _ohlcv = pd.DataFrame()

        if not investor_df.empty:
            recent = investor_df.tail(5)
            # 어느 날짜까지의 확정치인지 밝힌다. 안 밝히면 장중에 AI가
            # 전 거래일 수급을 '오늘 외국인이 샀다'로 옮겨 적는다.
            _span = ""
            try:
                _span = (f" (확정 구간 {recent.index.min():%Y-%m-%d}"
                         f"~{recent.index.max():%Y-%m-%d})")
            except Exception:
                pass
            # 순매수 열만 돈다. 같은 표에 거래량·종가가 붙어 있어서 전체 열을 돌면
            # '거래량 순매수 합계' 같은 말이 안 되는 줄이 생긴다.
            lines = [
                f"- 최근 {len(recent)}거래일 {col} 순매수 합계: {recent[col].sum():+,.0f}주{_span}"
                for col in INVESTOR_COLUMNS if col in recent.columns
            ]
            # 거래량은 수급보다 먼저 확정된다. _ohlcv는 위 병렬 배치에서 이미 받아둔 값이다.
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

        headlines = [n["제목"] for n in news_items]
        news_md = "\n".join(
            f"- {n['제목']}" + (f" [{n['시점']}]" if n.get("시점") else " [게재 시점 미확인]")
            + (f"\n  요약: {n['요약']}" if n["요약"] else "") for n in news_items
        )
        reports_md = "\n".join(
            f"- [{row['증권사']}] {row['제목']} ({row['작성일']})" for _, row in reports_df.iterrows()
        )
        trendforce_md = "\n".join(
            f"- {row['제목']} ({row['날짜']})" for _, row in trendforce_df.iterrows()
        )

        snap = snapshot or {}
        snapshot_lines = [
            f"- 컨센서스 목표주가(FnGuide): {snap.get('목표주가') or 'N/A'}원 "
            f"(투자의견 평균 {snap.get('투자의견') or 'N/A'}/5, 기준일 {snap.get('컨센서스일자') or '-'})",
            # 후행 PER이다. 재무 탭의 예상 PER(2026E)과 값이 다른 게 정상이라 밝혀 둔다.
            f"- PER {snap.get('PER') or 'N/A'} · EPS {snap.get('EPS') or 'N/A'} "
            "(최근 실적 기준 후행값. 재무 탭의 추정 PER과는 다른 지표다)",
            # 네이버가 주는 52주 고저는 장중 고가·저가 기준이라 종가 기록과 다를 수 있다.
            f"- 52주 최고 {snap.get('52주최고') or 'N/A'} / 최저 {snap.get('52주최저') or 'N/A'} (장중가 기준)",
            # 소진율(취득한도 대비)과 아래 수급표의 보유율(상장주식 대비)은 다른 값이다.
            # 둘 다 프롬프트에 들어가는데 이름이 비슷해서, 안 밝히면 같은 지표가 어긋난 걸로 읽힌다.
            f"- 시가총액 {snap.get('시가총액') or 'N/A'} · 외국인소진율 "
            f"{snap.get('외국인소진율') or 'N/A'} (취득한도 대비. 아래 수급표의 '외국인보유율'은 "
            "상장주식 대비라 값이 조금 다른 게 정상이다)",
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


        adr_md = ""
        if ticker == ADR_HOST_TICKER:
            try:
                adr_q, adr_base = fetch_adr_quote(), fetch_adr_baseline()
                cur_price = _price["value"]
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
            f"{stock_name}({ticker})",
            time_label, price_summary, supply_summary, headlines, reports_md,
            _dram, community_md, _overheat,
            trendforce_md, snapshot_md, news_md, use_search,
            macro_md, sector_news_md, adr_md,
            disclosure_md, over_market_md, intraday_md, market_flow_md,
            capex_md, early_signal_md, recent_price_md, earnings_md, consensus_md,
            market_state_md, short_sale_md, foreign_desk_md, analyst_view_md,
            broker_targets_md, disclosure_view_md, financial_view_md,
            _stream_to=stream_to,
        )

        ai_analysis.save({
            "ticker": ticker,
            "stock_name": stock_name,
            "text": analysis,
            "time": time_label,
            "search_note": search_note,
            "headlines": headlines,
            "market_state": market_state_md,
            # 프롬프트에는 들어가는데 저장을 안 해서 '원본 데이터 보기'에서만 빠져 있었다.
            "overheat": _overheat,
            "dram": _dram,
            "recent_price": recent_price_md,
            "intraday": intraday_md,
            "over_market": over_market_md,
            "market_flow": market_flow_md,
            "early_signal": early_signal_md,
            "capex": capex_md,
            "earnings": earnings_md,
            "consensus": consensus_md,
            "short_sale": short_sale_md,
            "foreign_desk": foreign_desk_md,
            "analyst_view": analyst_view_md,
            "disclosure_view": disclosure_view_md,
            "financial_view": financial_view_md,
            "broker_targets": broker_targets_md,
            "community": community_md,
            "disclosure": disclosure_md,
            "macro": macro_md,
            "sector_news": sector_news_md,
            "adr": adr_md,
        })
        # 예약 시각에 실제로 돌았는지 나중에 확인할 수 있게 한 줄 남긴다.
        print(f"[ai_analysis] 생성 완료 {time_label} ({len(analysis):,}자)", flush=True)
        return True
    except Exception as exc:
        _log_ai_analysis_error(exc)
        return False


# ── 수집기용 ────────────────────────────────────────────────────────────────
# 예약 시각(정시)을 지날 때마다 한 번씩 만든다. 화면이 붙어 있든 없든 돈다.
#
# 08~20시 13회로 잡은 이유: 무료 등급이 모델당 하루 20회쯤이고, 리포트·공시·재무
# 요약이 llm.py로 0~5회를 먼저 쓴다. 24시간(24회)으로 벌리면 그 상한을 넘겨 하루
# 후반부가 폴백 모델로 떨어지는데, 그러면 thinking_level=high로 올려놓은 품질이
# 그날 오후에 통째로 사라진다. 시간대는 장 전(8시) · 장중 · 마감 후 수급 확정
# (16~19시) · 미국장 개장 전(20시)을 덮는다.
REFRESH_HOURS = [int(h) for h in os.environ.get(
    "AI_ANALYSIS_REFRESH_HOURS", "8,9,10,11,12,13,14,15,16,17,18,19,20").split(",") if h.strip()]

_last_slot: dt.datetime | None = None


def _last_passed_slot(hours: list[int], now: dt.datetime) -> dt.datetime | None:
    """hours 중 now 이전에 지난 가장 최근 시각. 오늘 첫 시각도 안 지났으면 None."""
    passed = [now.replace(hour=h, minute=0, second=0, microsecond=0)
              for h in sorted(hours)
              if now.replace(hour=h, minute=0, second=0, microsecond=0) <= now]
    return passed[-1] if passed else None


def tick(now: dt.datetime, ticker: str, stock_name: str, log=print) -> None:
    """예약 시각이 지났으면 분석을 새로 만든다. 조건이 안 맞으면 즉시 돌아간다.

    주말에도 돈다. 금요일 장이 끝난 뒤에도 미국 증시·DRAM 시세·공시는 계속 바뀌고,
    월요일 아침에 열었을 때 지난 금요일 분석이 떠 있으면 그게 더 헷갈린다.
    """
    global _last_slot
    slot = _last_passed_slot(REFRESH_HOURS, now)
    if slot is None or slot == _last_slot:
        return
    # 먼저 표시해 둔다. 생성이 실패해도 같은 시각을 매 순회마다 다시 시도하지 않는다
    # (한 번에 30~100초가 걸려서, 실패를 반복하면 수집기의 시세 기록이 밀린다).
    _last_slot = slot
    started = time.monotonic()
    try:
        made = build_and_save(ticker, stock_name)
    except Exception as exc:
        log(f"[AI분석] 실패: {type(exc).__name__}: {exc}")
        return
    took = time.monotonic() - started
    log(f"[AI분석] {slot:%H:%M} 예약분 생성 {'완료' if made else '실패'} ({took:.0f}초)")
