# 코드 지도

> **큰 파일을 통째로 읽지 말 것.** 여기서 줄 번호를 찾아 그 범위만 열면 된다.
> 줄 번호는 편집하면 밀린다 — 어긋나면 `grep -n '^def 이름' 파일`로 다시 잡는다.
> 아래에 없는 함수는 전부 그 위 구간에 속한 보조 함수다.

세 파일로 나뉘어 있다. **streamlit을 import하는 건 `app.py` 하나뿐이다.**

| 파일 | 줄 | 역할 |
|---|---|---|
| `app.py` | 4,608 | 화면. 탭 렌더와 자동 새로고침 |
| `ai_inputs.py` | 2,508 | AI 분석 재료를 만드는 계층. 수집기와 공용 |
| `ai_report.py` | 356 | 재료를 모아 프롬프트에 태우고 저장 |

---

# ai_inputs.py — 재료 계층

streamlit을 import하지 않는다. 그래서 수집기가 화면 없이 그대로 부를 수 있다.
캐시는 걸려 있지 않다 — 화면 쪽 ttl은 `app.py`의 `_CACHED` 표가 다시 씌운다.

**LLM 게이트웨이** `_is_quota_error` 142 · `_gemini_models_to_try` 146 · `_mark_exhausted` 155 · `_gemini_gen_config` 162 · `_looks_truncated` 169 · `_is_thinking_unsupported` 180 · `_is_timeout` 195 · **`_stream_gemini` 200** · `_call_gemini` 285

**시세·수급** `fetch_current_price` 321 · `fetch_intraday_price` 325 · `fetch_investor_netbuy` 366 **(오래된 것부터 온다 — 최신순 아님)** · `fetch_daily_ohlcv` 391 · `fetch_stock_snapshot` 698 · `fetch_market_flow` 936 · `fetch_foreign_desk` 963 · `fetch_foreign_hold_ratio` 1753

**백테스트** `fetch_backtest_history` 436 · `fetch_backtest_history_live` 484 · `run_overheat_backtest` 528 · `run_boolean_pattern_backtest` 600

**뉴스·공시·리포트** `fetch_news_with_summary` 885 · `fetch_disclosures` 1049 **(목록+오늘 본문만. 그 앞 15건은 `disclosure.py` 정리본 담당)** · `fetch_sector_news` 1212 · `fetch_trendforce_news` 1256 · `fetch_analyst_reports` 1291 · `fetch_community_posts` 1349 · `fetch_dc_gallery_posts` 1378

**DRAM·재무·Capex** `fetch_dram_module_prices` 1493 · `fetch_dram_chip_prices` 1506 · `fetch_fnguide_page` 1594 · `fetch_short_balance` 1598 · `fetch_bigtech_capex` 1619

**ADR·매크로** `fetch_macro_summary` 660 · `fetch_adr_quote` 772 · `fetch_adr_baseline` 833 · `fetch_earnings_calendar` 1854

**요약 만들기** — 프롬프트에 그대로 들어가는 문장을 만든다
`build_foreign_desk_summary` 1000 · `build_market_flow_summary` 1025 · `build_community_summary` 1097 · `build_over_market_summary` 1137 · `build_intraday_summary` 1169 · `build_market_state_summary` 1644 · `build_capex_summary` 1677 · `build_early_signal_summary` 1698 · `build_recent_price_summary` 1770 · `build_consensus_trend_summary` 1898 · `build_short_sale_summary` 1945 · **`build_price_context` 2277** · **`build_overheat_summary` 2325** · **`build_dram_summary` 2383**

굵게 표시한 셋은 화면 의존을 대신하려고 새로 만든 것이다.
`build_price_context`는 화면이 `session_state`에 남기던 장 상태·현재가를,
나머지 둘은 **탭 렌더가 전역에 채우던 요약**을 탭 없이 만든다.

**프롬프트 본체 — `generate_ai_analysis` 2001** (약 250줄 f-string)
여기가 품질의 중심이다. 새 재료를 넣을 땐 **입력 섹션과 출력 형식을 같이** 고쳐야 한다
(출력에 자리가 없으면 모델이 한 글자도 안 쓴다 — CONTEXT §5).
`_tab_summary` 2405는 계산이 안 된 항목을 '없다'고 못박아 넘긴다.

---

# ai_report.py — 생성·저장

`_log_ai_analysis_error` · **`build_and_save(ticker, stock_name, ...)`** — 재료를 병렬로 모아
프롬프트에 태우고 `ai_analysis.save()`. **`tick(now, ...)`** — 예약 시각(정시)마다 한 번.

`REFRESH_HOURS`는 08~20시 13회. 늘리기 전에 CONTEXT §4의 무료 한도 이야기를 볼 것.
`stock_name`은 **코드가 붙지 않은 이름**이다(이름표는 여기서 `이름(코드)`으로 만든다).

수집기와 화면 버튼이 둘 다 이 함수를 부른다. **화면은 예약 시각에 만들지 않는다** —
양쪽이 다 만들면 하루 26회가 되어 무료 한도를 넘긴다.

---

# app.py — 화면

## 골격 (이 순서로 실행된다)

| 줄 | 무엇 |
|---|---|
| 33~ | `ai_inputs`에서 이름 가져오기 + `_CACHED`로 캐시 ttl 다시 씌우기 |
| 167 | `ALL_TAB_LABELS` — 탭 14개 정의 |
| 570 | `DEFAULT_HIDDEN_TAB_LABELS` — 기본 숨김 6개 |
| 1518 | `render_current_price` (558줄. 현재가·장중차트·ADR) |
| 2085~ | 자동 새로고침 시각 상수 (AI 분석은 여기 없다 — 수집기 소관) |
| 2148 | `REFRESH_GROUPS` — 수급·DRAM·Capex 3묶음 |
| 2406 | 탭 생성 + Plotly 폭 재측정 iframe (`st.iframe`은 `height=0`을 안 받는다) |
| 4572 | `_TAB_RENDERERS` — 라벨 → 렌더 함수. 그 뒤가 디스패치 루프이고, **루프가 끝난 뒤** 버튼발 AI 생성 |

## 탭 렌더

| 탭 | 줄 |
|---|---|
| 수급 현황 | `_render_tab_supply` 2436 |
| 가격 과열도 | `_render_tab_overheat` 2546 |
| 선물 경보 | `_render_tab_futures` 2808 |
| 통합 신호 | `_render_tab_composite` 2984 |
| 매매 신호 | `_render_tab_signal` 3130 |
| 하락 조기신호 | `_render_tab_decline` 3311 |
| 상승 조기신호 | `_render_tab_rally` 3455 |
| DRAM 시세 | `_render_tab_dram` 3614 |
| 빅테크 Capex | `_render_tab_capex` 3730 |
| 재무 데이터 | `_render_tab_financials` 3899 (재무 요약 `_render_financial_digest` 3852) |
| 공시 | `_render_tab_disclosure` 4130 |
| 애널리스트 | `_render_tab_analyst` 4202 (증권사별 목표주가 `_render_broker_targets` 4023) |
| 커뮤니티 | `_render_tab_community` 4275 |
| AI 분석 | `_render_tab_ai` 4421 |

`_render_saved_ai_analysis` 4528 — 파일만 읽어 그린다. 기준 시각·경과를 표시하고 60분이 넘으면 경고.
**여기서 만들지 않는다.**

차트 보조: `_render_dram_price_table` 1262 · `_render_foreign_desk_line` 1064 · `_render_dram_trend_chart` 2377
