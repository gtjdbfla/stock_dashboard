# app.py 지도 (380KB · 6,900줄 · 함수 154개)

> **app.py를 통째로 읽지 말 것.** 여기서 줄 번호를 찾아 그 범위만 열면 된다.
> 줄 번호는 편집하면 밀린다 — 어긋나면 `grep -n '^def 이름' app.py`로 다시 잡는다.
> 아래에 없는 함수는 전부 그 위 구간에 속한 보조 함수다.

## 골격 (이 순서로 실행된다)

| 줄 | 무엇 |
|---|---|
| 64 | `ALL_TAB_LABELS` — 탭 14개 정의 |
| 707 | `DEFAULT_HIDDEN_TAB_LABELS` — 기본 숨김 6개 |
| 4168~4173 | 자동 새로고침 시각 상수 4종 |
| 4229 | `REFRESH_GROUPS` — (키, 이름, 시각, 갱신함수) 4묶음 |
| 6880 | `_TAB_RENDERERS` — 라벨 → 렌더 함수 |
| 그 뒤 | 탭 렌더 루프 → **루프가 끝난 뒤** AI 분석 생성 |

## 1. LLM 호출 (163~396)

app 자체 게이트웨이. `llm.py`와 **별개**이므로 설정을 따로 맞춰야 한다.

`_is_quota_error` 163 · `_gemini_models_to_try` 172 · `_mark_exhausted` 181 · `_gemini_gen_config` 201 · `_looks_truncated` 216 · `_md_safe` 227 · `_md_stream_safe` 238 · `_is_thinking_unsupported` 247 · `_is_timeout` 271 · **`_stream_gemini` 276** · **`_call_gemini` 349**

## 2. UI 헬퍼 (397~457)

`_subheader_with_help` 397 · `_bold_label_with_help` 408 · `_metric_with_help` 417 · `_style_chart_mobile` 440

## 3. 시세·수급 수집 (458~1480)

`fetch_stock_search` 458 · `fetch_current_price` 722 · `fetch_intraday_price` 727 · `fetch_investor_netbuy` 769 **(오래된 것부터 온다 — 최신순 아님)** · `fetch_daily_ohlcv` 795

**과열도 백테스트** `fetch_backtest_history` 841 · `forward_max_drawdown` 900 · `forward_max_gain` 911 · `two_proportion_ztest` 922 · `run_overheat_backtest` 938 · `run_overheat_threshold_strategy` 1010

**선물·통합신호(프롬프트에서 제외됨)** `fetch_futures_foreign_history` 1069 · `run_futures_decline_backtest` 1136 · `run_boolean_pattern_backtest` 1197 · `build_composite_dataset` 1315 · `compute_composite` 1392 · `compute_flow_signal` 1413

`fetch_yahoo_history` 1248 · `fetch_macro_summary` 1277

## 4. 종목 상태·ADR·뉴스 (1481~1760)

`fetch_stock_snapshot` 1481 · `fetch_adr_quote` 1616 · `fetch_adr_intraday` 1678 · `fetch_adr_baseline` 1699 · `_to_number` 1733 · **`_clean_text` 1751** (`<mark>` 조각 붙는 것 방지 — `get_text(strip=True)` 쓰지 말 것) · `fetch_news_with_summary` 1757

## 5. 수급·공시·커뮤니티 원본 (1809~2480)

`fetch_market_flow` 1832 · `fetch_foreign_desk` 1877 · `build_foreign_desk_summary` 1936 · `build_market_flow_summary` 1961 · **`fetch_disclosures` 1986** (목록+오늘 본문만. 그 앞 15건은 `disclosure.py` 정리본 담당 — 둘 다 본문 넣으면 중복) · `build_community_summary` 2034 · `build_over_market_summary` 2074 · `build_intraday_summary` 2106 · `fetch_sector_news` 2152 · `fetch_trendforce_news` 2197 · `fetch_analyst_reports` 2235 · `fetch_community_posts` 2295 · `classify_sentiment` 2308 · `fetch_dc_gallery_posts` 2348 · `render_wordcloud_image` 2434 · `curate_good_dc_posts` 2447

## 6. DRAM (2482~2708)

`_parse_dram_table` 2509 · `fetch_dram_module_prices` 2542 · `fetch_dram_chip_prices` 2556 · `_signed_pct` 2571 · **`_stale_note` 2597** (갱신 멈춤 경고) · `_period_change_pct` 2629 · `save_dram_snapshot` 2646

## 7. 재무·Capex (2709~2825)

`fetch_fnguide_page` 2763 · `fetch_financials` 2767 · `fetch_short_balance` 2774 · `fetch_target_price_history` 2787 · `fetch_bigtech_capex` 2797

## 8. AI 분석 재료 만들기 (2826~3277)

`build_market_state_summary` 2826 · `build_capex_summary` 2885 · `build_early_signal_summary` 2906 · `fetch_foreign_hold_ratio` 2962 · `build_recent_price_summary` 2979 · `fetch_earnings_calendar` 3063 · `record_consensus_snapshot` 3094 · `build_consensus_trend_summary` 3135 · `build_short_sale_summary` 3182 · `_price_summary_fallback` 3219 · `_log_ai_analysis_error` 3234 · **`_dated_digest` 3243** (미리 만든 요약에 기준일 붙이기) · **`_tab_summary` 3260 / `missing_tab_summaries` 3267** (탭 안 그려졌을 때 '계산되지 않음' 못박기)

## 9. **프롬프트 본체 — `generate_ai_analysis` 3278~3532**

여기가 품질의 중심이다. 250줄짜리 f-string 하나. 새 재료를 넣을 땐 **입력 섹션과 출력 형식을 같이** 고쳐야 한다(출력에 자리가 없으면 모델이 한 글자도 안 쓴다).

## 10. 화면 상단 (3533~4167)

`_live_deviation` 3533 · `_korea_session_now` 3556 · `_note_optional_failure` 3577 · `render_current_price` 3601

## 11. 새로고침 체계 (4168~4431)

`_refresh_market_data_caches` 4183 · `_refresh_dram_caches` 4206 · `_refresh_bigtech_capex_cache` 4214 · **`_mark_ai_analysis_due` 4219** (여기서 만들지 않고 표시만 남긴다) · `_refresh_all_indicator_caches` 4237 · `_last_passed_schedule_slot` 4257 · `_next_schedule_slot` 4267 · `_get_group_refresh_state` 4288 · `_post_close_catch_up` 4315 · `_auto_refresh_indicators` 4338

## 12. 탭 렌더 (4432~6879)

| 탭 | 줄 |
|---|---|
| 수급 현황 | `_render_tab_supply` 4516 |
| 가격 과열도 | `_render_tab_overheat` 4626 |
| 선물 경보 | `_render_tab_futures` 4888 |
| 통합 신호 | `_render_tab_composite` 5064 |
| 매매 신호 | `_render_tab_signal` 5210 |
| 하락 조기신호 | `_render_tab_decline` 5391 |
| 상승 조기신호 | `_render_tab_rally` 5535 |
| DRAM 시세 | `_render_tab_dram` 5694 |
| 빅테크 Capex | `_render_tab_capex` 5810 |
| 재무 데이터 | `_render_tab_financials` 5979 |
| 공시 | `_render_tab_disclosure` 6210 |
| 애널리스트 | `_render_tab_analyst` 6282 (증권사별 목표주가 `_render_broker_targets` 6103) |
| 커뮤니티 | `_render_tab_community` 6355 |
| AI 분석 | `_render_tab_ai` 6501 |

차트 보조: `_add_regime_shading` 4432 · `_add_downtrend_shading` 4447 · `_render_dram_trend_chart` 4457 · `_render_dram_price_table` 2614 · `_fin_style` 5962

## 13. AI 분석 생성·표시 (6606~끝)

**`_build_and_save_ai_analysis` 6606** — 재료 모아 프롬프트 태우고 `ai_analysis.save()`.
**`_render_saved_ai_analysis` 6836** — 파일만 읽어 그린다. 기준 시각·경과 표시, 60분 넘으면 경고.

호출 지점은 파일 **맨 끝**, 탭 렌더 루프 다음이다. 이 위치가 핵심이다 — 과열도·DRAM 요약은 그 탭이 그려질 때 전역에 채워지므로, 더 앞(자동 새로고침 fragment)에서 만들면 '계산되지 않음'으로 빠진다.
