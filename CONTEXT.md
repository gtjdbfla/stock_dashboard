# SK하이닉스 대시보드 — 작업 컨텍스트

> 새 세션은 **이 파일만** 읽는다. app.py를 뒤져야 하면 [MAP.md](MAP.md)에서 줄 번호를 찾아 그 범위만 연다.
> **변경 이력은 쓰지 않는다** — "지금 이렇게 되어 있다"만. 이력은 git log 담당.
> 잘라낼 기준은 크기가 아니라 종류다: **"언제 무엇을 바꿨다"는 줄이 보이면 지운다.**
> §4·§5의 실측 숫자는 길어도 남긴다 — 이게 이미 끝난 실험을 다시 돌리는 걸 막는다(그게 제일 비싸다).

## 1. 목적

SK하이닉스(000660) 단일 종목을 수급·과열도·DRAM 시세·공시·애널리스트·재무까지 한 화면에서 보고, 마지막에 **AI가 그날의 상태를 서술**하는 Streamlit 대시보드. **사용자의 최우선 목표는 언제나 "AI 분석의 품질"이다.**

## 2. 배포

git 저장소 — `origin https://github.com/gtjdbfla/stock_dashboard.git`, `main`. 소스 `C:\Directory\claude\stock_dashboard\`, 서버 `~/stock_dashboard/`.

| 컨테이너 | 역할 | 포트 |
|---|---|---|
| `hynix-dashboard` | Streamlit 화면 | `8501:8501` (LAN) |
| `hynix-collector` | 같은 이미지, `command: python collector.py` | 없음 |
| `hynix-caddy` | 리버스 프록시 | `127.0.0.1:8080` |

- **외부 공개 = Tailscale Funnel 443 → localhost:8080 → Caddy → Streamlit 8501.** HTTPS는 Funnel이 처리하므로 Caddy는 평문만 받고, `127.0.0.1`에 묶어 직접 노출을 막는다. 현재 **로그인 없음**(되돌리려면 Caddyfile `basic_auth` + `.env`의 `DASHBOARD_USER`/`DASHBOARD_PASSWORD_HASH`).
- `duckdns_update.sh`가 크론(`*/10`)으로 공인 IP 갱신.
- **배포는 Claude가 직접 한다**(사용자 상시 위임): 소스 디렉터리 반영 → `docker compose build && up -d` → md5 대조·헬스체크. **`docker cp`로 컨테이너만 패치 금지**(재빌드에 날아간다).
- `.env`는 gitignore. 필수 키는 `GEMINI_API_KEY` 하나. **값을 파일·커밋에 절대 쓰지 말 것.**

## 3. 파일 지도

`app.py`는 4,605줄이다(→ MAP.md). 나머지 모듈은 **전부 streamlit을 import하지 않는다** — 수집기에서도 돌아야 하므로.

| 파일 | 역할 |
|---|---|
| `collector.py` | 프리장/애프터장 상시 수집 **별도 프로세스** |
| `over_market.py` | 시간외(NXT) 시세 수집·파싱. app·collector 공용 |
| `flow_probe.py` | 외국인·기관 일별 수급을 마감 후 5분 간격 적재 |
| `analyst_digest.py` | 리포트 목록 + AI 요약 |
| `analyst_targets.py` | 증권사별 목표주가 → 컨센서스 직접 집계 |
| `disclosure.py` | 공시 목록·본문 + AI 요약 (KOSCOM, 네이버 중계) |
| `financial_digest.py` | 재무 요약 AI |
| `fnguide.py` | FnGuide `snpFinancial` 임베디드 JSON 파싱 |
| `ai_inputs.py` | **AI 분석 재료 계층.** app.py에서 떼어낸 함수 72개. streamlit 미import |
| `ai_report.py` | **분석 생성·저장 오케스트레이터.** 수집기와 화면 버튼이 같이 쓴다 |
| `daily_history.py` | 일별 시세·수급 이력 **CSV 저장/로드만**. 수집기가 채우고 화면이 읽는다 |
| `ai_analysis.py` | AI 분석 결과 **저장/로드만**. app을 import하지 않는다(순환 방지) |
| `llm.py` | `analyst_digest`·`financial_digest` **전용** LLM 호출 |
| `data/` | DRAM 이력 등. gitignore |

**탭 14개**(app.py:64), 기본 숨김 6개(app.py:707): 매매 신호·선물 경보·통합 신호·하락/상승 조기신호·커뮤니티.
**자동 새로고침**: 수급·과열도 16~19시 / DRAM 13·16·20시 / Capex 16시. 이 셋은 화면이 맡는다.
**AI 분석은 수집기가 08~20시 매시 정각에 만든다**(`ai_report.REFRESH_HOURS`). 화면은 저장된 것을 읽기만 하고,
예약 생성을 하지 않는다 — 양쪽이 다 만들면 하루 26회가 되어 무료 한도를 넘긴다. 버튼은 즉시 생성한다.

## 4. 확정된 결정 — 건드리지 말 것

- **`GEMINI_THINKING_LEVEL=high`가 AI 품질의 유일한 레버.** low는 나열된 줄에서 이름↔숫자 짝을 틀린다(한미반도체 -2.54% ← 실제 -2.49%, 옆 종목 값을 끌어옴). 프롬프트로 "검산해라"라고 지시해도 안 고쳐진다 — **설정으로만 고쳐진다.** 실측 low/high: 첫 글자 6.0/21.7초, 완료 21.8/34.9초, 출처태그 10/14개.
- **`GEMINI_STALL_SEC=120`.** high는 첫 글자까지 21.7~49.7초로 편차가 크다. 짧게 잡으면 붐비는 날 정상 응답이 '지연'으로 버려지고 폴백 모델 답이 나와 품질을 올린 의미가 사라진다.
- **모델 체인은 `3.6-flash → 3.7-flash → 3.5-flash → 3.5-flash-lite → flash-lite-latest`.** 무료 한도는 **모델마다 따로** 잡히므로, 온전한 모델을 체인에 둬야 기본 모델이 소진돼도 품질이 유지된다(예전 체인은 곧장 lite로 떨어졌고 2026-09-08 13시에 실제로 그랬다).
  - `gemini-3.7-flash`는 **쓸 수 있다**. 실제 프롬프트(18,547자) 실측: 첫 글자 13.4초, 완료 23.8초, 2,757자, 10개 섹션·7개 출처 갈래를 모두 채우고 근거 인용 14건. (이전 기록의 '빈 응답'은 낡았다.)
  - `gemini-3.8-flash`는 넣지 않는다. 짧은 프롬프트엔 답하지만 실제 프롬프트로는 두 번 다 `currently experiencing high demand`만 돌려줬다.
  - `gemini-pro-latest`·`gemini-3.1-pro-preview`는 **여전히 빈 응답**(0.3~0.5초 즉시).
  - **3.6 vs 3.7의 품질 우열은 아직 안 가려졌다.** 같은 프롬프트로 나란히 재야 하는데 그날 3.6 한도가 이미 소진돼 있었다. 다시 잴 땐 반드시 **같은 프롬프트**로 할 것 — 시각이 다르면 장중가가 달라져서 '지어낸 숫자' 지표가 그대로 망가진다.
- **`BODY_CHARS`(리포트 본문)는 2,000자 유지.** 5,000자로 늘려도 인용 숫자가 똑같았고(목표가 4개, 추정표 0건) 프롬프트만 9,288→20,984자로 불면서 증권사 언급은 오히려 7→6곳으로 줄었다. 목표주가·투자의견은 **표지(앞 2,000자)**에 다 있다.
- **프롬프트 길이는 속도와 무관하다 — 다이어트로 시간을 줄이려 하지 말 것.** 18,261자 38.9초 / 17,678자 49.7초 / 15,633자 48.1초. 그날 API가 붐비는 정도가 지배한다. 줄일 이유가 있다면 속도가 아니라 정보 중복 때문이어야 한다.
- **통합 신호·선물 경보는 프롬프트에서 뺐다**(사용자 판단: "맞지 않는 지표"). 탭은 남아 있지만 프롬프트에는 안 들어간다. **요청 없이 되살리지 말 것.**
- **공매도 잔고·대차잔고는 넣지 않는다.** 출처를 전부 훑고 무료로는 불가능하다고 결론냈고 사용자도 "포기"에 합의했다. 요청이 다시 오면 이 결론을 먼저 말하고 유일하게 남은 길(KRX OpenAPI 키 발급)만 제시한다. **스크래핑 우회는 제안하지 않는다.** 수급 압력은 외국인·기관 순매수, 보유율 추이, 코스피 전체 수급, 하락 조기신호로 상당 부분 커버된다.
- **목표주가는 네이버 리서치 상세가 유일한 길**(`m.stock.naver.com/api/research/company/{researchId}`의 `goalPrice`). 한경 컨센서스·wisereport·PDF 파싱은 전부 실측하고 탈락. 컨센서스는 최근 3개월 **증권사별 최신 1건씩만 평균**(다작 증권사가 평균을 좌우하지 않게). FnGuide와 값이 다른 건 정상(모집단 차이) — 통일하지 말고 라벨에 어느 집계인지 박는다.
- **`llm.py`와 app의 `_stream_gemini`는 별개다. 양쪽 설정을 따로 맞춰야 한다.**
- **수집기가 별도 컨테이너인 이유:** Streamlit은 브라우저가 접속해야 스크립트를 실행한다. 대시보드 안에 넣으면 아무도 안 보는 아침 프리장(08~09시)이 통째로 빈다.

## 5. 반복해서 걸린 함정

- **네이버 레거시 종목 페이지는 2026-09-10 저녁에 죽었다.** `finance.naver.com/item/frgn.naver`·`/item/board.naver`·`/research/company_list.naver`가 전부 `<table>` 0개짜리 같은 껍데기(116KB)를 200으로 돌려준다. `pd.read_html`이 lxml에서 "No tables found"로 실패하고 bs4로 넘어가다 html5lib가 없어 **`ImportError: Missing optional dependency 'html5lib'`**로 터졌다 — 에러 이름만 보면 의존성 문제 같지만 **html5lib를 깔아도 안 고쳐진다**(표가 없으니까). 같은 증상이 또 나오면 먼저 `resp.text.count("<table")`을 찍어 볼 것.
  - 옮긴 곳: 일별 수급·종가·거래량 → `m.stock.naver.com/api/stock/{code}/trend`(pageSize 최대 60, `page`는 무시되고 **`bizdate`가 커서** — 그 날짜보다 앞선 구간을 준다). 리서치 목록 → `/api/research/stock/{code}`(pageSize·page 정상), 상세 → `/api/research/company/{researchId}`(`goalPrice`·`opinion`·`attachUrl`을 구조화해서 준다. 정규식 파싱이 통째로 사라졌다).
  - **살아 있는 것**: `sise/investorDealTrendDay.naver`·`sise/programDealTrendDay.naver`·`item/sise_day.naver`(시장 전체·일봉 쪽 레거시 페이지는 그대로 표를 준다).
  - **못 살린 것**: 거래원 외국계추정합(`fetch_foreign_desk`, `FOREIGN_DESK_ENABLED=False`)과 종목토론방(`_fetch_board_page`). 모바일 앱에 이 화면 자체가 없다 — 종목 페이지가 부르는 API 212개를 훑어도 경로가 없었다.
- **`개인` 순매수를 `-(기관+외국인)`으로 유도하지 말 것.** 옛 frgn 표가 개인을 안 줘서 그렇게 썼는데 기타법인이 빠진다. 2026-08-20부터 하루 **약 64만주씩 거의 일정하게** 어긋났다(자사주 매입으로 보인다). 새 trend API는 `individualPureBuyQuant`를 직접 준다. `daily_history`도 개인을 저장한다(파생값이라 빼뒀던 걸 되돌렸다).
- **다크 화면의 작은 SVG 글자, 대비를 극단으로 올리지 말 것.** Plotly updatemenu 라벨 이력: (1) 흰색·`font-weight:600` → 가짜 굵기(synthetic bold)가 뭉개져 "글자가 깨진 것처럼 보인다"는 지적. (2) `font-weight` 대신 `paint-order: stroke fill`+같은 색 0.5px 외곽선으로 두께를 줬더니 이번엔 그 외곽선 자체가 "또 깨져 보인다"는 지적(검정 박스+흰 글자 조합에서). 두 번 다 원인은 굵기 트릭이 아니라 **배경과 글자 사이의 대비가 너무 컸던 것**이었다 — 극단적 대비(순검정/순백)는 작은 SVG 글자에서 항상 번짐(halation)을 만들고, 그걸 트릭으로 메우려 하면 트릭 자체가 또 깨져 보인다. 최종 해법은 트릭을 걷어내고 **대비를 애초에 낮추는 것**: 박스는 페이지 배경(#0e1117)보다 살짝만 밝은 톤(#1c1f26), 테두리는 이 앱이 실제 링크색으로 쓰는 옅은 파랑(#3d9df3, opacity 0.45)만 둘러 존재를 알리고, 글자는 순백 대신 옅은 회색(#e3e6ea)에 `font-weight:400`·외곽선 없이 그대로 둔다. `paint-order`/외곽선 트릭은 다시 꺼내지 말 것.
- **드롭다운 라벨은 항목들이 공유하는 토큰을 떼고 보여준다**(`_short_item_labels`). `DDR5 16Gb (2Gx8) 4800/5600` 26자가 들어가면 상자가 차트 폭의 절반(209/458px)을 먹는다. 공통 토큰을 떼면 136px. 전체 이름은 바로 위 표에 있다.
- **Caddy에 `reverse_proxy 컨테이너명:포트` 직접 지정 금지.** 재빌드로 IP가 바뀌어도 옛 IP를 물고 502를 낸다. 반드시 `dynamic a { name ... refresh 5s }`.
- **데이터를 프롬프트에 넣는 것만으로는 모델이 안 쓴다 — 출력 형식에 자리를 만들어야 한다.** 재무 요약을 넣었더니 한 글자도 인용하지 않았다(10개 섹션 어디에도 쓸 자리가 없었다). 출처 갈래에 `[재무]`를 추가하고 '지금 위치'에 한 줄을 요구하자 바로 인용했다. **새 재료엔 반드시 출력 형식도 같이 손볼 것.**
- **탭 렌더에 묶인 요약을 조심할 것.** `overheat_summary`·`dram_summary`는 전역이라 **그 탭이 그려질 때만** 채워진다. 그래서 AI 분석 생성은 반드시 **탭 렌더 루프 다음**에서 한다(MAP.md §13). 새 요약은 탭과 무관한 `build_*`로 만드는 쪽이 안전하다.
- **캐시를 '없을 때만 받기'로 짜지 말 것.** 한 번 실패해 껍데기가 저장되면 영영 그대로다. `url not in cache`가 아니라 `len(cache.get(url,"")) < 최소길이`(`BODY_MIN_CHARS=300`). `analyst_targets`는 목표가가 빈 리포트를 `RETRY_EMPTY_DAYS=7`일치만 재시도한다(컨콜 후기 등은 몇 번을 열어도 None).
- **미리 만든 요약에는 기준일을 붙여 넘긴다**(`_dated_digest()`). 며칠 전 요약이 그대로 쓰이는 게 정상인데, 날짜를 안 주면 모델이 '오늘'로 읽는다.
- **이름이 비슷한 지표를 혼동하지 말 것**(전부 정상이고 정의가 다를 뿐): 외국인**소진율**(취득한도 대비, `totalInfos.foreignRate`) vs **보유율**(상장주식 대비, `dealTrendInfos.foreignerHoldRatio`) · PER 후행 vs 재무탭 추정 · 52주 고저는 **장중가** 기준이라 종가 최대·최소와 다르다.
- **`fetch_investor_netbuy`는 오래된 것부터** 온다. 최신순으로 착각해 "값이 안 맞는다"고 오진한 적 있다.
- **"KRX가 응답이 없다"는 오진 반복 금지.** 원인은 파이썬 stdout 버퍼링이었고 KRX는 멀쩡하다.
- **탭 안의 위젯은 그 탭이 `@st.fragment`일 때만 그 탭만 다시 그린다.** 프래그먼트가 아니면 위젯 하나를 건드려도 스크립트 전체가 다시 돌아 보이는 탭 8개가 통째로 재렌더된다. 실측: DRAM '표시 품목' 라디오 전환 **13.7초 → 0.84초**. 탭 렌더 함수 14개에 전부 걸어 뒀다. **안쪽에 또 걸지 말 것** — 중첩 프래그먼트는 Streamlit이 막는다(`_render_dram_trend_chart`처럼 탭이 부르는 보조 함수가 그 대상이다).
- **차트 뷰만 바꾸는 위젯은 서버로 보내지 말고 Plotly `updatemenus`로 브라우저에서 처리한다.** DRAM 품목 전환은 프래그먼트 안이라도 라디오라서 탭이 통째로 다시 돌았다(CSV 재기록·표 2개·차트 2개 → 약 4초). 품목 전부를 trace로 넣고 첫 개만 `visible=True`, 나머지는 드롭다운이 `visible` 배열만 토글 → **9~150ms, 서버 왕복 0**. 함정 둘: (1) `_style_chart_mobile`이 모든 축에 `fixedrange=True`를 거는데 그 y축에서 visible 토글 시 "axis scaling" 오류 → 그 차트만 `fig.update_yaxes(fixedrange=False)`. (2) Streamlit plotly 테마가 updatemenu를 밝게 칠해 다크 화면에서 글자가 안 보임 → `.updatemenu-item-text{fill:...!important}` 등 CSS로 직접 박음(SVG라 CSS가 닿는다). **이 선택자를 `.js-plotly-plot`으로만 걸지 말 것** — 페이지의 모든 Plotly 차트가 이 클래스를 공유한다. DRAM 드롭다운용으로 걸었던 전역 규칙이 장중 주가 추이 차트의 본주/ADR(SKHY) 전환 버튼까지 덮어써서, 원래 파란 글자(#4a8ec2)가 검게 바뀌어 다크 배경에서 안 보이게 된 적이 있다(2026-09-11). Streamlit이 `st.plotly_chart(..., key=...)`에 `st-key-<key>` 클래스를 붙여 주므로, 그 차트만 겨냥하려면 `div[class*="st-key-chart_dram_"] .updatemenu-item-text`처럼 감싸는 div로 범위를 좁힌다.
- **`st.tabs`는 보이는 탭을 전부 한 번에, 동기로 그린다.** 사용자는 한 탭만 보는데 8개 조회가 순서대로 쌓여 콜드 로드가 20초였다. 지금은 **새 세션의 첫 렌더에서 열려 있는 첫 탭만** 그리고 `st.rerun()`으로 나머지를 채운다(`_all_tabs_rendered` 세션 키). 첫 화면 **20.0초 → 5.5초**(재방문 1.8초), 탭 전환은 그대로 즉시. **느린 걸 상단에 두지 말 것** — `_live_deviation`이 괴리율 백분위 하나 때문에 700일(네이버 ~35페이지, 5~7초)을 받고 있었고, 그게 첫 화면을 통째로 막았다(250일로 줄임. 700일은 지연 렌더되는 과열도 탭이 따로 받는다).
- **`ai_inputs` 안에서 서로 부르는 함수는 app.py의 `_CACHED` 래핑을 못 탄다.** 모듈 내부 참조라 원본이 불린다. 실제로 `fetch_adr_quote`가 `_fetch_adr_bars`(야후 5일치 분봉)를 캐시 없이 매번 받고 있었다. 공유가 필요하면 `ai_inputs.<이름> = globals()['<이름>']`로 되써야 한다(대시보드 프로세스에만 적용되고 수집기와 무관).
- **콜드 로드가 느려지면 추측하지 말고 `BOOT_PROFILE=1`.** docker-compose의 dashboard `environment`에 넣고 재배포하면 import·상단·프래그먼트 3개·탭별 소요가 `[boot]` 줄로 docker logs에 찍힌다. 평소엔 꺼져 있다.
- **콜드 로드가 느리면 `BOOT_PROFILE=1`로 먼저 재라**(docker-compose의 dashboard `environment`에 넣고 재배포). app.py가 import·상단·프래그먼트·탭별 소요를 `[boot]`로 남긴다. 추측으로 뒤지지 말 것.
- **`st.tabs`는 보이는 탭을 전부 즉시, 순차로 그린다.** 사용자는 한 탭만 보는데 8개 조회 비용을 처음에 다 치른다. 그래서 **새 세션의 첫 렌더는 첫 탭만** 그리고 곧바로 `st.rerun()`으로 나머지를 채운다(app.py 탭 디스패치 루프). 실측 콜드 로드 **~20초 → 첫 화면 4.1초**, 나머지는 그 뒤 1초 안에 채워진다.
- **화면에서 같은 데이터를 두 번 받고 있지 않은지 보라.** `_live_deviation`(상단)과 가격 과열도 탭이 둘 다 `fetch_backtest_history_live(700)`을 불렀다. 상단은 250일이면 충분해서 그렇게 줄였다.
- **헬스체크 200은 "앱이 멀쩡하다"는 뜻이 아니다.** `/_stcore/health`는 Streamlit 서버가 살아 있으면 200을 준다. 탭 렌더 루프(파일 뒤쪽)보다 **앞에서** 예외가 나면 헤더·장중차트까지만 그려지고 탭 8개가 통째로 빈 채 뜨는데, 헬스체크·`docker ps`는 healthy로 나온다. 배포 확인은 md5+헬스체크로 끝내지 말고 **브라우저에서 탭 안에 내용이 있는지**까지 봐라(`document.querySelectorAll('.js-plotly-plot').length` — 정상 13개, 깨졌을 때 1개).
- **`st.iframe`은 `height=0`을 거부한다**(양의 정수·`"stretch"`·`"content"`만). `st.components.v1.html`은 0을 받아줬다. 화면에 안 보이게 스크립트만 심을 때는 `height=1`.
- **`overMarketPriceInfo`는 세션이 `OPEN`일 때만 값을 준다.** 프리장·애프터장이 끝나는 순간 필드가 사라져서, 화면도 그 값에 기대 그리면 지표가 통째로 안 보이게 된다. 정규장도 안 돌고 있을 때는 수집기가 20초 간격으로 남긴 tick(`load_over_market_ticks`, 최근 7일치를 통째로 봄)의 마지막 값을 `상태 라벨: 마감`으로 대신 보여준다. **'오늘'로만 좁히면 안 된다** — 애프터장이 자정 근처까지 이어지는 날엔 그 tick이 어제 날짜로 찍혀서 자정을 넘기자마자 다시 사라진다. SKHY(ADR)도 같은 증상이 있었는데, `fetch_adr_quote()`의 `prev_close`가 세션과 무관하게 이미 '본장(정규장) 종가' 그 자체라서 별도 조회 없이 그 값을 나란히 보여주면 된다.
- **본장(`prev_close`) 자체의 등락률이 필요하면 `_adr_baselines`가 세 번째 값으로 주는 `host_prev_close`를 쓴다** — `prev_close`가 가리키는 거래일의 바로 전날 종가라, 이미 있는 5일치 봉에서 한 번 더 골라내는 것뿐이고 추가 네트워크 호출이 없다. **좁은 중첩 컬럼(`st.columns` 안에 또 `st.columns`)에 괄호 붙은 라벨("SKHY (미국장 마감)")을 넣으면 줄바꿈돼서 그 지표의 값만 옆 칸보다 몇 px 아래로 밀린다** — `getBoundingClientRect()`로 직접 재보기 전엔 눈으로 잘 안 잡힌다. 나란히 보여줄 지표는 괄호 없는 짧은 라벨("SKHY 마감")을 따로 만들 것.

## 6. 미해결 / 다음에 할 것

- **무료 한도가 13회 일정에 빠듯하다.** 2026-09-08에 예약 13회 + 디제스트 + 시험 호출로 3.6이 13시경 소진됐다. 지금은 체인의 3.7이 받아 주지만, 그것까지 소진되면 lite로 내려간다. 며칠 로그를 보고 08~20시가 실제로 감당되는지 판단할 것.
- 리포트 PDF는 표지가 이미지면 6페이지를 다 읽어도 143자(OCR 없이는 불가). 목표주가는 네이버 상세에서 따로 얻으므로 치명적이지 않다.
- 네이버가 싣지 않는 증권사(LS증권 등)는 컨센서스에 안 잡힌다. `SECTOR_NEWS_QUERIES`의 "{name} 투자의견" 뉴스로만 보완 중.
