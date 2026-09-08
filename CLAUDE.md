# 작업 전 필독

1. `CONTEXT.md` — 목적·배포·확정된 결정·반복해서 걸린 함정. **먼저 읽는다.**
2. `MAP.md` — 세 파일(`app.py` 4,608줄 · `ai_inputs.py` 2,508줄 · `ai_report.py` 356줄)의 줄 번호 지도.

**큰 파일을 통째로 읽지 말 것.** MAP.md에서 줄 번호를 찾아 그 범위만 연다.
streamlit을 import하는 건 `app.py` 하나뿐이다 — 재료·생성 코드는 `ai_inputs.py`/`ai_report.py`에 있다.
새로 알게 된 사실은 CONTEXT.md에 쓴다(§4 결정 / §5 함정). 변경 이력은 쓰지 않는다 — git log 담당.
