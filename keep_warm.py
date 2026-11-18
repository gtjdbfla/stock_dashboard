"""대시보드 캐시 예열. **streamlit을 import하지 않는다**(수집기가 쓴다).

배경(2026-11-18, 사용자 신고 "처음 열 때 느리다"): 대시보드는 브라우저가 붙어야 스크립트를 돌리고, 외부 조회 결과는
`st.cache_data`(ttl 60초~1시간)에만 있다. 한동안 아무도 안 열면 캐시가 다 식어서 다음 접속자가 DRAM·재무·애널리스트
조회와 과열도 계산을 전부 직접 기다린다(실측: 재시작 직후 첫 접속 7초, 그다음 접속 2초).

그래서 수집기가 몇 분마다 대시보드에 '가짜 브라우저 접속'을 열어 스크립트를 한 번 끝까지 돌리고 끊는다. 실제 접속자가
거치는 경로 그대로라 캐시 키가 정확히 맞는다(별도의 예열 목록을 따로 관리할 필요가 없다).

AI 분석은 수집기(ai_report.tick)가 만들고 화면은 읽기만 하므로 이 접속이 모델 쿼터를 쓰지 않는다.
"""
import os
import time

from websockets.sync.client import connect

URL = os.environ.get("KEEP_WARM_URL", "ws://hynix-dashboard:8501/_stcore/stream")
ORIGIN = os.environ.get("KEEP_WARM_ORIGIN", "http://hynix-dashboard:8501")
INTERVAL_SEC = int(os.environ.get("KEEP_WARM_INTERVAL_SEC", "240"))   # 가장 긴 잦은 캐시(15분)보다 훨씬 짧게
IDLE_SEC = 4         # 이만큼 새 메시지가 없으면 스크립트가 끝난 것으로 본다(끝 신호를 해석하지 않으려고)
MAX_SEC = 90         # 무슨 일이 있어도 이 안에 끊는다
# BackMsg(rerun_script{query_string: ""})를 직렬화한 2바이트. streamlit 프로토콜을 import하지 않으려고 미리 박아뒀다.
RERUN_SCRIPT = bytes.fromhex("5a00")

_last = {"at": None}


def warm_once() -> tuple[float, int]:
    """대시보드에 접속해 스크립트를 한 번 돌리고 (걸린 초, 받은 바이트)를 돌려준다."""
    t0 = time.monotonic()
    got = 0
    with connect(URL, subprotocols=["streamlit"], origin=ORIGIN, max_size=None, open_timeout=10) as ws:
        ws.send(RERUN_SCRIPT)
        while time.monotonic() - t0 < MAX_SEC:
            try:
                msg = ws.recv(timeout=IDLE_SEC)
            except TimeoutError:
                break
            got += len(msg)
    return time.monotonic() - t0, got


def tick(log=print) -> None:
    """수집기 루프에서 부른다. 마지막 접속 뒤 INTERVAL_SEC가 지났을 때만 돈다."""
    now = time.monotonic()
    if _last["at"] is not None and now - _last["at"] < INTERVAL_SEC:
        return
    _last["at"] = now
    try:
        took, got = warm_once()
    except Exception as exc:
        # 대시보드가 재시작 중이면 실패하는 게 정상이다. 다음 주기에 다시 한다.
        log(f"[캐시예열] 접속 실패: {type(exc).__name__}: {exc}")
        return
    log(f"[캐시예열] 대시보드 스크립트 1회 실행 — {took:.1f}s, {got / 1e6:.2f}MB")
