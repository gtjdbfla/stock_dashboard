"""프리장/애프터장 시세를 상시 기록하는 독립 프로세스.

Streamlit은 브라우저 세션이 붙어야만 스크립트를 실행한다. 그래서 대시보드 안에 스레드를 띄우면
아무도 화면을 안 보는 새벽·아침에는 수집이 시작되지 않아, 정작 프리장(08:00~09:00)이 비어버린다.
이 스크립트를 별도 컨테이너로 띄워 화면 접속과 무관하게 계속 돌린다.
"""
import datetime as dt
import os
import sys
import time

import analyst_digest
import analyst_targets
import disclosure
import financial_digest
import flow_probe
import over_market as om

# 로그 한 줄 때문에 수집기가 죽는 일이 없게 한다.
# (한글·특수문자가 콘솔 인코딩과 안 맞으면 print가 UnicodeEncodeError로 터진다)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


def log(msg: str) -> None:
    now = dt.datetime.now(om.KST).strftime("%Y-%m-%d %H:%M:%S")
    try:
        print(f"[{now}] {msg}", flush=True)
    except UnicodeEncodeError:
        print(f"[{now}] {msg.encode('ascii', 'replace').decode('ascii')}", flush=True)


# 거래 시간대에는 이만큼마다 상태를 한 줄 남긴다.
# 예전에는 '기록에 성공했을 때만' 로그를 찍어서, 하루 종일 한 건도 못 담은 날은
# 로그가 통째로 비었다. 그러면 수집기가 죽은 건지, 원래 장이 안 열린 건지(공휴일 등)
# 구분할 방법이 없다. 0건인 날도 0건이라고 말하게 한다.
HEARTBEAT_SEC = 1800

# 아침에 리포트 요약을 확인할 종목. 대시보드 기본 종목과 같게 둔다.
DIGEST_TICKER = os.environ.get("ANALYST_DIGEST_TICKER", "000660")
DIGEST_STOCK_NAME = os.environ.get("ANALYST_DIGEST_NAME", "SK하이닉스(000660)")


def main() -> None:
    log(f"수집기 시작 | 종목={om.COLLECT_TICKERS} | 주기={om.POLL_SEC}s | 저장={om.TICK_FILE}")
    last_state = None
    day = None
    recorded = skipped = failed = 0
    last_beat = 0.0

    def summary() -> str:
        return f"기록 {recorded}건 / 휴장·미체결 {skipped}회 / 오류 {failed}회"

    while True:
        now = dt.datetime.now(om.KST)
        if now.date() != day:                   # 날짜가 바뀌면 그날 집계를 마무리하고 초기화
            if day is not None:
                log(f"{day} 집계: {summary()}")
            day, recorded, skipped, failed = now.date(), 0, 0, 0

        # 마감 후 투자자 수급이 어느 경로에 먼저 올라오는지 기록한다.
        # 수집 시간대(~20:30) 판정보다 앞에 둔다. 탐침 창이 21:00까지라 대기 모드에서도 돌아야 한다.
        try:
            flow_probe.tick(now, log=log)
        except Exception as exc:                # 탐침 때문에 시세 수집이 멈추면 안 된다
            log(f"수급탐침 오류: {type(exc).__name__}: {exc}")

        # 애널리스트 리포트 요약. 아침에 하루 한 번 확인해서 새 리포트가 있을 때만 다시 만든다.
        # 대시보드에서만 확인하면 화면을 열기 전까지 옛 요약이 그대로 보인다.
        try:
            analyst_digest.tick(now, DIGEST_TICKER, DIGEST_STOCK_NAME, log=log)
        except Exception as exc:
            log(f"리포트요약 오류: {type(exc).__name__}: {exc}")

        # 증권사별 목표주가 누적. 새 리포트의 상세 페이지만 열어 목표가를 뽑아 쌓는다.
        # 이걸 모아야 컨센서스를 직접 계산할 수 있다(FnGuide 값은 주 1회만 갱신된다).
        try:
            analyst_targets.tick(now, DIGEST_TICKER, log=log)
        except Exception as exc:
            log(f"목표주가수집 오류: {type(exc).__name__}: {exc}")

        # 공시 요약. 공시는 뉴스보다 빠르고 숫자가 확정적이라 새로 뜨면 바로 정리해 둔다.
        try:
            disclosure.tick(now, DIGEST_TICKER, DIGEST_STOCK_NAME, log=log)
        except Exception as exc:
            log(f"공시요약 오류: {type(exc).__name__}: {exc}")

        # 재무 요약도 같은 방식으로. 화면에서 만들면 생성되는 동안 페이지가 통째로 멈춘다.
        try:
            financial_digest.tick(now, DIGEST_TICKER, DIGEST_STOCK_NAME, log=log)
        except Exception as exc:
            log(f"재무요약 오류: {type(exc).__name__}: {exc}")

        if not om.in_collect_window(now):
            if last_state != "idle":
                log(f"거래 시간대 아님 - 대기 모드 (오늘까지 {summary()})")
                last_state = "idle"
            time.sleep(om.IDLE_SEC)
            continue

        if last_state != "active":
            log("거래 시간대 진입 - 수집 시작")
            last_state = "active"
            last_beat = time.monotonic()

        for ticker in om.COLLECT_TICKERS:
            try:
                if om.record_once(ticker):
                    recorded += 1
                else:
                    # 시간외 장이 안 열렸거나(공휴일·주말 대체) 아직 체결이 없는 정상 상황
                    skipped += 1
            except Exception as exc:            # 네트워크 오류로 수집기가 죽으면 안 된다
                failed += 1
                if failed % 20 == 1:            # 같은 오류가 쏟아질 때 로그를 채우지 않게
                    log(f"{ticker} 수집 실패: {type(exc).__name__}: {exc}")

        if time.monotonic() - last_beat >= HEARTBEAT_SEC:
            log(f"수집 중 | {summary()}")
            last_beat = time.monotonic()
        time.sleep(om.POLL_SEC)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("종료")
        sys.exit(0)
