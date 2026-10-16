"""전향 검증 장부: 낙폭 관리 규칙(signal_lab.RISK_RULES)의 신호를 **그날 장 마감 뒤에** 한 줄씩 기록한다.

왜 필요한가: 규칙 4개는 2026-10-16에 SK하이닉스 과거 데이터에서 골랐고, 다른 종목에서는 재현되지 않았다.
과거 백테스트는 이미 '봐 버린' 데이터라 더 이상 증거가 못 된다. 앞으로 실제로 어떻게 되는지를 기록해 판정한다.

규칙(사전 등록 — 바꾸지 말 것):
  * 기록은 마감 뒤 확정 수급이 들어온 날에만, 하루 한 줄. **한 번 쓴 줄은 고치지 않는다. 과거 날짜를 채워 넣지 않는다(백필 금지).**
  * 규칙 정의는 signal_lab.RISK_RULES 그대로. 조건·파라미터를 바꾸려면 새 장부 파일로 시작한다.
  * 성적은 기록된 목표 비중을 다음 날 종가에 체결했다고 가정(편도 0.15%)해 단순 보유와 비교한다.
  * 판정은 기록이 6개월(약 120거래일) 쌓인 뒤에. 그 전에는 "아직 판단 이르다"로만 읽는다.
streamlit을 import하지 않는다(수집기에서 돈다).
"""
import datetime as dt
import os

import numpy as np
import pandas as pd

import signal_lab as sl

LEDGER_FILE = os.environ.get("SIGNAL_LEDGER_FILE", "data/signal_ledger.csv")
LEDGER_TICKER = "000660"
MIN_DAYS_TO_JUDGE = 120
COLUMNS = ["날짜", "기록시각", "종가", "기관기울기", "SOX전일종가", "SOX200일선", "괴리율80", "변동성20", "비중_C2", "비중_P6", "비중_R3", "비중_R2"]

RECORD_FROM = dt.time(16, 30)
RECORD_TO = dt.time(23, 0)
RETRY_MIN = 30

_last_attempt: dt.datetime | None = None


def load(path: str | None = None) -> pd.DataFrame:
    p = path or LEDGER_FILE
    if not os.path.exists(p):
        return pd.DataFrame(columns=COLUMNS)
    try:
        df = pd.read_csv(p, parse_dates=["날짜"])
    except Exception:
        return pd.DataFrame(columns=COLUMNS)
    return df.sort_values("날짜").reset_index(drop=True)


def make_row(hist: pd.DataFrame, sox: pd.DataFrame | None, now: dt.datetime) -> dict | None:
    """확정된 마지막 날(hist의 마지막 행) 기준 한 줄. 수급이 아직 안 찼거나 SOX가 없으면 None."""
    h = hist.sort_values("날짜").reset_index(drop=True)
    last = h.iloc[-1]
    flows = [float(last.get(c) or 0) for c in ("기관", "외국인", "개인")]
    if not any(abs(v) > 0 for v in flows):
        return None
    rw = sl.risk_rule_weights(h, sox)
    w, info = rw["weights"], rw["info"]
    if "C2" not in w:
        return None
    g = lambda k: (None if np.isnan(info[k][-1]) else float(info[k][-1]))
    return {"날짜": pd.Timestamp(last["날짜"]).date().isoformat(), "기록시각": now.strftime("%Y-%m-%d %H:%M"),
            "종가": float(last["종가"]), "기관기울기": g("inst_slope"), "SOX전일종가": g("sox_prev"), "SOX200일선": g("sox_ma200"),
            "괴리율80": g("dev80"), "변동성20": g("sig20"),
            "비중_C2": float(w["C2"][-1]), "비중_P6": float(w["P6"][-1]), "비중_R3": float(w["R3"][-1]), "비중_R2": float(w["R2"][-1])}


def append_once(row: dict, path: str | None = None) -> bool:
    """같은 날짜가 이미 있으면 쓰지 않는다(수정 금지). 새로 썼으면 True."""
    p = path or LEDGER_FILE
    existing = load(p)
    if not existing.empty and row["날짜"] in set(existing["날짜"].dt.date.astype(str)):
        return False
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    new = pd.DataFrame([row], columns=COLUMNS)
    out = pd.concat([existing.assign(날짜=existing["날짜"].dt.date.astype(str)), new], ignore_index=True) if not existing.empty else new
    tmp = f"{p}.tmp"
    out.to_csv(tmp, index=False)
    os.replace(tmp, p)
    return True


def tick(now: dt.datetime, log=print, path: str | None = None) -> None:
    """마감 뒤(16:30~23:00) 확정 수급이 오늘 날짜로 들어오면 한 줄 기록. 평일 휴장일엔 조용히 끝난다."""
    global _last_attempt
    if now.weekday() >= 5 or not (RECORD_FROM <= now.time() <= RECORD_TO):
        return
    if _last_attempt is not None and (now - _last_attempt) < dt.timedelta(minutes=RETRY_MIN):
        return
    led = load(path)
    if not led.empty and led["날짜"].max().date() == now.date():
        return
    _last_attempt = now
    import ai_inputs                       # 순환 import 방지: 늦게 부른다
    hist = ai_inputs.fetch_long_history_live(LEDGER_TICKER, years=25)
    if hist.empty or pd.Timestamp(hist["날짜"].iloc[-1]).date() != now.date():
        return                             # 오늘 행이 아직 없다(휴장 또는 집계 전)
    try:
        sox = ai_inputs.fetch_us_daily("%5ESOX")
    except Exception as exc:
        log(f"[신호장부] SOX 시세 실패: {type(exc).__name__}: {exc}")
        return
    row = make_row(hist, sox, now)
    if row is None:
        return
    if append_once(row, path):
        log(f"[신호장부] {row['날짜']} 기록 — 비중 C2 {row['비중_C2']:.0f} / P6 {row['비중_P6']:.0f} / "
            f"R3 {row['비중_R3']:.2f} / R2 {row['비중_R2']:.2f}")


def evaluate(led: pd.DataFrame, cost: float = sl.RISK_COST) -> dict:
    """기록된 목표 비중을 다음 날 종가에 체결했다고 보고 단순 보유와 비교한다. 기록된 종가만 쓴다."""
    out = {"days": len(led), "ok": False}
    if len(led) < 5:
        return out
    price = led["종가"].astype(float).to_numpy()
    ret = np.r_[0.0, price[1:] / price[:-1] - 1]
    n = len(led)

    def perf(w):
        pos = np.zeros(n)
        pos[2:] = np.asarray(w, float)[: n - 2]          # t 종가 기록 → t+1 체결 → t+2 수익부터
        turn = np.abs(np.diff(pos, prepend=0))
        r = (pos * ret - cost * turn)[2:]
        if len(r) < 1:
            return None
        eq = np.cumprod(1 + r)
        return {"total": float(eq[-1] - 1), "mdd": float((eq / np.maximum.accumulate(eq) - 1).min()),
                "exposure": float(pos[2:].mean())}

    out["buy_hold"] = perf(np.ones(n))
    out["rules"] = {k: perf(led[f"비중_{k}"].fillna(0).to_numpy()) for k in ("C2", "P6", "R3", "R2")}
    out["start"] = led["날짜"].iloc[0]
    out["end"] = led["날짜"].iloc[-1]
    out["ok"] = out["buy_hold"] is not None
    out["judgeable"] = len(led) >= MIN_DAYS_TO_JUDGE
    return out


# ── 괴리율 구간별 이후 결과(사전 등록) ───────────────────────────────────────────
# 옛 괴리율 그래프("괴리율이 클수록 하락 확률이 늘어난다")는 과거 10년에서만 정비례였고 독립 사건이 9~15번뿐이라,
# 장부에 기록되는 괴리율(80일선)로 앞으로 실제로 이어지는지 본다. 구간·기간·기준은 여기서 고정하고 바꾸지 않는다.
DEV_EDGES = [-9.0, -0.10, 0.0, 0.10, 0.20, 0.30, 9.0]
DEV_LABELS = ["-10% 미만", "-10~0%", "0~10%", "10~20%", "20~30%", "30% 이상"]
DEV_HORIZON = 15
DEV_THR = 0.10
DEV_MIN_PER_BIN = 20


def bin_outcomes(price, dev, horizon: int = DEV_HORIZON, thr: float = DEV_THR) -> pd.DataFrame:
    """고정 괴리율 구간별로, 이후 horizon거래일 안에 -thr 이하 하락 / +thr 이상 상승한 비율과 마지막 날 평균 수익.
    결과가 확정된 날(뒤에 horizon일이 더 있는 날)만 센다."""
    price = np.asarray(price, float)
    dev = np.asarray(dev, float)
    dd, up, ret = sl.forward_stats(price, horizon)
    known = ~np.isnan(dd) & ~np.isnan(dev)
    rows = []
    for i, lab in enumerate(DEV_LABELS):
        m = known & (dev >= DEV_EDGES[i]) & (dev < DEV_EDGES[i + 1])
        n = int(m.sum())
        rows.append({"구간": lab, "n": n,
                     "하락": float((dd[m] <= -thr).mean()) if n else None,
                     "상승": float((up[m] >= thr).mean()) if n else None,
                     "평균수익": float(ret[m].mean()) if n else None})
    return pd.DataFrame(rows)


def deviation_table(led: pd.DataFrame) -> pd.DataFrame:
    """장부에 기록된 종가·괴리율(80일선)만으로 구간별 결과를 센다."""
    if len(led) < 2:
        return bin_outcomes([], [])
    return bin_outcomes(led["종가"].astype(float).to_numpy(), led["괴리율80"].astype(float).to_numpy())
