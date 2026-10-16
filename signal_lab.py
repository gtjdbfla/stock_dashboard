"""조기신호·괴리율 백테스트의 계산부. **streamlit과 네트워크를 쓰지 않는다.**

화면(app.py)과 AI 재료(ai_inputs.py)가 같은 숫자를 보게 하려고 계산을 여기 한 곳에 둔다.
2026-10-16에 SK하이닉스 5년·10년·21년 이력으로 다시 검증한 결과(CONTEXT.md §5)를 반영했다:

- p값은 **원형 이동 순열검정**이다. 연속한 날짜를 독립 표본으로 세는 z검정은 p를 크게
  과소평가한다(하락 신호: z검정 p=0.001 → 순열 p=0.03~0.16). 신호의 군집을 그대로 두고
  라벨과의 짝만 깬다.
- 분위는 **점-시점**이다: 각 날짜의 분위 경계를 그날까지의 과거 분포만으로 정한다.
  전체 구간 분위는 미래 분포가 섞여 효과가 부풀려진다(괴리율 상위 10%: 하락 효과가
  전체 분위로는 보이다가 점-시점으로는 사라졌다).
- 라벨은 고정 7%와 함께 변동성 정규화(σ20·√h 이상 하락)를 같이 본다. 일간 σ가 3.5%까지
  가는 종목이라 고정 %만으로는 국면에 따라 기준율이 6%~58%로 흔들린다.
"""
import numpy as np
import pandas as pd

PERM_N = 500            # 순열 횟수. 화면 응답과 정밀도의 타협(p 해상도 ≈ 0.002)
MIN_PIT = 250           # 점-시점 분위를 쓰기 시작하는 최소 과거 일수
TRADING_DAYS = 245


# ── 공통 도구 ───────────────────────────────────────────────────────────────

def forward_stats(price, horizon: int):
    """각 날짜 t의 이후 horizon일 안에서 (최대 낙폭, 최대 상승, 마지막 날 수익률). 모두 t 종가 대비.
    마지막 horizon일은 아직 알 수 없어 NaN."""
    v = np.asarray(price, dtype=float)
    n = len(v)
    dd = np.full(n, np.nan)
    up = np.full(n, np.nan)
    ret = np.full(n, np.nan)
    if n <= horizon:
        return dd, up, ret
    win = np.lib.stride_tricks.sliding_window_view(v[1:], horizon)[: n - horizon]
    base = v[: n - horizon]
    dd[: n - horizon] = win.min(axis=1) / base - 1
    up[: n - horizon] = win.max(axis=1) / base - 1
    ret[: n - horizon] = win[:, -1] / base - 1
    return dd, up, ret


def rolling_slope(series, window: int) -> np.ndarray:
    """누적합에 직선을 맞춘 기울기(= 하루 평균 순매수). ai_inputs._rolling_slope와 같은 값이다."""
    cum = np.cumsum(np.asarray(series, dtype=float))
    x = np.arange(window) - (window - 1) / 2
    out = np.full(len(cum), np.nan)
    if len(cum) >= window:
        out[window - 1:] = np.correlate(cum, x, mode="valid") / (x ** 2).sum()
    return out


def episodes(sig, gap: int = 5) -> int:
    """연속 신호를 gap일 이내로 이어진 하나의 사건으로 묶어 센다."""
    idx = np.flatnonzero(np.asarray(sig, dtype=bool))
    if len(idx) == 0:
        return 0
    return int(1 + (np.diff(idx) > gap).sum())


def perm_p(signal, label, horizon: int, expect: str = "greater",
           n_perm: int = PERM_N, seed: int = 0):
    """원형 이동 순열검정 p값. 신호군과 나머지의 라벨 평균 차이가 우연보다 큰가.

    expect="greater": 신호군 평균이 더 높다는 가설(단측), "less": 더 낮다는 가설.
    신호 배열을 통째로 돌려(np.roll) 군집·자기상관을 보존한 채 라벨과의 짝만 깬다.
    반환: (관측 차이, p) — 표본이 모자라면 (None, None).
    """
    s = np.asarray(signal, dtype=bool)
    y = np.asarray(label, dtype=float)
    ok = ~np.isnan(y)
    n = len(s)

    def diff(sv):
        m = sv & ok
        r = (~sv) & ok
        if m.sum() < 5 or r.sum() < 5:
            return np.nan
        return y[m].mean() - y[r].mean()

    obs = diff(s)
    if np.isnan(obs) or n < 6 * horizon:
        return (None if np.isnan(obs) else float(obs)), None
    rng = np.random.default_rng(seed)
    shifts = rng.integers(2 * horizon, n - 2 * horizon, size=n_perm)
    stats = np.array([diff(np.roll(s, int(k))) for k in shifts])
    stats = stats[~np.isnan(stats)]
    if len(stats) < 50:
        return float(obs), None
    hit = (stats >= obs).sum() if expect == "greater" else (stats <= obs).sum()
    return float(obs), float((hit + 1) / (len(stats) + 1))


def pit_bucket(series, k: int = 5, min_periods: int = MIN_PIT) -> np.ndarray:
    """각 날짜를 '그날 이전' 과거 분포만으로 k분위(1..k) 중 하나에 넣는다. 부족하면 NaN."""
    s = pd.Series(np.asarray(series, dtype=float))
    cuts = [s.expanding(min_periods=min_periods).quantile(j / k).shift(1) for j in range(1, k)]
    bucket = np.ones(len(s))
    for c in cuts:
        bucket += (s.to_numpy() > c.to_numpy()).astype(float)
    bucket[np.isnan(cuts[0].to_numpy()) | s.isna().to_numpy()] = np.nan
    return bucket


# ── 수급 조기신호 ───────────────────────────────────────────────────────────

SIGNAL_WINDOW = 20      # 기관·외국인 누적 순매수 기울기 창 (20~40일이 유효, 5·10일은 무효였다)
VOL_WINDOW = 20
HORIZON = 10            # 이후 거래일
THRESHOLD = 0.07        # 큰 폭 상승·하락 기준


def flow_frame(hist: pd.DataFrame, window: int = SIGNAL_WINDOW) -> pd.DataFrame:
    """날짜별 수급 특징. 기울기는 20일 평균 거래량으로 나눠 '거래량 대비 하루 평균 순매수'로 만든다."""
    h = hist.sort_values("날짜").reset_index(drop=True)
    price = h["종가"].astype(float)
    avg_vol = h["거래량"].astype(float).rolling(VOL_WINDOW).mean()
    f = pd.DataFrame({"날짜": h["날짜"], "종가": price})
    for key, col in (("inst", "기관"), ("frgn", "외국인"), ("retail", "개인")):
        raw = rolling_slope(h[col].astype(float), window)
        f[f"{key}_slope"] = raw
        f[f"{key}_n"] = raw / avg_vol.to_numpy()
    f["vr"] = h["거래량"].astype(float) / avg_vol
    f["sig20"] = price.pct_change().rolling(20).std()
    return f


def make_labels(f: pd.DataFrame, horizon: int = HORIZON, thr: float = THRESHOLD) -> pd.DataFrame:
    dd, up, ret = forward_stats(f["종가"], horizon)
    sc = f["sig20"].to_numpy() * np.sqrt(horizon)
    known = ~np.isnan(dd)
    L = pd.DataFrame({
        "ret": ret,
        "dn": np.where(known, (dd <= -thr).astype(float), np.nan),
        "up": np.where(known, (up >= thr).astype(float), np.nan),
        "dnz": np.where(known & ~np.isnan(sc), (dd / sc <= -1.0).astype(float), np.nan),
    })
    return L


def _rule_row(name, kind, expect, sig, L, horizon, recent_mask=None):
    sig = np.asarray(sig, dtype=bool)
    row = {"name": name, "kind": kind, "expect": ("up" if expect == "up" else "down"), "now": bool(sig[-1]) if len(sig) else False,
           "n_days": int(sig.sum()), "n_events": episodes(sig)}
    for lab in ("up", "dn", "dnz"):
        y = L[lab].to_numpy()
        ok = ~np.isnan(y)
        m, r = sig & ok, (~sig) & ok
        # 가설 방향: 상승 신호는 상승↑·하락↓, 하락 신호는 하락↑·상승↓ (사전에 정한 경제적 방향)
        want = "greater" if ((lab == "up") == (expect == "up")) else "less"
        obs, p = perm_p(sig, y, horizon, expect=want)
        row[lab] = {"sig": float(y[m].mean()) if m.sum() >= 5 else None,
                    "rest": float(y[r].mean()) if r.sum() >= 5 else None, "p": p}
    if recent_mask is not None:       # 최근 구간(예: 5년)만으로 같은 검정을 한 번 더 — 시기 의존성 확인용
        lab = "up" if expect == "up" else "dn"
        y = np.where(recent_mask, L[lab].to_numpy(), np.nan)
        _, p = perm_p(sig, y, horizon, expect="greater")
        ok = ~np.isnan(y)
        m, r = sig & ok, (~sig) & ok
        row["recent"] = {"sig": float(y[m].mean()) if m.sum() >= 5 else None,
                         "rest": float(y[r].mean()) if r.sum() >= 5 else None, "p": p}
    return row


def early_signal_report(hist: pd.DataFrame, horizon: int = HORIZON, thr: float = THRESHOLD,
                        recent_days: int = 1225) -> dict:
    """하락·상승 조기신호 화면과 AI 재료가 함께 쓰는 검증 결과.

    핵심 신호는 둘 다 **기관 20일 누적 순매수 기울기**다(10년: 기울기>0이면 상승 7% 42% vs 23%,
    <0이면 변동성 대비 큰 하락 23% vs 14%). 옛 신호(외국인 기울기<0 + 거래량, 기관>0 + 개인<0 + 거래량)는
    같은 검정에서 무정보여서 '참고' 행으로만 남겼다.
    """
    f = flow_frame(hist)
    L = make_labels(f, horizon, thr)
    n = len(f)
    out = {"n": n, "start": f["날짜"].iloc[0] if n else None, "end": f["날짜"].iloc[-1] if n else None,
           "horizon": horizon, "threshold": thr, "ok": False}
    if n < MIN_PIT + 2 * horizon + 60:
        return out
    out["ok"] = True
    inst, frgn, retail = f["inst_slope"].to_numpy(), f["frgn_slope"].to_numpy(), f["retail_slope"].to_numpy()
    vr = f["vr"].to_numpy()
    inst_pos = np.nan_to_num(inst, nan=0.0) > 0
    inst_neg = np.nan_to_num(inst, nan=0.0) < 0
    heavy = np.nan_to_num(vr, nan=0.0) > 1.0
    recent = (np.arange(n) >= n - recent_days)

    out["base"] = {lab: float(np.nanmean(L[lab])) for lab in ("up", "dn", "dnz")}
    cur = {"inst_slope": float(inst[-1]), "inst_n": float(f["inst_n"].iloc[-1]),
           "frgn_slope": float(frgn[-1]), "vr": float(vr[-1]), "retail_slope": float(retail[-1]),
           "price": float(f["종가"].iloc[-1])}
    # 점-시점 5분위: 오늘 기관 기울기가 과거 분포에서 어디쯤인가
    bucket = pit_bucket(f["inst_n"])
    cur["bucket"] = int(bucket[-1]) if not np.isnan(bucket[-1]) else None
    out["current"] = cur

    rows = []
    for q in range(1, 6):
        m = bucket == q
        y = {lab: L[lab].to_numpy() for lab in ("up", "dn", "dnz", "ret")}
        known = m & ~np.isnan(y["up"])
        rows.append({"분위": q, "표본": int(known.sum()),
                     "상승": float(y["up"][known].mean()) if known.sum() else None,
                     "하락": float(y["dn"][known].mean()) if known.sum() else None,
                     "변동성대비하락": float(np.nanmean(y["dnz"][known])) if known.sum() else None,
                     "평균수익": float(np.nanmean(y["ret"][known])) if known.sum() else None})
    out["quintiles"] = pd.DataFrame(rows)

    out["rules"] = [
        _rule_row("기관 20일 기울기 > 0", "상승", "up", inst_pos, L, horizon, recent),
        _rule_row("기관 > 0 + 거래량 평균 이상", "상승", "up", inst_pos & heavy, L, horizon, recent),
        _rule_row("[옛 상승 신호] 기관 > 0 + 개인 < 0 + 거래량 평균 이상", "참고", "up",
                  inst_pos & (np.nan_to_num(retail, nan=0.0) < 0) & heavy, L, horizon, recent),
        _rule_row("기관 20일 기울기 < 0", "하락", "down", inst_neg, L, horizon, recent),
        _rule_row("기관 < 0 + 거래량 평균 이상", "하락", "down", inst_neg & heavy, L, horizon, recent),
        _rule_row("[옛 하락 신호] 외국인 < 0 + 거래량 평균 이상", "참고", "down",
                  (np.nan_to_num(frgn, nan=0.0) < 0) & heavy, L, horizon, recent),
        _rule_row("외국인 20일 기울기 < 0", "참고", "down", np.nan_to_num(frgn, nan=0.0) < 0, L, horizon, recent),
    ]
    out["frame"] = f
    return out


# ── 괴리율(가격 과열도) ──────────────────────────────────────────────────────

def run_overheat_pit(df: pd.DataFrame, price_col: str, ma_window: int, horizon: int,
                     quantile: float = 0.2, drawdown_threshold: float = 0.05,
                     gain_threshold: float | None = None, side: str = "high") -> dict:
    """ai_inputs.run_overheat_backtest와 같은 키를 돌려주되 **점-시점 분위 + 순열 p값**을 쓴다.

    - 각 날짜가 상위(하위) quantile 구간인지는 그날 이전 과거 분포만으로 판정한다.
    - hi_cutoff는 화면의 '현재 기준 경계'(전체 이력 분포)로, 오늘 어느 구간인지 보여주는 용도다.
    - 상승 확률과 이후 horizon일 평균수익도 같이 돌려준다. 과열 구간은 하락 확률만이 아니라
      상승 확률과 평균수익도 같이 높았다(10년: 상승 60~64% vs 30%, 평균수익 +6.5% vs +2.1%).
    """
    gain_threshold = drawdown_threshold if gain_threshold is None else gain_threshold
    price = df[price_col].astype(float).reset_index(drop=True)
    dev = (price / price.rolling(ma_window).mean() - 1)
    dd, up, ret = forward_stats(price, horizon)
    result = {
        "ma_window": ma_window, "horizon": horizon, "quantile": quantile, "side": side,
        "drawdown_threshold": drawdown_threshold, "gain_threshold": gain_threshold, "n": 0,
        "hi_n": 0, "rest_n": 0, "hi_rate": None, "rest_rate": None, "base_rate": None, "p_value": None,
        "hi_up_rate": None, "rest_up_rate": None, "base_up_rate": None, "up_p_value": None,
        "hi_ret": None, "rest_ret": None, "hi_cutoff": None, "current_deviation": None,
        "current_regime": None, "point_in_time": True,
    }
    dev_all = dev.dropna()
    if len(dev_all):
        result["current_deviation"] = float(dev_all.iloc[-1])
        result["hi_cutoff"] = float(dev_all.quantile(1 - quantile if side == "high" else quantile))

    q_cut = 1 - quantile if side == "high" else quantile
    cut = dev.expanding(min_periods=MIN_PIT).quantile(q_cut).shift(1).to_numpy()
    dv = dev.to_numpy()
    valid = ~np.isnan(cut) & ~np.isnan(dv) & ~np.isnan(dd)
    result["n"] = int(valid.sum())
    if result["n"] < 30:
        return result
    grp = (dv >= cut) if side == "high" else (dv <= cut)
    grp = grp & valid
    rest = (~grp) & valid
    down = (dd <= -drawdown_threshold).astype(float)
    gain = (up >= gain_threshold).astype(float)
    result["hi_n"], result["rest_n"] = int(grp.sum()), int(rest.sum())
    if grp.sum() == 0 or rest.sum() == 0:
        return result
    result["hi_rate"], result["rest_rate"] = float(down[grp].mean()), float(down[rest].mean())
    result["base_rate"] = float(down[valid].mean())
    result["hi_up_rate"], result["rest_up_rate"] = float(gain[grp].mean()), float(gain[rest].mean())
    result["base_up_rate"] = float(gain[valid].mean())
    result["hi_ret"], result["rest_ret"] = float(ret[grp].mean()), float(ret[rest].mean())
    ylab = np.where(valid, down, np.nan)
    _, result["p_value"] = perm_p(grp, ylab, horizon, expect="greater")
    ylab = np.where(valid, gain, np.nan)
    _, result["up_p_value"] = perm_p(grp, ylab, horizon, expect="greater")
    cdev = result["current_deviation"]
    cc = result["hi_cutoff"]
    in_group = cdev is not None and cc is not None and ((cdev >= cc) if side == "high" else (cdev <= cc))
    result["current_regime"] = (f"과열 (상위 {quantile:.0%})" if side == "high" else f"침체 (하위 {quantile:.0%})") if in_group else "평상시"
    return result


# ── 이동평균 추세추종 전략 ────────────────────────────────────────────────────

def _perf(daily: np.ndarray) -> dict:
    daily = np.asarray(daily, dtype=float)
    if len(daily) < 2:
        return {"cagr": None, "mdd": None, "sharpe": None, "total": None}
    eq = np.cumprod(1 + daily)
    years = len(daily) / TRADING_DAYS
    mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
    sd = daily.std()
    return {"cagr": float(eq[-1] ** (1 / years) - 1) if years > 0 else None, "mdd": mdd,
            "sharpe": float(daily.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else None,
            "total": float(eq[-1] - 1)}


def run_ma_strategy(df: pd.DataFrame, ma_window: int, period_start, band: float = 0.03,
                    lag: int = 1, cost: float = 0.003) -> dict:
    """주가가 이동평균 위일 때만 보유하는 추세추종 전략.

    - 진입: 괴리율 >= +band, 청산: 괴리율 < -band (밴드가 휩쏘를 줄인다 — 80일선에서 5개 하위기간
      모두 샤프가 개선됐고 연 거래가 6.3→1.8회가 됐다).
    - lag: 신호가 난 날 종가 기준으로 lag일 뒤 종가에 체결(기본 1 = 다음 날). 옛 화면은 당일 종가
      판정·당일 체결(lag=0)이라 낙관적이었다.
    - cost: 왕복 비용(매도세+수수료+슬리피지). 청산할 때 한 번 뺀다.
    이동평균은 period_start 이전 이력을 포함해 계산한다(워밍업 손실 없음).
    """
    d = df.sort_values("날짜").reset_index(drop=True)
    price = d["종가"].astype(float).to_numpy()
    dev = (d["종가"].astype(float) / d["종가"].astype(float).rolling(ma_window).mean() - 1).to_numpy()
    dates = d["날짜"].to_numpy()
    n = len(d)
    start_idx = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(period_start)), side="left"))
    result = {"n_days": max(0, n - start_idx), "trades": [], "still_open": False, "unrealized_return": None,
              "equity_curve": None, "bh_curve": None, "strategy": None, "buy_hold": None,
              "period_start": None, "period_end": None, "state": None, "pending": None,
              "ma_window": ma_window, "band": band, "lag": lag, "cost": cost}
    first = max(start_idx, ma_window)           # 괴리율이 정의되는 첫 날부터
    if n - first < 30:
        return result

    # 신호 → 체결 일정
    in_pos = False
    fills = {}                                  # 체결 인덱스 → 'buy'/'sell'
    pending = None
    for i in range(first, n):
        if not in_pos and dev[i] >= band:
            in_pos = True
            j = i + lag
            if j <= n - 1:
                fills[j] = "buy"
            else:
                pending = ("buy", dates[i])
        elif in_pos and dev[i] < -band:
            in_pos = False
            j = i + lag
            if j <= n - 1:
                fills[j] = "sell"
            else:
                pending = ("sell", dates[i])
    # lag로 같은 날 사고팔기가 겹치면 마지막 지시가 이긴다(위에서 dict로 덮어씀)

    held = False
    entry = None
    trades = []
    daily = np.zeros(n)
    for t in range(first, n):
        if held:
            daily[t] = price[t] / price[t - 1] - 1
        action = fills.get(t)
        if action == "buy" and not held:
            held, entry = True, (dates[t], price[t])
        elif action == "sell" and held:
            held = False
            ret = price[t] / entry[1] - 1
            daily[t] -= cost                    # 왕복 비용은 청산일에 반영
            trades.append({"buy_date": pd.Timestamp(entry[0]), "sell_date": pd.Timestamp(dates[t]),
                           "buy_price": float(entry[1]), "sell_price": float(price[t]),
                           "ret": float((1 + ret) * (1 - cost) - 1)})
            entry = None
    seg = slice(first, n)
    bh = np.zeros(n)
    bh[first + 1:] = price[first + 1:] / price[first:-1] - 1
    result["trades"] = trades
    result["still_open"] = held
    if held and entry is not None:
        result["unrealized_return"] = float(price[-1] / entry[1] - 1)
    result["strategy"] = _perf(daily[seg][1:])
    result["buy_hold"] = _perf(bh[seg][1:])
    result["period_start"], result["period_end"] = pd.Timestamp(dates[first]), pd.Timestamp(dates[-1])
    result["equity_curve"] = pd.DataFrame({"날짜": dates[seg], "자산가치": np.cumprod(1 + daily[seg])})
    result["bh_curve"] = pd.DataFrame({"날짜": dates[seg], "자산가치": np.cumprod(1 + bh[seg])})
    result["state"] = "보유" if held else "현금"
    result["pending"] = pending
    result["years"] = (n - first) / TRADING_DAYS
    result["trades_per_year"] = len(trades) / result["years"] if result["years"] > 0 else None
    result["current_deviation"] = float(dev[-1]) if not np.isnan(dev[-1]) else None
    return result


# ── 가격 과열도 탭: 지금 상황 + 지표별 경향성 ────────────────────────────────────

def pit_percentile(x, min_periods: int = MIN_PIT) -> np.ndarray:
    """각 값이 '그 이전' 과거 값들 중 몇 %보다 큰가(0~1). 과거가 min_periods개 미만이면 NaN."""
    import bisect
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    srt: list = []
    for i, v in enumerate(x):
        if np.isnan(v):
            continue
        if len(srt) >= min_periods:
            out[i] = bisect.bisect_left(srt, v) / len(srt)
        bisect.insort(srt, v)
    return out


def shift_corr_p(x, y, horizon: int):
    """x(연속값)와 y(0/1)의 순위상관과, 모든 원형 이동(2·horizon 이상)을 다 해 본 정확 순열 p(양측).

    FFT로 모든 이동의 상관을 한 번에 구한다 — 무작위 순열보다 빠르고 재현 가능하다.
    이동할 때마다 짝이 맞는 구간이 달라지지만 정규화는 고정해 관측값과 같은 식으로 비교한다.
    """
    from scipy.stats import rankdata
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    vx, vy = ~np.isnan(x), ~np.isnan(y)
    if (vx & vy).sum() < 100 or n < 8 * horizon:
        return None, None

    def z(a, v):
        out = np.zeros(n)
        r = rankdata(a[v])
        out[v] = (r - r.mean()) / (r.std() + 1e-12)
        return out

    zx, zy = z(x, vx), z(y, vy)
    norm = np.sqrt((zx ** 2).sum() * (zy ** 2).sum())
    cc = np.fft.irfft(np.fft.rfft(zy) * np.conj(np.fft.rfft(zx)), n=n) / norm   # cc[k] = Σ zx[i]·zy[i+k]
    obs = float(cc[0])
    ks = np.arange(2 * horizon, n - 2 * horizon)
    sims = cc[ks]
    p = float((1 + (np.abs(sims) >= abs(obs)).sum()) / (1 + len(sims)))
    return obs, p


def wilson_ci(k: float, n: int, z: float = 1.96):
    if n <= 0:
        return None
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return float(max(0.0, c - h)), float(min(1.0, c + h))


def drop_incomplete_last_row(hist: pd.DataFrame, now=None) -> pd.DataFrame:
    """오늘 행이 아직 덜 찼으면(장중이거나 수급이 0) 뺀다. 수급 지표는 하루가 덜 찬 값을 섞으면 틀어진다."""
    import datetime as _dt
    if hist.empty:
        return hist
    now = now or _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=9)))
    last = hist.iloc[-1]
    if pd.Timestamp(last["날짜"]).date() != now.date():
        return hist
    flows = [float(last.get(c) or 0) for c in ("기관", "외국인", "개인")]
    if now.time() < _dt.time(15, 45) or not any(abs(v) > 0 for v in flows):
        return hist.iloc[:-1].reset_index(drop=True)
    return hist


# 지표 정의: (키, 이름, 방향성 있음?, 설명). 방향성 없음 = 하락·상승 둘 다 커지는 변동폭 지표.
INDICATORS = [
    ("flow_heat", "수급 과열 점수", True, "개인 순매수 비중↑ · 기관·외국인 순매수 비중↓ 를 합친 점수(과거 백분위 평균)"),
    ("retail", "개인 순매수 비중 (20일)", True, "20일간 개인 순매수 ÷ 거래량"),
    ("inst_slope", "기관 20일 순매수 기울기", True, "기관 20일 누적 순매수의 기울기 ÷ 평균 거래량"),
    ("inst", "기관 순매수 비중 (20일)", True, "20일간 기관 순매수 ÷ 거래량"),
    ("frgn", "외국인 순매수 비중 (20일)", True, "20일간 외국인 순매수 ÷ 거래량"),
    ("dev", "이동평균 괴리율", True, "종가 ÷ 이동평균 − 1"),
    ("vol", "변동성 (20일)", False, "일간 수익률의 20일 표준편차 — 하락·상승 둘 다 커지는 변동폭 지표"),
]


def _bucket(pct: np.ndarray, k: int = 5) -> np.ndarray:
    b = np.floor(pct * k) + 1
    return np.where(np.isnan(pct), np.nan, np.clip(b, 1, k))


def situation_report(hist: pd.DataFrame, ma_window: int = 80, horizon: int = 10, thr: float = 0.07) -> dict:
    """가격 과열도 탭의 '지금 상황': 지표별로 지금이 과거 분포의 어디쯤이고, 그 위치에서 이후 하락·상승 확률이 얼마였나.

    - 모든 분위는 점-시점이다(그날 이전 분포만 사용). 확률은 그 분위에 있던 모든 과거 날짜의 실제 비율이다.
    - 경향성은 지표(백분위)와 이후 하락·상승 여부의 순위상관과 정확 순열 p다. 하락·상승 기준은 horizon일 안에
      종가 대비 최대 낙폭 <= -thr / 최대 상승 >= +thr.
    - 헤드라인: 하락 확률 = 수급 과열 점수(방향성 지표 중 하락과 가장 강한 경향), 상승 확률 = 기관이
      20일간 순매수 우위일 때(조기신호 탭과 같은 기준 — 10년 검증 42% vs 23%).
    """
    h = hist.sort_values("날짜").reset_index(drop=True)
    out = {"ok": False, "horizon": horizon, "thr": thr, "ma_window": ma_window, "n": len(h)}
    if len(h) < MIN_PIT + 3 * horizon + 100:
        return out
    price = h["종가"].astype(float)
    vol = h["거래량"].astype(float)
    r = price.pct_change()

    def share(col):
        return (h[col].astype(float).rolling(20).sum() / vol.rolling(20).sum()).to_numpy()

    f = flow_frame(h)
    raw = {
        "retail": share("개인"), "inst": share("기관"), "frgn": share("외국인"),
        "inst_slope": f["inst_n"].to_numpy(),
        "dev": (price / price.rolling(ma_window).mean() - 1).to_numpy(),
        "vol": r.rolling(20).std().to_numpy(),
    }
    pct = {k: pit_percentile(v) for k, v in raw.items()}
    heat = np.vstack([pct["retail"], 1 - pct["inst"], 1 - pct["frgn"]]).T
    raw["flow_heat"] = np.where(np.isnan(heat).any(axis=1), np.nan, heat.mean(axis=1))
    pct["flow_heat"] = pit_percentile(raw["flow_heat"])

    dd, up, _ = forward_stats(price, horizon)
    known = ~np.isnan(dd)
    y_dn = np.where(known, (dd <= -thr).astype(float), np.nan)
    y_up = np.where(known, (up >= thr).astype(float), np.nan)
    out["base"] = {"dn": float(np.nanmean(y_dn)), "up": float(np.nanmean(y_up))}
    out["asof"] = h["날짜"].iloc[-1]
    out["start"] = h["날짜"].iloc[0]

    indicators = []
    for key, name, directional, desc in INDICATORS:
        pc = pct[key]
        b = _bucket(pc)
        now_pct = float(pc[-1]) if not np.isnan(pc[-1]) else None
        now_b = int(b[-1]) if not np.isnan(b[-1]) else None
        rows = []
        for q in range(1, 6):
            m = (b == q) & known
            nq = int(m.sum())
            rows.append({"분위": q, "표본": nq,
                         "하락": float(y_dn[m].mean()) if nq else None,
                         "상승": float(y_up[m].mean()) if nq else None})
        rho_dn, p_dn = shift_corr_p(pc, y_dn, horizon)
        rho_up, p_up = shift_corr_p(pc, y_up, horizon)
        indicators.append({
            "key": key, "name": name, "directional": directional, "desc": desc,
            "raw": float(raw[key][-1]) if not np.isnan(raw[key][-1]) else None,
            "pct": now_pct, "bucket": now_b, "table": pd.DataFrame(rows),
            "rho_dn": rho_dn, "p_dn": p_dn, "rho_up": rho_up, "p_up": p_up,
            "strength": max(abs(rho_dn or 0), abs(rho_up or 0)),
        })
    indicators.sort(key=lambda d: d["strength"], reverse=True)
    out["indicators"] = indicators
    by_key = {d["key"]: d for d in indicators}

    # 헤드라인: 하락 확률(수급 과열 점수 현재 분위)
    heat_ind = by_key["flow_heat"]
    if heat_ind["bucket"] is not None:
        row = heat_ind["table"][heat_ind["table"]["분위"] == heat_ind["bucket"]].iloc[0]
        k = row["하락"] * row["표본"]
        out["headline_dn"] = {"rate": float(row["하락"]), "n": int(row["표본"]), "bucket": heat_ind["bucket"],
                              "ci": wilson_ci(k, int(row["표본"])), "basis": "수급 과열 점수",
                              "up_rate": float(row["상승"])}
    # 헤드라인: 상승 확률(기관 20일 순매수 우위 여부)
    inst_s = f["inst_slope"].to_numpy()
    ok_s = ~np.isnan(inst_s)
    if ok_s[-1]:
        state = inst_s > 0
        now_state = bool(state[-1])
        m = (state == now_state) & ok_s & known
        n_s = int(m.sum())
        out["headline_up"] = {"rate": float(y_up[m].mean()), "n": n_s, "ci": wilson_ci(y_up[m].sum(), n_s),
                              "state": "순매수 우위" if now_state else "순매도 우위", "basis": "기관 20일 기울기",
                              "dn_rate": float(y_dn[m].mean())}

    # 괴리율 × 수급 과열 점수 3x3
    d_p, f_p = pct["dev"], pct["flow_heat"]
    okg = ~np.isnan(d_p) & ~np.isnan(f_p) & known
    tri = lambda a: np.clip(np.floor(a * 3), 0, 2)
    Dt, Ft = tri(d_p), tri(f_p)
    grid_rate = np.full((3, 3), np.nan)
    grid_n = np.zeros((3, 3), dtype=int)
    for i in range(3):
        for j in range(3):
            m = okg & (Dt == i) & (Ft == j)
            grid_n[i, j] = int(m.sum())
            if m.sum() >= 30:
                grid_rate[i, j] = y_dn[m].mean()
    cur = (None, None)
    if not np.isnan(d_p[-1]) and not np.isnan(f_p[-1]):
        cur = (int(tri(d_p)[-1]), int(tri(f_p)[-1]))
    out["grid"] = {"rate": grid_rate, "n": grid_n, "current": cur}
    out["ok"] = True
    return out


# ── 낙폭 관리 규칙(SK하이닉스 한정) ─────────────────────────────────────────────
# 2026-10-16 사전 등록 후보 32개를 같은 기준으로 평가해 하이닉스에서 통과한 4개. 같은 규칙을 삼성전자·국내 대형주
# 26종목·마이크론에 옮기면 대부분 보유보다 못했으므로 '하이닉스에서만 통한 참고'로만 쓴다(CONTEXT.md §5).
# 규칙은 여기서 고정한다 — 결과를 본 뒤 조건을 바꾸지 말 것(전향 검증 장부가 이 정의를 그대로 기록한다).

RISK_RULES = [
    ("C2", "기관 20일 순매수 우위 + SOX가 200일선 위",
     "기관 20일 누적 순매수 기울기>0 이고 필라델피아 반도체지수(SOX) 전날 종가가 200일선 위일 때만 보유"),
    ("P6", "80일선 ±3% 밴드 추세추종",
     "종가가 80일선 대비 +3% 이상으로 올라오면 보유, -3% 아래로 내려가면 현금(밴드 안에서는 직전 상태 유지)"),
    ("R3", "변동성 목표 비중 × 기관 순매수 우위",
     "기관 20일 순매수 우위일 때만, 비중 = min(1, 과거 20일 변동성 중앙값 ÷ 최근 20일 변동성)"),
    ("R2", "변동성 목표 비중",
     "항상 보유하되 비중 = min(1, 과거 20일 변동성 중앙값 ÷ 최근 20일 변동성) — 변동성이 커지면 비중을 줄임"),
]
RISK_COST = 0.0015          # 편도 비용 0.15% (왕복 0.3%)


def align_prev(dates: pd.Series, ext: pd.DataFrame, col: str = "close") -> pd.Series:
    """미국·환율 시세를 한국 날짜에 붙인다. **한국 날짜보다 앞선 날**의 값만 쓴다(한국 종가 시점에 이미 끝난 값)."""
    e = ext[["날짜", col]].copy()
    e["날짜"] = pd.to_datetime(e["날짜"]).astype("datetime64[ns]")
    e = e.sort_values("날짜").rename(columns={"날짜": "_d"})
    left = pd.DataFrame({"날짜": pd.to_datetime(pd.Series(dates)).astype("datetime64[ns]").to_numpy()})
    m = pd.merge_asof(left, e, left_on="날짜", right_on="_d", allow_exact_matches=False, direction="backward")
    return m[col].astype(float).reset_index(drop=True)


def _band_state(dev: np.ndarray, b: float) -> np.ndarray:
    s = np.zeros(len(dev))
    cur = False
    for i, v in enumerate(dev):
        if np.isnan(v):
            continue
        if not cur and v >= b:
            cur = True
        elif cur and v < -b:
            cur = False
        s[i] = 1.0 if cur else 0.0
    return s


def risk_rule_weights(h: pd.DataFrame, sox: pd.DataFrame | None) -> dict:
    """날짜별 목표 비중(0~1). 신호는 그날 종가 기준이고 체결은 다음 날 종가다."""
    h = h.sort_values("날짜").reset_index(drop=True)
    price = h["종가"].astype(float)
    inst_pos = (np.nan_to_num(rolling_slope(h["기관"].astype(float), SIGNAL_WINDOW)) > 0).astype(float)
    r20 = price.pct_change().rolling(20).std()
    med = r20.expanding(250).median().shift(1)
    wvt = np.clip(np.nan_to_num((med / r20).to_numpy()), 0, 1)
    dev80 = (price / price.rolling(80).mean() - 1).to_numpy()
    out = {"P6": _band_state(dev80, 0.03), "R3": wvt * inst_pos, "R2": wvt}
    info = {"inst_slope": rolling_slope(h["기관"].astype(float), SIGNAL_WINDOW), "dev80": dev80,
            "sig20": r20.to_numpy(), "wvt": wvt}
    if sox is not None and len(sox) > 250:
        s = align_prev(h["날짜"], sox)
        ma200 = s.rolling(200).mean()
        sox_up = np.nan_to_num((s - ma200).to_numpy()) > 0
        out["C2"] = (inst_pos.astype(bool) & sox_up).astype(float)
        info.update({"sox_prev": s.to_numpy(), "sox_ma200": ma200.to_numpy()})
    return {"weights": out, "info": info}


def _weights_perf(price: np.ndarray, w: np.ndarray, mask: np.ndarray, lag: int = 1, cost: float = RISK_COST) -> dict:
    n = len(price)
    ret = np.r_[0.0, price[1:] / price[:-1] - 1]
    pos = np.zeros(n)
    pos[1 + lag:] = np.asarray(w, float)[: n - 1 - lag]
    turn = np.abs(np.diff(pos, prepend=0))
    r = (pos * ret - cost * turn)[mask]
    if len(r) < 60:
        return {}
    eq = np.cumprod(1 + r)
    yrs = len(r) / TRADING_DAYS
    sd = r.std()
    return {"cagr": float(eq[-1] ** (1 / yrs) - 1), "mdd": float((eq / np.maximum.accumulate(eq) - 1).min()),
            "sharpe": float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else None,
            "exposure": float(pos[mask].mean())}


def risk_rules_report(hist: pd.DataFrame, sox: pd.DataFrame | None, start="2006-06-01") -> dict:
    """규칙별 현재 상태(오늘 종가 기준 목표 비중)와 21년·10년 성적. 비교 기준은 단순 보유(같은 체결·비용 가정)."""
    h = hist.sort_values("날짜").reset_index(drop=True)
    out = {"ok": False}
    if len(h) < 1500:
        return out
    rw = risk_rule_weights(h, sox)
    price = h["종가"].astype(float).to_numpy()
    d = pd.to_datetime(h["날짜"])
    m_all = np.asarray(d >= pd.Timestamp(start))
    m_10 = np.asarray(d >= pd.Timestamp("2016-10-14"))
    bh_all = _weights_perf(price, np.ones(len(h)), m_all)
    bh_10 = _weights_perf(price, np.ones(len(h)), m_10)
    rules = []
    for key, name, desc in RISK_RULES:
        w = rw["weights"].get(key)
        row = {"key": key, "name": name, "desc": desc, "weight_now": None, "all": {}, "ten": {}}
        if w is not None:
            row["weight_now"] = float(w[-1])
            row["all"] = _weights_perf(price, w, m_all)
            row["ten"] = _weights_perf(price, w, m_10)
        rules.append(row)
    out.update({"ok": True, "asof": h["날짜"].iloc[-1], "rules": rules, "bh_all": bh_all, "bh_ten": bh_10,
                "info": {k: (float(v[-1]) if len(v) and not np.isnan(v[-1]) else None) for k, v in rw["info"].items()},
                "weights": rw["weights"]})
    return out
