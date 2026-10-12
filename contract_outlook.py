"""TrendForce 보도자료의 메모리 고정가(계약가) 전망을 모아 검증하고, 분기별 추정 지수를 만든다.
**streamlit을 import하지 않는다.**

고정가의 실측치는 공개되지 않는다(DRAMeXchange 상세·이력은 회원 전용, 모바일·서버 계약가는 가격을 숨긴다).
다만 TrendForce가 분기마다 보도자료로 "범용 DRAM 계약가 4Q26 +10~15% QoQ"처럼 **범위로 전망**을 낸다.
이 전망을 날짜순으로 쌓으면 ① 분기별 전망표 ② 전망이 올라가는지(상향)·내려가는지(하향) ③ 전망 중간값을
이어 곱한 누적 추정 지수를 얻는다. 실측이 아니라 **전망의 누적**이라는 점이 한계다.

숫자는 LLM에게 맡기지 않고 세 겹으로 막는다(hbm.py와 같은 방식).
1. LLM은 하한·상한·분기를 구조화해 내고 **원문 인용**을 같이 낸다.
2. 코드가 인용이 본문에 글자 그대로 있는지, 하한·상한 숫자가 그 인용에 있는지, 분기 표기가 본문에 있는지 대조한다.
3. 방향(상승/하락)은 LLM 말을 믿지 않고 숫자의 부호로 코드가 정한다.
지수·상향/하향·둔화/가속은 전부 코드가 계산한다.
"""
import datetime as dt
import hashlib
import json
import os
import re

import hbm
import llm

STORE = os.environ.get("CONTRACT_OUTLOOK_FILE", "data/contract_outlook.json")

LOOKBACK_DAYS = 270            # 1Q26~4Q26 전망이 모두 들어오도록 넉넉히
MAX_DOCS_PER_TICK = 3          # LLM을 부르는 문서 수 상한(무료 한도 보호)
MAX_FETCH_PER_TICK = 30
MAX_ATTEMPTS = 3               # 하루 시도 상한
MAX_TOTAL_ATTEMPTS = 12
CHECK_EVERY_MIN = 180
BACKLOG_EVERY_MIN = 20

CATEGORIES = ("범용 DRAM", "서버 DRAM", "PC DRAM", "모바일 DRAM", "그래픽 DRAM", "컨슈머 DRAM", "HBM",
              "NAND", "클라이언트 SSD", "엔터프라이즈 SSD")
BASES = ("QoQ", "YoY", "HoH", "MoM")

_PERIOD = re.compile(r"^(20\d\d)(Q[1-4]|H[12])$")
_ORDINAL = {"1": "first", "2": "second", "3": "third", "4": "fourth"}

_last_tick: dt.datetime | None = None
_backlog = False


# ── 저장 ────────────────────────────────────────────────────────────────────

def load() -> dict:
    try:
        with open(STORE, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def save(payload: dict) -> None:
    try:
        os.makedirs(os.path.dirname(STORE) or ".", exist_ok=True)
        tmp = f"{STORE}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        os.replace(tmp, STORE)
    except OSError:
        pass


# ── 검증 ────────────────────────────────────────────────────────────────────

def _period_in_text(period: str, folded_text: str) -> bool:
    """정규화한 분기('2026Q3')가 본문·제목 어딘가에 실제로 적혀 있는지('3Q26', 'third quarter of 2026' 등)."""
    m = _PERIOD.match(period)
    y, p = m.groups()
    yy, n = y[2:], p[1]
    if p[0] == "Q":
        cands = [f"{n}q{yy}", f"{n}q{y}", f"q{n}{y}", f"q{n}{yy}", f"{_ORDINAL[n]}quarterof{y}", f"{_ORDINAL[n]}quarter"]
    else:
        cands = [f"{n}h{yy}", f"{n}h{y}", f"h{n}{y}", f"{_ORDINAL[n]}halfof{y}", f"{_ORDINAL[n]}half"]
    return any(c in folded_text for c in cands)


# 인용 문장에 품목을 가리키는 말이 있어야 그 품목의 전망으로 인정한다. LLM이 NOR 플래시 문장을 NAND로,
# 일반론 문장을 서버 DRAM으로 붙이는 오분류를 거른다. 인용문만 본다(제목에는 여러 품목이 섞여 있다).
_CATEGORY_WORDS = {
    "범용 DRAM": ("dram",),
    "서버 DRAM": ("server",),
    "PC DRAM": ("pc dram", "pc oem", "notebook", "pc "),
    "모바일 DRAM": ("mobile", "lpddr", "smartphone", "handset"),
    "그래픽 DRAM": ("graphics", "gddr"),
    "컨슈머 DRAM": ("consumer", "ddr4", "ddr3", "ddr2"),
    "HBM": ("hbm",),
    "NAND": ("nand",),
    "클라이언트 SSD": ("client ssd", "ssd", "notebook"),
    "엔터프라이즈 SSD": ("enterprise ssd", "ssd"),
}


_OTHER_PRODUCT_WORDS = ("nand", "nor flash", "ssd", "wafer", "emmc", "ufs")


def category_ok(category: str, quote: str) -> bool:
    q = quote.lower()
    if any(w in q for w in _CATEGORY_WORDS.get(category, ())):
        return True
    # 범용 DRAM 전망은 'moderate contract price increases to 13-18% QoQ'처럼 품목명 없이 앞 문장을 이어 받는 일이 흔하다.
    # 다른 품목 표현이 없고 계약가 얘기일 때만 허용한다.
    return (category == "범용 DRAM" and "contract price" in q
            and not any(w in q for w in _OTHER_PRODUCT_WORDS))


def _num_tokens(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text.replace(",", "")))


def _as_token(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def validate_forecast(raw: dict, body: str, title: str, pub_date: str) -> tuple[dict | None, str]:
    """LLM이 낸 전망 하나를 원문과 대조한다. (통과한 항목 or None, 탈락 사유)."""
    if not isinstance(raw, dict):
        return None, "형식"
    category = str(raw.get("category") or "").strip()
    period = str(raw.get("period") or "").strip().upper()
    basis = {b.upper(): b for b in BASES}.get(str(raw.get("basis") or "").strip().upper(), "")
    quote = re.sub(r"\s+", " ", str(raw.get("quote") or "")).strip()
    if category not in CATEGORIES:
        return None, "품목"
    if not _PERIOD.match(period):
        return None, "분기 형식"
    qualitative = raw.get("low") is None and raw.get("high") is None
    if qualitative and not basis:
        basis = "-"                                       # 숫자 없는 방향 전망은 비교 기준이 없다
    elif basis not in BASES:
        return None, "기준(QoQ 등)"
    if len(quote) < 20:
        return None, "인용 짧음"
    folded_body = hbm._fold(body)
    if hbm._fold(quote) not in folded_body:
        return None, "인용이 원문에 없음"
    if not category_ok(category, quote):
        return None, "인용에 품목 표현이 없음"
    if not _period_in_text(period, folded_body + hbm._fold(title)):
        return None, f"분기 {period}이 본문에 없음"
    pub_year = int(pub_date[:4])
    if not pub_year - 1 <= int(period[:4]) <= pub_year + 2:
        return None, "분기가 발표 시점과 안 맞음"

    low, high = raw.get("low"), raw.get("high")
    try:
        low = None if low is None else float(low)
        high = None if high is None else float(high)
    except (TypeError, ValueError):
        return None, "숫자 형식"
    nums = _num_tokens(quote)
    for v in (low, high):
        if v is not None and (abs(v) > 400 or _as_token(abs(v)) not in nums):
            return None, f"숫자 {v:g}이 인용에 없음"
    if low is not None and high is not None and low > high:
        low, high = high, low
    if low is None and high is not None:
        return None, "하한 없이 상한만"

    if low is None:                                      # 숫자 없는 서술형 전망 — 방향 라벨은 검증할 수 없어 비운다
        direction = ""
    else:                                                # 방향은 숫자 부호로 코드가 정한다
        mid = low if high is None else (low + high) / 2
        direction = "상승" if mid > 0.5 else ("하락" if mid < -0.5 else "보합")
    return {"category": category, "period": period, "basis": basis, "low": low, "high": high,
            "direction": direction, "quote": quote}, ""


# ── 추출 ────────────────────────────────────────────────────────────────────

def _prompt(doc: dict, body: str) -> str:
    return f"""다음은 TrendForce 보도자료 "{doc['title']}" ({doc['date']}) 의 본문이다.
**메모리 계약가(contract price) 전망**만 골라 JSON 배열로만 답해라(설명·코드펜스 금지). 계약가가 아닌 현물가·매출·출하량은 제외한다.
계약가 전망이 없으면 [] 만 답해라.

각 원소의 형식:
{{"category": "{'|'.join(CATEGORIES)}" 중 하나,
 "period": "전망 대상 기간. 2026Q3 또는 2026H2 형식(연도 4자리 + Q1~Q4 또는 H1~H2)",
 "basis": "QoQ|YoY|HoH|MoM 중 하나(본문에 쓰인 비교 기준)",
 "low": 변동률 하한 숫자(%, 부호 포함, 예 13) 또는 null,
 "high": 변동률 상한 숫자(%) 또는 null. 단일 수치면 low와 같게. 범위면 둘 다 채운다,
 "direction": "상승|하락|보합" — 숫자가 없는 방향 전망일 때만 의미가 있다,
 "quote": "근거가 되는 본문 문장을 **한 글자도 고치지 말고 그대로 복사**(250자 이내)"}}

규칙:
- 'conventional DRAM'이나 품목 수식이 없는 'DRAM'은 범용 DRAM(HBM 제외)이다. 'overall NAND Flash'는 NAND다.
- 본문에 있는 숫자만 쓰고 환산·반올림하지 마라. "13–18%"는 low 13, high 18이다. "more than 100%"처럼 하한만 있으면 high는 null이다.
- 숫자가 없는 "계속 오를 전망" 같은 문장은 low·high를 null로 두고 direction만 채워라.
- 같은 문장에서 여러 품목·분기가 나오면 원소를 나눠라. 같은 내용을 중복하지 마라.

본문:
{body[:hbm.BODY_CHARS]}"""


def extract(doc: dict, body: str) -> tuple[list[dict], dict]:
    text, _ = llm.call(_prompt(doc, body), min_usable=2)
    kept, dropped, seen = [], {}, set()
    for raw in hbm._parse_json_list(text)[:40]:
        item, why = validate_forecast(raw, body, doc["title"], doc["date"])
        if item is None:
            dropped[why] = dropped.get(why, 0) + 1
            continue
        key = (item["category"], item["period"], item["basis"], item["low"], item["high"])
        if key in seen:
            continue
        seen.add(key)
        item["id"] = hashlib.sha1((doc["url"] + item["category"] + item["period"] + item["quote"]).encode("utf-8")).hexdigest()[:8]
        kept.append(item)
    return kept, dropped


# ── 분석: 전부 코드가 계산한다 ─────────────────────────────────────────────

def _period_key(period: str) -> tuple[int, int]:
    """2026Q3 -> (2026, 3), 2026H2 -> (2026, 2.5 대신 하반기 끝 분기 4)"""
    y = int(period[:4])
    if period[4] == "Q":
        return y, int(period[5])
    return y, 2 if period[5] == "1" else 4


def _prev_quarter(period: str) -> str:
    y, q = int(period[:4]), int(period[5])
    return f"{y - 1}Q4" if q == 1 else f"{y}Q{q - 1}"


def flat_forecasts(store: dict) -> list[dict]:
    """문서 정보가 붙은 전망 목록. 같은 문서에서 같은 (품목·분기·기준)이 중복이면 첫 항목만."""
    rows, seen = [], set()
    for url, d in store.get("docs", {}).items():
        if d.get("status") != "extracted":
            continue
        for f in d.get("forecasts", []):
            k = (url, f["category"], f["period"], f["basis"])
            if k in seen:
                continue
            seen.add(k)
            if not category_ok(f["category"], f["quote"]):
                continue
            rows.append({**f, "date": d["date"], "title": d["title"], "url": url})
    rows.sort(key=lambda r: (r["date"], r["url"]))
    return rows


def mid_of(f: dict) -> float | None:
    if f["low"] is None:
        return None
    return f["low"] if f["high"] is None else (f["low"] + f["high"]) / 2


def history_by_key(forecasts: list[dict], basis: str = "QoQ") -> dict[tuple[str, str], list[dict]]:
    """(품목, 분기) -> 그 전망이 발표된 순서대로의 숫자 전망들."""
    out: dict[tuple[str, str], list[dict]] = {}
    for f in forecasts:
        if f["basis"] == basis and f["low"] is not None:
            out.setdefault((f["category"], f["period"]), []).append(f)
    return out


def latest_table(forecasts: list[dict]) -> dict[str, dict[str, dict]]:
    """품목 -> 분기 -> {cur: 최신 전망, prev: 직전 전망 or None, revision: 중간값 변화(%p) or None}."""
    table: dict[str, dict[str, dict]] = {}
    for (cat, period), seq in history_by_key(forecasts).items():
        cur = seq[-1]
        closed = [f for f in seq if f["high"] is not None]
        # '60% 초과'처럼 상한이 없는 전망은 중간값이 없다 — 닫힌 범위끼리만 개정폭을 잰다.
        prev = closed[-2] if cur["high"] is not None and len(closed) >= 2 else None
        rev = round(mid_of(cur) - mid_of(prev), 2) if prev is not None else None
        table.setdefault(cat, {})[period] = {"cur": cur, "prev": prev, "revision": rev, "count": len(seq)}
    return table


def pace_change(cat_row: dict[str, dict]) -> dict | None:
    """가장 최근 두 연속 분기의 전망 중간값을 비교해 증가폭이 둔화/가속인지 판정."""
    periods = sorted((p for p in cat_row if _PERIOD.match(p) and p[4] == "Q"), key=_period_key)
    for i in range(len(periods) - 1, 0, -1):
        a, b = periods[i - 1], periods[i]
        if _prev_quarter(b) != a:
            continue
        ma, mb = mid_of(cat_row[a]["cur"]), mid_of(cat_row[b]["cur"])
        if ma is None or mb is None:
            continue
        diff = round(mb - ma, 2)
        label = "둔화" if diff <= -1.0 else ("가속" if diff >= 1.0 else "비슷")
        return {"from": a, "to": b, "from_mid": ma, "to_mid": mb, "diff": diff, "label": label}
    return None


def build_index(cat_row: dict[str, dict]) -> dict | None:
    """연속 분기 전망을 이어 곱해 누적 지수를 만든다(시작 직전 분기말 = 100).

    low/high 열은 각 분기의 하한·상한을 그대로 이어 곱한 범위다. 모든 분기가 최저(최고)로 실현되는
    극단이라 실제 범위보다 넓게 나온다. 상한이 없는 전망('more than')이 끼면 거기서 끊는다."""
    qs = sorted((p for p, v in cat_row.items() if p[4] == "Q" and v["cur"]["high"] is not None), key=_period_key)
    if not qs:
        return None
    chain = [qs[-1]]                                           # 가장 최근 분기에서 거꾸로 이어지는 구간
    for p in reversed(qs[:-1]):
        if _prev_quarter(chain[0]) == p:
            chain.insert(0, p)
        else:
            break
    if len(chain) < 2:
        return None
    base = _prev_quarter(chain[0])
    pts = [{"period": base, "low": 100.0, "mid": 100.0, "high": 100.0}]
    for p in chain:
        f = cat_row[p]["cur"]
        last = pts[-1]
        pts.append({"period": p,
                    "low": last["low"] * (1 + f["low"] / 100),
                    "mid": last["mid"] * (1 + mid_of(f) / 100),
                    "high": last["high"] * (1 + f["high"] / 100)})
    return {"base": base, "points": pts}


# ── 현물가 변화 vs 고정가 전망 ──────────────────────────────────────────────

# 현물가 품목이 어느 고정가 전망 품목과 짝인지. 칩·UDIMM은 범용 DRAM, RDIMM은 서버 DRAM으로 본다.
# 같은 품목이 아니라 '같은 시장의 대리 지표'다(고정가 전망은 품목 믹스 전체의 평균이고 현물가는 대표 규격 하나).
SPOT_PROXY = {
    "DDR5 16Gb (2Gx8) 4800/5600": "범용 DRAM",
    "DDR4 16Gb (2Gx8) 3200": "범용 DRAM",
    "DDR5 UDIMM 16GB 4800/5600": "범용 DRAM",
    "DDR5 RDIMM 32GB 4800/5600": "서버 DRAM",
}
SPOT_MAX_AGE_DAYS = 15          # 분기 경계에서 이만큼 넘게 묵은 값이면 쓰지 않는다(모듈은 사이트가 주 1회 갱신)
MOMENTUM_THRESHOLD = 1.0        # 최근 4주와 직전 4주 변화율 차이가 이 %p 이상이면 가속/둔화


def _quarter_end(period: str) -> dt.date:
    import calendar
    y, q = int(period[:4]), int(period[5])
    return dt.date(y, q * 3, calendar.monthrange(y, q * 3)[1])


def _spot_series(history, item: str):
    """품목의 일별 마지막 값(Series, 날짜 인덱스). 하루에 여러 번 기록되면 마지막 값만 쓴다."""
    s = history[history["품목"] == item].sort_values("날짜")
    if s.empty:
        return s["평균가(USD)"]
    s = s.assign(일자=s["날짜"].dt.normalize()).groupby("일자")["평균가(USD)"].last()
    return s.astype(float)


def _value_at(series, date: dt.date):
    import pandas as pd
    cutoff = pd.Timestamp(date)
    s = series[series.index <= cutoff]
    if s.empty or (cutoff - s.index[-1]).days > SPOT_MAX_AGE_DAYS:
        return None
    return float(s.iloc[-1])


def _label(diff: float) -> str:
    return "둔화" if diff <= -MOMENTUM_THRESHOLD else ("가속" if diff >= MOMENTUM_THRESHOLD else "비슷")


def compare_spot(history, table: dict, today: dt.date) -> dict:
    """분기별 현물가 변화율과 같은 분기 고정가 전망을 나란히 놓고, 현물가의 최근 모멘텀을 잰다.

    - 분기 변화율 = (분기 끝 또는 오늘의 값) / (직전 분기 끝 값) - 1. 시작 값이 없으면(이력이 짧으면) None.
    - 모멘텀 = 최근 4주 변화율과 그 직전 4주 변화율. 현물가가 고정가보다 먼저 움직인다는 가설을 눈으로 볼 수 있게 한다.
    - 카테고리별 판정은 두 모멘텀의 방향(둔화/가속/비슷)이 같은지만 말한다. 예측이 아니다."""
    import pandas as pd
    cur_q = f"{today.year}Q{(today.month - 1) // 3 + 1}"
    periods = [cur_q]
    for _ in range(2):
        periods.insert(0, _prev_quarter(periods[0]))

    rows = []
    for item, cat in SPOT_PROXY.items():
        s = _spot_series(history, item)
        if s.empty:
            continue
        row = {"item": item, "category": cat, "quarters": {}}
        for p in periods:
            q_end = _quarter_end(p)
            start = _value_at(s, _quarter_end(_prev_quarter(p)))
            end = _value_at(s, min(q_end, today))
            pct = (end / start - 1) * 100 if start and end else None
            cell = table.get(cat, {}).get(p)
            contract = cell["cur"] if cell and cell["cur"]["high"] is not None else None
            partial = today < q_end
            row["quarters"][p] = {
                "spot_pct": pct, "partial": partial,
                "elapsed_days": (today - _quarter_end(_prev_quarter(p))).days if partial else None,
                "contract": contract, "contract_mid": mid_of(contract) if contract else None,
            }
        last = s.index[-1]
        t0 = last.date()
        v0, v4, v8 = (_value_at(s, t0), _value_at(s, t0 - dt.timedelta(days=28)), _value_at(s, t0 - dt.timedelta(days=56)))
        row["last_date"] = t0
        row["recent4"] = (v0 / v4 - 1) * 100 if v0 and v4 else None
        row["prior4"] = (v4 / v8 - 1) * 100 if v4 and v8 else None
        rows.append(row)

    # 판정은 품목별로 낸다. DDR5와 DDR4처럼 세대가 다르면 현물가 방향이 반대일 수 있어 평균 내면 가려진다.
    verdicts = []
    for cat in dict.fromkeys(SPOT_PROXY.values()):
        crow = [r for r in rows if r["category"] == cat and r["recent4"] is not None and r["prior4"] is not None]
        if not crow:
            continue
        pace = pace_change(table[cat]) if cat in table else None
        items = [{"item": r["item"], "recent4": r["recent4"], "prior4": r["prior4"],
                  "label": _label(r["recent4"] - r["prior4"])} for r in crow]
        match = sum(1 for i in items if pace and i["label"] == pace["label"]) if pace else None
        verdicts.append({"category": cat, "items": items, "pace": pace, "match": match, "n_items": len(items)})
    latest = max((r["last_date"] for r in rows), default=None)
    return {"periods": periods, "rows": rows, "verdicts": verdicts, "latest_spot_date": latest}


def rebased_spot(history, since: dt.date) -> dict:
    """품목별 현물가를 since 시점 = 100으로 맞춘 일별 시계열({품목: (날짜들, 값들)})."""
    import pandas as pd
    out = {}
    for item in SPOT_PROXY:
        s = _spot_series(history, item)
        if s.empty:
            continue
        part = s[s.index >= pd.Timestamp(since)]
        if part.empty:
            continue
        base = _value_at(s, since)
        if base is None:                        # since 이전 기록이 없으면 첫 기록을 기준으로
            base = float(part.iloc[0])
        out[item] = ([d.date() for d in part.index], [float(v) / base * 100 for v in part.values])
    return out


# ── AI 분석 프롬프트용 요약 ─────────────────────────────────────────────────

PROMPT_CATEGORIES = ("범용 DRAM", "서버 DRAM", "NAND", "모바일 DRAM", "PC DRAM", "컨슈머 DRAM", "HBM",
                     "클라이언트 SSD", "엔터프라이즈 SSD", "그래픽 DRAM")
PROMPT_QUARTERS_BACK = 3        # 현재 분기를 포함해 최근 4분기까지 보여준다(둔화·가속 흐름이 보이도록)


def _range_text(f: dict) -> str:
    if f["high"] is None:
        return f">{f['low']:g}%"
    return f"{f['low']:g}%" if f["low"] == f["high"] else f"{f['low']:g}~{f['high']:g}%"


def summary_md(store: dict | None = None) -> str:
    """AI 분석 프롬프트에 넣을 고정가 전망. 파일을 읽기만 한다.

    검증을 통과한 전망(원문 인용 확인)만 쓰고, 상향/하향·둔화/가속·누적 지수는 코드가 계산한 값이다.
    LLM이 쓴 글이 아니라 숫자와 날짜만 넘긴다."""
    store = store if store is not None else load()
    forecasts = flat_forecasts(store)
    table = latest_table(forecasts)
    if not table:
        return ""
    today = hbm._now().date()
    first_q = f"{today.year}Q{(today.month - 1) // 3 + 1}"
    for _ in range(PROMPT_QUARTERS_BACK):
        first_q = _prev_quarter(first_q)
    lo_key = _period_key(first_q)
    cur_q = f"{today.year}Q{(today.month - 1) // 3 + 1}"
    prev_q = _prev_quarter(cur_q)

    lines, newest = [], ""
    for cat in PROMPT_CATEGORIES:
        row = table.get(cat)
        if not row:
            continue
        periods = sorted((p for p in row if p[4] == "Q" and _period_key(p) >= lo_key), key=_period_key)
        # 최신 전망이 지난 분기보다 오래된 품목(예: 1Q26에서 멈춘 PC DRAM)은 지금 상황처럼 읽히지 않게 뺀다.
        if not periods or _period_key(periods[-1]) < _period_key(prev_q):
            continue
        parts = []
        for p in periods:
            cell = row[p]
            txt = f"{p} {_range_text(cell['cur'])}"
            if cell["revision"] is not None and abs(cell["revision"]) >= 0.5:
                txt += f"(직전 발표 대비 {'상향' if cell['revision'] > 0 else '하향'} {abs(cell['revision']):g}%p)"
            parts.append(txt)
            newest = max(newest, cell["cur"]["date"])
        pace = pace_change(row)
        tail = f" → 증가 속도 {pace['label']}({pace['from']} 중간 {pace['from_mid']:g}% → {pace['to']} {pace['to_mid']:g}%)" if pace else ""
        lines.append(f"- {cat} QoQ: " + " / ".join(parts) + tail)

    idx_parts = []
    for cat in ("범용 DRAM", "NAND"):
        idx = build_index(table[cat]) if cat in table else None
        if idx:
            last = idx["points"][-1]
            idx_parts.append(f"{cat} {idx['base']}=100 기준 {last['period']}까지 약 {last['mid']:.0f}(극단 {last['low']:.0f}~{last['high']:.0f})")
    if not lines:
        return ""
    n_docs = sum(1 for d in store.get("docs", {}).values() if d.get("status") == "extracted")
    out = [f"(TrendForce 보도자료 {n_docs}건에서 원문 인용이 확인된 전망 · 수집 기준 {store.get('checked_at', '시점 미상')} · "
           f"가장 최근 발표 {newest})", *lines]
    if idx_parts:
        out.append("- 전망 중간값을 이어 곱한 누적 추정 지수: " + "; ".join(idx_parts))
    out.append("※ 고정가(계약가)의 실측은 공개되지 않는다. 위는 TrendForce의 **전망**이고 발표마다 바뀐다 — 실제 계약가라고 쓰지 말고 "
               "'전망'임과 발표일을 밝혀라. 서버 DRAM·HBM의 달러 수준은 데이터에 없다. 숫자를 새로 계산하거나 평균 내지 마라.")
    return "\n".join(out)


# ── 수집기 진입점 ───────────────────────────────────────────────────────────

def _pending(store: dict) -> list[tuple[str, dict]]:
    today = hbm._now().strftime("%Y-%m-%d")
    rows = [(u, d) for u, d in store.get("docs", {}).items()
            if d.get("status") == "pending" and d.get("total_attempts", 0) < MAX_TOTAL_ATTEMPTS
            and (d.get("attempt_day") != today or d.get("attempts", 0) < MAX_ATTEMPTS)]
    rows.sort(key=lambda x: x[1]["date"], reverse=True)
    return rows


def run_once(log=print) -> dict:
    store = load()
    store.setdefault("docs", {})
    cutoff = (hbm._now().date() - dt.timedelta(days=LOOKBACK_DAYS)).isoformat()

    first = not store["docs"]
    try:
        listed = hbm.list_trendforce_research(pages=30 if first else 3)
        store["source_status"] = {"ok": True, "at": hbm._now().strftime("%Y-%m-%d %H:%M")}
    except Exception as exc:
        listed = []
        store["source_status"] = {"ok": False, "at": hbm._now().strftime("%Y-%m-%d %H:%M"),
                                  "error": f"{type(exc).__name__}: {str(exc)[:120]}"}
        log(f"[고정가] 목록 실패: {type(exc).__name__}: {exc}")
    new_docs = 0
    for d in listed:
        if d["date"] < cutoff or d["url"] in store["docs"]:
            continue
        store["docs"][d["url"]] = {"url": d["url"], "date": d["date"], "title": d["title"],
                                   "status": "pending", "attempts": 0, "forecasts": []}
        new_docs += 1

    processed = fetched = 0
    for url, d in _pending(store):
        if processed >= MAX_DOCS_PER_TICK or fetched >= MAX_FETCH_PER_TICK:
            break
        fetched += 1
        today_s = hbm._now().strftime("%Y-%m-%d")
        if d.get("attempt_day") != today_s:
            d["attempt_day"], d["attempts"] = today_s, 0
        d["attempts"] = d.get("attempts", 0) + 1
        d["total_attempts"] = d.get("total_attempts", 0) + 1
        try:
            body = hbm.fetch_trendforce_body(url)
            if not re.search(r"contract price", body, re.I):
                d["status"] = "skipped"                     # 계약가 얘기가 아닌 문서
                continue
            forecasts, dropped = extract(d, body)
            d["forecasts"], d["dropped"], d["status"] = forecasts, dropped, "extracted"
            d["extracted_at"] = hbm._now().strftime("%Y-%m-%d %H:%M")
            processed += 1
            log(f"[고정가] {d['date']} 전망 {len(forecasts)}건 통과"
                + (f" (탈락 {sum(dropped.values())}건: {dropped})" if dropped else ""))
        except Exception as exc:
            log(f"[고정가] 문서 처리 실패({d['attempts']}/{MAX_ATTEMPTS}) {d['title'][:40]}: {type(exc).__name__}: {exc}")
        finally:
            save(store)

    store["checked_at"] = hbm._now().strftime("%Y-%m-%d %H:%M")
    save(store)
    return {"new_docs": new_docs, "processed": processed, "pending": len(_pending(store))}


def tick(now: dt.datetime, log=print) -> None:
    """수집기가 매번 부른다. 주기가 안 됐으면 즉시 돌아간다."""
    global _last_tick, _backlog
    if not 6 <= now.hour < 24:
        return
    if _last_tick is not None:
        every = BACKLOG_EVERY_MIN if _backlog else CHECK_EVERY_MIN
        if (now - _last_tick).total_seconds() < every * 60:
            return
    _last_tick = now
    _backlog = run_once(log)["pending"] > 0
