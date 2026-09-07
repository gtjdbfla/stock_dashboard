"""증권사별 목표주가를 모아서 컨센서스를 직접 계산한다.

FnGuide가 주는 컨센서스(wcomp의 snpTargetChart)는 주 1회만 갱신된다. 실제로
2026-08-31에 LS증권이 목표주가를 330만->240만으로 내렸을 때, FnGuide 값은
2026-08-28자 3,317,917원에 그대로 멈춰 있었다. 그래서 원자료를 직접 모은다.

출처는 네이버 리서치 '리포트 상세' 페이지다. 목록에는 목표가가 없지만 상세에는
`목표가 2,800,000 | 투자의견 매수` 형태로 구조화돼 있어서, PDF를 열지 않고도 뽑힌다.
(PDF 본문 파싱도 되지만 표지가 이미지인 리포트에서 64자만 나오는 등 실패가 잦다.
 상세 페이지는 그런 리포트도 값을 준다.)

한계 하나는 분명히 해 둔다: 네이버가 싣는 증권사만 잡힌다. LS증권처럼 미수록
증권사가 목표주가를 바꾼 건은 여기 안 들어온다(그쪽은 뉴스 질의가 맡는다).
"""
import csv
import datetime as dt
import os
import re
import statistics
import time

import requests
from bs4 import BeautifulSoup

KST = dt.timezone(dt.timedelta(hours=9))
BASE = "https://finance.naver.com/research/"
STORE = os.environ.get("ANALYST_TARGET_FILE", "data/analyst_targets.csv")
_UA = {"User-Agent": "Mozilla/5.0"}

# 컨센서스에 넣을 기간. 반년 전 목표가까지 섞으면 평균이 과거에 끌려간다.
# (실측: 전 구간 57건 평균 2,804,035원 vs 최근 3개월 기준은 그보다 한참 높다.)
CONSENSUS_MONTHS = int(os.environ.get("CONSENSUS_MONTHS", "3"))
FIELDS = ["nid", "작성일", "증권사", "목표주가", "투자의견", "제목"]
# 목표가가 빈 리포트를 며칠까지 다시 열어볼지. 이보다 오래된 건은 확정으로 본다.
RETRY_EMPTY_DAYS = int(os.environ.get("TARGET_RETRY_EMPTY_DAYS", "7"))

_TP = re.compile(r"목표가\s*([\d,]{5,12})")
_OP = re.compile(r"투자의견\s*([가-힣A-Za-z.]{1,12})")


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


def list_reports(ticker: str, pages: int = 2) -> list[dict]:
    """리포트 목록에서 nid·증권사·작성일을 뽑는다. 목표가는 여기 없다."""
    out: list[dict] = []
    for page in range(1, pages + 1):
        r = requests.get(BASE + "company_list.naver",
                         params={"searchType": "itemCode", "itemCode": ticker,
                                 "page": page},
                         headers=_UA, timeout=15)
        r.encoding = "euc-kr"
        soup = BeautifulSoup(r.text, "html.parser")
        for tr in soup.select("table.type_1 tr"):
            tds = tr.select("td")
            if len(tds) < 5:
                continue
            a = tds[1].select_one("a[href*='company_read']")
            date = tds[4].get_text(strip=True)
            if not a or not re.match(r"\d{2}\.\d{2}\.\d{2}", date):
                continue
            m = re.search(r"nid=(\d+)", a["href"])
            if not m:
                continue
            out.append({
                "nid": m.group(1),
                "작성일": "20" + date.replace(".", "-"),
                "증권사": tds[2].get_text(strip=True),
                "제목": a.get_text(strip=True),
            })
    return out


def fetch_target(nid: str) -> tuple[int | None, str | None]:
    """리포트 상세에서 (목표주가, 투자의견). 없으면 (None, None)."""
    try:
        r = requests.get(BASE + "company_read.naver",
                         params={"nid": nid, "page": 1}, headers=_UA, timeout=15)
        r.encoding = "euc-kr"
        txt = _text(r.text)
        tp = _TP.search(txt)
        op = _OP.search(txt)
        price = int(tp.group(1).replace(",", "")) if tp else None
        # '투자의견 없음'은 목표가를 안 준 리포트(산업 코멘트 등)라 의견도 버린다.
        opinion = op.group(1).strip() if op else None
        if opinion == "없음":
            opinion = None
        return price, opinion
    except Exception:
        return None, None


def load() -> list[dict]:
    if not os.path.exists(STORE):
        return []
    try:
        with open(STORE, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _save(rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(STORE) or ".", exist_ok=True)
    tmp = STORE + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in sorted(rows, key=lambda x: str(x.get("작성일")), reverse=True):
            w.writerow({k: r.get(k, "") for k in FIELDS})
    os.replace(tmp, STORE)


def refresh(ticker: str, pages: int = 2, log=print) -> int:
    """새로 올라온 리포트만 상세를 열어 목표가를 붙이고 누적한다. 반환: 새로 담은 건수.

    이미 담은 nid는 상세를 다시 열지 않는다. 리포트는 나온 뒤 값이 바뀌지 않으므로
    한 번 받으면 그만이고, 매번 30건씩 다시 여는 건 네이버에도 예의가 아니다.
    """
    have = {r["nid"]: r for r in load()}
    added = 0
    # 목표가가 비어 있는 건은 최근 것만 다시 열어 본다. 조회가 실패한 날 빈 값으로 저장되면
    # 'nid in have'에 걸려 영영 다시 안 보는데, 그 한 건이 통째로 컨센서스에서 빠진다.
    # 오래된 건은 재시도해도 소용없다 — 목표가를 아예 안 준 리포트(컨퍼런스콜 후기 등)라
    # 몇 번을 열어도 None이다. 그래서 최근 것만 확인하고 나머지는 확정으로 둔다.
    retry_after = (dt.datetime.now(KST) - dt.timedelta(days=RETRY_EMPTY_DAYS)).strftime("%Y-%m-%d")
    for it in list_reports(ticker, pages=pages):
        old_row = have.get(it["nid"])
        if old_row is not None:
            filled = str(old_row.get("목표주가") or "").strip()
            if filled or str(old_row.get("작성일") or "") < retry_after:
                continue
        price, opinion = fetch_target(it["nid"])
        have[it["nid"]] = {**it, "목표주가": price or "", "투자의견": opinion or ""}
        added += 1
        time.sleep(0.25)          # 상세를 연달아 여니 간격을 둔다
    if added:
        _save(list(have.values()))
        log(f"[targets] 새 리포트 {added}건 (누적 {len(have)}건)")
    return added


def latest_by_broker(months: int = CONSENSUS_MONTHS,
                     now: dt.datetime | None = None) -> dict[str, dict]:
    """기간 안에서 증권사별 '가장 최근' 목표주가 한 건씩.

    같은 증권사가 여러 번 냈으면 최신 것만 남긴다. 그래야 리포트를 자주 내는
    증권사(미래에셋이 5월에만 5건)가 평균을 좌우하지 않는다.
    """
    now = now or dt.datetime.now(KST)
    cutoff = (now - dt.timedelta(days=months * 31)).strftime("%Y-%m-%d")
    best: dict[str, dict] = {}
    for r in load():
        tp = str(r.get("목표주가") or "").strip()
        date = str(r.get("작성일") or "")
        if not tp or date < cutoff:
            continue
        brk = str(r.get("증권사") or "").strip()
        if not brk:
            continue
        if brk not in best or date > best[brk]["작성일"]:
            best[brk] = {**r, "목표주가": int(float(tp))}
    return best


def consensus(months: int = CONSENSUS_MONTHS,
              now: dt.datetime | None = None) -> dict:
    """직접 계산한 컨센서스. 값이 없으면 빈 dict."""
    best = latest_by_broker(months=months, now=now)
    vals = [v["목표주가"] for v in best.values()]
    if not vals:
        return {}
    return {
        "평균": round(statistics.mean(vals)),
        # 한 곳이 크게 어긋난 값을 내면 평균이 끌려간다. 중앙값을 같이 둔다.
        "중앙값": round(statistics.median(vals)),
        "최고": max(vals),
        "최저": min(vals),
        "기관수": len(vals),
        "기준개월": months,
        "최신일": max(v["작성일"] for v in best.values()),
    }


def summary_md(months: int = CONSENSUS_MONTHS) -> str:
    """AI 프롬프트에 넣을 증권사별 목표주가 표."""
    best = latest_by_broker(months=months)
    if not best:
        return ""
    c = consensus(months=months)
    lines = [f"직접 집계한 컨센서스(최근 {months}개월, {c['기관수']}곳): "
             f"평균 {c['평균']:,}원 · 중앙값 {c['중앙값']:,}원 "
             f"· 최고 {c['최고']:,}원 · 최저 {c['최저']:,}원"]
    for brk, r in sorted(best.items(), key=lambda x: -x[1]["목표주가"]):
        op = f" {r['투자의견']}" if r.get("투자의견") else ""
        lines.append(f"- {brk} {r['목표주가']:,}원{op} ({r['작성일']})")
    return "\n".join(lines)


# 리포트는 장 시작 전(07~08시)과 장중에 올라온다. 자주 볼 필요는 없지만
# 하루 한 번이면 오전에 나온 목표주가 변경을 저녁까지 놓친다. 2시간마다 본다.
CHECK_FROM = dt.time(7, 30)
CHECK_TO = dt.time(20, 0)
CHECK_EVERY_MIN = 120
_last_check_at: dt.datetime | None = None


def tick(now: dt.datetime, ticker: str, log=print) -> None:
    """수집기에서 주기적으로 부른다. 새 리포트만 상세를 열어 목표가를 누적한다."""
    global _last_check_at
    if now.weekday() >= 5 or not (CHECK_FROM <= now.time() <= CHECK_TO):
        return
    if _last_check_at and (now - _last_check_at).total_seconds() < CHECK_EVERY_MIN * 60:
        return
    _last_check_at = now
    refresh(ticker, log=log)
