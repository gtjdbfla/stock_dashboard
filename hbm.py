"""HBM 펀더멘털 자료를 모아 검증하고 정리한다. **streamlit을 import하지 않는다.**

HBM은 현물가가 없다(고객사와 연·분기 단위로 협상하는 계약가). 그래서 매일 움직이는 숫자 대신
'날짜와 출처가 붙은 사실'을 누적한다. 수치를 LLM이 지어내거나 잘못 옮기는 게 가장 큰 위험이라
세 겹으로 막는다.

1. 숫자는 가능한 한 코드가 직접 뽑는다(마이크론 SEC 실적 표). LLM에는 숫자를 맡기지 않는다.
2. 문장에서 사실을 뽑을 때는 LLM에게 **원문 인용(quote)을 같이** 내게 하고, 그 인용이 본문에
   글자 그대로 있는지, 주장(claim)에 쓴 숫자가 인용에 있는지 코드가 확인한다. 하나라도 틀리면 버린다.
3. 종합 분석은 검증을 통과한 사실 목록 안에서만 쓰게 하고, 문장마다 근거 번호([F3])를 달게 한 뒤
   없는 번호는 코드가 지운다.

출처 등급: A=회사 자료(마이크론 실적·Prepared Remarks), B=TrendForce 리서치 보도자료,
C=언론 보도를 인용한 기사(TrendForce News, 네이버 뉴스 — 'reportedly').
수집기(collector.py)가 `tick()`으로 채우고, 화면은 `load()`로 파일만 읽는다.
"""
import datetime as dt
import hashlib
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup

import llm

KST = dt.timezone(dt.timedelta(hours=9))
STORE = os.environ.get("HBM_DIGEST_FILE", "data/hbm_digest.json")

UA_BROWSER = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
UA_SEC = {"User-Agent": "PersonalDashboard contact@example.com"}      # SEC는 UA 선언을 요구한다
HTTP_TIMEOUT = (5, 25)

TRENDFORCE_RESEARCH_LIST = "https://www.trendforce.com/presscenter/news?page={page}"
TRENDFORCE_NEWS_HOME = "https://www.trendforce.com/news/"
SKHYNIX_NEWSROOM = "https://news.skhynix.co.kr"
SKHYNIX_LISTS = ("/category/press/", "/category/ir/", "/category/tech-and-ai/", "/?s=HBM")
MICRON_CIK = "0000723125"
MICRON_IR_HOME = "https://investors.micron.com/"
MICRON_REMARKS_URL = "https://s25.q4cdn.com/621799436/files/doc_financials/{fy}/q{q}/Q{q}-FY{yy}-Prepared-Remarks.pdf"
NAVER_NEWS_URL = "https://search.naver.com/search.naver"
HEADLINE_QUERIES = ["HBM4", "SK하이닉스 HBM", "삼성전자 HBM", "마이크론 HBM", "엔비디아 HBM"]

TOPICS = ("가격", "수급·생산능력", "점유율·경쟁", "제품·로드맵", "고객·퀄", "수요", "실적·마진", "리스크")
DIRECTIONS = ("호재", "중립", "악재")
TIER_LABEL = {"A": "회사 자료", "B": "TrendForce 리서치", "C": "언론 보도 인용"}
SOURCE_LABEL = {
    "skhynix_newsroom": "SK하이닉스 뉴스룸",
    "micron_remarks": "마이크론 Prepared Remarks",
    "trendforce_research": "TrendForce 보도자료",
    "trendforce_news": "TrendForce News",
}
SOURCE_TIER = {"skhynix_newsroom": "A", "micron_remarks": "A", "trendforce_research": "B", "trendforce_news": "C"}

LOOKBACK_DAYS = 150            # 이보다 오래된 문서는 새로 받지 않는다
FACT_WINDOW_DAYS = 200         # 종합 분석에 쓰는 사실의 범위
BODY_CHARS = 14000             # 한 문서에서 LLM에 넣는 본문 상한
MAX_FACTS_PER_DOC = 8
MAX_FETCH_PER_TICK = 30
MAX_DOCS_PER_TICK = 3          # 한 번에 LLM으로 읽는 문서 수 — 무료 한도(모델당 하루 20회)를 아낀다
MAX_ATTEMPTS = 3               # 하루 시도 상한. 무료 한도가 소진된 날 문서가 영영 버려지지 않게 날짜별로 센다
MAX_TOTAL_ATTEMPTS = 12
MAX_SYNTH_FACTS = 70
# 종합 분석에 넣는 사실의 출처별 상한. 날짜순으로만 자르면 글이 많이 올라오는 곳(회사 뉴스룸의 전시·수상 소식)이
# 시장 분석을 밀어낸다. 출처마다 최신 사실부터 이만큼씩 담는다.
SOURCE_QUOTA = {"micron_remarks": 14, "trendforce_research": 34, "skhynix_newsroom": 14, "trendforce_news": 8}
CHECK_EVERY_MIN = 180          # 평소 확인 주기
BACKLOG_EVERY_MIN = 20         # 읽을 문서가 밀려 있을 때
SYNTH_MIN_GAP_HOURS = 6
HEADLINE_KEEP = 40

_last_tick: dt.datetime | None = None


# ── 저장 ────────────────────────────────────────────────────────────────────

def load() -> dict:
    try:
        with open(STORE, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def save(payload: dict) -> None:
    """수집기가 쓰고 화면이 읽는다. 반쯤 쓰인 파일을 읽지 않게 통째로 바꾼다."""
    try:
        os.makedirs(os.path.dirname(STORE) or ".", exist_ok=True)
        tmp = f"{STORE}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        os.replace(tmp, STORE)
    except OSError:
        pass


def _now() -> dt.datetime:
    return dt.datetime.now(KST)


# ── 원문 검증 ───────────────────────────────────────────────────────────────

_QUOTE_FOLD = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "‑": "-", " ": " "})


def _fold(text: str) -> str:
    """인용 비교용 정규화. PDF는 줄바꿈 자리에 'high- capacity'처럼 공백·하이픈이 끼어서,
    공백과 하이픈을 전부 없애고 소문자로 맞춘 뒤 비교한다."""
    return re.sub(r"[\s\-]+", "", text.translate(_QUOTE_FOLD)).lower()


_NUM_PAT = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> list[str]:
    """숫자 토큰(쉼표 제거). 한 자리 수는 문장 속 서수·품목명(HBM4, 8-Hi)과 섞여 거르지 않으면 오탐이 많다."""
    out = []
    for tok in _NUM_PAT.findall(text):
        tok = tok.replace(",", "").rstrip(".")
        if len(tok.replace(".", "")) >= 2:
            out.append(tok)
    return out


def validate_fact(raw: dict, body: str, title: str = "") -> tuple[dict | None, str]:
    """LLM이 낸 사실 하나를 원문과 대조한다. (통과한 사실 or None, 탈락 사유)."""
    if not isinstance(raw, dict):
        return None, "형식"
    topic = str(raw.get("topic") or "").strip()
    claim = re.sub(r"\s+", " ", str(raw.get("claim") or "")).strip()
    quote = re.sub(r"\s+", " ", str(raw.get("quote") or "")).strip()
    subject = re.sub(r"\s+", " ", str(raw.get("subject") or "")).strip()[:30] or "업계"
    direction = str(raw.get("direction") or "").strip()
    if topic not in TOPICS:
        return None, "주제"
    if len(claim) < 8:
        return None, "주장 없음"
    if len(quote) < 20:
        return None, "인용 짧음"
    folded_body = _fold(body)
    if _fold(quote) not in folded_body:
        return None, "인용이 원문에 없음"
    # 주장에 쓴 숫자가 인용(또는 제목)에 그대로 있어야 한다. 옮기다 틀리거나 지어낸 숫자를 거른다.
    # 연도(2027 등)만은 문맥에서 따오는 경우가 많아, 인용 대신 본문 전체에서 확인한다.
    haystack = _fold(quote + " " + title).replace(",", "")
    body_hay = folded_body.replace(",", "")
    for tok in _numbers(claim):
        where = body_hay if re.fullmatch(r"20\d\d", tok) else haystack
        if _fold(tok) not in where:
            return None, f"숫자 {tok}이 인용에 없음"
    if direction not in DIRECTIONS:
        direction = "중립"
    return {"topic": topic, "subject": subject, "claim": claim, "quote": quote, "direction": direction}, ""


# ── 문서 목록 ───────────────────────────────────────────────────────────────

def _get(url: str, headers: dict | None = None, **kw) -> requests.Response:
    r = requests.get(url, headers=headers or UA_BROWSER, timeout=HTTP_TIMEOUT, **kw)
    r.raise_for_status()
    return r


def list_trendforce_research(pages: int = 5) -> list[dict]:
    """TrendForce 보도자료(press center). 리서치팀이 낸 전망·집계라 B등급."""
    found: dict[str, dict] = {}
    for page in range(1, pages + 1):
        soup = BeautifulSoup(_get(TRENDFORCE_RESEARCH_LIST.format(page=page)).text, "html.parser")
        new = 0
        for a in soup.find_all("a", href=True):
            m = re.match(r"^(?:https://www\.trendforce\.com)?(/presscenter/news/(\d{8})-\d+\.html)", a["href"])
            title = re.sub(r"\s+", " ", a.get_text(" ", strip=True))
            if not m or len(title) < 15:
                continue
            url = "https://www.trendforce.com" + m.group(1)
            if url not in found:
                found[url] = {"url": url, "date": f"{m.group(2)[:4]}-{m.group(2)[4:6]}-{m.group(2)[6:]}",
                              "title": title, "source": "trendforce_research"}
                new += 1
        if new == 0:
            break
    return list(found.values())


def list_trendforce_news() -> list[dict]:
    """TrendForce News 첫 화면(최근 7~9건). 태그·카테고리 페이지는 오래된 사본이라 못 쓴다.
    다른 매체 보도를 옮긴 기사('reportedly')라 C등급 — 그래서 매번 훑어 누적한다."""
    soup = BeautifulSoup(_get(TRENDFORCE_NEWS_HOME).text, "html.parser")
    found: dict[str, dict] = {}
    for a in soup.find_all("a", href=True):
        m = re.search(r"(https://www\.trendforce\.com)?/news/(\d{4})/(\d{2})/(\d{2})/([\w-]+)", a["href"])
        title = re.sub(r"\s+", " ", a.get_text(" ", strip=True))
        if not m or len(title) < 15 or title == "View More":
            continue
        url = f"https://www.trendforce.com/news/{m.group(2)}/{m.group(3)}/{m.group(4)}/{m.group(5)}"
        found.setdefault(url, {"url": url, "date": f"{m.group(2)}-{m.group(3)}-{m.group(4)}",
                               "title": re.sub(r"^\[[^\]]+\]\s*", "", title), "source": "trendforce_news"})
    return list(found.values())


def list_skhynix_newsroom(backfill: bool = False) -> list[dict]:
    """SK하이닉스 공식 뉴스룸. 회사가 직접 내는 글이라 A등급 — 단, 홍보성 글이 많아서
    사실 추출 단계에서 수치·계획만 건지고 수식어는 버린다."""
    found: dict[str, dict] = {}
    today = _now().strftime("%Y-%m-%d")
    for path in SKHYNIX_LISTS:
        urls = [SKHYNIX_NEWSROOM + path]
        if backfill and "?" not in path:
            urls.append(SKHYNIX_NEWSROOM + path + "page/2/")
        for url in urls:
            soup = BeautifulSoup(_get(url).text, "html.parser")
            for a in soup.find_all("a", href=True):
                m = re.match(r"^https://news\.skhynix\.co\.kr/([\w%-]+)/?$", a["href"])
                text = re.sub(r"\s+", " ", a.get_text(" ", strip=True))
                if not m or m.group(1) in ("category", "tag", "page", "en") or len(text) < 14:
                    continue
                dm = re.search(r"(\d{4}-\d{2}-\d{2})\s*$", text)
                title = text[:dm.start()].strip() if dm else text
                link = f"{SKHYNIX_NEWSROOM}/{m.group(1)}/"
                found.setdefault(link, {"url": link, "date": dm.group(1) if dm else today, "title": title,
                                        "source": "skhynix_newsroom", "date_unsure": not dm})
    return list(found.values())


def fetch_skhynix_body(url: str) -> tuple[str, str | None]:
    """뉴스룸 본문과 게시일(메타태그). 목록에 날짜가 없는 글의 날짜를 여기서 바로잡는다."""
    soup = BeautifulSoup(_get(url).text, "html.parser")
    meta = soup.find("meta", property="article:published_time")
    date = meta["content"][:10] if meta and meta.get("content") else None
    paras = [re.sub(r"\s+", " ", p.get_text(" ", strip=True)) for p in soup.find_all("p")]
    return "\n".join(t for t in paras if len(t) >= 25), date


_BOILERPLATE = ("TrendForce News operates independently", "Copyright", "All rights reserved")


def fetch_trendforce_body(url: str) -> str:
    soup = BeautifulSoup(_get(url).text, "html.parser")
    paras = []
    for p in soup.find_all("p"):
        t = re.sub(r"\s+", " ", p.get_text(" ", strip=True))
        if len(t) >= 25 and not t.startswith(_BOILERPLATE):
            paras.append(t)
    return "\n".join(paras)


def _pdf_text(url: str) -> str | None:
    from pypdf import PdfReader
    r = requests.get(url, headers=UA_BROWSER, timeout=(5, 40))
    if r.status_code != 200 or r.content[:4] != b"%PDF":
        return None
    return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(r.content)).pages)


# ── 마이크론: 숫자는 코드가 직접 ────────────────────────────────────────────

_BU = (("CMBU", "Cloud Memory"), ("CDBU", "Core Data Center"),
       ("MCBU", "Mobile and Client"), ("AEBU", "Automotive and Embedded"))


def parse_micron_release(text: str) -> dict:
    """실적 보도자료(Exhibit 99.1) 본문에서 분기 매출과 사업부 표를 읽는다. {분기: {...}}.

    한 보도자료에 당분기·직전분기·전년동기 세 열이 있어 한 번에 세 분기가 나온다.
    사업부 매출 합이 전체 매출과 1% 안에서 안 맞으면 표를 잘못 읽은 것이라 통째로 버린다."""
    head = re.search(r"Quarterly Business Unit Financial Results (FQ\d-\d\d) (FQ\d-\d\d) (FQ\d-\d\d)", text)
    total = re.search(r"Revenue \$ ([\d,]+) \$ ([\d,]+) \$ ([\d,]+) \$ [\d,]+ \$ [\d,]+ \$ [\d,]+", text)
    if not head or not total:
        return {}
    quarters = list(head.groups())
    revenue = [int(x.replace(",", "")) for x in total.groups()]
    out = {q: {"quarter": q, "revenue": revenue[i]} for i, q in enumerate(quarters)}
    for code, name in _BU:
        m = re.search(
            rf"{name} Business Unit Revenue \$ ([\d,]+) \$ ([\d,]+) \$ ([\d,]+) "
            rf"Gross margin (\d+) % (\d+) % (\d+) % Operating margin (\d+) % (\d+) % (\d+) %", text)
        if not m:
            return {}
        g = m.groups()
        for i, q in enumerate(quarters):
            out[q][f"{code}_rev"] = int(g[i].replace(",", ""))
            out[q][f"{code}_gm"] = int(g[3 + i])
            out[q][f"{code}_om"] = int(g[6 + i])
    for q, row in out.items():
        parts = sum(row[f"{c}_rev"] for c, _ in _BU)
        if abs(parts - row["revenue"]) > row["revenue"] * 0.01:
            return {}
    return out


def fetch_micron_quarters(limit: int = 6) -> tuple[list[dict], dict[str, str]]:
    """SEC EDGAR의 실적 8-K(Item 2.02)에서 분기 표를 모은다.
    반환: (분기 목록 — 오래된 순, {분기: 실적 발표일})."""
    subs = _get(f"https://data.sec.gov/submissions/CIK{MICRON_CIK}.json", UA_SEC).json()["filings"]["recent"]
    filings = [(subs["filingDate"][i], subs["accessionNumber"][i]) for i in range(len(subs["form"]))
               if subs["form"][i] == "8-K" and "2.02" in (subs.get("items") or [""] * 999)[i]][:limit]
    merged: dict[str, dict] = {}
    release_date: dict[str, str] = {}
    for date, acc in reversed(filings):                      # 오래된 것부터 — 새 공시가 덮어쓴다
        nodash = acc.replace("-", "")
        idx = _get(f"https://www.sec.gov/Archives/edgar/data/{int(MICRON_CIK)}/{nodash}/", UA_SEC).text
        links = re.findall(rf'href="(/Archives/edgar/data/{int(MICRON_CIK)}/{nodash}/[^"]*(?:ex99|ex-99|pressrelease)[^"]*\.htm)"',
                           idx, re.I)
        if not links:
            continue
        text = re.sub(r"\s+", " ", BeautifulSoup(_get("https://www.sec.gov" + links[0], UA_SEC).text,
                                                 "html.parser").get_text(" ", strip=True))
        parsed = parse_micron_release(text)
        current = re.search(r"Quarterly Business Unit Financial Results (FQ\d-\d\d)", text)
        for q, row in parsed.items():
            row["source_date"] = date
            merged[q] = row
        if current and parsed:
            release_date[current.group(1)] = date
    def key(q):                                              # FQ4-26 -> (26, 4)
        m = re.match(r"FQ(\d)-(\d\d)", q["quarter"])
        return (int(m.group(2)), int(m.group(1)))
    return sorted(merged.values(), key=key), release_date


def discover_micron_remarks(release_date: dict[str, str], known: dict) -> list[dict]:
    """Prepared Remarks PDF 주소는 규칙이 있는 분기도, 없는 분기도 있다(FY26 Q3·Q4는 규칙대로,
    Q1·Q2는 404). IR 첫 화면에서 최신 링크를 찾고, 규칙으로 추측한 주소도 시도한다.
    404는 일주일에 한 번만 다시 본다."""
    cands: dict[str, str] = {}
    try:
        for href in re.findall(r'(?:https:)?(//s25\.q4cdn\.com/[^"\']*?Q(\d)-FY(\d\d)-Prepared-Remarks\.pdf)',
                               _get(MICRON_IR_HOME).text, re.I):
            cands[f"FQ{href[1]}-{href[2]}"] = "https:" + href[0]
    except requests.RequestException:
        pass
    for fy in (2025, 2026, 2027):
        for q in (1, 2, 3, 4):
            label = f"FQ{q}-{fy % 100:02d}"
            cands.setdefault(label, MICRON_REMARKS_URL.format(fy=fy, q=q, yy=fy % 100))
    today = _now().date()
    out = []
    for label, url in cands.items():
        if label not in release_date:                       # 발표 안 된 분기
            continue
        if url in known.get("docs", {}):
            continue
        tried = known.get("remarks_tried", {}).get(url)
        if tried and (today - dt.date.fromisoformat(tried)).days < 7:
            continue
        known.setdefault("remarks_tried", {})[url] = today.isoformat()
        out.append({"url": url, "date": release_date[label], "source": "micron_remarks",
                    "title": f"마이크론 {label} Prepared Remarks"})
    return out


# ── 네이버 뉴스 헤드라인 (C등급, 본문 없음) ────────────────────────────────

def refresh_headlines(store: dict) -> int:
    """HBM 제목이 든 기사만 모은다. 본문과 주소가 없어 사실 추출은 하지 않고 제목만 보여준다."""
    import ai_inputs                                           # 순환 방지: 필요할 때만 가져온다
    seen = {h["title"] for h in store.get("headlines", [])}
    added = 0
    today = _now().strftime("%Y-%m-%d")
    for q in HEADLINE_QUERIES:
        try:
            items = ai_inputs.fetch_news_with_summary(q, count=8)
        except Exception:
            continue
        for it in items:
            title = it.get("제목", "")
            if "HBM" not in title.upper() or title in seen:
                continue
            seen.add(title)
            store.setdefault("headlines", []).insert(0, {"title": title, "when": it.get("시점", ""),
                                                         "first_seen": today})
            added += 1
    store["headlines"] = store.get("headlines", [])[:HEADLINE_KEEP]
    return added


# ── LLM으로 사실 추출 ───────────────────────────────────────────────────────

def _extraction_prompt(doc: dict, body: str) -> str:
    return f"""다음은 [{SOURCE_LABEL[doc['source']]}] "{doc['title']}" ({doc['date']}) 의 본문이다.
HBM(고대역폭 메모리)에 관한 **구체적 사실**만 골라 JSON 배열로만 답해라(설명·코드펜스 금지). 최대 {MAX_FACTS_PER_DOC}개.
HBM과 직접 관련된 내용이 없으면 [] 만 답해라.

각 원소의 형식:
{{"topic": "{'|'.join(TOPICS)}" 중 하나,
 "subject": "그 사실의 주체(SK하이닉스, 삼성전자, 마이크론, 엔비디아, 업계 등)",
 "claim": "한국어 한 문장. 숫자와 단위는 본문 그대로 쓰고 환산하지 마라",
 "quote": "claim의 근거가 되는 본문 문장을 **한 글자도 고치지 말고 그대로 복사**(200자 이내)",
 "direction": "HBM 시장의 수익성 환경(가격·수요·공급 타이트함)에 우호적이면 호재, 비우호적(가격 하락·수요 둔화·공급 과잉)이면 악재. 특정 기업의 제품·증설·고객 소식이라 시장 전체에 대한 영향이 불분명하면 중립"}}

규칙:
- 본문에 없는 내용, 네가 아는 배경지식은 쓰지 마라.
- 전망·추정은 claim에 '전망'이라고, 'reportedly'로 전해진 소문은 '~라고 보도'라고 표시해라.
- 같은 내용을 중복해서 쓰지 마라.

본문:
{body[:BODY_CHARS]}"""


def _parse_json_list(text: str) -> list:
    """모델 답에서 JSON 배열을 꺼낸다. 배열이 없거나 깨졌으면 예외로 올린다 —
    '사실 0건'으로 조용히 확정해 버리면 그 문서는 영영 다시 읽히지 않는다."""
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise ValueError("답에 JSON 배열이 없음")
    try:
        data = json.loads(m.group())
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 해석 실패: {exc.msg}") from exc
    if not isinstance(data, list):
        raise ValueError("JSON이 배열이 아님")
    return data


def extract_facts(doc: dict, body: str, log=print) -> tuple[list[dict], dict]:
    """본문에서 사실을 뽑아 원문 대조를 통과한 것만 돌려준다. (사실 목록, 탈락 사유별 개수)."""
    text, model = llm.call(_extraction_prompt(doc, body), min_usable=2)
    kept, dropped = [], {}
    for raw in _parse_json_list(text)[:MAX_FACTS_PER_DOC * 2]:
        fact, why = validate_fact(raw, body, doc["title"])
        if fact is None:
            dropped[why] = dropped.get(why, 0) + 1
            continue
        fact["id"] = hashlib.sha1((doc["url"] + fact["quote"]).encode("utf-8")).hexdigest()[:8]
        kept.append(fact)
        if len(kept) >= MAX_FACTS_PER_DOC:
            break
    return kept, dropped


# ── 종합 분석 ───────────────────────────────────────────────────────────────

def _micron_lines(quarters: list[dict]) -> list[str]:
    """종합 분석에 근거로 넘기는 마이크론 숫자(코드가 계산한 값)."""
    lines = []
    for i, q in enumerate(quarters[-4:]):
        prev = quarters[-4:][i - 1] if i else None
        qoq = f", 직전 분기 대비 {q['CMBU_rev'] / prev['CMBU_rev'] - 1:+.0%}" if prev else ""
        lines.append(f"{q['quarter']}: 전체 매출 ${q['revenue'] / 1000:.2f}B, 클라우드 메모리 사업부(CMBU, HBM 포함) "
                     f"매출 ${q['CMBU_rev'] / 1000:.2f}B{qoq}, 매출총이익률 {q['CMBU_gm']}%, 영업이익률 {q['CMBU_om']}%")
    return lines


def collect_fact_list(store: dict, balanced: bool = True) -> list[dict]:
    """검증을 통과한 사실 목록(최신순). balanced=True면 종합 분석용으로 출처별 상한을 적용하고 번호(F1...)를 붙인다.
    balanced=False는 화면 타임라인용 — 기간 안의 사실 전부."""
    cutoff = (_now().date() - dt.timedelta(days=FACT_WINDOW_DAYS)).isoformat()
    rows = []
    for url, d in store.get("docs", {}).items():
        if d.get("status") != "extracted" or d["date"] < cutoff:
            continue
        for f in d.get("facts", []):
            rows.append({**f, "date": d["date"], "source": d["source"], "tier": SOURCE_TIER[d["source"]],
                         "title": d["title"], "url": url})
    tier_rank = {"A": 0, "B": 1, "C": 2}
    rows.sort(key=lambda r: (r["date"], -tier_rank[r["tier"]]), reverse=True)
    if not balanced:
        return rows
    taken: dict[str, int] = {}
    picked = []
    for r in rows:                                   # 최신순으로 훑으며 출처별 상한까지만 담는다
        if taken.get(r["source"], 0) >= SOURCE_QUOTA.get(r["source"], 10):
            continue
        taken[r["source"]] = taken.get(r["source"], 0) + 1
        picked.append(r)
    picked = picked[:MAX_SYNTH_FACTS]
    for i, r in enumerate(picked, 1):
        r["fid"] = f"F{i}"
    return picked


def _coverage_line(facts: list[dict]) -> str:
    n: dict[str, int] = {}
    for f in facts:
        n[f["source"]] = n.get(f["source"], 0) + 1
    return ", ".join(f"{SOURCE_LABEL[k]} {v}건" for k, v in sorted(n.items(), key=lambda x: -x[1]))


def _synthesis_prompt(facts: list[dict], micron: list[str]) -> str:
    fact_lines = "\n".join(
        f"[{f['fid']}] ({f['date']}, {TIER_LABEL[f['tier']]}·{SOURCE_LABEL[f['source']]}, {f['subject']}, {f['topic']}, {f['direction']}) {f['claim']}"
        for f in facts)
    micron_lines = "\n".join(f"[M{i}] {l}" for i, l in enumerate(micron, 1)) or "(마이크론 숫자 없음)"
    return f"""너는 SK하이닉스 HBM 사업의 펀더멘털을 점검하는 애널리스트다. 아래 **검증된 사실 목록만** 근거로 쓴다.
각 사실은 원문 인용이 확인된 것이고, 등급은 A=회사 자료 / B=TrendForce 리서치 / C=언론 보도 인용(소문 가능)이다.

[사실 목록] (출처별 건수: {_coverage_line(facts)} — 한쪽 출처에 치우쳤으면 그 한계를 결론에 밝혀라)
{fact_lines}

[마이크론 실적 숫자 — 코드가 SEC 보도자료에서 직접 뽑은 값]
{micron_lines}

아래 형식으로 한국어로 작성해라.

## 한 줄 결론
HBM 펀더멘털이 지금 어느 쪽으로 기울었는지 2문장 이내.

## 가격·수익성
## 공급·생산능력
## 경쟁 구도·점유율
## 제품 로드맵·고객 퀄
## 리스크·반론
## SK하이닉스에 주는 시사점
(이 항목만 해석을 허용한다. 해석 문장은 "해석:"으로 시작하고, 그 해석의 바탕이 된 사실 번호를 붙여라. SK하이닉스 자체의 발언·수치가 목록에 없으면 그 사실을 먼저 밝혀라.)

규칙(반드시 지켜라):
- 모든 사실 문장 끝에 근거 번호를 [F3] 또는 [F3][M1]처럼 붙여라. 위 목록에 없는 번호를 만들지 마라.
- 목록에 없는 내용은 쓰지 마라. 근거가 없는 소제목은 "이번 자료에는 근거가 없다"고만 써라.
- C등급(보도 인용)에만 의존한 내용은 "보도에 따르면"이라고 밝히고, A·B등급과 어긋나면 두 쪽을 모두 적어라.
- 날짜가 오래된 사실(60일 이상)은 날짜를 밝혀라. 전망은 전망이라고 써라.
- '호재/악재' 표시는 HBM 시장의 가격·수요·공급 환경 기준이다. 경쟁사의 가격 인상 계약은 업계 가격 환경의 신호로 읽고, 경쟁 심화로 단정하지 마라.
- 매수/매도 권유는 하지 마라. 숫자를 새로 계산하지 말고 사실에 있는 것만 인용해라.
- LaTeX를 쓰지 마라."""


def synthesize(facts: list[dict], quarters: list[dict]) -> dict:
    micron = _micron_lines(quarters)
    text, model = llm.call(_synthesis_prompt(facts, micron))
    valid = {f["fid"] for f in facts} | {f"M{i}" for i in range(1, len(micron) + 1)}
    bad = [c for c in re.findall(r"\[((?:F|M)\d+)\]", text) if c not in valid]
    for c in set(bad):
        text = text.replace(f"[{c}]", "")                  # 없는 번호는 지운다
    body_lines = [l for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]
    cited = sum(1 for l in body_lines if re.search(r"\[(?:F|M)\d+\]", l))
    return {"text": text.strip(), "model": model, "generated_at": _now().strftime("%Y-%m-%d %H:%M"),
            "fact_ids": [f["fid"] for f in facts], "micron_lines": micron, "invalid_citations_removed": len(bad),
            "cited_line_ratio": round(cited / len(body_lines), 2) if body_lines else 0.0}


# ── AI 분석 프롬프트용 요약 ─────────────────────────────────────────────────

# 주제별로 몇 건씩 넣을지. 프롬프트가 길수록 생성이 급격히 느려져서(CONTEXT §4) 전체를 1,800자 안쪽으로 묶는다.
PROMPT_TOPIC_QUOTA = (("가격", 3), ("수급·생산능력", 2), ("수요", 2), ("점유율·경쟁", 2),
                      ("제품·로드맵", 1), ("고객·퀄", 1), ("리스크", 2), ("실적·마진", 1))
PROMPT_WINDOW_DAYS = 75
_PROMPT_SOURCE = {"micron_remarks": "마이크론", "skhynix_newsroom": "SK하이닉스 뉴스룸(회사 발표)",
                  "trendforce_research": "TrendForce 리서치", "trendforce_news": "언론 보도 인용"}


def summary_md(store: dict | None = None) -> str:
    """AI 분석 프롬프트에 넣을 HBM 재료. 파일을 읽기만 한다.

    종합 분석 글(LLM이 쓴 글)은 넣지 않고 **검증된 사실을 직접** 넣는다 — AI가 쓴 글을 또 AI가
    요약하면 오류가 겹친다. 호재/악재 라벨도 AI의 판단이라 뺐다. 원문 인용은 길어서 뺐고 날짜·출처만 남긴다."""
    store = store if store is not None else load()
    facts = collect_fact_list(store, balanced=False)
    if not facts:
        return ""
    today = _now().date()
    recent = [f for f in facts if (today - dt.date.fromisoformat(f["date"])).days <= PROMPT_WINDOW_DAYS] or facts[:12]
    lines = []
    for topic, n in PROMPT_TOPIC_QUOTA:
        cands = [f for f in recent if f["topic"] == topic]       # collect_fact_list가 이미 최신순
        cands.sort(key=lambda f: f["tier"] == "C")               # 안정 정렬 — 회사·리서치 자료를 보도 인용보다 먼저
        for f in cands[:n]:
            src = _PROMPT_SOURCE[f["source"]]
            lines.append(f"- [{topic}] ({f['date']}, {src}) {f['claim']}")
    quarters = store.get("micron_quarters", [])
    micron = _micron_lines(quarters)[-2:] if quarters else []
    head = (f"(HBM 자료 정리 기준 {store.get('checked_at', '시점 미상')} · 원문 인용이 확인된 사실 {len(facts)}건 중 "
            f"{len(lines)}건 · 가장 최근 사실 {facts[0]['date']})")
    out = [head, *lines]
    if micron:
        out.append("- [마이크론 실적 — SEC 공시 표에서 직접 읽은 숫자] " + " / ".join(micron))
    out.append("※ SK하이닉스 자체의 HBM 점유율·가격·매출은 공개 자료에 없다. 위 사실은 업계·경쟁사·고객 쪽 자료이니 "
               "SK하이닉스에 어떻게 이어지는지는 추론이라고 밝히고, 사실 날짜를 같이 써라.")
    return "\n".join(out)


# ── 수집기 진입점 ───────────────────────────────────────────────────────────

def _pending(store: dict) -> list[tuple[str, dict]]:
    """읽을 문서. 등급이 높은 것(A→B→C) 먼저, 같은 등급 안에서는 최신 먼저."""
    today = _now().strftime("%Y-%m-%d")
    rows = [(u, d) for u, d in store.get("docs", {}).items()
            if d.get("status") == "pending" and d.get("total_attempts", 0) < MAX_TOTAL_ATTEMPTS
            and (d.get("attempt_day") != today or d.get("attempts", 0) < MAX_ATTEMPTS)]
    rows.sort(key=lambda x: x[1]["date"], reverse=True)
    rows.sort(key=lambda x: SOURCE_TIER[x[1]["source"]])          # 안정 정렬 — 등급 안의 최신순 유지
    return rows


def _note_error(store: dict, source: str, exc: Exception | None) -> None:
    errs = store.setdefault("source_status", {})
    errs[source] = ({"ok": True, "at": _now().strftime("%Y-%m-%d %H:%M")} if exc is None else
                    {"ok": False, "at": _now().strftime("%Y-%m-%d %H:%M"),
                     "error": f"{type(exc).__name__}: {str(exc)[:120]}"})


def run_once(log=print) -> dict:
    """한 번 훑어서 새 문서를 등록하고, 읽을 문서를 몇 개 처리하고, 필요하면 종합을 다시 만든다."""
    store = load()
    store.setdefault("docs", {})
    cutoff = (_now().date() - dt.timedelta(days=LOOKBACK_DAYS)).isoformat()

    # 1) 새 문서 등록 (목록만 — 본문은 나중에)
    discovered: list[dict] = []
    first_backfill = not any(d["source"] == "trendforce_research" for d in store["docs"].values())
    for name, fn in (("trendforce_research", lambda: list_trendforce_research(pages=14 if first_backfill else 3)),
                     ("trendforce_news", list_trendforce_news),
                     ("skhynix_newsroom", lambda: list_skhynix_newsroom(
                         backfill=not any(d["source"] == "skhynix_newsroom" for d in store["docs"].values())))):
        try:
            discovered += fn()
            _note_error(store, name, None)
        except Exception as exc:
            _note_error(store, name, exc)
            log(f"[HBM] {name} 목록 실패: {type(exc).__name__}: {exc}")
    last_micron = store.get("micron_checked_at")
    if not last_micron or (_now() - dt.datetime.fromisoformat(last_micron)).total_seconds() > 6 * 3600:
        try:
            quarters, release_date = fetch_micron_quarters()
            if quarters:
                store["micron_quarters"] = quarters
                store["micron_release_dates"] = release_date
            store["micron_checked_at"] = _now().isoformat(timespec="seconds")
            discovered += discover_micron_remarks(store.get("micron_release_dates", {}), store)
            _note_error(store, "micron", None)
        except Exception as exc:
            _note_error(store, "micron", exc)
            log(f"[HBM] 마이크론 확인 실패: {type(exc).__name__}: {exc}")
    try:
        added = refresh_headlines(store)
        _note_error(store, "naver_news", None)
        if added:
            log(f"[HBM] 뉴스 헤드라인 {added}건 추가")
    except Exception as exc:
        _note_error(store, "naver_news", exc)

    new_docs = 0
    for d in discovered:
        if d["date"] < cutoff or d["url"] in store["docs"]:
            continue
        store["docs"][d["url"]] = {**d, "status": "pending", "attempts": 0, "facts": []}
        new_docs += 1

    # 2) 읽을 문서 처리 (등급 높은 것·최신 먼저). 상한은 LLM을 부르는 문서에만 걸고,
    #    HBM 얘기가 아니라 건너뛰는 문서는 네트워크 한 번뿐이라 한 바퀴에 넉넉히 거른다.
    processed = fetched = 0
    for url, d in _pending(store):
        if processed >= MAX_DOCS_PER_TICK or fetched >= MAX_FETCH_PER_TICK:
            break
        fetched += 1
        today_s = _now().strftime("%Y-%m-%d")
        if d.get("attempt_day") != today_s:
            d["attempt_day"], d["attempts"] = today_s, 0
        d["attempts"] = d.get("attempts", 0) + 1
        d["total_attempts"] = d.get("total_attempts", 0) + 1
        try:
            if d["source"] == "micron_remarks":
                body = _pdf_text(url)
                if body is None:                                    # 주소가 틀렸거나 아직 안 올라온 분기
                    store["docs"].pop(url, None)                    # remarks_tried가 일주일간 재시도를 막는다
                    continue
            elif d["source"] == "skhynix_newsroom":
                body, posted = fetch_skhynix_body(url)
                if posted and d.get("date_unsure"):
                    d["date"], d["date_unsure"] = posted, False
            else:
                body = fetch_trendforce_body(url)
            d["hbm_mentions"] = len(re.findall(r"HBM", body, re.I))
            need = {"micron_remarks": 0, "trendforce_research": 1}.get(d["source"], 2)
            if d["hbm_mentions"] < need:
                d["status"] = "skipped"                             # HBM 얘기가 아닌 문서
                continue
            facts, dropped = extract_facts(d, body, log)
            d["facts"], d["dropped"], d["status"] = facts, dropped, "extracted"
            d["extracted_at"] = _now().strftime("%Y-%m-%d %H:%M")
            processed += 1
            log(f"[HBM] {SOURCE_LABEL[d['source']]} {d['date']} 사실 {len(facts)}건 통과"
                + (f" (탈락 {sum(dropped.values())}건: {dropped})" if dropped else ""))
        except Exception as exc:
            log(f"[HBM] 문서 처리 실패({d['attempts']}/{MAX_ATTEMPTS}) {d['title'][:40]}: {type(exc).__name__}: {exc}")
        finally:
            save(store)                                            # 한 건씩 저장 — 중간에 죽어도 진행이 남는다

    # 3) 종합 분석 (사실이 바뀌었을 때만, 최소 간격을 둔다)
    facts = collect_fact_list(store)
    fp = hashlib.sha1("|".join(sorted(f["id"] for f in facts)).encode()).hexdigest()
    synth = store.get("synthesis") or {}
    gap_ok = True
    if synth.get("generated_at"):
        age = _now() - dt.datetime.strptime(synth["generated_at"], "%Y-%m-%d %H:%M").replace(tzinfo=KST)
        gap_ok = age.total_seconds() > SYNTH_MIN_GAP_HOURS * 3600
    backlog = len(_pending(store))
    prev_n = len(synth.get("facts", []))
    # 읽을 문서가 남아 있으면 기다린다(처음 몇 건만으로 종합하면 한쪽 출처로 쏠린다). 첫 종합은 사실이
    # 8건 이상 모였을 때, 이미 종합이 있으면 사실이 30% 넘게 늘었거나 최소 간격이 지났을 때 다시 만든다.
    ready = (len(facts) >= 8 if not synth else (gap_ok or len(facts) >= prev_n * 1.3))
    if facts and fp != synth.get("fingerprint") and ready and (backlog == 0 or not synth and len(facts) >= 30):
        try:
            result = synthesize(facts, store.get("micron_quarters", []))
            result["fingerprint"] = fp
            result["facts"] = facts                                  # 번호(F#)가 나중에 밀리지 않게 그때의 목록을 같이 둔다
            store["synthesis"] = result
            log(f"[HBM] 종합 분석 갱신 (사실 {len(facts)}건, 근거 표기 {result['cited_line_ratio']:.0%}, "
                f"잘못된 번호 {result['invalid_citations_removed']}개 제거)")
        except Exception as exc:
            log(f"[HBM] 종합 분석 실패: {type(exc).__name__}: {exc}")

    store["checked_at"] = _now().strftime("%Y-%m-%d %H:%M")
    save(store)
    return {"new_docs": new_docs, "processed": processed, "pending": backlog}


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


_backlog = False
