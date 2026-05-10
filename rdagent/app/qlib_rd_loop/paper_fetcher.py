from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode

import requests
from requests import RequestException

from rdagent.log import rdagent_logger as logger


ARXIV_API_URL = "http://export.arxiv.org/api/query"
DEFAULT_ARXIV_QUERY = (
    "(cat:q-fin.ST OR cat:q-fin.PM OR cat:q-fin.TR) AND "
    "(all:factor OR all:alpha OR all:predictor OR all:signal OR all:\"return prediction\" "
    "OR all:\"cross-sectional return\")"
)
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
DEFAULT_MANIFEST_NAME = "_arxiv_manifest.jsonl"
POSITIVE_SUMMARY_PATTERNS = (
    "factor",
    "alpha",
    "signal",
    "predictor",
    "cross-sectional",
    "return prediction",
    "stock return",
    "asset pricing",
    "characteristic",
    "anomaly",
    "momentum",
    "reversal",
    "liquidity",
    "volatility",
    "microstructure",
)
NEGATIVE_SUMMARY_PATTERNS = (
    "portfolio optimization",
    "option pricing",
    "derivative pricing",
    "risk management",
    "execution strategy",
    "order execution",
    "hedging",
    "market making",
    "cryptocurrency trading",
    "reinforcement learning for trading",
    "llm trading agent",
)


@dataclass(frozen=True)
class ArxivPaper:
    arxiv_id: str
    title: str
    summary: str
    published: str
    updated: str
    pdf_url: str
    entry_id: str
    authors: tuple[str, ...]


def _looks_like_factor_mining_paper(paper: ArxivPaper) -> bool:
    text = f"{paper.title} {paper.summary}".lower()
    positive_hits = sum(1 for pattern in POSITIVE_SUMMARY_PATTERNS if pattern in text)
    negative_hits = sum(1 for pattern in NEGATIVE_SUMMARY_PATTERNS if pattern in text)
    if positive_hits == 0:
        return False
    if negative_hits >= 2 and positive_hits < 2:
        return False
    if "factor" in text or "alpha" in text or "cross-sectional" in text:
        return True
    return positive_hits >= 2 and negative_hits == 0


def _build_arxiv_query(base_query: str, days_back: int | None = None) -> str:
    query = base_query.strip() or DEFAULT_ARXIV_QUERY
    if days_back is None or days_back <= 0:
        return query
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days_back)
    start_str = start_dt.strftime("%Y%m%d%H%M")
    end_str = end_dt.strftime("%Y%m%d%H%M")
    return f"({query}) AND submittedDate:[{start_str} TO {end_str}]"


def _sanitize_filename(text: str, max_len: int = 120) -> str:
    clean = re.sub(r"\s+", "_", text.strip())
    clean = re.sub(r"[^A-Za-z0-9._-]+", "", clean)
    clean = clean.strip("._-")
    if not clean:
        clean = "paper"
    return clean[:max_len]


def _manifest_path(target_dir: Path, manifest_path: str | Path | None = None) -> Path:
    if manifest_path is not None:
        return Path(manifest_path)
    return target_dir / DEFAULT_MANIFEST_NAME


def _load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _write_manifest(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(item, ensure_ascii=False) for item in records)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def fetch_latest_arxiv_papers(
    *,
    query: str = DEFAULT_ARXIV_QUERY,
    max_results: int = 20,
    days_back: int | None = 30,
    timeout: int = 30,
) -> list[ArxivPaper]:
    search_query = _build_arxiv_query(query, days_back=days_back)
    params = {
        "search_query": search_query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "start": 0,
        "max_results": max_results,
    }
    url = f"{ARXIV_API_URL}?{urlencode(params)}"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    papers: list[ArxivPaper] = []
    for entry in root.findall("atom:entry", ARXIV_NS):
        entry_id = (entry.findtext("atom:id", default="", namespaces=ARXIV_NS) or "").strip()
        title = " ".join((entry.findtext("atom:title", default="", namespaces=ARXIV_NS) or "").split())
        summary = " ".join((entry.findtext("atom:summary", default="", namespaces=ARXIV_NS) or "").split())
        published = (entry.findtext("atom:published", default="", namespaces=ARXIV_NS) or "").strip()
        updated = (entry.findtext("atom:updated", default="", namespaces=ARXIV_NS) or "").strip()
        authors = tuple(
            author_name.strip()
            for author_name in (
                author.findtext("atom:name", default="", namespaces=ARXIV_NS)
                for author in entry.findall("atom:author", ARXIV_NS)
            )
            if author_name.strip()
        )
        pdf_url = ""
        for link in entry.findall("atom:link", ARXIV_NS):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "").strip()
                break
        if not pdf_url and entry_id:
            pdf_url = entry_id.replace("/abs/", "/pdf/") + ".pdf"
        if not entry_id:
            continue
        arxiv_id = entry_id.rstrip("/").split("/")[-1]
        papers.append(
            ArxivPaper(
                arxiv_id=arxiv_id,
                title=title,
                summary=summary,
                published=published,
                updated=updated,
                pdf_url=pdf_url,
                entry_id=entry_id,
                authors=authors,
            )
        )
    return papers


def sync_latest_factor_papers(
    *,
    target_dir: str | Path,
    query: str = DEFAULT_ARXIV_QUERY,
    max_results: int = 20,
    download_limit: int | None = None,
    days_back: int | None = 30,
    manifest_path: str | Path | None = None,
    timeout: int = 60,
) -> dict:
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest = _manifest_path(target, manifest_path)
    existing_records = _load_manifest(manifest)
    existing_ids = {record.get("arxiv_id") for record in existing_records if record.get("arxiv_id")}

    papers = fetch_latest_arxiv_papers(
        query=query,
        max_results=max_results,
        days_back=days_back,
        timeout=timeout,
    )
    papers = [paper for paper in papers if _looks_like_factor_mining_paper(paper)]

    downloaded: list[dict] = []
    failed_downloads: list[dict] = []
    seen_records = {record.get("arxiv_id"): record for record in existing_records if record.get("arxiv_id")}
    remaining_downloads = download_limit if download_limit is not None and download_limit >= 0 else None

    for paper in papers:
        if paper.arxiv_id in existing_ids:
            continue
        if remaining_downloads is not None and remaining_downloads <= 0:
            break
        published_prefix = paper.published[:10].replace("-", "") if paper.published else "unknown"
        filename = f"{published_prefix}_{paper.arxiv_id}_{_sanitize_filename(paper.title)}.pdf"
        pdf_path = target / filename
        try:
            response = requests.get(paper.pdf_url, timeout=timeout)
            response.raise_for_status()
            pdf_path.write_bytes(response.content)
        except RequestException as exc:
            logger.warning(f"Failed to download arXiv paper {paper.arxiv_id} from {paper.pdf_url}: {exc}")
            failed_downloads.append(
                {
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "pdf_url": paper.pdf_url,
                    "error": str(exc),
                }
            )
            continue
        record = {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "summary": paper.summary,
            "published": paper.published,
            "updated": paper.updated,
            "pdf_url": paper.pdf_url,
            "entry_id": paper.entry_id,
            "authors": list(paper.authors),
            "local_pdf_path": str(pdf_path.resolve()),
            "synced_at": datetime.now().isoformat(timespec="seconds"),
            "source_type": "arxiv",
            "query": query,
        }
        seen_records[paper.arxiv_id] = record
        downloaded.append(record)
        if remaining_downloads is not None:
            remaining_downloads -= 1

    ordered_records = sorted(
        seen_records.values(),
        key=lambda item: (item.get("published", ""), item.get("arxiv_id", "")),
        reverse=True,
    )
    _write_manifest(manifest, ordered_records)
    logger.info(
        f"Synced {len(downloaded)} new arXiv factor paper(s) into {target}. Failed downloads: {len(failed_downloads)}. "
        f"Query={query!r}, max_results={max_results}, days_back={days_back}"
    )
    return {
        "downloaded_count": len(downloaded),
        "failed_count": len(failed_downloads),
        "failed_downloads": failed_downloads,
        "manifest_path": str(manifest.resolve()),
        "downloaded_paths": [record["local_pdf_path"] for record in downloaded],
        "query": query,
        "target_dir": str(target.resolve()),
    }
