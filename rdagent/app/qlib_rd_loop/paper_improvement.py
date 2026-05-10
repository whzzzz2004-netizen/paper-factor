from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rdagent.components.document_reader.document_reader import load_and_process_pdfs_by_langchain
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.qlib.knowledge_router import (
    PAPER_IMPROVEMENT_DROPBOX,
    PAPER_IMPROVEMENT_SUMMARY_PATH,
    ensure_knowledge_v2_dirs,
)


def _load_existing_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning(f"Skip invalid paper improvement record line: {line[:120]}")
    return records


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
    path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")


def _summarize_paper_for_factor_improvement(report_path: Path, report_content: str) -> dict[str, Any]:
    system_prompt = (
        "You are building a factor-improvement knowledge base for quantitative research. "
        "Read the paper content and summarize only reusable knowledge for improving factors. "
        "Do not produce implementation code."
    )
    user_prompt = f"""
Paper path: {report_path}
Paper title: {report_path.stem}

Please extract a compact JSON object with the following fields:
- title: paper title
- summary: 3-5 sentence summary focusing on factor improvement value
- applicable_tags: list of short tags such as momentum, liquidity, volatility, minute_input, microstructure
- improvement_ideas: list of concise actionable ideas for improving factors
- cautions: list of pitfalls, constraints, leakage risks, data requirements, or assumptions
- evidence_snippets: list of 1-3 short supporting snippets or claims paraphrased from the paper

Paper content:
{report_content[:30000]}
"""
    response = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        json_mode=True,
        json_target_type=dict[str, Any],
    )
    summary = json.loads(response)
    summary["source_path"] = str(report_path.resolve())
    summary["source_type"] = "paper_improvement"
    return summary


def ingest_factor_improvement_papers(report_folder: str | None = None) -> int:
    ensure_knowledge_v2_dirs()
    folder = Path(report_folder) if report_folder is not None else PAPER_IMPROVEMENT_DROPBOX
    folder.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(folder.rglob("*.pdf"))
    if not pdf_files:
        logger.info(f"No factor-improvement papers found under {folder}")
        return 0

    existing_records = _load_existing_records(PAPER_IMPROVEMENT_SUMMARY_PATH)
    existing_by_path = {record.get("source_path"): record for record in existing_records if record.get("source_path")}

    updated_count = 0
    for pdf_path in pdf_files:
        logger.info(f"Ingesting factor-improvement paper: {pdf_path}")
        docs_dict = load_and_process_pdfs_by_langchain(str(pdf_path))
        report_content = "\n".join(docs_dict.values()).strip()
        if not report_content:
            logger.warning(f"Skip empty paper content: {pdf_path}")
            continue
        summary = _summarize_paper_for_factor_improvement(pdf_path, report_content)
        existing_by_path[str(pdf_path.resolve())] = summary
        updated_count += 1

    ordered_records = sorted(existing_by_path.values(), key=lambda record: record.get("source_path", ""))
    _write_records(PAPER_IMPROVEMENT_SUMMARY_PATH, ordered_records)
    logger.info(
        f"Ingested {updated_count} factor-improvement paper(s) into {PAPER_IMPROVEMENT_SUMMARY_PATH}"
    )
    return updated_count


def main(report_folder: str | None = None) -> int:
    return ingest_factor_improvement_papers(report_folder=report_folder)
