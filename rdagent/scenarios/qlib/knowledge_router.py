from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


FACTOR_OUTPUT_DIR = Path.cwd() / "git_ignore_folder" / "factor_outputs"
RESEARCH_STORE_DIR = Path.cwd() / "git_ignore_folder" / "research_store"
KNOWLEDGE_V2_DIR = RESEARCH_STORE_DIR / "knowledge_v2"
PAPER_IMPROVEMENT_DROPBOX = Path.cwd() / "papers" / "factor_improvement"

IMPLEMENTATION_EXPERIENCE_KB_PATH = RESEARCH_STORE_DIR / "knowledge" / "costeer_knowledge_base.pkl"
FACTOR_MANIFEST_PATH = FACTOR_OUTPUT_DIR / "manifest.csv"
FACTOR_LEADERBOARD_PATH = FACTOR_OUTPUT_DIR / "leaderboard.csv"
PAPER_IMPROVEMENT_SUMMARY_PATH = KNOWLEDGE_V2_DIR / "paper_improvement" / "paper_improvement_ideas.jsonl"
ERROR_CASES_PATH = KNOWLEDGE_V2_DIR / "error_cases" / "factor_error_cases.jsonl"
FAILED_FACTOR_SUMMARY_PATH = KNOWLEDGE_V2_DIR / "failed_factors" / "failed_factor_ideas.jsonl"


@dataclass(frozen=True)
class KnowledgeSource:
    name: str
    path: Path
    description: str


@dataclass(frozen=True)
class KnowledgeRoute:
    step: str
    purpose: str
    sources: tuple[KnowledgeSource, ...]


def ensure_knowledge_v2_dirs() -> None:
    for path in [
        KNOWLEDGE_V2_DIR / "paper_improvement",
        KNOWLEDGE_V2_DIR / "error_cases",
        KNOWLEDGE_V2_DIR / "failed_factors",
        PAPER_IMPROVEMENT_DROPBOX,
    ]:
        path.mkdir(parents=True, exist_ok=True)


FACTOR_WORKFLOW_ROUTES: tuple[KnowledgeRoute, ...] = (
    KnowledgeRoute(
        step="factor_hypothesis_generation",
        purpose="提出新的因子方向时，优先参考已有强因子表现和因子改进文献摘要，避免重复想法。",
        sources=(
            KnowledgeSource(
                name="factor_leaderboard",
                path=FACTOR_LEADERBOARD_PATH,
                description="已正式入库因子的 IC 排行榜和逻辑摘要。",
            ),
            KnowledgeSource(
                name="factor_manifest",
                path=FACTOR_MANIFEST_PATH,
                description="所有已入库因子的元数据、标签、来源和路径。",
            ),
            KnowledgeSource(
                name="failed_factor_ideas",
                path=FAILED_FACTOR_SUMMARY_PATH,
                description="历史失败因子的简要逻辑与失败原因，用于避免重复踩坑。",
            ),
            KnowledgeSource(
                name="paper_improvement_ideas",
                path=PAPER_IMPROVEMENT_SUMMARY_PATH,
                description="关于如何改进因子的论文摘要、启发和适用标签。",
            ),
        ),
    ),
    KnowledgeRoute(
        step="factor_experiment_expansion",
        purpose="把 hypothesis 展开为具体可实现因子时，参考已有强因子标签和文献摘要，控制方向和粒度。",
        sources=(
            KnowledgeSource(
                name="factor_leaderboard",
                path=FACTOR_LEADERBOARD_PATH,
                description="高 IC 因子及其逻辑摘要，用于避免重复和提供方向参考。",
            ),
            KnowledgeSource(
                name="failed_factor_ideas",
                path=FAILED_FACTOR_SUMMARY_PATH,
                description="失败因子摘要，用于避开低 IC 或实现脆弱的方向。",
            ),
            KnowledgeSource(
                name="paper_improvement_ideas",
                path=PAPER_IMPROVEMENT_SUMMARY_PATH,
                description="文献中的改进策略、可借鉴的计算思路和适用场景。",
            ),
        ),
    ),
    KnowledgeRoute(
        step="factor_coding_and_fixing",
        purpose="实现和修复因子代码时，只参考实现经验和错误案例，不混入论文全文。",
        sources=(
            KnowledgeSource(
                name="implementation_experience",
                path=IMPLEMENTATION_EXPERIENCE_KB_PATH,
                description="CoSTEER 历史成功/失败实现、代码和反馈经验。",
            ),
            KnowledgeSource(
                name="error_cases",
                path=ERROR_CASES_PATH,
                description="结构化错误案例摘要，例如 MultiIndex、rolling、分钟数据、leakage 等。",
            ),
        ),
    ),
    KnowledgeRoute(
        step="factor_selection_and_modeling",
        purpose="选因子、组池、建模时，优先看因子排行榜、标签和来源，不直接看实现细节。",
        sources=(
            KnowledgeSource(
                name="factor_leaderboard",
                path=FACTOR_LEADERBOARD_PATH,
                description="按 IC 排名的候选因子列表。",
            ),
            KnowledgeSource(
                name="factor_manifest",
                path=FACTOR_MANIFEST_PATH,
                description="因子标签、来源、粒度和文件路径。",
            ),
        ),
    ),
)


def iter_routes() -> tuple[KnowledgeRoute, ...]:
    ensure_knowledge_v2_dirs()
    return FACTOR_WORKFLOW_ROUTES


def render_route_map() -> str:
    lines: list[str] = []
    for route in iter_routes():
        lines.append(f"[{route.step}] {route.purpose}")
        for source in route.sources:
            status = "exists" if source.path.exists() else "missing"
            lines.append(f"- {source.name}: {source.path} ({status}) - {source.description}")
        lines.append("")
    return "\n".join(lines).strip()


def _load_jsonl(path: Path) -> list[dict]:
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


def load_top_factor_records(limit: int = 8) -> list[dict]:
    if not FACTOR_LEADERBOARD_PATH.exists():
        return []
    leaderboard = pd.read_csv(FACTOR_LEADERBOARD_PATH)
    if leaderboard.empty:
        return []
    if "accepted" in leaderboard.columns:
        accepted = leaderboard["accepted"]
        if accepted.dtype == object:
            accepted = accepted.astype(str).str.lower().isin({"1", "true", "yes", "y"})
        leaderboard = leaderboard[accepted.fillna(False)]
        if leaderboard.empty:
            return []
    if "ic_score" in leaderboard.columns:
        leaderboard["ic_score"] = pd.to_numeric(leaderboard["ic_score"], errors="coerce")
        leaderboard = leaderboard.sort_values("ic_score", ascending=False, na_position="last")
    result: list[dict] = []
    for _, row in leaderboard.head(limit).iterrows():
        result.append(
            {
                "factor_name": row.get("factor_name"),
                "ic_score": row.get("ic_score"),
                "logic_summary": row.get("logic_summary"),
                "tags": row.get("tags"),
                "source_type": row.get("source_type"),
            }
        )
    return result


def load_paper_improvement_records(limit: int = 6) -> list[dict]:
    records = _load_jsonl(PAPER_IMPROVEMENT_SUMMARY_PATH)
    return records[:limit]


def load_failed_factor_records(limit: int = 6) -> list[dict]:
    records = _load_jsonl(FAILED_FACTOR_SUMMARY_PATH)
    if not records:
        return []
    records = sorted(records, key=lambda x: x.get("updated_at", ""), reverse=True)
    return records[:limit]


def upsert_failed_factor_record(record: dict) -> None:
    ensure_knowledge_v2_dirs()
    existing = _load_jsonl(FAILED_FACTOR_SUMMARY_PATH)
    factor_name = record.get("factor_name")
    if factor_name:
        existing = [item for item in existing if item.get("factor_name") != factor_name]
    existing.append(record)
    FAILED_FACTOR_SUMMARY_PATH.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in existing) + "\n",
        encoding="utf-8",
    )


def _format_records(title: str, records: Iterable[dict], keys: list[str]) -> str:
    records = list(records)
    if not records:
        return f"{title}: none"
    lines = [f"{title}:"]
    for idx, record in enumerate(records, start=1):
        parts = []
        for key in keys:
            value = record.get(key)
            if value in (None, "", [], {}):
                continue
            parts.append(f"{key}={value}")
        if parts:
            lines.append(f"{idx}. " + "; ".join(parts))
    return "\n".join(lines)


def build_factor_generation_knowledge_brief(
    leaderboard_limit: int = 8,
    failed_limit: int = 6,
    paper_limit: int = 5,
) -> str:
    ensure_knowledge_v2_dirs()
    sections = [
        _format_records(
            "Accepted factors to avoid duplicating and to learn from",
            load_top_factor_records(limit=leaderboard_limit),
            ["factor_name", "ic_score", "logic_summary", "tags", "source_type"],
        ),
        _format_records(
            "Failed factor directions to avoid repeating blindly",
            load_failed_factor_records(limit=failed_limit),
            ["factor_name", "failure_reason", "ic_score", "logic_summary", "tags"],
        ),
        _format_records(
            "Paper-derived improvement ideas",
            load_paper_improvement_records(limit=paper_limit),
            ["title", "summary", "applicable_tags", "improvement_ideas"],
        ),
    ]
    return "\n\n".join(sections)


def build_factor_generation_knowledge_summary(
    leaderboard_limit: int = 3,
    failed_limit: int = 4,
    paper_limit: int = 2,
    improvement_ideas_per_paper: int = 2,
) -> str:
    """Return a shorter knowledge snippet for proposal prompts.

    The full route map and full paper summaries are useful for humans, but they
    make factor-generation prompts much longer than necessary.
    """
    ensure_knowledge_v2_dirs()

    top_factors = load_top_factor_records(limit=leaderboard_limit)
    factor_lines = ["Accepted factor examples to avoid duplicating:"]
    if not top_factors:
        factor_lines.append("- none")
    else:
        for record in top_factors:
            parts = [str(record.get("factor_name"))]
            ic_score = record.get("ic_score")
            if pd.notna(ic_score):
                parts.append(f"IC={float(ic_score):.4f}")
            logic_summary = record.get("logic_summary")
            if isinstance(logic_summary, str) and logic_summary:
                parts.append(logic_summary[:160])
            factor_lines.append("- " + " | ".join(parts))

    failed_records = load_failed_factor_records(limit=failed_limit)
    failed_lines = ["Failed factor directions to avoid repeating blindly:"]
    if not failed_records:
        failed_lines.append("- none")
    else:
        for record in failed_records:
            parts = [str(record.get("factor_name"))]
            failure_reason = record.get("failure_reason")
            if failure_reason:
                parts.append(f"reason={failure_reason}")
            ic_score = record.get("ic_score")
            if ic_score not in (None, "") and pd.notna(ic_score):
                parts.append(f"IC={float(ic_score):.4f}")
            logic_summary = record.get("logic_summary")
            if isinstance(logic_summary, str) and logic_summary:
                parts.append(logic_summary[:160])
            failed_lines.append("- " + " | ".join(parts))

    paper_records = load_paper_improvement_records(limit=paper_limit)
    paper_lines = ["Factor-improvement paper ideas you may borrow:"]
    if not paper_records:
        paper_lines.append("- none")
    else:
        for record in paper_records:
            title = record.get("title") or "untitled"
            tags = record.get("applicable_tags") or []
            ideas = record.get("improvement_ideas") or []
            trimmed_ideas = ideas[:improvement_ideas_per_paper]
            summary = record.get("summary") or ""
            line = f"- {title}"
            if tags:
                line += f" | tags={tags}"
            if summary:
                line += f" | {summary[:160]}"
            paper_lines.append(line)
            for idea in trimmed_ideas:
                paper_lines.append(f"  idea: {idea}")

    return "\n".join(factor_lines + [""] + failed_lines + [""] + paper_lines)
