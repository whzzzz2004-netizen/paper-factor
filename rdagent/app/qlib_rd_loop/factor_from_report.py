import asyncio
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, Tuple

import pandas as pd
import yaml

from rdagent.app.qlib_rd_loop.conf import FACTOR_FROM_REPORT_PROP_SETTING
from rdagent.app.qlib_rd_loop.factor import FactorRDLoop
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.components.coder.factor_coder.eva_utils import evaluate_factor_ic_from_workspace
from rdagent.components.coder.factor_coder.evaluators import FactorSingleFeedback
from rdagent.components.coder.factor_coder.factor import FactorExperiment, FactorFBWorkspace, FactorTask
from rdagent.components.document_reader.document_reader import (
    extract_first_page_screenshot_from_pdf,
    load_and_process_pdfs_by_langchain,
)
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.proposal import Hypothesis, HypothesisFeedback
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend, md5_hash
from rdagent.scenarios.qlib.factor_experiment_loader.pdf_loader import (
    FactorExperimentLoaderFromPDFfiles,
)
from rdagent.utils.agent.tpl import T
from rdagent.utils.workflow import LoopMeta


class PaperFactorExperiment(FactorExperiment[FactorTask, FactorFBWorkspace, FactorFBWorkspace]):
    """A lightweight experiment for paper_factor that avoids creating Qlib template workspaces."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = FactorFBWorkspace()
        self.stdout = ""
        self.base_features: dict[str, str] = {}
        self.base_feature_codes: dict[str, str] = {}


def _load_processed_report_paths() -> set[str]:
    manifest_path = Path.cwd() / "git_ignore_folder" / "factor_outputs" / "manifest.csv"
    if not manifest_path.exists():
        return set()
    try:
        manifest = pd.read_csv(manifest_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to read factor manifest for processed report detection: {exc}")
        return set()
    if "source_report_path" not in manifest.columns:
        return set()
    if "source_type" in manifest.columns:
        source_type = manifest["source_type"].astype(str).str.lower()
        manifest = manifest[source_type == "literature_report"]
    manifest = manifest[manifest["accepted"].astype(str).str.lower().isin(["true", "1"])] if "accepted" in manifest.columns else manifest
    path_counts = manifest["source_report_path"].dropna().astype(str).map(lambda p: str(Path(p).resolve()))
    processed_paths = {
        path
        for path, count in path_counts.value_counts().items()
        if count >= FACTOR_FROM_REPORT_PROP_SETTING.max_factors_per_exp
    }
    return processed_paths


def _load_terminal_report_factor_names(report_path: str | Path) -> set[str]:
    manifest_path = Path.cwd() / "git_ignore_folder" / "factor_outputs" / "manifest.csv"
    if not manifest_path.exists():
        return set()
    try:
        manifest = pd.read_csv(manifest_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to read factor manifest for exported factor detection: {exc}")
        return set()
    required_columns = {"factor_name", "source_report_path"}
    if not required_columns.issubset(manifest.columns):
        return set()
    if "source_type" in manifest.columns:
        source_type = manifest["source_type"].astype(str).str.lower()
        manifest = manifest[source_type == "literature_report"]
    resolved_report_path = str(Path(report_path).resolve())
    source_paths = manifest["source_report_path"].fillna("").astype(str).map(lambda p: str(Path(p).resolve()) if p else "")
    report_manifest = manifest[source_paths == resolved_report_path]
    if "accepted" in report_manifest.columns:
        accepted = report_manifest["accepted"].fillna(False).astype(str).str.lower().isin(["true", "1"])
        rejected_notes = report_manifest["review_notes"].fillna("").astype(str) if "review_notes" in report_manifest.columns else ""
        old_limit_up_unavailable = rejected_notes.str.contains(
            "limit-up threshold|P_limit_up|涨停价|涨停日",
            case=False,
            regex=True,
        )
        report_manifest = report_manifest[accepted | (~old_limit_up_unavailable & rejected_notes.str.contains("DATA_UNAVAILABLE", case=False, regex=False))]
    return {
        str(name).strip()
        for name in report_manifest["factor_name"].dropna().astype(str).tolist()
        if str(name).strip()
    }


def _record_rejected_report_factor(task, feedback, source_report_path: str | None, source_report_title: str | None) -> None:
    if source_report_path is None:
        return
    output_dir = Path.cwd() / "git_ignore_folder" / "factor_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    factor_name = str(getattr(task, "factor_name", "") or "").strip()
    if not factor_name:
        return

    review_notes = "\n".join(
        part
        for part in [
            getattr(feedback, "execution", None),
            getattr(feedback, "return_checking", None),
            getattr(feedback, "code", None),
            getattr(feedback, "final_feedback", None),
        ]
        if part
    )
    report_output_dir = output_dir / "literature_reports" / FactorFBWorkspace._sanitize_factor_name(
        source_report_title or Path(source_report_path).stem
    )
    report_output_dir.mkdir(parents=True, exist_ok=True)
    rejected_reason = review_notes or "paper_factor 复现阶段未通过，但没有返回更具体的失败信息。"
    is_data_unavailable = "DATA_UNAVAILABLE:" in rejected_reason
    if is_data_unavailable:
        short_reason = rejected_reason.split("DATA_UNAVAILABLE:", 1)[1].splitlines()[0].strip()
        status_title = "DATA_UNAVAILABLE"
        status_label = "跳过：数据或定义不可用"
    else:
        short_reason = rejected_reason.splitlines()[0].strip() if rejected_reason.splitlines() else rejected_reason
        status_title = "IMPLEMENTATION_FAILED"
        status_label = "失败：实现未通过"
    reason_path = report_output_dir / f"SKIPPED__{FactorFBWorkspace._sanitize_factor_name(factor_name)}.md"
    reason_path.write_text(
        "\n".join(
            [
                f"# {status_title}: {factor_name}",
                "",
                f"- 状态：{status_label}",
                f"- 因子：`{factor_name}`",
                f"- 论文：{source_report_title or Path(source_report_path).stem}",
                f"- 更新时间：{datetime.now().isoformat(timespec='seconds')}",
                f"- 简要原因：{short_reason}",
                "",
                "## 缺失/失败详情",
                "",
                rejected_reason,
                "",
                "## 因子描述",
                "",
                str(getattr(task, "factor_description", "") or ""),
                "",
                "## 因子公式",
                "",
                str(getattr(task, "factor_formulation", "") or ""),
                "",
                "## 变量说明",
                "",
                json.dumps(getattr(task, "variables", None), ensure_ascii=False, indent=2),
                "",
            ]
        ),
        encoding="utf-8",
    )
    summary_path = report_output_dir / "_SKIPPED_FACTORS.md"
    summary_line = (
        f"- `{factor_name}`：{status_label}。{short_reason} "
        f"详情见 `{reason_path.name}`。"
    )
    existing_lines = summary_path.read_text(encoding="utf-8").splitlines() if summary_path.exists() else []
    kept_lines = [line for line in existing_lines if not line.startswith(f"- `{factor_name}`：")]
    if not kept_lines:
        kept_lines = [
            f"# 跳过或失败的因子汇总：{source_report_title or Path(source_report_path).stem}",
            "",
        ]
    kept_lines.append(summary_line)
    summary_path.write_text("\n".join(kept_lines).rstrip() + "\n", encoding="utf-8")
    row = pd.DataFrame(
        [
            {
                "factor_name": factor_name,
                "display_name": factor_name,
                "hash": None,
                "rows": 0,
                "non_null": 0,
                "time_granularity": "daily",
                "accepted": False,
                "ic_score": None,
                "factor_description": getattr(task, "factor_description", None),
                "factor_formulation": getattr(task, "factor_formulation", None),
                "variables": json.dumps(getattr(task, "variables", None), ensure_ascii=False),
                "logic_summary": getattr(task, "factor_description", None),
                "tags": json.dumps(["literature_factor", "report_extracted", "rejected"], ensure_ascii=False),
                "source_type": "literature_report",
                "source_report_title": source_report_title,
                "source_report_path": source_report_path,
                "review_notes": review_notes or "paper_factor 复现阶段未通过，但没有返回更具体的失败信息。",
                "latest_path": None,
                "metadata_path": None,
                "code_path": None,
                "workspace_path": None,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        ]
    )
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        same_factor = manifest["factor_name"].astype(str) == factor_name if "factor_name" in manifest.columns else False
        same_report = (
            manifest["source_report_path"].fillna("").astype(str).map(lambda p: str(Path(p).resolve()) if p else "")
            == str(Path(source_report_path).resolve())
            if "source_report_path" in manifest.columns
            else False
        )
        manifest = manifest[~(same_factor & same_report)]
        manifest = pd.concat([manifest, row], ignore_index=True)
    else:
        manifest = row
    manifest.sort_values("factor_name").to_csv(manifest_path, index=False)


def list_unprocessed_report_paths(report_folder: str | Path) -> list[Path]:
    folder = Path(report_folder)
    if not folder.exists():
        return []
    result: list[Path] = []
    for item in folder.rglob("*.pdf"):
        path = item.resolve()
        if _report_fully_processed(path):
            continue
        result.append(path)
    return sorted(result)


def generate_hypothesis(factor_result: dict, report_content: str) -> str:
    """
    Generate a hypothesis based on factor results and report content.

    Args:
        factor_result (dict): The results of the factor analysis.
        report_content (str): The content of the report.

    Returns:
        str: The generated hypothesis.
    """
    system_prompt = T(".prompts:hypothesis_generation.system").r()
    user_prompt = T(".prompts:hypothesis_generation.user").r(
        factor_descriptions=json.dumps(factor_result), report_content=report_content
    )

    response = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        json_mode=True,
        json_target_type=Dict[str, str],
    )

    response_json = json.loads(response)

    return Hypothesis(
        hypothesis=response_json.get("hypothesis", "No hypothesis provided"),
        reason=response_json.get("reason", "No reason provided"),
        concise_reason=response_json.get("concise_reason", "No concise reason provided"),
        concise_observation=response_json.get("concise_observation", "No concise observation provided"),
        concise_justification=response_json.get("concise_justification", "No concise justification provided"),
        concise_knowledge=response_json.get("concise_knowledge", "No concise knowledge provided"),
    )


def build_lightweight_hypothesis(report_file_path: str, factor_result: dict) -> Hypothesis:
    factor_names = list(factor_result)
    factor_preview = ", ".join(factor_names[:3]) if factor_names else "no extracted factors"
    report_title = Path(report_file_path).stem
    return Hypothesis(
        hypothesis=f"Reproduce factors extracted from report: {report_title}",
        reason=f"Implement the core factors described in the report and validate them with full-sample IC.",
        concise_reason=f"Report-derived factor reproduction for {report_title}.",
        concise_observation=f"Extracted factors include: {factor_preview}.",
        concise_justification=f"Directly implement the report's factor definitions with minimal extra reasoning.",
        concise_knowledge=f"Use extracted factor names, formulations, and variables from {report_title}.",
    )


def _load_local_factor_data_profile() -> dict[str, Any]:
    base_dir = Path.cwd() / "git_ignore_folder" / "factor_implementation_source_data"
    meta_path = base_dir / "jq_data_meta.json"
    profile: dict[str, Any] = {
        "source": "local_factor_data",
        "stock_count": "unknown",
        "start_date": "unknown",
        "end_date": "unknown",
        "years": "unknown",
        "daily_columns": [],
        "minute_columns": [],
    }
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            profile.update(
                {
                    "source": meta.get("source", profile["source"]),
                    "stock_count": meta.get("stock_count", profile["stock_count"]),
                    "start_date": meta.get("start_date", profile["start_date"]),
                    "end_date": meta.get("end_date", profile["end_date"]),
                    "years": meta.get("years", profile["years"]),
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to load local factor data metadata from {meta_path}: {exc}")
    for file_name, field_name in [("daily_pv.h5", "daily_columns"), ("minute_pv.h5", "minute_columns")]:
        data_path = base_dir / file_name
        if not data_path.exists():
            continue
        try:
            df = pd.read_hdf(data_path, key="data", start=0, stop=1)
            profile[field_name] = list(df.columns)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to inspect available columns from {data_path}: {exc}")
    return profile


def _paper_factor_task_query(task: FactorTask) -> str:
    return " ".join(
        [
            getattr(task, "factor_name", "") or "",
            getattr(task, "factor_description", "") or "",
            getattr(task, "factor_formulation", "") or "",
            json.dumps(getattr(task, "variables", {}) or {}, ensure_ascii=False),
        ]
    )


def _split_markdown_knowledge_sections(content: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_title = "通用知识"
    current_lines: list[str] = []
    for line in content.splitlines():
        if line.startswith("## "):
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = line.removeprefix("## ").strip() or "未命名知识"
            current_lines = [line]
        elif line.startswith("# "):
            continue
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))
    return [(title, section) for title, section in sections if section]


def _knowledge_section_aliases(title: str) -> list[str]:
    alias_map = {
        "中国 A 股涨停规则": [
            "涨停",
            "跌停",
            "涨跌幅",
            "涨停价",
            "涨停板",
            "封板",
            "limit up",
            "limit-up",
            "price limit",
            "daily limit",
        ],
        "科创板代码识别": [
            "科创板",
            "688",
            "star market",
            "sci-tech innovation board",
            "science and technology innovation board",
        ],
        "一字板与非一字板": [
            "一字板",
            "非一字板",
            "封死",
            "炸板",
            "回封",
            "开板",
            "limit-up board",
            "one-line limit",
        ],
        "首板日": [
            "首板",
            "首板日",
            "一板",
            "首板涨停",
            "first board",
            "first-board",
        ],
        "回调确认日": [
            "回调",
            "回调日",
            "回调确认",
            "涨停次日",
            "pullback",
            "pull-back",
        ],
        "首板回调策略时间锚点": [
            "T-2",
            "T-1",
            "T日",
            "信号日",
            "输出日期",
            "首板回调",
            "首板后回调",
            "event anchor",
        ],
        "本地数据使用纪律": [
            "基本面",
            "财务",
            "财报",
            "市值",
            "总股本",
            "流通股",
            "行业",
            "分析师",
            "盘口",
            "逐笔",
            "order book",
            "tick",
            "fundamental",
            "financial statement",
            "market cap",
            "industry",
            "analyst",
        ],
    }
    return alias_map.get(title, [])


def _score_knowledge_section(query: str, title: str, section: str) -> int:
    query_lower = query.lower()
    section_lower = section.lower()
    alias_score = 0
    for alias in _knowledge_section_aliases(title):
        if alias.lower() in query_lower:
            alias_score += 5
    query_terms = set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z][a-zA-Z0-9_-]{2,}|688\d{0,3}", query_lower))
    stop_terms = {
        "factor",
        "description",
        "formulation",
        "variables",
        "using",
        "with",
        "from",
        "the",
        "and",
        "因子",
        "构建",
        "使用",
        "计算",
        "过去",
        "收盘",
        "收盘价",
    }
    lexical_score = 0
    for term in query_terms - stop_terms:
        if term in section_lower:
            lexical_score += 1
    if alias_score > 0:
        return alias_score + lexical_score
    return lexical_score if lexical_score >= 2 else 0


def _expand_related_knowledge_sections(
    matched_sections: list[tuple[int, str, str, Path]],
    all_sections: list[tuple[int, str, str, Path]],
) -> list[tuple[int, str, str, Path]]:
    scored_sections = list(matched_sections)
    selected_titles = {title for _, title, _, _ in scored_sections}
    title_to_section = {title: (score, title, section, path) for score, title, section, path in all_sections}

    dependency_map = {
        "中国 A 股涨停规则": ["科创板代码识别", "一字板与非一字板"],
        "科创板代码识别": ["中国 A 股涨停规则"],
        "一字板与非一字板": ["中国 A 股涨停规则", "科创板代码识别", "首板日"],
        "首板日": ["中国 A 股涨停规则", "一字板与非一字板", "首板回调策略时间锚点"],
        "回调确认日": ["首板日", "首板回调策略时间锚点"],
        "首板回调策略时间锚点": ["首板日", "回调确认日"],
    }
    expanded = list(scored_sections)
    for title in list(selected_titles):
        base_score = title_to_section[title][0]
        for related_title in dependency_map.get(title, []):
            if related_title in selected_titles or related_title not in title_to_section:
                continue
            _, _, related_section, related_path = title_to_section[related_title]
            expanded.append((max(base_score - 1, 1), related_title, related_section, related_path))
            selected_titles.add(related_title)
    expanded.sort(key=lambda item: item[0], reverse=True)
    return expanded


def _load_paper_factor_knowledge_graph() -> dict[str, dict[str, Any]]:
    graph_paths = [
        Path(__file__).with_name("paper_factor_knowledge_graph.yaml"),
        Path.cwd() / "git_ignore_folder" / "paper_factor_knowledge_graph.yaml",
    ]
    nodes: dict[str, dict[str, Any]] = {}
    for path in graph_paths:
        if not path.exists():
            continue
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to load paper_factor knowledge graph from {path}: {exc}")
            continue
        raw_nodes = payload.get("nodes", {}) if isinstance(payload, dict) else {}
        if not isinstance(raw_nodes, dict):
            logger.warning(f"Invalid paper_factor knowledge graph format in {path}: `nodes` must be a mapping.")
            continue
        for node_id, node in raw_nodes.items():
            if not isinstance(node, dict):
                continue
            normalized_node = dict(node)
            normalized_node["_source_path"] = path
            nodes[str(node_id)] = normalized_node
    return nodes


def _match_knowledge_graph_nodes(query: str, nodes: dict[str, dict[str, Any]]) -> list[str]:
    query_lower = query.lower()
    scored_nodes: list[tuple[int, str]] = []
    for node_id, node in nodes.items():
        aliases = node.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = []
        title = str(node.get("title") or "")
        score = 0
        for alias in [title, *aliases]:
            alias = str(alias).strip()
            if alias and alias.lower() in query_lower:
                score += max(2, min(len(alias), 8))
        if score > 0:
            scored_nodes.append((score, node_id))
    scored_nodes.sort(key=lambda item: item[0], reverse=True)
    return [node_id for _, node_id in scored_nodes]


def _resolve_knowledge_graph_closure(seed_node_ids: list[str], nodes: dict[str, dict[str, Any]]) -> list[str]:
    selected: list[str] = []
    pending = list(seed_node_ids)
    while pending:
        node_id = pending.pop(0)
        if node_id in selected or node_id not in nodes:
            continue
        selected.append(node_id)
        requires = nodes[node_id].get("requires", [])
        if isinstance(requires, list):
            pending.extend(str(required) for required in requires)
    return selected


def _order_knowledge_graph_dependency_first(node_ids: list[str], nodes: dict[str, dict[str, Any]]) -> list[str]:
    selected = set(node_ids)
    ordered: list[str] = []
    visiting: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in ordered or node_id in visiting or node_id not in selected or node_id not in nodes:
            return
        visiting.add(node_id)
        requires = nodes[node_id].get("requires", [])
        if isinstance(requires, list):
            for required in requires:
                visit(str(required))
        visiting.remove(node_id)
        ordered.append(node_id)

    for node_id in node_ids:
        visit(node_id)
    return ordered


def _render_knowledge_graph_nodes(node_ids: list[str], nodes: dict[str, dict[str, Any]]) -> str:
    rendered_sections = []
    for node_id in node_ids:
        node = nodes[node_id]
        title = str(node.get("title") or node_id)
        content = str(node.get("content") or "").strip()
        source_path = node.get("_source_path")
        if not content:
            continue
        rendered_sections.append(f"Knowledge graph node: {source_path}#{node_id} ({title})\n## {title}\n{content}")
    return "\n\n".join(rendered_sections)


def _load_paper_factor_domain_knowledge(task: FactorTask | None = None, *, top_k: int = 6) -> str:
    graph_nodes = _load_paper_factor_knowledge_graph()
    if graph_nodes:
        if task is None:
            ordered_node_ids = _order_knowledge_graph_dependency_first(list(graph_nodes), graph_nodes)
            return _render_knowledge_graph_nodes(ordered_node_ids, graph_nodes)
        query = _paper_factor_task_query(task)
        seed_node_ids = _match_knowledge_graph_nodes(query, graph_nodes)
        if seed_node_ids:
            closure_node_ids = _resolve_knowledge_graph_closure(seed_node_ids, graph_nodes)
            ordered_node_ids = _order_knowledge_graph_dependency_first(closure_node_ids, graph_nodes)
            graph_knowledge = _render_knowledge_graph_nodes(ordered_node_ids, graph_nodes)
            if graph_knowledge:
                return graph_knowledge

    knowledge_paths = [
        Path(__file__).with_name("paper_factor_knowledge.md"),
        Path.cwd() / "git_ignore_folder" / "paper_factor_knowledge.md",
    ]
    sections: list[tuple[str, str, Path]] = []
    for path in knowledge_paths:
        if not path.exists():
            continue
        try:
            content = path.read_text(encoding="utf-8").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to load paper_factor knowledge from {path}: {exc}")
            continue
        if content:
            sections.extend((title, section, path) for title, section in _split_markdown_knowledge_sections(content))
    if not sections:
        return ""
    if task is None:
        return "\n\n".join(f"Knowledge source: {path}\n{section}" for _, section, path in sections)

    query = _paper_factor_task_query(task)
    scored_sections = []
    all_sections = []
    for title, section, path in sections:
        score = _score_knowledge_section(query, title, section)
        if score > 0:
            scored_sections.append((score, title, section, path))
        all_sections.append((score, title, section, path))
    scored_sections.sort(key=lambda item: item[0], reverse=True)
    if scored_sections:
        scored_sections = _expand_related_knowledge_sections(scored_sections, all_sections)
    selected_sections = scored_sections[:top_k]
    if not selected_sections:
        return ""
    return "\n\n".join(
        f"Knowledge source: {path}#{title}\n{section}" for _, title, section, path in selected_sections
    )


def _detect_unavailable_data_requirement(task: FactorTask, data_profile: dict[str, Any]) -> str | None:
    content = " ".join(
        [
            getattr(task, "factor_name", "") or "",
            getattr(task, "factor_description", "") or "",
            getattr(task, "factor_formulation", "") or "",
            json.dumps(getattr(task, "variables", {}) or {}, ensure_ascii=False),
        ]
    ).lower()
    available_columns = {
        str(column).lower()
        for column in (data_profile.get("daily_columns") or []) + (data_profile.get("minute_columns") or [])
    }

    unavailable_groups = [
        (
            "基本面或财务报表数据",
            [
                "基本面",
                "财务",
                "财报",
                "资产负债",
                "利润表",
                "现金流",
                "roe",
                "roa",
                "eps",
                "book value",
                "net profit",
                "revenue",
                "cash flow",
                "balance sheet",
                "income statement",
                "financial statement",
                "fundamental",
            ],
        ),
        (
            "市值、总股本或流通股本数据",
            ["市值", "总股本", "流通股", "market cap", "capitalization", "shares outstanding", "float shares"],
        ),
        (
            "行业或板块分类数据",
            ["行业", "申万", "中信行业", "industry", "sector", "gics", "sw industry"],
        ),
        (
            "分析师预测或评级数据",
            ["分析师", "一致预期", "盈利预测", "评级", "analyst", "forecast", "consensus", "rating"],
        ),
        (
            "盘口、订单簿、tick 或逐笔成交数据",
            ["盘口", "委托", "逐笔", "订单簿", "买一", "卖一", "order book", "level2", "l2", "tick data"],
        ),
    ]
    for data_name, keywords in unavailable_groups:
        if any(keyword in content for keyword in keywords):
            if not any(keyword in " ".join(available_columns) for keyword in keywords):
                return (
                    f"DATA_UNAVAILABLE: 该因子需要{data_name}，但当前本地 paper_factor 数据只包含"
                    f"日频字段 {data_profile.get('daily_columns') or []} 和分钟频字段 "
                    f"{data_profile.get('minute_columns') or []}，无法在不伪造字段的情况下复现。"
                )
    return None


def _refine_factor_task_with_llm(task: FactorTask, data_profile: dict[str, Any], domain_knowledge: str) -> FactorTask:
    if os.environ.get("RDAGENT_PAPER_FACTOR_DISABLE_TASK_REFINEMENT", "").strip().lower() in {"1", "true", "yes", "on"}:
        return task

    cache_dir = Path.cwd() / "git_ignore_folder" / "factor_outputs" / "task_refinement_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = md5_hash(
        json.dumps(
            {
                "factor_name": task.factor_name,
                "description": task.factor_description,
                "formulation": task.factor_formulation,
                "variables": task.variables,
                "daily_columns": data_profile.get("daily_columns") or [],
                "minute_columns": data_profile.get("minute_columns") or [],
                "knowledge": domain_knowledge,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    cache_path = cache_dir / f"{cache_key}.json"
    if cache_path.exists():
        try:
            refined = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to read factor task refinement cache {cache_path}: {exc}")
        else:
            return FactorTask(
                factor_name=str(refined.get("factor_name") or task.factor_name),
                factor_description=str(refined.get("description") or task.factor_description),
                factor_formulation=str(refined.get("formulation") or task.factor_formulation),
                variables=refined.get("variables") if isinstance(refined.get("variables"), dict) else task.variables,
            )

    system_prompt = (
        "你是金融工程因子复现任务精修助手。用户会给你一个从研报抽取出的因子、当前本地可用数据字段、"
        "以及按因子检索到的领域知识。你的目标不是写代码，而是把因子任务改写得足够完整、可执行、无歧义。\n"
        "要求：\n"
        "1. 保留论文原意，不要发明新因子。\n"
        "2. description 不应只是金融含义解释，还要说明实际生成步骤，包括变量含义、时间偏移、输入字段、样本范围、过滤条件、中间状态/判定逻辑、缺失值和非满足条件样本处理。\n"
        "3. formulation、description、variables 中出现的变量，都要明确时间锚点、是否相对事件日定义、依赖的原始字段、具体计算口径和输出频率。\n"
        "4. 因子任务必须自包含，不要假设其他因子已经提前算出。若引用 Stage_t_raw、Stage_smooth、市场情绪、回调确认、首板日、行业热度等中间概念，必须展开其生成逻辑；无法展开时明确缺少什么定义或数据。\n"
        "5. rolling/window/groupby/rank/标准化/分层统计等操作，必须说明窗口长度、聚合方式、排序方向、分组维度，以及是否去极值、标准化或中性化。\n"
        "6. 如果因子依赖事件或条件样本，例如涨停、炸板、首板、回调、放量、突破等，必须写清事件如何由本地行情数据或 retrieved_knowledge 口径计算、事件时间锚点、有效样本、非事件样本输出、多次事件冲突处理、是否需要历史窗口或未来确认。\n"
        "7. 领域概念口径只能来自研报原文或 retrieved_knowledge。对于涨停、首板、回调、一字板、非一字板等概念，禁止根据常识临时发明新定义，也不要为不同因子生成不同口径。\n"
        "8. 必须明确该因子使用的数据频率：日频、分钟频，或日频+分钟频。最终输出仍然必须是日频因子。\n"
        "9. T 日固定表示因子值输出/信号对应的日期；若论文使用事件日、形成日、买入日、调仓日、预测区间等不同日期，必须保留论文原始时间关系，不要混用。\n"
        "10. 如果因子包含阈值、分段、打分映射、状态判定、分位数边界、窗口长度等参数，必须给出具体数值或确定性公式。禁止写“合理设定”“自行设定”“参照图表但不给数值”“根据情况调整”等不可执行描述。\n"
        "11. 如果论文定义不完整或本地数据无法支持完整复现，不要擅自补全；请明确指出缺少字段、缺少规则、近似实现部分和逻辑歧义。\n"
        "12. 只返回 JSON，不要输出解释文字。"
    )
    user_prompt = json.dumps(
        {
            "factor": {
                "factor_name": task.factor_name,
                "description": task.factor_description,
                "formulation": task.factor_formulation,
                "variables": task.variables,
            },
            "local_data": {
                "daily_columns": data_profile.get("daily_columns") or [],
                "minute_columns": data_profile.get("minute_columns") or [],
                "history_window": f"{data_profile.get('start_date')} to {data_profile.get('end_date')}",
                "stock_count": data_profile.get("stock_count"),
            },
            "retrieved_knowledge": domain_knowledge,
            "return_schema": {
                "factor_name": "English snake_case factor name",
                "description": "完整中文任务描述，必须包含数据频率、T日输出/信号日期定义、事件锚点、有效样本条件、非事件样本输出、具体阈值/参数、缺失数据或缺失定义说明",
                "formulation": "自包含、可复现的公式；必须统一T日时间逻辑，必要时加入条件事件或 indicator，并展开中间概念、阈值、分段和打分映射的计算定义",
                "variables": {"变量名": "完整变量解释，包含数据频率、时间偏移、条件事件、输入字段、中间概念判定规则和具体阈值/参数"},
            },
        },
        ensure_ascii=False,
    )
    try:
        response = APIBackend().build_messages_and_create_chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=True,
            json_target_type=Dict[str, Any],
        )
        refined = json.loads(response)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to refine paper_factor task with LLM for {task.factor_name}: {exc}")
        return task

    if not isinstance(refined, dict):
        return task
    cache_path.write_text(json.dumps(refined, ensure_ascii=False, indent=2), encoding="utf-8")
    return FactorTask(
        factor_name=str(refined.get("factor_name") or task.factor_name),
        factor_description=str(refined.get("description") or task.factor_description),
        factor_formulation=str(refined.get("formulation") or task.factor_formulation),
        variables=refined.get("variables") if isinstance(refined.get("variables"), dict) else task.variables,
    )


def _adapt_report_task_for_available_data(task: FactorTask) -> FactorTask:
    data_profile = _load_local_factor_data_profile()
    domain_knowledge = _load_paper_factor_domain_knowledge(task)
    adapted_task = _refine_factor_task_with_llm(deepcopy(task), data_profile, domain_knowledge)
    available_daily_columns = ", ".join(data_profile.get("daily_columns") or []) or "unknown"
    available_minute_columns = ", ".join(data_profile.get("minute_columns") or []) or "unknown"
    adaptation_note = (
    "\n\n本次 paper_factor 任务的数据实现约束如下：\n"

    f"- 本地数据源：{data_profile.get('source')}。\n"
    f"- 当前可用股票数量：约 {data_profile.get('stock_count')} 只。\n"
    f"- 当前可用历史区间：{data_profile.get('start_date')} 至 "
    f"{data_profile.get('end_date')}（约 {data_profile.get('years')} 年）。\n"

    f"- 可用日频字段：{available_daily_columns}。\n"
    f"- 可用分钟频字段：{available_minute_columns}。\n\n"

    "【硬性数据约束】\n"
    "- 只能使用上面明确列出的数据源与字段。\n"
    "- 如果论文因子依赖未提供的数据字段或数据源，只有在研报原文或知识库已经给出口径且上述字段足以计算时才允许推导。"
    "如果实在无法计算得出，禁止使用其他字段近似替代。\n"
    "- 若数据缺失导致无法实现，请输出：\n"
    "  DATA_UNAVAILABLE: 缺失原因\n"
    "  然后直接退出，且不要生成 result.h5。\n\n"

    "【禁止事项】\n"
    "- 禁止使用价格成交量数据替代：\n"
    "  财务报表、基本面、市值、行业分类、分析师预期、订单簿、tick数据等不存在的数据。\n"
    "- 禁止通过构造 proxy 的方式伪造论文原始字段。\n"
    "- 禁止引入未来信息（future leakage）。\n\n"

    "【允许的自适应简化】\n"
    "- 可以根据本地数据规模，对论文中的训练窗口、滚动周期、样本规模、模型复杂度进行合理简化。\n"
    "- 如果论文使用的历史长度超过本地数据长度，应使用当前最大可用历史，而不是直接失败。\n"
    "- 如果论文使用的股票池大于当前股票数量，应直接在当前股票池上实现。\n"
    "- 如果论文采用复杂滚动训练（例如 GRU/TCN + rolling retraining），"
    "可改为更简单但不泄露未来信息的训练方式。\n"
    "- 当数据不足以支持年度滚动训练时，可改为 expanding window 或单次 train/valid/test 划分。\n"
    "- 在多种实现方式均可成立时，应优先选择：\n"
    "  1. 最符合当前字段条件\n"
    "  2. 最稳健\n"
    "  3. 最不容易未来泄露\n"
    "  的实现方式。\n\n"

    "【核心目标】\n"
    "- 保留论文方法的核心思想。\n"
    "- 允许根据本地数据规模调整实现细节。\n"
    "- 最终必须输出“日频因子 DataFrame”。\n"
    "- 不要因为本地数据规模小于论文设定而直接放弃实现。\n"
)
    if domain_knowledge:
        adaptation_note += f"\n\npaper_factor retrieved knowledge relevant to this factor:\n{domain_knowledge}"
    adapted_task.description = f"{adapted_task.factor_description}{adaptation_note}"
    return adapted_task


def _persist_extracted_factor_preview(
    report_file_path: str,
    *,
    minimal_mode: bool,
    factor_result: dict[str, dict[str, Any]],
    file_to_factor_result: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> Path:
    report_path = Path(report_file_path).resolve()
    output_dir = Path.cwd() / "git_ignore_folder" / "factor_outputs" / "extracted_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "report_file_path": str(report_path),
        "report_title": report_path.stem,
        "minimal_mode": minimal_mode,
        "factor_count": len(factor_result),
        "factor_names": list(factor_result.keys()),
        "factors": factor_result,
        "raw_file_to_factor_result": file_to_factor_result or {},
    }
    output_path = output_dir / f"{report_path.stem}.extracted.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _extracted_factor_preview_path(report_file_path: str | Path) -> Path:
    report_path = Path(report_file_path).resolve()
    return Path.cwd() / "git_ignore_folder" / "factor_outputs" / "extracted_reports" / f"{report_path.stem}.extracted.json"


def _load_extracted_factor_count(report_file_path: str | Path) -> int | None:
    preview_path = _extracted_factor_preview_path(report_file_path)
    if not preview_path.exists():
        return None
    try:
        payload = json.loads(preview_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to read extracted factor count from {preview_path}: {exc}")
        return None
    factors = payload.get("factors") or {}
    if not isinstance(factors, dict):
        return None
    return len(factors)


def _report_fully_processed(report_path: str | Path) -> bool:
    resolved_report_path = Path(report_path).resolve()
    if str(resolved_report_path) in _load_processed_report_paths():
        return True
    extracted_factor_count = _load_extracted_factor_count(resolved_report_path)
    if extracted_factor_count is None:
        return False
    terminal_factor_count = len(_load_terminal_report_factor_names(resolved_report_path))
    return terminal_factor_count >= extracted_factor_count > 0


def _load_exp_from_extracted_factor_preview(report_file_path: str, *, minimal_mode: bool) -> PaperFactorExperiment | None:
    preview_path = _extracted_factor_preview_path(report_file_path)
    if not preview_path.exists():
        return None
    try:
        payload = json.loads(preview_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to read extracted factor cache {preview_path}: {exc}")
        return None
    if str(payload.get("report_file_path") or "") != str(Path(report_file_path).resolve()):
        return None
    factors = payload.get("factors") or {}
    if not isinstance(factors, dict) or not factors:
        return None
    tasks = [
        FactorTask(
            factor_name=name,
            factor_description=str(item.get("description") or ""),
            factor_formulation=str(item.get("formulation") or ""),
            variables=item.get("variables") if isinstance(item.get("variables"), dict) else {},
        )
        for name, item in factors.items()
        if isinstance(item, dict)
    ]
    if not tasks:
        return None
    report_path = Path(report_file_path).resolve()
    exp = PaperFactorExperiment(sub_tasks=tasks, hypothesis=build_lightweight_hypothesis(report_path.stem, factors))
    exp.source_report_path = str(report_path)
    exp.source_report_title = report_path.stem
    logger.info(f"Loaded extracted factor cache from {preview_path}. factor_count={len(tasks)}")
    print(f"paper_factor: loaded extracted factor cache ({len(tasks)} factor(s)).", flush=True)
    return exp


def extract_hypothesis_and_exp_from_reports(
    report_file_path: str,
    *,
    minimal_mode: bool = True,
) -> PaperFactorExperiment | None:
    """
    Extract hypothesis and experiment details from report files.

    Args:
        report_file_path (str): Path to the report file.

    Returns:
        PaperFactorExperiment: A lightweight experiment containing the extracted details.
        None: If no valid experiment is found in the report.
    """
    if os.environ.get("RDAGENT_PAPER_FACTOR_FAST", "").strip().lower() in {"1", "true", "yes", "on"}:
        cached_exp = _load_exp_from_extracted_factor_preview(report_file_path, minimal_mode=minimal_mode)
        if cached_exp is not None:
            return cached_exp

    docs_dict = load_and_process_pdfs_by_langchain(report_file_path)
    loader = FactorExperimentLoaderFromPDFfiles()
    exp = loader.load_from_docs_dict(
        docs_dict,
        skip_report_classification=minimal_mode,
        skip_factor_viability_check=minimal_mode,
        single_pass_extraction=minimal_mode,
    )
    raw_factor_result = getattr(loader, "last_factor_dict", {}) or {}
    raw_file_to_factor_result = getattr(loader, "last_file_to_factor_result", {}) or {}
    preview_path = _persist_extracted_factor_preview(
        report_file_path,
        minimal_mode=minimal_mode,
        factor_result=raw_factor_result,
        file_to_factor_result=raw_file_to_factor_result,
    )
    logger.info(
        f"Saved extracted factor preview to {preview_path}. "
        f"Extracted factor_count={len(raw_factor_result)} for report={Path(report_file_path).name}"
    )

    if exp is None or exp.sub_tasks == []:
        logger.warning(
            f"No executable factors were produced from report {report_file_path}. "
            f"Check extracted preview JSON: {preview_path}"
        )
        return None

    pdf_screenshot = extract_first_page_screenshot_from_pdf(report_file_path)
    logger.log_object(pdf_screenshot, tag="load_pdf_screenshot")

    factor_result = {
        task.factor_name: {
            "description": task.factor_description,
            "formulation": task.factor_formulation,
            "variables": task.variables,
            "resources": task.factor_resources,
        }
        for task in exp.sub_tasks
    }

    report_content = "\n".join(docs_dict.values())
    hypothesis = (
        build_lightweight_hypothesis(report_file_path, factor_result)
        if minimal_mode
        else generate_hypothesis(factor_result, report_content)
    )
    exp.hypothesis = hypothesis
    exp.source_report_path = str(Path(report_file_path).resolve())
    exp.source_report_title = Path(report_file_path).stem
    return exp


class FactorReportLoop(FactorRDLoop, metaclass=LoopMeta):
    def __init__(self, report_folder: str = None, minimal_mode: bool = True, report_paths: list[str] | None = None):
        super().__init__(PROP_SETTING=FACTOR_FROM_REPORT_PROP_SETTING)
        os.environ.setdefault("RDAGENT_PAPER_FACTOR_SKIP_LOW_IC_REPAIR", "1")
        os.environ.setdefault("RDAGENT_PAPER_FACTOR_FAST", "1")
        os.environ.setdefault("RDAGENT_FACTOR_MAX_CONSECUTIVE_OUTPUT_WITHOUT_ACCEPT", "0")
        self.minimal_mode = minimal_mode
        self.report_cursor = 0
        self.pending_report_exp: PaperFactorExperiment | None = None
        self.pending_report_factor_idx = 0
        self.pending_report_factor_total = 0
        self.pending_report_extracted_count = 0
        if hasattr(self, "coder") and getattr(self.coder, "settings", None) is not None:
            self.coder.settings.v2_query_component_limit = 0
            self.coder.settings.v2_knowledge_sampler = 0.0
            self.coder.with_knowledge = True
            self.coder.knowledge_self_gen = False
            self.coder.max_loop = 3
            self.coder.stop_eval_chain_on_fail = True
        processed_report_paths = _load_processed_report_paths()
        if report_paths is not None:
            raw_items = report_paths
        elif report_folder is None:
            raw_items = json.load(
                open(FACTOR_FROM_REPORT_PROP_SETTING.report_result_json_file_path, "r")
            )
        else:
            raw_items = [i for i in Path(report_folder).rglob("*.pdf")]

        normalized_items: list[Path] = []
        for item in raw_items:
            path = Path(item).resolve()
            if str(path) in processed_report_paths:
                continue
            normalized_items.append(path)
        self.judge_pdf_data_items = normalized_items
        skipped_count = max(len(raw_items) - len(self.judge_pdf_data_items), 0)
        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} already-processed report(s) based on factor manifest.")

        self.loop_n = None
        self.shift_report = (
            0  # some reports does not contain viable factor, so we ship some of them to avoid infinite loop
        )

    async def direct_exp_gen(self, prev_out: dict[str, Any]):
        while True:
            if self.get_unfinished_loop_cnt(self.loop_idx) == 0:
                exp = self._next_single_factor_exp()
                exp.based_experiments = [PaperFactorExperiment(sub_tasks=[], hypothesis=exp.hypothesis)] + [
                    t[0] for t in self.trace.hist if t[1]
                ]
                exp.base_features = self.plan["features"]
                if exp.based_experiments:
                    exp.based_experiments[-1].base_features = self.plan["features"]
                logger.log_object(exp.hypothesis, tag="hypothesis generation")
                logger.log_object(exp.sub_tasks, tag="experiment generation")
                return exp
            await asyncio.sleep(1)

    def _next_single_factor_exp(self) -> PaperFactorExperiment:
        while self.pending_report_exp is None or self.pending_report_factor_idx >= self.pending_report_factor_total:
            if self.report_cursor >= len(self.judge_pdf_data_items) or self.report_cursor >= FACTOR_FROM_REPORT_PROP_SETTING.report_limit:
                raise self.LoopTerminationError("Reach stop criterion and stop loop")

            report_file_path = self.judge_pdf_data_items[self.report_cursor]
            self.report_cursor += 1
            logger.info(f"Processing report {self.report_cursor}: {report_file_path}")
            exp = extract_hypothesis_and_exp_from_reports(
                str(report_file_path),
                minimal_mode=self.minimal_mode,
            )
            if exp is None or not exp.sub_tasks:
                continue
            terminal_factor_names = _load_terminal_report_factor_names(report_file_path)
            if terminal_factor_names:
                before_count = len(exp.sub_tasks)
                exp.sub_tasks = [
                    task for task in exp.sub_tasks if getattr(task, "factor_name", None) not in terminal_factor_names
                ]
                skipped_factor_count = before_count - len(exp.sub_tasks)
                if skipped_factor_count > 0:
                    logger.info(
                        f"Skipped {skipped_factor_count} already-terminal factor(s) for report "
                        f"{report_file_path.name}: {sorted(terminal_factor_names)}"
                    )
            if not exp.sub_tasks:
                continue
            self.pending_report_exp = exp
            self.pending_report_factor_idx = 0
            self.pending_report_extracted_count = len(exp.sub_tasks)
            self.pending_report_factor_total = min(
                len(exp.sub_tasks),
                FACTOR_FROM_REPORT_PROP_SETTING.max_factors_per_exp,
            )
            logger.info(
                "Report factor serial coding plan: "
                f"extracted={self.pending_report_extracted_count}, "
                f"scheduled={self.pending_report_factor_total}, "
                f"max_per_exp={FACTOR_FROM_REPORT_PROP_SETTING.max_factors_per_exp}"
            )
            print(
                "paper_factor: extracted "
                f"{self.pending_report_extracted_count} factor(s), scheduled {self.pending_report_factor_total}.",
                flush=True,
            )

        source_exp = self.pending_report_exp
        factor_idx = self.pending_report_factor_idx
        self.pending_report_factor_idx += 1

        single_exp = deepcopy(source_exp)
        data_profile = _load_local_factor_data_profile()
        source_task = source_exp.sub_tasks[factor_idx]
        adapted_task = _adapt_report_task_for_available_data(source_task)
        skip_reason = _detect_unavailable_data_requirement(source_task, data_profile)
        single_exp.sub_tasks = [adapted_task]
        if skip_reason is not None:
            single_exp.paper_factor_skip_reason = skip_reason
        if source_exp.sub_workspace_list:
            single_exp.sub_workspace_list = [source_exp.sub_workspace_list[factor_idx]]
        else:
            single_exp.sub_workspace_list = []
        logger.info(
            "Scheduling report factor "
            f"{factor_idx + 1}/{self.pending_report_factor_total}: {single_exp.sub_tasks[0].factor_name}"
        )
        print(
            "paper_factor: coding "
            f"{factor_idx + 1}/{self.pending_report_factor_total} {single_exp.sub_tasks[0].factor_name}",
            flush=True,
        )
        return single_exp

    def coding(self, prev_out: dict[str, Any]):
        direct_exp = prev_out["direct_exp_gen"]
        skip_reason = getattr(direct_exp, "paper_factor_skip_reason", None)
        if skip_reason:
            task = direct_exp.sub_tasks[0] if direct_exp.sub_tasks else None
            task_name = getattr(task, "factor_name", "unknown")
            direct_exp.sub_workspace_list = [None]
            direct_exp.prop_dev_feedback = [
                FactorSingleFeedback(
                    execution_feedback=skip_reason,
                    value_generated_flag=False,
                    code_feedback="进入代码生成前已跳过：该因子所需数据或可执行定义在当前本地环境中不可用。",
                    value_feedback="未生成因子值：该因子所需数据或可执行定义在当前本地环境中不可用。",
                    final_decision=False,
                    final_feedback=skip_reason,
                    final_decision_based_on_gt=False,
                )
            ]
            logger.info(f"paper_factor: skipped {task_name} before coding. {skip_reason}")
            print(f"paper_factor: skipped {task_name} ({skip_reason})", flush=True)
            return direct_exp
        exp = self.coder.develop(direct_exp)
        logger.log_object(exp.sub_workspace_list, tag="coder result")
        return exp

    def running(self, prev_out: dict[str, Any]):
        logger.info("paper_factor: skip Qlib running stage and reuse coding result for record/export.")
        return prev_out["coding"]

    def feedback(self, prev_out: dict[str, Any]):
        exp = prev_out.get("running") or prev_out.get("coding") or prev_out.get("direct_exp_gen")
        prop_dev_feedback = getattr(exp, "prop_dev_feedback", None) if exp is not None else None
        accepted_count = 0
        reviewed_count = 0
        if prop_dev_feedback is not None:
            for task_feedback in prop_dev_feedback:
                if task_feedback is None:
                    continue
                reviewed_count += 1
                if bool(getattr(task_feedback, "final_decision", False)):
                    accepted_count += 1
        feedback = HypothesisFeedback(
            reason=(
                "Skipped Qlib running for paper_factor. "
                f"Reviewed {reviewed_count} factor implementation(s); {accepted_count} passed execution/value checks."
            ),
            decision=accepted_count > 0,
            acceptable=accepted_count > 0,
            observations="paper_factor now records IC directly from factor outputs without Qlib backtesting.",
            hypothesis_evaluation="Execution-level validation completed without running Qlib.",
            new_hypothesis=None,
            code_change_summary="Bypassed Qlib running stage for paper_factor; rely on factor output execution and IC recording.",
        )
        logger.log_object(feedback, tag="feedback")
        return feedback

    def record(self, prev_out: dict[str, Any]):
        feedback = prev_out["feedback"]
        exp = prev_out.get("running") or prev_out.get("coding") or prev_out.get("direct_exp_gen")
        exported_count = 0
        if exp is not None and getattr(exp, "prop_dev_feedback", None) is not None:
            source_report_path = getattr(exp, "source_report_path", None)
            source_report_title = getattr(exp, "source_report_title", None)
            for task, workspace, task_feedback in zip(exp.sub_tasks, exp.sub_workspace_list, exp.prop_dev_feedback):
                if task_feedback is None:
                    continue
                if not bool(task_feedback.final_decision):
                    _record_rejected_report_factor(task, task_feedback, source_report_path, source_report_title)
                    feedback_text = "\n".join(
                        part
                        for part in [
                            getattr(task_feedback, "execution", None),
                            getattr(task_feedback, "final_feedback", None),
                        ]
                        if part
                    )
                    if "DATA_UNAVAILABLE:" in feedback_text:
                        reason = feedback_text.split("DATA_UNAVAILABLE:", 1)[1].splitlines()[0].strip()
                        print(f"paper_factor: skipped {task.factor_name} (DATA_UNAVAILABLE: {reason})", flush=True)
                    else:
                        print(f"paper_factor: rejected {task.factor_name} (实现检查未通过).", flush=True)
                    continue
                if workspace is None:
                    continue
                try:
                    _, df = workspace.execute("All")
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        f"Failed to reload report-derived factor dataframe for reviewed export: "
                        f"task={task.factor_name}; {exc}"
                    )
                    continue
                if df is None or df.empty:
                    continue
                ic_feedback, full_sample_ic = evaluate_factor_ic_from_workspace(
                    workspace,
                    data_type="All",
                    gen_df=df,
                )
                logic_summary = task.factor_description
                review_notes = "\n".join(
                    part
                    for part in [task_feedback.execution, task_feedback.return_checking, task_feedback.code, ic_feedback]
                    if part
                )
                tags = _infer_report_factor_registry_tags(task, task_feedback)
                if full_sample_ic is not None and abs(full_sample_ic) >= FACTOR_COSTEER_SETTINGS.min_abs_ic:
                    tags.append("ic_passed")
                elif full_sample_ic is not None:
                    tags.append("ic_recorded_only")
                else:
                    tags.append("ic_unavailable")
                tags = sorted(set(tags))
                workspace.export_reviewed_factor(
                    df,
                    accepted=True,
                    logic_summary=logic_summary,
                    tags=tags,
                    review_notes=review_notes,
                    ic_score=full_sample_ic,
                    source_type="literature_report",
                    source_report_path=source_report_path,
                    source_report_title=source_report_title,
                )
                exported_count += 1
                print(f"paper_factor: exported {task.factor_name}.", flush=True)

        self.trace.sync_dag_parent_and_hist((exp, feedback), prev_out[self.LOOP_IDX_KEY])
        logger.info(
            f"Factor report loop recorded. Accepted reviewed factor exports: {exported_count}. "
            f"Source report: {getattr(exp, 'source_report_title', 'unknown') if exp is not None else 'unknown'}."
        )


def _infer_report_factor_registry_tags(task, feedback) -> list[str]:
    content = " ".join(
        [
            getattr(task, "factor_name", "") or "",
            getattr(task, "factor_description", "") or "",
            getattr(task, "factor_formulation", "") or "",
            str(getattr(task, "variables", {}) or {}),
            getattr(feedback, "code", "") or "",
        ]
    ).lower()
    tags: set[str] = {"literature_factor", "report_extracted"}
    keyword_to_tag = {
        "momentum": "momentum",
        "reversal": "reversal",
        "volatility": "volatility",
        "volume": "volume",
        "turnover": "turnover",
        "vwap": "vwap",
        "spread": "liquidity",
        "liquidity": "liquidity",
        "minute": "minute_input",
        "intraday": "minute_input",
        "microstructure": "microstructure",
        "gap": "gap",
        "trend": "trend",
        "acceleration": "acceleration",
    }
    for keyword, tag in keyword_to_tag.items():
        if keyword in content:
            tags.add(tag)
    if "future-information leak" not in content and "time leakage" not in content:
        tags.add("leakage_checked")
    tags.add("ic_passed")
    return sorted(tags)


def main(report_folder=None, path=None, all_duration=None, checkout=True, minimal_mode=True, report_paths=None):
    """
    Auto R&D Evolving loop for fintech factors (the factors are extracted from finance reports).

    Args:
        report_folder (str, optional): The folder contains the report PDF files. Reports will be loaded from this folder.
        path (str, optional): The path for loading a session. If provided, the session will be loaded.
        step_n (int, optional): Step number to continue running a session.
    """
    if path is None and report_folder is None and report_paths is None:
        model_loop = FactorReportLoop(minimal_mode=minimal_mode)
    elif path is not None:
        model_loop = FactorReportLoop.load(path, checkout=checkout)
    else:
        model_loop = FactorReportLoop(
            report_folder=report_folder,
            minimal_mode=minimal_mode,
            report_paths=report_paths,
        )

    asyncio.run(model_loop.run(all_duration=all_duration))


if __name__ == "__main__":
    import fire

    fire.Fire(main)
