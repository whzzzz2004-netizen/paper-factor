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
    get_paper_factor_pdf_reader_mode,
    load_and_process_pdfs_by_langchain,
    load_and_process_pdfs_for_paper_factor,
)
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.exception import CoderError
from rdagent.core.proposal import Hypothesis, HypothesisFeedback
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend, md5_hash
from rdagent.scenarios.qlib.factor_experiment_loader.pdf_loader import (
    FactorExperimentLoaderFromPDFfiles,
)
from rdagent.scenarios.qlib.experiment.data_schema import (
    filter_field_schema,
    format_field_schema_for_prompt,
    load_factor_field_schema,
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
    report_folder = RD_AGENT_SETTINGS.project_path / "papers" / "inbox"
    if not report_folder.exists():
        return set()
    return {str(path.resolve()) for path in report_folder.rglob("*.pdf") if _report_fully_processed(path)}


def _load_terminal_report_factor_names(report_path: str | Path) -> set[str]:
    report_title = Path(report_path).resolve().stem
    report_dir = (
        RD_AGENT_SETTINGS.project_path
        / "git_ignore_folder"
        / "factor_outputs"
        / "literature_reports"
        / FactorFBWorkspace._sanitize_factor_name(report_title)
    )

    # 从缓存中获取因子列表
    preview_path = _extracted_factor_preview_path(report_path)
    factors = {}
    if preview_path.exists():
        try:
            payload = json.loads(preview_path.read_text(encoding="utf-8"))
            factors = payload.get("factors") or {}
            if not isinstance(factors, dict):
                factors = {}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Failed to read extracted factor cache for terminal detection: {exc}")

    terminal_names: set[str] = set()

    # 从缓存因子列表中检查终端状态
    if factors:
        for factor_name in factors:
            safe_name = FactorFBWorkspace._sanitize_factor_name(str(factor_name))
            if (report_dir / f"{safe_name}.parquet").exists() or (report_dir / f"SKIPPED__{safe_name}.md").exists():
                terminal_names.add(str(factor_name))
    else:
        # 缓存为空时，扫描目录中的实际文件来发现终端因子
        if report_dir.exists():
            for f in report_dir.glob("*.parquet"):
                terminal_names.add(f.stem)
            for f in report_dir.glob("SKIPPED__*.md"):
                name = f.stem.removeprefix("SKIPPED__")
                terminal_names.add(name)
        if terminal_names:
            logger.info(
                f"Found {len(terminal_names)} terminal factor(s) from directory scan "
                f"(cache was empty): {sorted(terminal_names)}"
            )

    return terminal_names


def _record_rejected_report_factor(task, feedback, source_report_path: str | None, source_report_title: str | None) -> None:
    if source_report_path is None:
        return
    output_dir = RD_AGENT_SETTINGS.project_path / "git_ignore_folder" / "factor_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
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
    base_dir = RD_AGENT_SETTINGS.project_path / "git_ignore_folder" / "factor_implementation_source_data"
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
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Failed to load local factor data metadata from {meta_path}: {exc}")
    for freq, field_name in [("daily", "daily_columns"), ("minute", "minute_columns")]:
        stock_list_path = base_dir / "stock_data" / freq / "stock_list.json"
        if not stock_list_path.exists():
            continue
        try:
            import json as _json
            with open(stock_list_path) as _f:
                stocks = _json.load(_f)
            if stocks:
                sample_path = base_dir / "stock_data" / freq / f"{stocks[0]}.parquet"
                if sample_path.exists():
                    df = pd.read_parquet(sample_path, columns=None).head(1)
                    profile[field_name] = list(df.columns)
                    if hasattr(df.index, 'names'):
                        profile[f"{field_name.replace('columns', 'index_names')}"] = list(df.index.names)
        except (OSError, KeyError, ValueError) as exc:
            logger.warning(f"Failed to inspect available columns from per-stock parquet: {exc}")
    field_schema = filter_field_schema(load_factor_field_schema(base_dir), profile["daily_columns"])
    profile["daily_column_schema"] = field_schema
    profile["daily_column_schema_text"] = format_field_schema_for_prompt(field_schema)
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
        RD_AGENT_SETTINGS.project_path / "git_ignore_folder" / "paper_factor_knowledge_graph.yaml",
    ]
    nodes: dict[str, dict[str, Any]] = {}
    for path in graph_paths:
        if not path.exists():
            continue
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (yaml.YAMLError, OSError) as exc:
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
        RD_AGENT_SETTINGS.project_path / "git_ignore_folder" / "paper_factor_knowledge.md",
    ]
    sections: list[tuple[str, str, Path]] = []
    for path in knowledge_paths:
        if not path.exists():
            continue
        try:
            content = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
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


def _judge_factor_data_availability_with_llm(
    task: FactorTask,
    data_profile: dict[str, Any],
    domain_knowledge: str,
) -> str | None:
    """Return availability status string.

    Returns None for AVAILABLE, "DATA_UNAVAILABLE: ..." when data is truly missing,
    "DEFINITION_INCOMPLETE: ..." when definition is unclear but LLM can fill defaults.
    Only DATA_UNAVAILABLE should skip the factor; DEFINITION_INCOMPLETE enters coding.
    """
    if os.environ.get("RDAGENT_PAPER_FACTOR_DISABLE_DATA_AVAILABILITY_JUDGE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None

    # Cache availability check results to avoid repeated LLM calls on re-runs
    cache_dir = RD_AGENT_SETTINGS.project_path / "git_ignore_folder" / "factor_outputs" / "availability_cache"
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
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            result_str = cached.get("result")
            logger.info(f"Using cached availability check for {task.factor_name}: {result_str}")
            return result_str
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Failed to read availability cache {cache_path}: {exc}")

    system_prompt = (
        "你是金融工程因子复现前的数据可用性审查员。用户会给你一个从研报抽取出的因子定义、"
        "当前本地可用数据字段、字段含义和相关领域知识。你的任务只判断：仅依赖这些本地数据，是否足以进入代码实现。\n"
        "判定规则：\n"
        "1. 只能使用 local_data 中列出的字段、字段说明，以及能由这些字段或 retrieved_knowledge 明确推导出的派生量；禁止假设隐藏表或额外字段。\n"
        "2. local_data.daily_column_schema 包含每个日频字段的中文名、公式、英文名、来源等信息。很多字段使用抽象名称（如 $盈利因子1、$成长因子14），必须通过 schema 查看其实际含义来判断是否可用，不能仅凭列名就判定缺失。\n"
        "3. 如果因子能由现成字段直接取得，或能由日频/分钟频字段通过明确公式、rolling、rank、聚合、标准化、训练过程得到，应判定 AVAILABLE。\n"
        "4. 分钟数据中的 $vwap 可以通过加权平均聚合为日频均价，因此不要因为缺少日频vwap就判定不可用。\n"
        "5. 日频基本面数据（如 $盈利因子1=毛利率、$成长因子1=营收同比增长率）虽然以日频存储，但包含季度变化点，可以据此推导TTM、季度环比、同比等口径。不要因为因子要求TTM或季度口径就判定日频基本面数据不可用。\n"
        "6. 不允许用无关代理变量替代必需数据。例如没有 turnover/turnover_rate 或股本字段时，不能用 volume 代替换手率。\n"
        "7. 只判断是否进入代码实现，不写代码，不评价因子好坏。只返回 JSON。\n"
        "8. 【极其重要】数据为MultiIndex DataFrame结构，股票代码(instrument)和日期(datetime)是索引(index)，不是列(columns)。"
        "不要因为列名中没有股票代码就判定缺少股票代码字段！股票代码可通过 df.index.get_level_values('instrument') 获取。"
        "同理，日期可通过 df.index.get_level_values('datetime') 获取。这是最常见的误判，请务必注意！\n"
        "9. 【重要】分钟数据文件(minute_pv.h5)包含 $open,$high,$low,$close,$volume,$vwap,$factor,$return 共8个字段。"
        "如果因子公式涉及分钟级别的价格、成交量、均价、收益率计算，这些字段都可用。不要因为因子描述中写了'分钟数据'就判定 DATA_UNAVAILABLE。\n"
        "10. 【重要】字段 $turnover_rate 就是换手率（日频）。如果因子需要换手率数据，使用 $turnover_rate。"
        "如果因子需要'流通股本'来从成交量计算换手率，可以直接使用 $turnover_rate 字段，无需反推股本。\n"
        "11. 【重要】'市值中性化'、'行业中性化'、'截面标准化'、'Z-score标准化'、'Rank标准化'等操作属于标准数据处理流程，"
        "不需要额外数据字段。即使缺少行业分类数据，也可以做市值中性化或纯截面标准化。缺少行业中性化条件不应判为 DATA_UNAVAILABLE。\n"
        "12. 【宽松原则】除非因子明确需要以下不可用数据，否则一律判为 AVAILABLE 或 DEFINITION_INCOMPLETE：\n"
        "    - 逐笔成交数据（tick-level trade data），而非分钟K线\n"
        "    - 订单簿数据（order book / level-2 data）\n"
        "    - 龙虎榜数据、融资融券数据、北向资金数据\n"
        "    - 财务报告原文中的非结构化文本信息\n"
        "    - 行业分类标签（如申万一级行业）\n"
        "    - 机构持仓数据\n"
        "对于上述宽松原则的例外：如果因子需要单独的外生变量（如GDP、CPI、利率等宏观数据），这些确实不可用，判DATA_UNAVAILABLE。\n"
        "关于深度学习/机器学习因子的特别规则（极其重要）：\n"
        "- 以下情况全部属于 DEFINITION_INCOMPLETE，不是 DATA_UNAVAILABLE：\n"
        "  · 缺少预训练模型参数/权重（可以从头训练）\n"
        "  · 缺少 hidden size、层数、dilation_base 等超参数（可以用常见默认值）\n"
        "  · 缺少 loss 函数定义（可以用 MSE/MAE 等常见默认）\n"
        "  · 缺少标准化/归一化方法（可以用 z-score 默认）\n"
        "  · 缺少训练轮数、学习率等训练细节（可以用常见默认值）\n"
        "  · 预测标签定义有歧义（如10日均价涨跌幅，可以用合理默认解释）\n"
        "  · 周度/月度数据聚合口径未明确（可以从日频数据聚合）\n"
        "- 只有以下情况才属于 DATA_UNAVAILABLE：\n"
        "  · 输入字段本身不存在（如需要逐笔成交但只有分钟K线）\n"
        "  · 需要的原始数据列在 local_data 中确实没有（如需要 $roe 但没有）\n"
        "13. 【极其重要】历史数据时间长度不足（如研报要求9年但本地只有2.5年）不是 DATA_UNAVAILABLE 的理由。"
        "即使只有几个月的数据，coder 也可以用可用数据训练和预测。只要字段存在，就判 AVAILABLE 或 DEFINITION_INCOMPLETE。\n"
        "14. 【极其重要】禁止猜测数据质量。你没有实际读取完整数据文件，无法知道某列是否有 NaN、是否全为 0 等。"
        "只判断字段名是否存在。数据值质量问题（NaN、missing）由运行时代码自行处理。\n"
        "15. 因子描述中提到的训练窗口、历史长度等需求，如果本地数据不能满足，coder 会用可用数据适配。"
        "不要因此判 DATA_UNAVAILABLE。\n"
        "- 简单来说：只要输入字段（OHLCV、volume、turnover_rate 等）在 local_data 中列出来了，其他一切缺失都算 DEFINITION_INCOMPLETE。\n"
        "返回状态说明：\n"
        "- AVAILABLE：字段和定义都够，直接进入代码生成。\n"
        "- DEFINITION_INCOMPLETE：因子定义不够清楚（如 DL 模型缺 hidden size、loss、标签、标准化方法），但大模型可以凭知识尝试补全，进入代码生成。\n"
        "- DATA_UNAVAILABLE：真的缺输入字段（如需要逐笔成交但只有分钟K线，需要 $roe 但没有），无法实现，跳过。\n"
        "【总结】当你不确定时，倾向于 AVAILABLE 或 DEFINITION_INCOMPLETE，而不是 DATA_UNAVAILABLE。只有当明确检测到上述不可用数据时，才返回 DATA_UNAVAILABLE。"
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
                "daily_column_schema": data_profile.get("daily_column_schema") or {},
                "minute_columns": data_profile.get("minute_columns") or [],
                "daily_index": data_profile.get("daily_index_names") or ["datetime", "instrument"],
                "minute_index": data_profile.get("minute_index_names") or ["datetime", "instrument"],
                "history_window": f"{data_profile.get('start_date')} to {data_profile.get('end_date')}",
                "stock_count": data_profile.get("stock_count"),
                "data_structure_hint": "数据为MultiIndex DataFrame，股票代码(instrument)和日期(datetime)在索引中，不在列中。可通过 df.index.get_level_values('instrument') 和 df.index.get_level_values('datetime') 获取。",
            },
            "retrieved_knowledge": domain_knowledge,
            "return_schema": {
                "status": "AVAILABLE / DEFINITION_INCOMPLETE / DATA_UNAVAILABLE",
                "reason": "中文理由；DATA_UNAVAILABLE 必须说明缺少的字段；DEFINITION_INCOMPLETE 必须说明哪些定义不清",
            },
        },
        ensure_ascii=False,
    )
    judge_model = os.environ.get("RDAGENT_PAPER_FACTOR_AVAILABILITY_JUDGE_MODEL", "").strip()
    backend = APIBackend()
    if judge_model:
        from rdagent.oai.llm_conf import LLM_SETTINGS
        original_model = LLM_SETTINGS.chat_model
        LLM_SETTINGS.chat_model = judge_model
    try:
        _last_exc: Exception | None = None
        for _attempt in range(3):
            try:
                response = backend.build_messages_and_create_chat_completion(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    json_mode=True,
                    json_target_type=Dict[str, Any],
                )
                result = json.loads(response)
                break
            except Exception as exc:
                _last_exc = exc
                logger.warning(f"LLM data availability check attempt {_attempt + 1}/3 failed for {task.factor_name}: {exc}")
        else:
            raise RuntimeError(
                f"LLM data availability check failed for {task.factor_name} after 3 attempts: {_last_exc}"
            ) from _last_exc
    finally:
        if judge_model:
            LLM_SETTINGS.chat_model = original_model

    if not isinstance(result, dict):
        result_str = None
    else:
        status = str(result.get("status") or "").strip().upper()
        reason = str(result.get("reason") or "").strip()
        if status == "DATA_UNAVAILABLE":
            result_str = f"DATA_UNAVAILABLE: {reason or '该因子所需数据在当前本地环境中不可用。'}"
        elif status == "DEFINITION_INCOMPLETE":
            result_str = f"DEFINITION_INCOMPLETE: {reason or '因子定义不够清楚，将由代码生成阶段补全默认实现。'}"
        else:
            result_str = None

    # Cache the result
    try:
        cache_path.write_text(json.dumps({"result": result_str}, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        logger.warning(f"Failed to write availability cache {cache_path}: {exc}")

    return result_str


def _refine_factor_task_with_llm(task: FactorTask, data_profile: dict[str, Any], domain_knowledge: str) -> FactorTask:
    if os.environ.get("RDAGENT_PAPER_FACTOR_DISABLE_TASK_REFINEMENT", "").strip().lower() in {"1", "true", "yes", "on"}:
        return task

    cache_dir = RD_AGENT_SETTINGS.project_path / "git_ignore_folder" / "factor_outputs" / "task_refinement_cache"
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
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Failed to read factor task refinement cache {cache_path}: {exc}")
        else:
            return FactorTask(
                factor_name=str(refined.get("factor_name") or task.factor_name),
                factor_description=str(refined.get("description") or task.factor_description),
                factor_formulation=str(refined.get("formulation") or task.factor_formulation),
                variables=refined.get("variables") if isinstance(refined.get("variables"), dict) else task.variables,
                factor_type=getattr(task, 'factor_type', 'single_stock'),
            )

    system_prompt = (
        "你是金融工程因子复现任务精修助手。用户会给你一个从研报抽取出的因子、当前本地可用数据字段、"
        "以及按因子检索到的领域知识。你的目标不是写代码，而是把因子任务改写得足够完整、可执行、无歧义。\n"
        "要求：\n"
        "1. 保留论文原意，不要发明新因子。\n"
        "2. description 不应只是金融含义解释，还要说明实际生成步骤，包括变量含义、时间偏移、输入字段、样本范围、过滤条件、中间状态/判定逻辑、缺失值和非满足条件样本处理。\n"
        "3. formulation、description、variables 中出现的变量，都要明确时间锚点、是否相对事件日定义、依赖的原始字段、具体计算口径和输出频率。\n"
        "4. 因子任务必须自包含，不要假设其他因子已经提前算出。后续代码生成只能看到当前因子的 description、formulation、variables，不能再查看研报原文或其他因子定义。若引用 Stage_t_raw、Stage_smooth、市场情绪、回调确认、首板日、行业热度等中间概念，必须展开其生成逻辑；无法展开时明确缺少什么定义或数据。\n"
        "5. 禁止写“同 XXX 因子”“同上”“其他参数同”“具体架构见研报”“参考前文”“见表 X/图 X”等外部引用式描述。若当前因子与另一个因子共享模型结构、输入字段、标准化方式、标签定义、训练目标、窗口设置、阈值或事件口径，必须在当前因子的 description/formulation/variables 中完整重复展开。如果研报没有给出可展开细节，要明确写缺少哪些参数或定义。\n"
        "6. 对机器学习/深度学习因子，必须尽量定义好模型结构和训练目标，包括输入字段、输入窗口、预测 horizon、标签定义、样本构造、标准化方式、模型类型、层数、loss、随机种子等。名称中的信息必须显式展开。若研报没有给出 hidden size、训练轮数等工程细节，可以写明缺失并给出常见、简洁、可复现的默认实现建议；不要因此把任务改成不可实现。只有输入字段、标签或训练目标本身无法确定时，才标记定义不足。\n"
        "7. rolling/window/groupby/rank/标准化/分层统计等操作，必须说明窗口长度、聚合方式、排序方向、分组维度，以及是否去极值、标准化或中性化。\n"
        "8. 如果因子依赖事件或条件样本，例如涨停、炸板、首板、回调、放量、突破等，必须写清事件如何由本地行情数据或 retrieved_knowledge 口径计算、事件时间锚点、有效样本、非事件样本输出、多次事件冲突处理、是否需要历史窗口或未来确认。\n"
        "9. 领域概念口径只能来自研报原文或 retrieved_knowledge。对于涨停、首板、回调、一字板、非一字板等概念，禁止根据常识临时发明新定义，也不要为不同因子生成不同口径。\n"
        "10. 必须明确该因子使用的数据频率：日频、分钟频，或日频+分钟频。最终输出仍然必须是日频因子。\n"
        "11. T 日固定表示因子值输出/信号对应的日期；若论文使用事件日、形成日、买入日、调仓日、预测区间等不同日期，必须保留论文原始时间关系，不要混用。\n"
        "12. 如果因子包含阈值、分段、打分映射、状态判定、分位数边界、窗口长度等参数，必须给出具体数值或确定性公式。禁止写“合理设定”“自行设定”“参照图表但不给数值”“根据情况调整”等不可执行描述。\n"
        "13. 如果论文定义有小的工程细节缺失，应优先给出保守默认实现建议并明确标注；如果核心数据、核心标签或事件口径缺失到无法写代码，再指出缺少字段、缺少规则和逻辑歧义。\n"
        "14. ⚠️ local_data.history_window 是本地实际可用的数据时间范围（通常仅约2.5年）。"
        "因子描述中如果包含按年度滚动训练（如'每年用前N年数据训练'）、多年训练集（如'前9年数据作为训练集'）"
        "等需要超出本地数据范围的训练方案，必须改为'使用全部可用数据，按时间顺序分割为训练集/验证集'。"
        "不要保留需要多年数据的原始方案。\n"
        "15. 只返回 JSON，不要输出解释文字。"
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
                "daily_column_schema": data_profile.get("daily_column_schema") or {},
                "minute_columns": data_profile.get("minute_columns") or [],
                "daily_index": data_profile.get("daily_index_names") or ["datetime", "instrument"],
                "minute_index": data_profile.get("minute_index_names") or ["datetime", "instrument"],
                "history_window": f"{data_profile.get('start_date')} to {data_profile.get('end_date')}",
                "stock_count": data_profile.get("stock_count"),
                "data_structure_hint": "数据为MultiIndex DataFrame，股票代码(instrument)和日期(datetime)在索引中，不在列中。可通过 df.index.get_level_values('instrument') 和 df.index.get_level_values('datetime') 获取。",
            },
            "retrieved_knowledge": domain_knowledge,
            "return_schema": {
                "factor_name": "English snake_case factor name",
                "description": "完整中文任务描述，必须包含数据频率、T日输出/信号日期定义、事件锚点、有效样本条件、非事件样本输出、具体阈值/参数、共享设定的完整展开、ML/DL模型结构与训练目标、缺失数据或缺失定义说明；禁止同上/同某因子/见研报等引用式描述",
                "formulation": "自包含、可复现的公式；必须统一T日时间逻辑，必要时加入条件事件或 indicator，并展开中间概念、阈值、分段、打分映射、模型输入输出、标签和训练目标的计算定义；禁止外部引用式描述",
                "variables": {"变量名": "完整变量解释，包含数据频率、时间偏移、条件事件、输入字段、中间概念判定规则、具体阈值/参数、ML/DL模型参数和缺失定义说明"},
            },
        },
        ensure_ascii=False,
    )
    _last_exc: Exception | None = None
    refined = None
    for _attempt in range(3):
        try:
            response = APIBackend().build_messages_and_create_chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_mode=True,
                json_target_type=Dict[str, Any],
            )
            refined = json.loads(response)
            break
        except Exception as exc:
            _last_exc = exc
            logger.warning(f"LLM task refinement attempt {_attempt + 1}/3 failed for {task.factor_name}: {exc}")
    if refined is None:
        raise RuntimeError(
            f"LLM task refinement failed for {task.factor_name} after 3 attempts: {_last_exc}"
        ) from _last_exc

    if not isinstance(refined, dict):
        return task
    cache_path.write_text(json.dumps(refined, ensure_ascii=False, indent=2), encoding="utf-8")
    return FactorTask(
        factor_name=str(refined.get("factor_name") or task.factor_name),
        factor_description=str(refined.get("description") or task.factor_description),
        factor_formulation=str(refined.get("formulation") or task.factor_formulation),
        variables=refined.get("variables") if isinstance(refined.get("variables"), dict) else task.variables,
        factor_type=getattr(task, 'factor_type', 'single_stock'),
    )


def _adapt_report_task_for_available_data(task: FactorTask) -> FactorTask:
    data_profile = _load_local_factor_data_profile()
    domain_knowledge = _load_paper_factor_domain_knowledge(task)
    adapted_task = _refine_factor_task_with_llm(deepcopy(task), data_profile, domain_knowledge)
    available_daily_columns = ", ".join(data_profile.get("daily_columns") or []) or "unknown"
    available_minute_columns = ", ".join(data_profile.get("minute_columns") or []) or "unknown"
    daily_index_names = ", ".join(data_profile.get("daily_index_names") or []) or "unknown"
    minute_index_names = ", ".join(data_profile.get("minute_index_names") or []) or "unknown"
    schema_text = data_profile.get("daily_column_schema_text") or ""
    adaptation_note = (
        f"\n\n本地可用数据：{data_profile.get('source')}，约{data_profile.get('stock_count')}只股票，"
        f"{data_profile.get('start_date')}至{data_profile.get('end_date')}。"
        f"日频字段：{available_daily_columns}。分钟频字段：{available_minute_columns}。"
        f"\n重要：数据为MultiIndex结构，索引包含：{daily_index_names}（日频）、{minute_index_names}（分钟频）。"
        f"股票代码(instrument)在索引中，可通过 df.index.get_level_values('instrument') 获取。"
        f"日期(datetime)也在索引中，可通过 df.index.get_level_values('datetime') 获取。"
    )
    if schema_text:
        adaptation_note += f"\n\n日频字段说明（含公式、英文名、来源）：\n{schema_text}"
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
    output_dir = RD_AGENT_SETTINGS.project_path / "git_ignore_folder" / "factor_outputs" / "extracted_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "report_file_path": str(report_path),
        "report_title": report_path.stem,
        "minimal_mode": minimal_mode,
        "pdf_reader": get_paper_factor_pdf_reader_mode(),
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
    return RD_AGENT_SETTINGS.project_path / "git_ignore_folder" / "factor_outputs" / "extracted_reports" / f"{report_path.stem}.extracted.json"


def _load_extracted_factor_count(report_file_path: str | Path) -> int | None:
    preview_path = _extracted_factor_preview_path(report_file_path)
    if not preview_path.exists():
        return None
    try:
        payload = json.loads(preview_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Failed to read extracted factor count from {preview_path}: {exc}")
        return None
    factors = payload.get("factors") or {}
    if not isinstance(factors, dict):
        return None
    return len(factors)


def _report_fully_processed(report_path: str | Path) -> bool:
    resolved_report_path = Path(report_path).resolve()
    terminal_factor_names = _load_terminal_report_factor_names(resolved_report_path)
    if not terminal_factor_names:
        return False
    extracted_factor_count = _load_extracted_factor_count(resolved_report_path)
    if extracted_factor_count is None or extracted_factor_count == 0:
        # 缓存因子数为 0 但有终端文件，说明之前的运行已处理完毕
        return True
    return len(terminal_factor_names) >= extracted_factor_count


def _classify_factors_with_llm(factors: dict) -> dict:
    """用 LLM 对提取的因子进行类型分类（single_stock / cross_section / deep_learning）。"""
    if not factors:
        return factors
    # 构建因子描述列表
    desc_lines = []
    for name, info in factors.items():
        desc = str(info.get("description") or "")[:200]
        form = str(info.get("formulation") or "")[:100]
        desc_lines.append(f"- {name}: {desc} | 公式: {form}")
    factor_descriptions = "\n".join(desc_lines)

    system_prompt = T(".prompts:classify_factor_type_system").r()
    user_prompt = T(".prompts:classify_factor_type_user").r(factor_descriptions=factor_descriptions)

    try:
        resp = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=True,
        )
        classifications = json.loads(resp)
        valid_types = {"single_stock", "cross_section", "deep_learning"}
        for name in factors:
            ft = str(classifications.get(name) or "single_stock").strip().lower()
            if ft not in valid_types:
                ft = "single_stock"
            factors[name]["factor_type"] = ft
        logger.info(f"Factor type classification: { {n: f.get('factor_type') for n, f in factors.items()} }")
    except Exception as exc:
        logger.warning(f"Factor type classification failed: {exc}. Defaulting to single_stock.")
        for name in factors:
            factors[name]["factor_type"] = "single_stock"
    return factors


def _load_exp_from_extracted_factor_preview(report_file_path: str, *, minimal_mode: bool) -> PaperFactorExperiment | None:
    preview_path = _extracted_factor_preview_path(report_file_path)
    if not preview_path.exists():
        return None
    try:
        payload = json.loads(preview_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, KeyError) as exc:
        logger.warning(f"Failed to read extracted factor cache {preview_path}: {exc}")
        return None
    if str(payload.get("report_file_path") or "") != str(Path(report_file_path).resolve()):
        return None
    current_pdf_reader = get_paper_factor_pdf_reader_mode()
    cached_pdf_reader = str(payload.get("pdf_reader") or "pymupdf").strip().lower()
    if cached_pdf_reader != current_pdf_reader:
        logger.info(
            f"Ignore extracted factor cache because PDF reader changed: "
            f"cached={cached_pdf_reader}, current={current_pdf_reader}."
        )
        return None
    factors = payload.get("factors") or {}
    if not isinstance(factors, dict) or not factors:
        return None
    # 第二步：用 LLM 对因子分类
    factors = _classify_factors_with_llm(factors)
    tasks = [
        FactorTask(
            factor_name=name,
            factor_description=str(item.get("description") or ""),
            factor_formulation=str(item.get("formulation") or ""),
            variables=item.get("variables") if isinstance(item.get("variables"), dict) else {},
            factor_type=str(item.get("factor_type") or "single_stock"),
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

    # Extract text from PDF and parse factors using LLM
    docs_dict = load_and_process_pdfs_for_paper_factor(report_file_path)
    loader = FactorExperimentLoaderFromPDFfiles()
    exp = loader.load_from_docs_dict(
        docs_dict,
        skip_report_classification=minimal_mode,
        skip_factor_viability_check=minimal_mode,
        single_pass_extraction=minimal_mode,
    )
    raw_factor_result = getattr(loader, "last_factor_dict", {}) or {}
    raw_file_to_factor_result = getattr(loader, "last_file_to_factor_result", {}) or {}
    # 用 LLM 对因子分类
    raw_factor_result = _classify_factors_with_llm(raw_factor_result)
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
        if RD_AGENT_SETTINGS.step_timeout <= 0:
            RD_AGENT_SETTINGS.step_timeout = 600  # 10 minutes per step default
        try:
            self.parallel_factor_n = max(
                1,
                int(
                    os.environ.get(
                        "RDAGENT_PAPER_FACTOR_PARALLEL_FACTORS",
                        str(FACTOR_FROM_REPORT_PROP_SETTING.max_factors_per_exp),
                    )
                ),
            )
        except ValueError:
            self.parallel_factor_n = max(1, FACTOR_FROM_REPORT_PROP_SETTING.max_factors_per_exp)
        if self.parallel_factor_n > 1:
            RD_AGENT_SETTINGS.step_semaphore = {
                "direct_exp_gen": 1,
                "coding": self.parallel_factor_n,
                "running": self.parallel_factor_n,
                "feedback": 1,
                "record": 1,
            }
            RD_AGENT_SETTINGS.multi_proc_n = 1
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
            self.coder.max_loop = 5
            self.coder.settings.v2_error_summary = True
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
        loop = asyncio.get_running_loop()
        while True:
            unfinished_loop_cnt = self.get_unfinished_loop_cnt(self.loop_idx)
            current_report_dispatched = (
                self.pending_report_exp is not None
                and self.pending_report_factor_idx >= self.pending_report_factor_total
            )
            if current_report_dispatched and unfinished_loop_cnt > 0:
                await asyncio.sleep(0.5)
                continue
            if unfinished_loop_cnt < self.parallel_factor_n:
                # Run blocking LLM calls in thread pool to avoid blocking the event loop
                exp = await loop.run_in_executor(None, self._next_single_factor_exp)
                exp.based_experiments = [PaperFactorExperiment(sub_tasks=[], hypothesis=exp.hypothesis)] + [
                    t[0] for t in self.trace.hist if t[1]
                ]
                exp.base_features = self.plan["features"]
                if exp.based_experiments:
                    exp.based_experiments[-1].base_features = self.plan["features"]
                logger.log_object(exp.hypothesis, tag="hypothesis generation")
                logger.log_object(exp.sub_tasks, tag="experiment generation")
                return exp
            await asyncio.sleep(0.1)

    @staticmethod
    def _is_deep_learning_report(report_path: Path) -> bool:
        """用 LLM 判断研报是否涉及深度学习（GRU/TCN/LSTM/Transformer 等），仅需是/否。"""
        try:
            backend = APIBackend()
            title = report_path.stem
            response = backend.build_messages_and_create_chat_completion(
                user_prompt=f"研报标题：{title}\n\n这是一篇关于深度学习（如 GRU、TCN、LSTM、Transformer、CNN、RNN、神经网络）的研报吗？只回答 yes 或 no。",
                system_prompt="你是一个分类助手。只回答 yes 或 no，不要解释。",
                json_mode=False,
            )
            result = "yes" in response.strip().lower()
            logger.info(f"Deep learning check for '{title}': {result} (response: {response.strip()[:50]})")
            return result
        except Exception as e:
            logger.warning(f"Deep learning check failed: {e}, defaulting to non-DL")
            return False

    def _next_single_factor_exp(self) -> PaperFactorExperiment:
        while self.pending_report_exp is None or self.pending_report_factor_idx >= self.pending_report_factor_total:
            if self.report_cursor >= len(self.judge_pdf_data_items) or self.report_cursor >= FACTOR_FROM_REPORT_PROP_SETTING.report_limit:
                raise self.LoopTerminationError("Reach stop criterion and stop loop")

            report_file_path = self.judge_pdf_data_items[self.report_cursor]
            self.report_cursor += 1
            logger.info(f"Processing report {self.report_cursor}: {report_file_path}")

            # 根据研报类型调整并行度
            _base_parallel = int(os.environ.get("RDAGENT_PAPER_FACTOR_PARALLEL_FACTORS", "3"))
            _dl_parallel = int(os.environ.get("RDAGENT_PAPER_FACTOR_DL_PARALLEL_FACTORS", "2"))
            if self._is_deep_learning_report(report_file_path):
                self.parallel_factor_n = _dl_parallel
            else:
                self.parallel_factor_n = _base_parallel
            RD_AGENT_SETTINGS.step_semaphore = {
                "direct_exp_gen": 1,
                "coding": self.parallel_factor_n,
                "running": self.parallel_factor_n,
                "feedback": 1,
                "record": 1,
            }
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
            self.parallel_factor_n = min(self.parallel_factor_n, self.pending_report_factor_total)
            logger.info(
                "Report factor parallel coding plan: "
                f"extracted={self.pending_report_extracted_count}, "
                f"scheduled={self.pending_report_factor_total}, "
                f"parallel={self.parallel_factor_n}, "
                f"max_per_exp={FACTOR_FROM_REPORT_PROP_SETTING.max_factors_per_exp}"
            )
            print(
                "paper_factor: extracted "
                f"{self.pending_report_extracted_count} factor(s), scheduled {self.pending_report_factor_total}, "
                f"parallel {self.parallel_factor_n}.",
                flush=True,
            )

        source_exp = self.pending_report_exp
        factor_idx = self.pending_report_factor_idx
        self.pending_report_factor_idx += 1

        single_exp = deepcopy(source_exp)
        data_profile = _load_local_factor_data_profile()
        source_task = source_exp.sub_tasks[factor_idx]
        domain_knowledge = _load_paper_factor_domain_knowledge(source_task)
        availability = _judge_factor_data_availability_with_llm(source_task, data_profile, domain_knowledge)
        skip_reason = availability if availability is not None and availability.startswith("DATA_UNAVAILABLE:") else None
        adapted_task = deepcopy(source_task) if skip_reason is not None else _adapt_report_task_for_available_data(source_task)
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
        if skip_reason is not None:
            print(
                "paper_factor: skip-before-coding "
                f"{factor_idx + 1}/{self.pending_report_factor_total} {single_exp.sub_tasks[0].factor_name}",
                flush=True,
            )
        else:
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

        # Handle CoderError: when all implementations fail, record factors as permanently failed
        from rdagent.utils.workflow.loop import LoopBase
        exc = prev_out.get(LoopBase.EXCEPTION_KEY)
        if exc is not None and isinstance(exc, CoderError) and exp is not None:
            source_report_path = getattr(exp, "source_report_path", None)
            source_report_title = getattr(exp, "source_report_title", None)
            for task in (exp.sub_tasks or []):
                factor_name = getattr(task, "factor_name", "unknown")
                _record_rejected_report_factor(
                    task,
                    FactorSingleFeedback(
                        execution_feedback=str(exc),
                        value_generated_flag=False,
                        code_feedback="所有实现均失败，已永久跳过。",
                        value_feedback="未生成因子值。",
                        final_decision=False,
                        final_feedback=f"CoderError: 所有实现均失败 ({factor_name})",
                        final_decision_based_on_gt=False,
                    ),
                    source_report_path,
                    source_report_title,
                )
                print(f"paper_factor: permanently failed {factor_name} (all implementations failed).", flush=True)

        if exp is not None and getattr(exp, "prop_dev_feedback", None) is not None:
            source_report_path = getattr(exp, "source_report_path", None)
            source_report_title = getattr(exp, "source_report_title", None)
            # Collect accepted factor export jobs for parallel execution
            export_jobs = []
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
                export_jobs.append((task, workspace, task_feedback, source_report_path, source_report_title))

            # Run accepted factor exports in parallel
            if export_jobs:
                from concurrent.futures import ThreadPoolExecutor, as_completed

                max_workers = min(len(export_jobs), 10)
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = {
                        pool.submit(_export_single_factor, *job): job[0].factor_name
                        for job in export_jobs
                    }
                    for future in as_completed(futures):
                        factor_name = futures[future]
                        try:
                            if future.result():
                                exported_count += 1
                                print(f"paper_factor: exported {factor_name}.", flush=True)
                        except Exception as exc:
                            logger.warning(f"Failed to export factor {factor_name}: {exc}")

        self.trace.sync_dag_parent_and_hist((exp, feedback), prev_out[self.LOOP_IDX_KEY])
        logger.info(
            f"Factor report loop recorded. Accepted reviewed factor exports: {exported_count}. "
            f"Source report: {getattr(exp, 'source_report_title', 'unknown') if exp is not None else 'unknown'}."
        )


def _export_single_factor(task, workspace, task_feedback, source_report_path, source_report_title) -> bool:
    """Export a single accepted factor. Returns True if exported successfully."""
    try:
        _, df = workspace.execute("All")
    except (OSError, ValueError) as exc:
        logger.warning(
            f"Failed to reload report-derived factor dataframe for reviewed export: "
            f"task={task.factor_name}; {exc}"
        )
        return False
    if df is None or df.empty:
        return False
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

    # Also export to remote mount: /mnt/remote_e/paper_factors/文献因子/<ReportTitle>/
    remote_base = Path("/mnt/remote_e/paper_factors/文献因子")
    if remote_base.exists():
        try:
            report_title_slug = FactorFBWorkspace._sanitize_factor_name(
                source_report_title or Path(source_report_path).stem
            ) if (source_report_title or source_report_path) else "unknown_report"
            remote_dir = remote_base / report_title_slug
            remote_dir.mkdir(parents=True, exist_ok=True)

            factor_name = FactorFBWorkspace._sanitize_factor_name(str(df.columns[0]))
            remote_parquet = remote_dir / f"{factor_name}.parquet"
            df.to_parquet(remote_parquet, engine="pyarrow")

            remote_meta = remote_dir / f"{factor_name}.meta.json"
            metadata = {
                "factor_name": factor_name,
                "display_name": str(df.columns[0]),
                "factor_description": getattr(task, "factor_description", None),
                "factor_formulation": getattr(task, "factor_formulation", None),
                "variables": getattr(task, "variables", None),
                "rows": len(df),
                "non_null": int(df.iloc[:, 0].notna().sum()),
                "ic_score": full_sample_ic,
                "tags": tags,
                "export_time": datetime.now().isoformat(timespec="seconds"),
            }
            if source_report_title:
                metadata["source_report_title"] = source_report_title
            if source_report_path:
                metadata["source_report_path"] = str(source_report_path)
            remote_meta.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        except (OSError, ValueError) as exc:
            logger.warning(f"Failed to export factor to remote mount {remote_base}: {exc}")
    return True


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
