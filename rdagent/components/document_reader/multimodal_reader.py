from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any

import fitz
from dashscope import MultiModalConversation

from rdagent.log import rdagent_logger as logger

MULTIMODAL_DEFAULT_MODEL = "qwen3.5-omni-plus-2026-03-15"


def get_multimodal_model() -> str:
    return os.environ.get("PAPER_FACTOR_MULTIMODAL_MODEL", MULTIMODAL_DEFAULT_MODEL).strip()


def is_multimodal_available() -> bool:
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    return bool(api_key)


def _pdf_page_to_base64(pdf_path: str | Path, page_num: int, zoom: float = 1.5) -> str | None:
    try:
        doc = fitz.open(str(pdf_path))
        if page_num >= len(doc):
            doc.close()
            return None
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img_bytes = pix.tobytes("png")
        doc.close()
        return base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        logger.warning(f"Failed to convert PDF page {page_num} to image: {e}")
        return None


def _try_parse_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def extract_factors_from_pdf_multimodal(
    pdf_path: str | Path,
    docs_dict: dict[str, str] | None = None,
    pages_per_batch: int = 8,
    max_batches: int = 4,
    skip_first_pages: int = 3,
) -> dict[str, dict[str, Any]]:
    """Extract factors from a PDF using Qwen Omni multimodal model.

    Uses page images so the model can see formulas, tables, and charts.

    Args:
        pdf_path: Path to the PDF file.
        docs_dict: Unused, kept for API compatibility.
        pages_per_batch: Number of page images per API call.
        max_batches: Maximum number of batches.
        skip_first_pages: Number of cover/TOC pages to skip.

    Returns factor dict:
        {"FactorName": {"description": ..., "formulation": ..., "variables": {...}}}
    """
    if not is_multimodal_available():
        raise ValueError("Multimodal extraction is not available. Set DASHSCOPE_API_KEY.")

    model = get_multimodal_model()
    pdf_path = Path(pdf_path).resolve()

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    doc.close()
    logger.info(
        f"Extracting factors via multimodal {model}: {pdf_path.name} "
        f"({total_pages} pages, skipping first {skip_first_pages})"
    )

    all_factors: dict[str, dict[str, Any]] = {}
    batch_num = 0

    while batch_num < max_batches:
        start_page = skip_first_pages + batch_num * pages_per_batch
        if start_page >= total_pages:
            break

        # Build page images for this batch
        content_items = []
        end_page = min(start_page + pages_per_batch, total_pages)
        for i in range(start_page, end_page):
            img_b64 = _pdf_page_to_base64(pdf_path, i, zoom=1.5)
            if img_b64:
                content_items.append({"image": f"data:image/png;base64,{img_b64}"})

        if not content_items:
            break

        # Build prompt with page images
        if batch_num == 0:
            prompt = _build_first_prompt()
        else:
            prompt = _build_followup_prompt(list(all_factors.keys()))
        content_items.append({"text": prompt})

        logger.info(f"  Batch {batch_num + 1}: pages {start_page + 1}–{end_page} of {total_pages}")

        parsed = None
        for retry in range(2):
            if retry > 0:
                # Use retry-specific prompt
                content_items[-1] = {"text": _build_retry_prompt()}

            try:
                response = MultiModalConversation.call(
                    model=model,
                    messages=[{"role": "user", "content": content_items}],
                    max_tokens=8192,
                )
            except Exception as e:
                logger.warning(f"Multimodal API call failed in batch {batch_num + 1}: {e}")
                if retry > 0:
                    break
                continue

            if response.status_code != 200:
                logger.warning(f"Multimodal API error in batch {batch_num + 1}: {response}")
                if retry > 0:
                    break
                continue

            result_text = ""
            for item in response.output.choices[0].message.content:
                if "text" in item:
                    result_text += item["text"]

            parsed = _try_parse_json(result_text)
            if parsed is None:
                logger.info(f"  (unparseable, retry {retry + 1})")
                continue

            factors = parsed.get("factors", [])
            if isinstance(factors, list) and len(factors) > 0:
                break  # got factors, no need to retry

            logger.info(f"  (empty factors, retry {retry + 1})")
            parsed = None  # make sure parsed is None for retry check
            continue

        if parsed is None:
            batch_num += 1
            continue

        factors = parsed.get("factors", [])
        if not isinstance(factors, list):
            batch_num += 1
            continue

        for f in factors:
            if not isinstance(f, dict):
                continue
            name = str(f.get("factor_name") or f.get("因子名称") or "").strip()
            formula = str(f.get("factor_formulation") or f.get("数学公式") or f.get("formulation") or "").strip()
            if not name or not formula or name in all_factors:
                continue

            # Normalize variables to dict format
            raw_vars = f.get("variables") or f.get("变量定义") or f.get("变量") or {}
            if isinstance(raw_vars, list):
                var_dict = {}
                for v in raw_vars:
                    if isinstance(v, dict):
                        vname = str(v.get("variable_name") or v.get("name") or v.get("变量名") or "").strip()
                        vdesc = str(v.get("variable_description") or v.get("description") or v.get("变量描述") or "").strip()
                        if vname:
                            var_dict[vname] = vdesc
                        elif vdesc:
                            var_dict[str(len(var_dict))] = vdesc
                    elif isinstance(v, str) and ":" in v:
                        parts = v.split(":", 1)
                        var_dict[parts[0].strip()] = parts[1].strip()
                raw_vars = var_dict
            elif not isinstance(raw_vars, dict):
                raw_vars = {}

            all_factors[name] = {
                "description": str(f.get("factor_description") or f.get("因子描述") or f.get("description") or "").strip(),
                "formulation": formula,
                "variables": raw_vars,
            }

        logger.info(f"  Found {len(factors)} factor(s) in batch {batch_num + 1} (total: {len(all_factors)})")
        batch_num += 1

    logger.info(f"Total factors extracted via multimodal: {len(all_factors)}")
    for name in all_factors:
        logger.info(f"  - {name}")
    return all_factors


def _build_first_prompt() -> str:
    return (
        "请分析这些研报页面，找出其中定义的所有选股因子（选股因子是用来选股的量化指标，"
        "有明确的数学计算公式）。\n\n"

        "首先列出你找到的所有因子名称和公式，然后再用JSON格式输出。\n\n"

        "IC、IR、夏普比率、最大回撤等是评估指标，不是选股因子，不要列入。\n\n"

        "最终输出格式如下：\n"
        '{"factors": [{"factor_name": "因子英文名", "factor_description": "详细描述", "factor_formulation": "公式", "variables": {"变量名": "含义"}}]}\n\n'

        "请先思考再输出，确保每个因子都有完整的英文名称和数学公式。"
    )


def _build_followup_prompt(existing_factors: list[str]) -> str:
    existing_str = str(existing_factors) if existing_factors else "无"
    return (
        "这是研报的后续页面。之前已提取的因子：" + existing_str + "。\n"
        "请继续从这些新页面中找出其他选股因子。不要重复已提取的因子。\n\n"

        "先列出你找到的所有因子名称和公式，然后用JSON格式输出。\n"

        '{"factors": [{"factor_name": "英文名", "factor_description": "描述", "factor_formulation": "公式", "variables": {"变量名": "含义"}}]}'
    )


def _build_retry_prompt() -> str:
    return (
        "请重新仔细阅读这些页面。这些是研报的技术内容，应该包含选股因子定义。\n"
        "请找出所有有明确计算公式的因子，包括但不限于基本的财务因子。\n\n"

        "如果页面中有任何数学公式定义的指标，请全部提取出来。\n"
        "IC、IR等评估指标不要提取。\n\n"

        '{"factors": [{"factor_name": "...", "factor_description": "...", "factor_formulation": "...", "variables": {"变量名": "含义"}}]}'
    )
