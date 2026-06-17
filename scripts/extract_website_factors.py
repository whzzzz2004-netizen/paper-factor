#!/usr/bin/env python3
"""
从网站/公众号文章提取因子 — 输入 URL，LLM 创造性提取因子，保存到 extracted_reports/。

用法:
  # 从 sources.json 批量处理
  python scripts/extract_website_factors.py

  # 直接处理单条 URL
  python scripts/extract_website_factors.py --url "https://..." --title "文章标题"

  # 仅提取，不自动启动 pipeline（默认自动启动）
  python scripts/extract_website_factors.py --extract-only

输入格式 (papers/website/sources.json):
  [
    {"title": "段永平谈商业模式", "url": "https://...", "source": "公众号"},
    {"title": "2025年宏观展望", "url": "https://..."}
  ]

输出:
  - git_ignore_folder/factor_outputs/extracted_reports/{title}.extracted.json
  - papers/website/{title}.md (虚拟源文件，供 pipeline 引用)
  - 随后自动启动 CoSTEER 编码 + 测试 + 导出
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
SOURCES_FILE = PROJECT_ROOT / "papers" / "website" / "sources.json"
WEBSITE_DIR = PROJECT_ROOT / "papers" / "website"
EXTRACTED_DIR = PROJECT_ROOT / "git_ignore_folder" / "factor_outputs" / "extracted_reports"

# 已有的日频/分钟频数据列（供 LLM 参考）
AVAILABLE_DAILY_FIELDS = (
    "open, close, high, low, volume, factor(复权因子), pct_chg(涨跌幅), "
    "pre_close(前收盘价), turnover_rate(换手率), jhjj_hsl(集合竞价换手率)"
)
AVAILABLE_FUNDAMENTAL_FIELDS = (
    "roe, roa, pe_ttm, pb, revenue_yoy, profit_yoy, gross_margin, net_margin, "
    "debt_to_asset, ocf_per_share, market_cap, circulating_market_cap, "
    "total_shares, float_shares"
)
AVAILABLE_MINUTE_FIELDS = "open, close, high, low, volume, vwap, factor(复权因子), return(分钟收益率)"


EXTRACT_PROMPT_SYSTEM = """你是一个量化因子提取助手。用户会给出一篇财经文章/公众号内容，你需要从中提取**可量化的 alpha 因子**。

【核心原则】
- 即使文章不直接讨论量化因子，也要尝试从文中识别出**可量化的概念、比率、模式或指标**，将其构造成因子。
- 例如：文章讨论"护城河"→可以构造"毛利率 vs 行业毛利率"因子；讨论"现金流"→可以构造经营现金流/市值因子。
- 因子必须基于我们已有的数据字段计算。输出时说明每个因子的构造思路和所用字段。
- 宁多勿少：每篇文章至少提取 1 个因子，最多 5 个。实在无法提取也要说明理由。

【可用数据字段】
日频价量：{daily_fields}
日频基本面：{fundamental_fields}
分钟频：{minute_fields}

【因子构造规则】
1. description 必须写出从原始数据到最终因子值的完整计算步骤（至少 3 步）
2. formulation 用 LaTeX，必须引用上述可用数据字段，不能凭空创造字段
3. 必须指定 factor_type（daily_single / cross_section / minute / minute_cross_section）
4. 必须指定 lookback_days（需要多少天历史数据）
5. 如果文章概念无法直接用现有数据实现，在 description 中说明"近似方案"，不能跳过
6. 对非量化文章：优先提取 business concept → data proxy → 因子构造 的完整推理链路

输出 JSON schema:
{{
    "summary": "文章核心内容摘要",
    "construct_approach": "因子的整体构造思路（从文章概念到可计算因子的推理过程）",
    "factors": {{
        "factor_name": {{
            "description": "完整计算步骤",
            "formulation": "LaTeX 公式",
            "variables": {{"var": "变量说明"}},
            "factor_type": "daily_single",
            "lookback_days": 0,
            "special_conditions": "剔除停牌、ST",
            "source_excerpt": "文章中与因子概念相关的原文段落（直接引用）"
        }}
    }}
}}
"""


def is_content_garbled(text: str) -> bool:
    """检测内容是否为乱码（加密/编码后的不可读文本）。"""
    if len(text) < 100:
        return False
    # 统计中文字符比例
    cjk_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    total_printable = sum(1 for c in text if c.isprintable())
    if total_printable == 0:
        return True
    cjk_ratio = cjk_chars / total_printable
    # 正常中文文章：中文字符占比通常 > 20%
    # 乱码/加密文本：几乎无中文字符或字符极重复
    return cjk_ratio < 0.05


def fetch_content(url: str, timeout: int = 30) -> tuple[str, str]:
    """抓取网页内容并提取正文文本。

    Returns:
        (page_title, clean_text) — 失败时返回 ("", "")
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        # 尝试从 Content-Type 或 <meta> 中检测编码
        if resp.encoding and resp.encoding.lower() != "utf-8":
            try:
                resp.encoding = resp.apparent_encoding or "utf-8"
            except Exception:
                resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")

        # 移除无用标签
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
            tag.decompose()

        # 取 title
        title_tag = soup.find("title")
        page_title = title_tag.get_text(strip=True) if title_tag else ""

        # 取正文：优先 article 和 main，再 fallback 到 body
        for selector in ["article", "main", ".article", ".content", ".post-content", ".rich_media_content"]:
            content_div = soup.select_one(selector)
            if content_div:
                text = content_div.get_text(separator="\n", strip=True)
                break
        else:
            text = soup.get_text(separator="\n", strip=True)

        # 清理：压缩空行、去掉过短的行
        lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 10]
        clean_text = "\n".join(lines[:500])  # 最多 500 行

        result = f"标题: {page_title}\n\n来源: {url}\n\n正文:\n{clean_text}"

        # 乱码检测
        if is_content_garbled(clean_text):
            print(f"  ⚠️ 页面内容疑似乱码/加密（中文字符比例过低），可能被反爬拦截", flush=True)
            return page_title, ""

        return page_title, result

    except requests.RequestException as e:
        print(f"  ⚠️ 抓取失败: {e}", flush=True)
        return "", ""
    except Exception as e:
        print(f"  ⚠️ 解析失败: {e}", flush=True)
        return "", ""


def extract_factors_with_llm(content: str, url: str) -> dict:
    """调用 LLM 从文章内容中提取因子。"""
    from rdagent.oai.llm_utils import APIBackend
    backend = APIBackend()

    # 构建 prompt
    system_prompt = EXTRACT_PROMPT_SYSTEM.format(
        daily_fields=AVAILABLE_DAILY_FIELDS,
        fundamental_fields=AVAILABLE_FUNDAMENTAL_FIELDS,
        minute_fields=AVAILABLE_MINUTE_FIELDS,
    )
    user_prompt = (
        f"请从以下文章内容中提取可量化的 alpha 因子。\n"
        f"即使文章没有直接讨论因子，也请尝试从文中概念推断出可计算的因子。\n\n"
        f"URL: {url}\n\n"
        f"文章内容:\n{content[:15000]}"  # 限制长度
    )

    try:
        from typing import Dict, Any
        response = backend.build_messages_and_create_chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=True,
            json_target_type=Dict[str, Any],
        )
        if response is None:
            print("  ⚠️ LLM 返回为空", flush=True)
            return {"summary": "", "factors": {}}

        # 尝试解析 JSON
        text = response if isinstance(response, str) else getattr(response, "content", str(response))
        # 提取 JSON 部分
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            text = text.rsplit("```", 1)[0]
        text = text.strip().lstrip("json").strip()

        result = json.loads(text)
        factors = result.get("factors", {})
        if isinstance(factors, dict):
            # 过滤掉无效因子
            valid = {}
            for name, item in factors.items():
                if isinstance(item, dict) and item.get("description"):
                    valid[name] = item
            result["factors"] = valid
        return result
    except json.JSONDecodeError as e:
        print(f"  ⚠️ LLM 返回非 JSON: {e}", flush=True)
        print(f"    原始返回: {str(text)[:200]}", flush=True)
        return {"summary": "", "factors": {}}
    except Exception as e:
        print(f"  ⚠️ LLM 调用失败: {e}", flush=True)
        return {"summary": "", "factors": {}}


def save_extracted_factors(title: str, url: str, source: str, extract_result: dict) -> Path | None:
    """保存提取结果到 extracted_reports/，同时创建虚拟源文件。"""
    from rdagent.oai.llm_utils import md5_hash

    if not extract_result.get("factors"):
        print(f"  ⚠️ 未提取到有效因子", flush=True)
        return None

    # 加来源前缀 "website__"，与研报 PDF 提取区分
    slug_base = "".join(c if c.isalnum() or c in ("-_") else "_" for c in title).strip("_")[:80]
    slug = f"website__{slug_base}"

    # 虚拟源文件（供 pipeline 引用）
    source_path = WEBSITE_DIR / f"{slug}.md"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        f"# {title}\n\n"
        f"来源: {source}\n"
        f"URL: {url}\n"
        f"提取时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"## 提取摘要\n{extract_result.get('summary', '')}\n\n"
        f"## 构造思路\n{extract_result.get('construct_approach', '')}\n"
    )

    # 转换为 extracted.json 格式
    factors_out = {}
    for name, item in extract_result["factors"].items():
        factor_type = item.get("factor_type", "daily_single")
        factors_out[name] = {
            "description": str(item.get("description", "")),
            "formulation": str(item.get("formulation", "")),
            "variables": item.get("variables", {}) if isinstance(item.get("variables"), dict) else {},
            "factor_type": factor_type if factor_type in (
                "daily_single", "cross_section", "minute", "minute_cross_section", "deep_learning"
            ) else "daily_single",
            "lookback_days": int(item.get("lookback_days", 0)),
            "special_conditions": str(item.get("special_conditions", "")),
            "source_excerpt": f"[来源: {source}] {item.get('source_excerpt', title)}",
        }

    # 保存 .extracted.json
    # pdf_reader 必须匹配 get_paper_factor_pdf_reader_mode() 的返回值
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    extracted_path = EXTRACTED_DIR / f"{slug}.extracted.json"
    payload = {
        "report_file_path": str(source_path.resolve()),
        "report_title": f"[网站] {title}",
        "pdf_reader": "pymupdf",
        "extract_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_url": url,
        "source_type": source or "website",
        "hash": md5_hash(str(factors_out)),
        "summary": extract_result.get("summary", ""),
        "construct_approach": extract_result.get("construct_approach", ""),
        "factors": factors_out,
    }
    extracted_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  ✅ 已保存: {extracted_path.name}", flush=True)

    return source_path


def process_url(title: str, url: str = "", source: str = "", text: str = "",
                extract_only: bool = False) -> bool:
    """处理单个条目：URL 抓取或直接文本 → LLM 提取 → 保存。

    Args:
        title: 文章标题。
        url: 网页 URL（可选，有 URL 优先抓取）。
        source: 来源（公众号/知乎/雪球等）。
        text: 直接文本内容（当 URL 不可用时手动粘贴）。
        extract_only: 仅提取，不启动 pipeline。
    """
    source = source or "website"
    print(f"\n{'='*60}", flush=True)
    print(f"  来源: {source}", flush=True)
    print(f"  标题: {title}", flush=True)
    if url:
        print(f"  URL:  {url}", flush=True)
    print(f"{'='*60}", flush=True)

    # 1. 获取内容
    content = ""
    page_title = ""
    if url:
        print(f"  抓取页面...", flush=True)
        page_title, content = fetch_content(url)
    if not content and text:
        print(f"  使用手动粘贴的文本...", flush=True)
        content = text
    if not content:
        print(f"  ❌ 无可用内容（网址无法抓取时请在 sources.json 中添加 text 字段）", flush=True)
        return False
    print(f"  内容长度: {len(content)} 字符", flush=True)

    # 2. LLM 提取
    print(f"  LLM 提取因子（可能需要 10-30 秒）...", flush=True)
    extract_result = extract_factors_with_llm(content, url or title)
    factor_count = len(extract_result.get("factors", {}))
    print(f"  提取到 {factor_count} 个因子", flush=True)

    # 3. 保存
    source_path = save_extracted_factors(title, url, source, extract_result)
    if source_path is None:
        return False

    # 4. 打印结果
    for fname, finfo in extract_result.get("factors", {}).items():
        print(f"    [{finfo.get('factor_type', '?')}] {fname}", flush=True)
        desc = finfo.get("description", "")[:120]
        if desc:
            print(f"      {desc}...", flush=True)

    # 5. 自动启动 pipeline
    if not extract_only and factor_count > 0:
        print(f"\n  启动 CoSTEER 编码流程...", flush=True)
        return run_pipeline(str(source_path))

    return True


def run_pipeline(source_path: str) -> bool:
    """调用 factor_from_report pipeline 进行 CoSTEER 编码 → 测试 → 导出。"""
    try:
        from rdagent.app.qlib_rd_loop.factor_from_report import main as run_factor_pipeline

        # 用环境变量控制 pipeline 行为
        os.environ.setdefault("RDAGENT_PAPER_FACTOR_SKIP_LOW_IC_REPAIR", "1")
        os.environ.setdefault("RDAGENT_PAPER_FACTOR_FAST", "1")
        os.environ.setdefault("QLIB_FACTOR_MAX_FACTORS_PER_EXP", "5")
        os.environ.setdefault("LOG_LLM_CHAT_CONTENT", "False")

        run_factor_pipeline(
            report_paths=[source_path],
            minimal_mode=True,
        )
        return True
    except Exception as e:
        print(f"  ⚠️ Pipeline 执行失败: {e}", flush=True)
        return False


def load_sources() -> list[dict]:
    """读取 sources.json。"""
    if not SOURCES_FILE.exists():
        WEBSITE_DIR.mkdir(parents=True, exist_ok=True)
        # 创建模板
        template = [
            {"title": "示例文章", "url": "https://example.com/article", "source": "公众号"},
            {"title": "手动粘贴", "text": "在此粘贴文章内容...", "source": "知乎"},
        ]
        SOURCES_FILE.write_text(json.dumps(template, indent=2, ensure_ascii=False))
        print(f"已创建模板文件: {SOURCES_FILE}")
        print(f"请编辑后重新运行")
        return []
    try:
        data = json.loads(SOURCES_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return [data]
        return data
    except (json.JSONDecodeError, OSError) as e:
        print(f"❌ 读取 sources.json 失败: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="从网站/公众号提取量化因子")
    parser.add_argument("--url", help="单条 URL（不传则从 sources.json 读取）")
    parser.add_argument("--title", default="", help="文章标题")
    parser.add_argument("--source", default="", help="来源（公众号/知乎/雪球等）")
    parser.add_argument("--text", default="", help="直接粘贴的文章内容（URL 不可用时）")
    parser.add_argument("--extract-only", action="store_true",
                        help="仅提取，不启动 CoSTEER pipeline")
    args = parser.parse_args()

    if args.url:
        title = args.title or args.url.split("/")[-1][:50]
        process_url(title, url=args.url, source=args.source, extract_only=args.extract_only)
    elif args.text:
        title = args.title or "手动输入"
        process_url(title, text=args.text, source=args.source, extract_only=args.extract_only)
    else:
        sources = load_sources()
        if not sources:
            return
        for i, entry in enumerate(sources):
            title = entry.get("title", f"来源_{i}")
            url = entry.get("url", "")
            source = entry.get("source", "")
            text = entry.get("text", "")
            if not url and not text:
                print(f"⚠️ 第 {i+1} 条缺少 url 和 text，跳过", flush=True)
                continue
            process_url(title, url=url, source=source, text=text, extract_only=args.extract_only)
            print(flush=True)


if __name__ == "__main__":
    main()
