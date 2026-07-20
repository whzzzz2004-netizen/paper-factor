#!/usr/bin/env python3
"""
Claude Code /factor skill 辅助脚本。
子命令：extract-pdf, wrap-template, run-test, export-factor, test-and-export, trigger-full, wait-full
"""

import argparse
import json
import os
import re

# Factor Memory Bank 路径
_FACTOR_MEMORY_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    ".claude/projects/-home-dministrator-paper-factor/memory/factor_memory.json"
)

# Domain Knowledge RAG
_DOMAIN_KNOWLEDGE_RAG = None
def _get_domain_rag():
    global _DOMAIN_KNOWLEDGE_RAG
    if _DOMAIN_KNOWLEDGE_RAG is None:
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from domain_knowledge_rag import DomainKnowledgeRAG
        _DOMAIN_KNOWLEDGE_RAG = DomainKnowledgeRAG()
        _DOMAIN_KNOWLEDGE_RAG.load_chunks_and_build()
    return _DOMAIN_KNOWLEDGE_RAG

def retrieve_domain_knowledge(query, top_k=3, min_score=0.01):
    """从领域知识库检索与 query 相关的知识块。

    知识库包含：A 股涨停规则、板块代码区间、数据列含义、因子编码惯例等。
    在编码因子时注入 LLM prompt，帮助 LLM 理解 A 股市场特殊规则。

    Args:
        query: 检索查询（如 "涨停阈值判断"、"怎么计算收益率"）
        top_k: 返回最多几个知识块
        min_score: 最低相似度阈值
    Returns:
        str: 格式化的知识文本，无匹配时返回空字符串
    """
    try:
        rag = _get_domain_rag()
        results = rag.retrieve(query, top_k=top_k, min_score=min_score)
        if not results:
            return ""
        lines = ["## 领域知识参考（RAG）"]
        for r in results:
            lines.append(f"### {r['source']} > {r['heading']} (relevance={r['score']})")
            lines.append(r['text'])
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return f"<!-- RAG 检索失败: {e} -->"

def _load_factor_memory():
    """加载因子记忆库"""
    path = os.path.abspath(_FACTOR_MEMORY_PATH)
    if not os.path.exists(path):
        return []
    try:
        return json.load(open(path, encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

def _save_factor_memory(memory):
    """保存因子记忆库"""
    path = os.path.abspath(_FACTOR_MEMORY_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(memory, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def find_similar_factors(factor_type, cols=None, lookback=None, top_k=3):
    """从记忆库中查找同类高分因子，返回格式化的参考文本

    Args:
        factor_type: 因子类型 (minute, daily, cross_section, minute_cs)
        cols: 使用的数据列列表 (可选，用于更精确匹配)
        lookback: lookback天数 (可选)
        top_k: 返回最多几个参考
    Returns:
        str: 格式化的参考文本，无匹配时返回空字符串
    """
    memory = _load_factor_memory()
    if not memory:
        return ""

    # 筛选同类因子，且 |alpha_tstat| > 2.0 的才算有效参考
    candidates = [
        m for m in memory
        if m.get("factor_type") == factor_type
        and abs(m.get("alpha_tstat") or 0) > 2.0
    ]

    if not candidates:
        return ""

    # 按 |alpha_tstat| 排序
    candidates.sort(key=lambda x: abs(x.get("alpha_tstat", 0) or 0), reverse=True)

    lines = ["## 参考记忆（同类高分因子）", "以下是与当前因子同类型的、效果较好的历史因子，可供参考其做法："]
    for m in candidates[:top_k]:
        name = m.get("name", "?")
        ic = m.get("ic_mean", "?")
        ir = m.get("ic_ir", "?")
        at = m.get("alpha_tstat", "?")
        desc = m.get("description", "") or ""
        form = m.get("formulation", "") or ""
        ic_str = f"{ic:.4f}" if isinstance(ic, (int, float)) else "?"
        ir_str = f"{ir:.3f}" if isinstance(ir, (int, float)) else "?"
        at_str = f"{at:.2f}" if isinstance(at, (int, float)) else "?"
        desc_str = f" — {desc[:120]}" if desc else ""
        lines.append(f"- {name}: IC={ic_str}, IR={ir_str}, Alpha t={at_str}{desc_str}")
        if form:
            lines.append(f"  formulation: {form[:200]}")

    lines.append("注意：如果上述参考与当前因子的原始描述不符，严格按照原始描述实现，不要照搬参考。")
    return "\n".join(lines)

def update_factor_memory(factor_name, report_name, factor_type, meta_path, full_meta_path):
    """因子全量运行完成后，记录到记忆库"""
    memory = _load_factor_memory()

    # 读取测试阶段 meta.json（含 description / formulation）
    test_meta = {}
    if meta_path and os.path.isfile(meta_path):
        try:
            test_meta = json.load(open(meta_path, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    # 读取全量 meta.json
    full_meta = {}
    if full_meta_path and os.path.isfile(full_meta_path):
        try:
            full_meta = json.load(open(full_meta_path, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    ev = full_meta.get("evaluation") or {}

    # description：优先 test_meta.description → test_meta.factor_description → full_meta.factor_description
    description = (
        test_meta.get("description")
        or test_meta.get("factor_description")
        or full_meta.get("factor_description")
        or ""
    )
    # formulation：优先 test_meta.formulation → test_meta.factor_formulation → full_meta.factor_formulation
    formulation = (
        test_meta.get("formulation")
        or test_meta.get("factor_formulation")
        or full_meta.get("factor_formulation")
        or ""
    )
    source_excerpt = test_meta.get("source_excerpt") or full_meta.get("source_excerpt") or ""

    record = {
        "name": factor_name,
        "report": report_name,
        "factor_type": factor_type,
        "description": description,
        "formulation": formulation,
        "source_excerpt": source_excerpt,
        "ic_mean": ev.get("ic_mean"),
        "rank_ic_mean": ev.get("rank_ic_mean"),
        "ic_ir": ev.get("ic_ir"),
        "alpha_tstat": full_meta.get("barra_analysis", {}).get("exposures", {}).get("alpha", {}).get("tstat"),
        "rows": full_meta.get("rows"),
        "stock_count": full_meta.get("stock_count"),
        "updated_at": full_meta.get("updated_at", ""),
    }

    # 更新或追加
    found = False
    for i, m in enumerate(memory):
        if m.get("name") == factor_name:
            memory[i] = record
            found = True
            break
    if not found:
        memory.append(record)

    # 按 |alpha_tstat| 排序
    memory.sort(key=lambda x: abs(x.get("alpha_tstat") or 0), reverse=True)
    _save_factor_memory(memory)
    return record
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
LITERATURE_REPORTS_DIR = PROJECT_ROOT / "git_ignore_folder" / "factor_outputs" / "literature_reports"

# ── 远程 E 盘配置 ──
SMB_HOST = "192.168.1.13"
SMB_SHARE = "E"
SMB_USER = "pc"
SMB_PASS = "123456"
CIFS_MOUNT = Path("/mnt/remote_e")


def _sudo_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    if "PYTHON_RUN_AS_ROOT" in os.environ:
        return subprocess.run(cmd, **kwargs)
    try:
        return subprocess.run(["sudo", "-n"] + cmd, **kwargs)
    except Exception:
        pass
    kwargs.pop("input", None)
    return subprocess.run(
        ["sudo", "-S"] + cmd,
        input=f"{SMB_PASS}\n".encode(),
        **kwargs,
    )


def _ensure_remote_mounted() -> bool:
    if CIFS_MOUNT.exists() and any(CIFS_MOUNT.iterdir()):
        return True
    print("  ⏳ 自动挂载远程 E 盘 ...", flush=True)
    try:
        CIFS_MOUNT.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"  ❌ 无法创建挂载点: {e}", flush=True)
        return False
    _sudo_run(["modprobe", "cifs"], capture_output=True, timeout=10)
    _sudo_run(["apt", "install", "-y", "cifs-utils"], capture_output=True, timeout=120)
    _base_opts = f"user={SMB_USER},password={SMB_PASS},uid={os.getuid()},gid={os.getgid()},file_mode=0644,dir_mode=0755,iocharset=utf8,noperm"
    for _vers in ("3.0", "2.1", "2.0", "1.0"):
        r = _sudo_run(
            ["mount", "-t", "cifs", f"//{SMB_HOST}/{SMB_SHARE}", str(CIFS_MOUNT), "-o", f"vers={_vers},{_base_opts}"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            print(f"  ✅ 已挂载 (vers={_vers})", flush=True)
            return True
    print(f"  ⚠️ 挂载失败，回退本地数据", flush=True)
    return False


def _detect_data_dir() -> Path:
    """Detect best available data directory (remote preferred, local fallback)."""
    candidates = [
        os.environ.get("FACTOR_DATA_DIR", ""),
        os.environ.get("RDAGENT_FACTOR_DATA_DIR", ""),
        "/mnt/remote_e/_paper_factor_unified/factor_implementation_source_data",
        str(PROJECT_ROOT / "git_ignore_folder" / "factor_implementation_source_data_1000"),
        str(PROJECT_ROOT / "git_ignore_folder" / "factor_implementation_source_data"),
        "E:\\_paper_factor_unified\\factor_implementation_source_data",
        "Z:\\_paper_factor_unified\\factor_implementation_source_data",
        "\\\\192.168.1.13\\E\\_paper_factor_unified\\factor_implementation_source_data",
    ]
    for p in candidates:
        if p and (Path(p) / "stock_data" / "daily").exists():
            return Path(p)
    # 都不存在 → 尝试自动挂载再查
    print("  ⏳ 未找到数据目录，尝试自动挂载远程 E 盘...", flush=True)
    if _ensure_remote_mounted():
        for p in candidates:
            if p and (Path(p) / "stock_data" / "daily").exists():
                return Path(p)
    return PROJECT_ROOT / "git_ignore_folder" / "factor_implementation_source_data_1000"


TEST_DATA_DIR = _detect_data_dir()

# Template type → attribute name on FactorFBWorkspace
TYPE_MAP = {
    "daily_single": "DAILY_FRAMEWORK_TEMPLATE",
    "cross_section": "CROSS_SECTION_FRAMEWORK_TEMPLATE",
    "minute": "MINUTE_FRAMEWORK_TEMPLATE",
    "minute_cross_section": "MINUTE_CROSS_SECTION_FRAMEWORK_TEMPLATE",
    "deep_learning": "DEEP_LEARNING_FRAMEWORK_TEMPLATE",
}


# ---------------------------------------------------------------------------
# Subcommand: extract-pdf
# ---------------------------------------------------------------------------
def cmd_extract_pdf(args):
    """Extract text from PDFs (pymupdf) or .md files, output JSON."""
    import fitz  # pymupdf

    result = {}
    for p in args.paths:
        path = Path(p).resolve()
        if not path.exists():
            result[str(path)] = ""
            continue
        if path.suffix == ".md":
            try:
                result[str(path)] = path.read_text(encoding="utf-8")
            except Exception:
                result[str(path)] = ""
        elif path.is_dir():
            for pdf_path in sorted(path.rglob("*.pdf")):
                try:
                    doc = fitz.open(pdf_path)
                    text = "\n".join(page.get_text() for page in doc)
                    result[str(pdf_path)] = text
                except Exception:
                    result[str(pdf_path)] = ""
        else:
            try:
                doc = fitz.open(path)
                text = "\n".join(page.get_text() for page in doc)
                result[str(path)] = text
            except Exception:
                result[str(path)] = ""
    print(json.dumps(result, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Subcommand: extract-website — 只抓取内容，不调 LLM
# ---------------------------------------------------------------------------
def cmd_extract_website(args):
    """Fetch URL content, output as JSON (no LLM). Claude does the extraction."""

    def _is_content_garbled(text: str) -> bool:
        if len(text) < 100:
            return False
        cjk_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_printable = sum(1 for c in text if c.isprintable())
        if total_printable == 0:
            return True
        return cjk_chars / total_printable < 0.05

    def _fetch_content(url: str, timeout: int = 30):
        import requests as _req
        from bs4 import BeautifulSoup as _BS
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        try:
            resp = _req.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            try:
                resp.encoding = resp.apparent_encoding or "utf-8"
            except Exception:
                resp.encoding = "utf-8"
            soup = _BS(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
                tag.decompose()
            title_tag = soup.find("title")
            page_title = title_tag.get_text(strip=True) if title_tag else ""
            for selector in ["article", "main", ".article", ".content", ".post-content", ".rich_media_content"]:
                content_div = soup.select_one(selector)
                if content_div:
                    text = content_div.get_text(separator="\n", strip=True)
                    break
            else:
                text = soup.get_text(separator="\n", strip=True)
            lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 10]
            clean_text = "\n".join(lines[:500])
            if _is_content_garbled(clean_text):
                print(f"  ⚠️ 页面内容疑似乱码/加密（中文字符比例过低）", flush=True)
                return page_title, ""
            return page_title, f"标题: {page_title}\n\n来源: {url}\n\n正文:\n{clean_text}"
        except Exception as e:
            print(f"  ⚠️ 抓取/解析失败: {e}", flush=True)
            return "", ""

    def _guess_source_from_url(url: str) -> str:
        from urllib.parse import urlparse
        try:
            netloc = urlparse(url).netloc.lower()
            known = {
                "mp.weixin.qq.com": "公众号", "weixin.qq.com": "公众号",
                "zhuanlan.zhihu.com": "知乎专栏", "zhihu.com": "知乎",
                "xueqiu.com": "雪球", "www.jianshu.com": "简书",
            }
            for k, v in known.items():
                if k in netloc:
                    return v
            return netloc.replace("www.", "")
        except Exception:
            return ""

    sources_json = PROJECT_ROOT / "papers" / "website" / "sources.json"
    if not sources_json.exists():
        print(json.dumps({"success": False, "error": "sources.json not found"}))
        return 1

    try:
        sources = json.loads(sources_json.read_text())
    except Exception as e:
        print(json.dumps({"success": False, "error": f"Failed to read sources.json: {e}"}))
        return 1

    idx = args.index
    if idx < 0 or idx >= len(sources):
        print(json.dumps({"success": False, "error": f"Index {idx} out of range (0-{len(sources)-1})"}))
        return 1

    src = _normalize_source(sources[idx], idx)
    url = src.get("url", "")
    title = src.get("title", "")
    source = src.get("source", "") or _guess_source_from_url(url)

    if not url:
        print(json.dumps({"success": False, "error": "No URL in sources.json entry"}))
        return 1

    print(f"  抓取: {title or url}", flush=True)
    page_title, content = _fetch_content(url)
    if not content:
        text = src.get("text", "")
        if text:
            print(f"  使用 sources.json 中的 text 字段", flush=True)
            content = f"标题: {title}\n\n来源: {url}\n\n正文:\n{text}"
        else:
            print(json.dumps({"success": False, "error": f"Failed to fetch: {url}"}))
            return 1

    # 输出内容（Claude 自行分析提取因子）
    result = {
        "success": True,
        "index": idx,
        "url": url,
        "title": title or page_title or url,
        "source": source,
        "content_length": len(content),
        "content": content[:30000],  # 不超过 30000 字符
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: wrap-template
# ---------------------------------------------------------------------------
def cmd_wrap_template(args):
    """Wrap user code into a full .code.py using the framework template."""
    # Lazy import to avoid heavy dependency chain
    sys.path.insert(0, str(PROJECT_ROOT))
    from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace

    type_key = args.type
    if type_key not in TYPE_MAP:
        print(f"ERROR: unknown type '{type_key}'. Choose from: {', '.join(TYPE_MAP)}", file=sys.stderr)
        return 1

    template = getattr(FactorFBWorkspace, TYPE_MAP[type_key])
    user_code = Path(args.code).read_text(encoding="utf-8")
    lookback = args.lookback
    load_cols = [c.strip() for c in args.cols.split(",")] if args.cols else None

    full_code = FactorFBWorkspace._build_factor_code(template, user_code, lookback, load_cols)

    output = Path(args.output) if args.output else None
    if output:
        output.write_text(full_code, encoding="utf-8")
        print(json.dumps({"output": str(output), "type": type_key, "lookback": lookback}))
    else:
        print(full_code)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: run-test
# ---------------------------------------------------------------------------
def cmd_run_test(args):
    """Run a .code.py against the 300-stock test dataset."""
    code_path = Path(args.code).resolve()
    if not code_path.exists():
        print(f"ERROR: code file not found: {code_path}", file=sys.stderr)
        return 1

    test_result = _run_test_in_tmpdir(code_path, timeout=args.timeout)

    # Cleanup tmpdir
    try:
        shutil.rmtree(test_result["tmpdir"], ignore_errors=True)
    except Exception:
        pass

    output = {k: v for k, v in test_result.items() if k != "tmpdir"}
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: export-factor
# ---------------------------------------------------------------------------
def cmd_export_factor(args):
    """Export factor to literature_reports directory structure."""
    code_path = Path(args.code).resolve()
    if not code_path.exists():
        print(f"ERROR: code file not found: {code_path}", file=sys.stderr)
        return 1

    report_name = args.report
    factor_name = args.factor

    factor_dir = LITERATURE_REPORTS_DIR / report_name / factor_name
    factor_dir.mkdir(parents=True, exist_ok=True)

    # Copy code.py
    dst_code = factor_dir / f"{factor_name}.code.py"
    shutil.copy2(code_path, dst_code)

    # Copy result.parquet if in same dir
    src_result = code_path.parent / "result.parquet"
    if src_result.exists():
        dst_result = factor_dir / f"{factor_name}.parquet"
        shutil.copy2(src_result, dst_result)
        has_result = True
    else:
        has_result = False

    # Build meta.json
    meta = {
        "factor_name": factor_name,
        "source_report": report_name,
        "created_at": __import__("datetime").datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "source_type": "literature_report",
        "factor_description": args.description or "",
        "factor_formulation": args.formulation or "",
        "source_report_title": args.source_report_title or "",
        "source_report_path": args.source_report_path or "",
        "source_excerpt": args.source_excerpt or "",
    }
    # Merge extra meta from --meta-json
    if args.meta_json:
        extra = json.loads(args.meta_json)
        # 避免 --meta-json 覆盖显式参数
        extra.pop("factor_description", None)
        extra.pop("factor_formulation", None)
        extra.pop("source_report_title", None)
        extra.pop("source_report_path", None)
        extra.pop("source_excerpt", None)
        meta.update(extra)

    dst_meta = factor_dir / f"{factor_name}.meta.json"
    dst_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    output = {
        "exported": True,
        "code": str(dst_code),
        "parquet": str(factor_dir / f"{factor_name}.parquet") if has_result else None,
        "meta": str(dst_meta),
        "directory": str(factor_dir),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Auto-detection helpers
# ---------------------------------------------------------------------------
# Function name → template type
_FUNC_TYPE_MAP = {
    "calc_factor_single_stock": "daily_single",
    "calc_factor_cross_section": "cross_section",
    "calc_factors_one_day": "minute",
    "calc_factor_minute_raw": "minute_cross_section",
    "cross_section_transform": "minute_cross_section",  # secondary
    "train_model": "deep_learning",
    "predict": "deep_learning",  # secondary
}


def detect_type_from_code(code: str) -> str | None:
    """Detect template type from function definitions in user code."""
    for func_name, type_key in _FUNC_TYPE_MAP.items():
        if re.search(r"\bdef\s+" + re.escape(func_name) + r"\s*\(", code):
            return type_key
    return None


def detect_lookback_from_code(code: str, default: int = 250) -> int:
    """Detect lookback from rolling/shifting window sizes in user code."""
    nums = set()
    for pattern in [
        r"\.rolling\s*\(\s*(\d+)\s*\)",       # .rolling(N)
        r"\.shift\s*\(\s*(\d+)\s*\)",           # .shift(N)
        r"\.diff\s*\(\s*(\d+)\s*\)",            # .diff(N)
        r"window\s*=\s*(\d+)",                  # window=N
        r"periods\s*=\s*(\d+)",                 # periods=N
        r"\.ewm\s*\([^)]*span\s*=\s*(\d+)",     # .ewm(span=N)
    ]:
        for m in re.finditer(pattern, code):
            nums.add(int(m.group(1)))
    if nums:
        return max(nums) + 10  # add buffer
    return default


# ---------------------------------------------------------------------------
# Subcommand: test-and-export
# ---------------------------------------------------------------------------
# 测试阶段非空值比率阈值：低于此值时认为因子输出全为空，触发代码重新检查
_NON_NULL_THRESHOLD = 0.01  # 1%


def _run_test_in_tmpdir(code_path: Path, timeout: int = 3600) -> dict:
    """Run a .code.py in a temp dir and return result dict + temp dir path."""
    env = os.environ.copy()
    env["FACTOR_DATA_DIR"] = str(TEST_DATA_DIR)
    env["FACTOR_N_WORKERS"] = "4"
    # PyTorch CPU may reference Intel VTune JIT profiling symbols (iJIT_NotifyEvent etc.)
    # that are missing on some systems; preload a stub to satisfy them.
    _itt_stub = Path(__file__).parent.parent / "lib" / "libittnotify_stub.so"
    if not _itt_stub.exists():
        _itt_stub = Path("/tmp/libittnotify_stub.so")
    if _itt_stub.exists():
        env["LD_PRELOAD"] = str(_itt_stub)

    tmpdir = tempfile.mkdtemp(prefix="factor_test_")
    try:
        proc = subprocess.run(
            [sys.executable, str(code_path)],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        returncode = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as e:
        returncode = -1
        stdout = e.stdout or ""
        stderr = e.stderr or ""

    result_path = Path(tmpdir) / "result.parquet"
    result_info = {}
    if result_path.exists():
        import pandas as pd
        df = pd.read_parquet(result_path)
        non_null_ratio = round(float(df.notna().mean().mean()), 4)
        result_info = {
            "result_exists": True,
            "result_shape": list(df.shape),
            "date_range": [str(df.index.min()), str(df.index.max())] if len(df) > 0 else None,
            "non_null_ratio": non_null_ratio,
        }
        # 全空值检查：如果非空比率低于阈值，视为因子逻辑有问题，需要重新检查代码
        if non_null_ratio < _NON_NULL_THRESHOLD:
            result_info["all_nan_warning"] = (
                f"因子输出几乎全为空值 (non_null_ratio={non_null_ratio:.4f} < {_NON_NULL_THRESHOLD})。"
                f"这通常意味着因子代码逻辑有问题（如数据列不存在、计算结果全为NaN、"
                f"或条件过滤过于严格导致无有效记录）。请仔细检查核心计算函数，"
                f"确保输入列名正确、计算过程无除零或NaN传播、筛选条件合理。"
            )
    else:
        result_info = {"result_exists": False}

    success = returncode == 0 and result_info.get("result_exists", False)
    if success and result_info.get("all_nan_warning"):
        success = False

    return {
        "tmpdir": tmpdir,
        "success": success,
        "returncode": returncode,
        "stdout_tail": stdout[-2000:] if stdout else "",
        "stderr_tail": stderr[-2000:] if stderr else "",
        **result_info,
    }


def cmd_test_and_export(args):
    """One-shot: wrap template → test → export. Auto-detect type and lookback."""
    code_path = Path(args.code).resolve()
    if not code_path.exists():
        print(json.dumps({"success": False, "error": f"code file not found: {code_path}"}))
        return 1

    user_code = code_path.read_text(encoding="utf-8")

    # Auto-detect type
    type_key = args.type
    if not type_key:
        type_key = detect_type_from_code(user_code)
    if not type_key:
        print(json.dumps({"success": False, "error": (
            "Cannot auto-detect type from function names. "
            "Please specify --type (daily_single, cross_section, minute, minute_cross_section, deep_learning)"
        )}))
        return 1

    # Auto-detect lookback：优先显式指定 → 其次 meta-json → 最后代码检测
    lookback = args.lookback
    if not lookback and args.meta_json:
        try:
            meta = json.loads(args.meta_json)
            lookback = meta.get("lookback_days")
        except (json.JSONDecodeError, AttributeError):
            lookback = None
    if not lookback:
        lookback = detect_lookback_from_code(user_code)

    # Wrap template
    sys.path.insert(0, str(PROJECT_ROOT))
    from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace

    template = getattr(FactorFBWorkspace, TYPE_MAP[type_key])
    load_cols = [c.strip() for c in args.cols.split(",")] if args.cols else None
    full_code = FactorFBWorkspace._build_factor_code(template, user_code, lookback, load_cols)

    # Write wrapped code to temp file
    wrapped_tmp = Path(tempfile.mktemp(suffix=".code.py", prefix="factor_wrapped_"))
    wrapped_tmp.write_text(full_code, encoding="utf-8")

    # Run test
    test_result = _run_test_in_tmpdir(wrapped_tmp, timeout=args.timeout)

    if not test_result["success"]:
        # Cleanup
        try:
            wrapped_tmp.unlink(missing_ok=True)
            shutil.rmtree(test_result["tmpdir"], ignore_errors=True)
        except Exception:
            pass
        extra_keys = ("result_exists", "result_shape", "non_null_ratio", "all_nan_warning")
        print(json.dumps({
            "success": False,
            "detected_type": type_key,
            "detected_lookback": lookback,
            "stderr_tail": test_result["stderr_tail"],
            "stdout_tail": test_result["stdout_tail"],
            "returncode": test_result["returncode"],
            **{k: v for k, v in test_result.items() if k in extra_keys},
        }, ensure_ascii=False, indent=2))
        return 0

    # Success → export
    report_name = args.report
    factor_name = args.factor
    factor_dir = LITERATURE_REPORTS_DIR / report_name / factor_name
    factor_dir.mkdir(parents=True, exist_ok=True)

    dst_code = factor_dir / f"{factor_name}.code.py"
    shutil.copy2(wrapped_tmp, dst_code)

    # Copy result.parquet from test tmpdir
    src_result = Path(test_result["tmpdir"]) / "result.parquet"
    has_result = src_result.exists()
    if has_result:
        dst_result = factor_dir / f"{factor_name}.parquet"
        shutil.copy2(src_result, dst_result)

    # Build meta.json with accepted=true
    meta = {
        "factor_name": factor_name,
        "source_report": report_name,
        "created_at": __import__("datetime").datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "source_type": "literature_report",
        "accepted": True,
        "factor_description": args.description or "",
        "factor_formulation": args.formulation or "",
        "source_report_title": args.source_report_title or "",
        "source_report_path": args.source_report_path or "",
        "source_excerpt": args.source_excerpt or "",
    }
    if args.meta_json:
        extra = json.loads(args.meta_json)
        # 避免 --meta-json 覆盖显式参数
        extra.pop("factor_description", None)
        extra.pop("factor_formulation", None)
        extra.pop("source_report_title", None)
        extra.pop("source_report_path", None)
        extra.pop("source_excerpt", None)
        meta.update(extra)

    dst_meta = factor_dir / f"{factor_name}.meta.json"
    dst_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Cleanup temp files
    try:
        wrapped_tmp.unlink(missing_ok=True)
        shutil.rmtree(test_result["tmpdir"], ignore_errors=True)
    except Exception:
        pass

    print(json.dumps({
        "success": True,
        "factor_name": factor_name,
        "report_name": report_name,
        "code": str(dst_code),
        "parquet": str(factor_dir / f"{factor_name}.parquet") if has_result else None,
        "meta": str(dst_meta),
        "directory": str(factor_dir),
        "detected_type": type_key,
        "detected_lookback": lookback,
        "result_shape": test_result.get("result_shape"),
    }, ensure_ascii=False, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: trigger-full
# ---------------------------------------------------------------------------
def cmd_trigger_full(args):
    """Run full-scale factor pipeline synchronously: compute → evaluate → Barra → LLM review → sync.

    Derives parameters from literature_reports directory structure, then calls run_full_pipeline().
    Blocking — waits until complete before returning.
    """
    code_path = Path(args.code).resolve()
    if not code_path.exists():
        print(f"ERROR: code file not found: {code_path}", file=sys.stderr)
        return 1

    factor_name = Path(code_path).stem.replace(".code", "")

    # Locate report name from directory structure
    # Expected: literature_reports/<report>/<factor>/<factor>.code.py
    report_dir = code_path.parent
    if report_dir.name == "literature_reports":
        print(f"ERROR: code.py must be inside literature_reports/<report>/<factor>/", file=sys.stderr)
        return 1
    report_name = report_dir.parent.name if report_dir.parent.name != "literature_reports" else report_dir.name
    meta_path = report_dir / f"{factor_name}.meta.json"

    # Read test meta
    test_meta = {}
    if meta_path.exists():
        try:
            test_meta = json.loads(meta_path.read_text())
        except Exception:
            pass

    source_excerpt = test_meta.get("source_excerpt", "")

    sys.path.insert(0, str(PROJECT_ROOT))
    from rdagent.app.qlib_rd_loop.factor_full_pipeline import (
        FULL_OUTPUT_BASE, detect_factor_type_from_code, run_full_pipeline,
    )

    factor_type = detect_factor_type_from_code(code_path.read_text())
    output_dir = FULL_OUTPUT_BASE / report_name / factor_name

    # 检查全量是否已存在
    if (output_dir / f"{factor_name}.parquet").exists():
        print(f"  全量结果已存在，跳过计算", flush=True)
        status = "success"
    else:
        ok = run_full_pipeline(
            factor_name=factor_name,
            code_path=code_path,
            output_dir=output_dir,
            factor_type=factor_type,
            test_meta=test_meta,
            source_excerpt=source_excerpt,
        )
        status = "success" if ok else "failed"

    # 全量成功 → 记录到因子记忆库
    if status == "success":
        try:
            full_meta_path = output_dir / f"{factor_name}.meta.json"
            code_text = code_path.read_text(encoding="utf-8")
            update_factor_memory(factor_name, report_name, factor_type, meta_path, full_meta_path)
        except Exception as e:
            print(f"  ⚠️ 因子记忆记录失败: {e}", file=sys.stderr)

    output = {
        "status": status,
        "factor_name": factor_name,
        "report_name": report_name,
        "code": str(code_path),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if status == "success" else 1


# ---------------------------------------------------------------------------
# Subcommand: run-full
# ---------------------------------------------------------------------------
def cmd_run_full(args):
    """Run full-scale factor pipeline directly from a .code.py + metadata.

    Unlike trigger-full, this does NOT depend on the literature_reports directory structure.
    It accepts flat parameters and writes output to the specified directory.
    """
    code_path = Path(args.code).resolve()
    if not code_path.exists():
        print(f"ERROR: code file not found: {code_path}", file=sys.stderr)
        return 1

    factor_name = args.factor_name
    report_name = args.report_name

    # Determine output directory
    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        sys.path.insert(0, str(PROJECT_ROOT))
        from rdagent.app.qlib_rd_loop.factor_full_pipeline import FULL_OUTPUT_BASE
        output_dir = FULL_OUTPUT_BASE / report_name / factor_name

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build test_meta: first try existing meta.json in output dir (from deploy-to-full),
    # then override with --meta file or individual CLI args
    test_meta = {}
    existing_meta_path = output_dir / f"{args.factor_name}.meta.json"
    if existing_meta_path.exists():
        try:
            existing = json.loads(existing_meta_path.read_text())
            # Only inherit text fields (not evaluation/barra which are computed)
            for k in ("factor_description", "factor_formulation", "source_excerpt",
                      "source_report_title", "source_report_path", "variables"):
                if k in existing and existing[k]:
                    test_meta[k] = existing[k]
        except Exception:
            pass

    if args.meta:
        meta_file = Path(args.meta)
        if meta_file.exists():
            try:
                test_meta = json.loads(meta_file.read_text())
            except Exception as e:
                print(f"ERROR: failed to read meta file: {e}", file=sys.stderr)
                return 1

    # Override individual fields from CLI args
    if args.description:
        test_meta["factor_description"] = args.description
    if args.formulation:
        test_meta["factor_formulation"] = args.formulation
    if args.source_excerpt:
        test_meta["source_excerpt"] = args.source_excerpt
    if args.source_report_title:
        test_meta["source_report_title"] = args.source_report_title

    source_excerpt = test_meta.get("source_excerpt", "")

    sys.path.insert(0, str(PROJECT_ROOT))
    from rdagent.app.qlib_rd_loop.factor_full_pipeline import run_full_pipeline

    ok = run_full_pipeline(
        factor_name=factor_name,
        code_path=code_path,
        output_dir=output_dir,
        factor_type=args.type,
        test_meta=test_meta,
        source_excerpt=source_excerpt,
    )

    output = {
        "status": "success" if ok else "failed",
        "factor_name": factor_name,
        "report_name": report_name,
        "code": str(code_path),
        "output_dir": str(output_dir),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Subcommand: wait-full
# ---------------------------------------------------------------------------
def cmd_wait_full(args):
    """Wait for all submitted FullPipeline tasks to complete."""
    from rdagent.app.qlib_rd_loop.factor_full_pipeline import FullPipelineExecutor

    executor = FullPipelineExecutor.get_instance(max_workers=1)
    timeout = getattr(args, 'timeout', 7200)
    executor.wait_for_completion(timeout=timeout)
    print(json.dumps({"status": "completed"}))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: deploy-to-full
# ---------------------------------------------------------------------------
def cmd_deploy_to_full(args):
    """Deploy tested factor to full-scale directory: copy code, patch DATA_DIR, inherit meta.

    Copies .code.py from literature_reports/<report>/<factor>/ to
    文献因子_全量/<report>/<factor>/, patches DATA_DIR from _1000 → full path,
    and copies/creates meta.json with pipeline_status='deployed'.
    Does NOT run full computation.
    """
    code_path = Path(args.code).resolve()
    if not code_path.exists():
        print(f"ERROR: code file not found: {code_path}", file=sys.stderr)
        return 1

    factor_name = code_path.stem.replace(".code", "")

    # Locate report + factor from directory structure
    # Expected: literature_reports/<report>/<factor>/<factor>.code.py
    test_factor_dir = code_path.parent
    report_dir = test_factor_dir.parent
    if report_dir.name == "literature_reports":
        print(f"ERROR: code.py must be inside literature_reports/<report>/<factor>/", file=sys.stderr)
        return 1
    report_name = report_dir.name

    # Target: 文献因子_全量/<report>/<factor>/
    sys.path.insert(0, str(PROJECT_ROOT))
    from rdagent.app.qlib_rd_loop.factor_full_pipeline import FULL_OUTPUT_BASE
    full_factor_dir = FULL_OUTPUT_BASE / report_name / factor_name
    full_factor_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy .code.py, patch DATA_DIR
    code_text = code_path.read_text(encoding="utf-8")
    patched = code_text.replace("factor_implementation_source_data_1000", "factor_implementation_source_data")

    # 注入多级 DATA_DIR 降级链（环境变量 → /mnt/remote_e → E:\ → .）
    # 使 .code.py 可在任意环境直接运行，无需设置 FACTOR_DATA_DIR
    _simple_dir = r'DATA_DIR = Path\(os\.environ\.get\("FACTOR_DATA_DIR"\) or os\.environ\.get\("RDAGENT_FACTOR_DATA_DIR"\) or "\."\)'
    if re.search(_simple_dir, patched):
        _subdir = "minute_by_date" if "minute_by_date" in patched else "daily"
        _fallback = (
            'DATA_DIR = Path(os.environ.get("FACTOR_DATA_DIR") or os.environ.get("RDAGENT_FACTOR_DATA_DIR") or "")\n'
            f'if not DATA_DIR or not (DATA_DIR/"stock_data"/"{_subdir}").exists():\n'
            f'    DATA_DIR = Path("/mnt/remote_e/_paper_factor_unified/factor_implementation_source_data")\n'
            f'    if not (DATA_DIR/"stock_data"/"{_subdir}").exists():\n'
            f'        DATA_DIR = Path("E:\\\\_paper_factor_unified\\\\factor_implementation_source_data")\n'
            f'        if not (DATA_DIR/"stock_data"/"{_subdir}").exists():\n'
            f'            DATA_DIR = Path("Z:\\\\_paper_factor_unified\\\\factor_implementation_source_data")\n'
            f'            if not (DATA_DIR/"stock_data"/"{_subdir}").exists():\n'
            f'                DATA_DIR = Path(r"\\\\\\\\192.168.1.13\\\\E\\\\_paper_factor_unified\\\\factor_implementation_source_data")\n'
            f'                if not (DATA_DIR/"stock_data"/"{_subdir}").exists():\n'
            f'                    DATA_DIR = Path(".")\n'
        )
        patched = re.sub(_simple_dir, _fallback, patched)

    full_code_path = full_factor_dir / f"{factor_name}.code.py"
    full_code_path.write_text(patched, encoding="utf-8")
    n_replaced = code_text.count("factor_implementation_source_data_1000")

    # 2. Inherit meta.json from test
    test_meta_path = test_factor_dir / f"{factor_name}.meta.json"
    meta = {}
    if test_meta_path.exists():
        try:
            meta = json.loads(test_meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    meta["pipeline_status"] = "deployed"
    meta["code_deployed_at"] = __import__("datetime").datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    meta["factor_type"] = args.type or meta.get("factor_type", "")
    meta["factor_name"] = factor_name
    meta["report_name"] = report_name

    full_meta_path = full_factor_dir / f"{factor_name}.meta.json"
    full_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "success": True,
        "factor_name": factor_name,
        "report_name": report_name,
        "code": str(full_code_path),
        "meta": str(full_meta_path),
        "directory": str(full_factor_dir),
        "data_dir_patches": n_replaced,
        "pipeline_status": "deployed",
    }, ensure_ascii=False, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: sync-full
# ---------------------------------------------------------------------------
def cmd_sync_full(args):
    """Sync deployed factors from 文献因子_全量 to remote.

    Only syncs .code.py, .meta.json, and .parquet files (skips checkpoints, logs, etc.).
    Supports --report and --factor for single-report sync, or --all for all.
    """
    sys.path.insert(0, str(PROJECT_ROOT))
    from rdagent.app.qlib_rd_loop.factor_full_pipeline import FULL_OUTPUT_BASE
    from scripts.sync_utils import upload_file, ensure_remote_writable

    if not ensure_remote_writable():
        print("ERROR: remote is not writable", file=sys.stderr)
        return 1

    from scripts.sync_utils import REMOTE_BASE_FULL

    SYNC_EXTS = {'.code.py', '.meta.json', '.parquet'}

    def _sync_factor_dir(factor_dir, remote_prefix):
        """Sync only allowed file types from a factor directory."""
        n = 0
        for f in factor_dir.iterdir():
            if f.is_dir():
                continue
            # Check if file extension matches any of SYNC_EXTS
            for ext in SYNC_EXTS:
                if f.name.endswith(ext):
                    remote_path = f"{remote_prefix}\\{f.name}"
                    if upload_file(f, remote_path):
                        n += 1
                    break
        return n

    if args.all:
        if not FULL_OUTPUT_BASE.exists():
            print("ERROR: full output base not found", file=sys.stderr)
            return 1
        report_dirs = sorted(FULL_OUTPUT_BASE.iterdir())
        total_files = 0
        for report_dir in report_dirs:
            if not report_dir.is_dir():
                continue
            remote_report = f"{REMOTE_BASE_FULL}\\{report_dir.name}"
            for factor_dir in sorted(report_dir.iterdir()):
                if not factor_dir.is_dir():
                    continue
                remote_factor = f"{remote_report}\\{factor_dir.name}"
                n = _sync_factor_dir(factor_dir, remote_factor)
                total_files += n
        print(f"OK: synced {total_files} files from {len(report_dirs)} reports")
    elif args.report:
        report_dir = FULL_OUTPUT_BASE / args.report
        if not report_dir.exists():
            print(f"ERROR: report not found: {report_dir}", file=sys.stderr)
            return 1
        if args.factor:
            factor_dir = report_dir / args.factor
            if not factor_dir.exists():
                print(f"ERROR: factor not found: {factor_dir}", file=sys.stderr)
                return 1
            remote_path = f"{REMOTE_BASE_FULL}\\{args.report}\\{args.factor}"
            n = _sync_factor_dir(factor_dir, remote_path)
            print(f"OK: synced {n} files for {args.report}/{args.factor}")
        else:
            remote_path = f"{REMOTE_BASE_FULL}\\{args.report}"
            total = 0
            for factor_dir in sorted(report_dir.iterdir()):
                if not factor_dir.is_dir():
                    continue
                remote_factor = f"{remote_path}\\{factor_dir.name}"
                total += _sync_factor_dir(factor_dir, remote_factor)
            print(f"OK: synced {total} files for report {args.report}")
    else:
        print("ERROR: specify --report or --all", file=sys.stderr)
        return 1

    return 0


# ---------------------------------------------------------------------------
# Subcommand: scan-pending
# ---------------------------------------------------------------------------
def cmd_scan_pending(args):
    """Scan for unprocessed papers, website sources, and ideas."""
    inbox_dir = PROJECT_ROOT / "papers" / "inbox"
    sources_json = PROJECT_ROOT / "papers" / "website" / "sources.json"
    ideas_json = PROJECT_ROOT / "papers" / "ideas" / "ideas.json"
    processed_json = LITERATURE_REPORTS_DIR.parent / "processed_reports.json"
    extracted_dir = LITERATURE_REPORTS_DIR.parent / "extracted_reports"

    # Load processed list
    processed_set = set()
    if processed_json.exists():
        try:
            processed_set = set(json.loads(processed_json.read_text()))
        except Exception:
            pass

    result = {"papers": [], "websites": [], "ideas": [], "fully_processed_papers": [], "fully_processed_websites": [], "fully_processed_ideas": []}

    # Scan inbox PDFs
    if inbox_dir.exists():
        for pdf_path in sorted(inbox_dir.glob("*.pdf")):
            pdf_name = pdf_path.name
            status, info = _check_report_status(pdf_path, processed_set, extracted_dir, LITERATURE_REPORTS_DIR)
            if status == "done":
                result["fully_processed_papers"].append(pdf_name)
            elif status == "partial":
                result["papers"].append({"file": str(pdf_path), "name": pdf_name, "status": "partial", **info})
            else:
                result["papers"].append({"file": str(pdf_path), "name": pdf_name, "status": "pending"})

    # Scan website sources.json
    if sources_json.exists():
        try:
            sources = json.loads(sources_json.read_text())
            for i, src in enumerate(sources):
                src = _normalize_source(src, i)
                slug = _make_source_slug(src, i)
                slug_name = f"{slug}.extracted.json"
                # Check if in processed_reports.json (skip/mark-done by slug or URL)
                if slug in processed_set or src.get("url", "") in processed_set:
                    result["fully_processed_websites"].append(slug)
                    continue
                # Check if already in literature_reports (递归检查子目录)
                report_dir = LITERATURE_REPORTS_DIR / slug
                _has_code = False
                if report_dir.exists():
                    _has_code = any(f.endswith(".code.py") for f in os.listdir(report_dir))
                    if not _has_code:
                        for _sub in report_dir.iterdir():
                            if _sub.is_dir() and any(f.endswith(".code.py") for f in os.listdir(_sub)):
                                _has_code = True
                                break
                if _has_code:
                    result["fully_processed_websites"].append(src.get("title", slug))
                elif (extracted_dir / slug_name).exists():
                    result["websites"].append({"index": i, "slug": slug, "title": src.get("title", ""), "url": src.get("url", ""), "status": "extracted"})
                else:
                    result["websites"].append({"index": i, "slug": slug, "title": src.get("title", ""), "url": src.get("url", ""), "status": "pending"})
        except Exception:
            pass

    # Scan ideas.json
    if ideas_json.exists():
        try:
            ideas = json.loads(ideas_json.read_text())
            for i, idea in enumerate(ideas):
                slug = f"idea__{i}"
                # Check if in processed_reports.json (mark-done skip)
                if slug in processed_set:
                    result["fully_processed_ideas"].append(idea.get("title", slug)[:60])
                    continue
                report_dir = LITERATURE_REPORTS_DIR / slug
                if report_dir.exists():
                    # 检查子目录是否有 .code.py
                    _has_code = any(f.endswith(".code.py") for f in os.listdir(report_dir))
                    if not _has_code:
                        for _sub in report_dir.iterdir():
                            if _sub.is_dir() and any(f.endswith(".code.py") for f in os.listdir(_sub)):
                                _has_code = True
                                break
                    if _has_code:
                        result["fully_processed_ideas"].append(idea.get("title", slug)[:60])
                        continue
                else:
                    result["ideas"].append({"index": i, "slug": slug, "title": idea.get("title"), "text": (idea.get("text") or idea.get("idea", ""))[:200], "status": "pending"})
        except Exception:
            pass

    # Summary
    total_pending = len(result["papers"]) + len(result["websites"]) + len(result["ideas"])
    result["summary"] = {
        "total_pending": total_pending,
        "papers_pending": len(result["papers"]),
        "websites_pending": len(result["websites"]),
        "ideas_pending": len(result["ideas"]),
        "fully_processed_papers": len(result["fully_processed_papers"]),
        "fully_processed_websites": len(result["fully_processed_websites"]),
        "fully_processed_ideas": len(result["fully_processed_ideas"]),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _sanitize_name(name: str) -> str:
    """Sanitize report/factor name for filesystem (same as FactorFBWorkspace._sanitize_factor_name)."""
    for ch in r'\/:*?"<>|':
        name = name.replace(ch, "_")
    return name.strip()


def _normalize_source(src, index: int) -> dict:
    """Normalize a sources.json entry to dict format (handles both string URLs and dicts)."""
    if isinstance(src, str):
        return {"url": src, "title": "", "source": ""}
    if isinstance(src, dict):
        return {"url": src.get("url", ""), "title": src.get("title", ""), "source": src.get("source", "")}
    return {"url": "", "title": f"未知来源_{index}", "source": ""}


def _make_source_slug(src, index: int) -> str:
    """Create a slug for a website source entry."""
    src = _normalize_source(src, index)
    title = src.get("title", "")
    if title:
        return _sanitize_name(f"website__{title[:80]}")
    url = src.get("url", "")
    if url:
        # Extract domain + path tail
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path_parts = parsed.path.strip("/").split("/")
        short = path_parts[-1][:40] if path_parts else parsed.hostname or f"src_{index}"
        return _sanitize_name(f"website__{short}")
    return f"website__{index}"


def _check_report_status(pdf_path, processed_set, extracted_dir, lit_dir) -> tuple:
    """Check if a report PDF is fully processed, partial, or unprocessed.
    Returns (status, info_dict) where status is 'done', 'partial', or 'pending'.
    """
    import unicodedata
    pdf_name = pdf_path.name
    report_title = pdf_path.stem

    # Normalize for comparison: strip common punctuation differences (Chinese/ASCII quotes, etc.)
    def _norm(s):
        s = unicodedata.normalize('NFKC', s)
        for ch in '\u201c\u201d\u2018\u2019""''':
            s = s.replace(ch, '')
        return s

    pdf_name_norm = _norm(pdf_name)
    for processed_name in processed_set:
        if _norm(processed_name) == pdf_name_norm:
            return "done", {}

    # Check extracted report cache
    extracted_json = extracted_dir / f"{report_title}.extracted.json"
    extracted_count = 0
    if extracted_json.exists():
        try:
            payload = json.loads(extracted_json.read_text(encoding="utf-8"))
            factors = payload.get("factors") or {}
            extracted_count = len(factors)
        except Exception:
            pass

    if extracted_count == 0:
        return "pending", {}

    # Check how many factors made it to literature_reports
    report_dir = lit_dir / _sanitize_name(report_title)
    terminal_count = 0
    if report_dir.exists():
        for factor_dir in report_dir.iterdir():
            if factor_dir.is_dir():
                if any(f.endswith(".code.py") for f in os.listdir(factor_dir)):
                    terminal_count += 1

    if terminal_count >= extracted_count:
        return "done", {"extracted": extracted_count, "terminal": terminal_count}
    else:
        return "partial", {"extracted": extracted_count, "terminal": terminal_count}


# ---------------------------------------------------------------------------
# Subcommand: mark-done
# ---------------------------------------------------------------------------
def cmd_mark_done(args):
    """Mark a paper as processed by adding it to processed_reports.json."""
    processed_json = LITERATURE_REPORTS_DIR.parent / "processed_reports.json"
    processed = []
    if processed_json.exists():
        try:
            processed = json.loads(processed_json.read_text())
        except Exception:
            pass

    name = args.name
    if name not in processed:
        processed.append(name)
        processed_json.write_text(json.dumps(processed, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"marked_done": name, "total_processed": len(processed)}))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: save-extracted
# ---------------------------------------------------------------------------
def cmd_save_extracted(args):
    """Save factor definitions JSON to extracted_reports/ (read from stdin)."""
    extracted_dir = LITERATURE_REPORTS_DIR.parent / "extracted_reports"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    content = sys.stdin.read()
    if not content.strip():
        print("ERROR: no content on stdin", file=sys.stderr)
        return 1
    path = extracted_dir / f"{args.name}.extracted.json"
    path.write_text(content, encoding="utf-8")
    print(f"OK: {path}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: show-columns — 显示可用数据列
# ---------------------------------------------------------------------------
def cmd_show_columns(args):
    """Show available columns in stock data parquet files with descriptions."""
    import pyarrow.parquet as _pq
    test_dir = TEST_DATA_DIR / "stock_data" / "daily"
    if test_dir.exists():
        files = sorted(test_dir.glob("*.parquet"))
        if files:
            schema = _pq.read_schema(files[0])
            cols = [n for n in schema.names if n not in ('datetime', 'instrument')]
            desc = {
                "open": "开盘价",
                "close": "收盘价",
                "high": "最高价",
                "low": "最低价",
                "factor": "复权因子",
                "volume": "成交量（股数）",
                "pct_chg": "涨跌幅（%）",
                "pre_close": "前收盘价",
                "turnover_rate": "换手率（%）",
                "roe": "净资产收益率",
                "roa": "总资产收益率",
                "pe_ttm": "滚动市盈率",
                "pb": "市净率",
                "revenue_yoy": "营业收入同比增速（%）",
                "profit_yoy": "净利润同比增速（%）",
                "gross_margin": "毛利率（%）",
                "net_margin": "净利率（%）",
                "debt_to_asset": "资产负债率（%）",
                "ocf_per_share": "每股经营活动现金流",
                "market_cap": "总市值",
                "circulating_market_cap": "流通市值",
                "total_shares": "总股本",
                "float_shares": "流通股本",
                "adjusted_profit": "调整后净利润",
                "gross_profit": "营业利润",
                "EMA5": "5日指数移动平均",
                "EMA10": "10日指数移动平均",
                "EMA20": "20日指数移动平均",
                "jhjj_hsl": "集合竞价换手率",
                "net_pct_main": "主力净流入占比",
                "net_pct_xl": "超大单净流入占比",
                "net_pct_l": "大单净流入占比",
                "net_pct_m": "中单净流入占比",
                "net_pct_s": "小单净流入占比",
                "net_amount_main": "主力净流入金额（元）",
                "amount": "成交金额（元）",
            }
            print("可用列及含义（日线数据）：")
            for c in cols:
                d = desc.get(c, "")
                print(f"  {c:30s} {d}")
            return 0
    print("ERROR: 无法找到数据文件")
    return 1


# ═══════════════════════════════════════════════════════════════════
# retrieve-domain-knowledge — 领域知识 RAG 检索
# ═══════════════════════════════════════════════════════════════════
def cmd_retrieve_knowledge(args):
    """检索领域知识（A 股规则、涨停机制、列定义等）"""
    query = " ".join(args.query)
    top_k = args.top_k
    min_score = args.min_score

    import sys as _sys
    _sys.path.insert(0, os.path.dirname(__file__))
    from domain_knowledge_rag import DomainKnowledgeRAG
    rag = DomainKnowledgeRAG()
    rag.load_chunks_and_build()
    results = rag.retrieve(query, top_k=top_k, min_score=min_score)

    if not results:
        print("无匹配结果")
        return 0

    for r in results:
        print(f"[{r['score']:.3f}] {r['source']} > {r['heading']}")
        print(f"  {r['text'][:200]}")
        print()

    if args.json:
        import json as _json
        print("---")
        print(_json.dumps(results, ensure_ascii=False, indent=2))

    return 0


# ---------------------------------------------------------------------------
# Subcommand: add-idea
# ---------------------------------------------------------------------------
def cmd_add_idea(args):
    """Add a factor idea to ideas.json. Auto-generates title from text if not provided."""
    ideas_json = PROJECT_ROOT / "papers" / "ideas" / "ideas.json"
    ideas_json.parent.mkdir(parents=True, exist_ok=True)

    ideas = []
    if ideas_json.exists():
        try:
            ideas = json.loads(ideas_json.read_text(encoding="utf-8"))
        except Exception:
            pass

    text = args.text
    title = args.title or None  # None = agent 处理时自动生成标题

    entry = {"title": title, "text": text}
    ideas.append(entry)
    ideas_json.write_text(json.dumps(ideas, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(entry, ensure_ascii=False))
    print(f"OK: added idea #{len(ideas)-1} ({title})", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Claude Code /factor skill helper — extract, wrap, test, export, run",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # extract-pdf
    p_pdf = sub.add_parser("extract-pdf", help="Extract text from PDFs/.md files to JSON")
    p_pdf.add_argument("paths", nargs="+", help="PDF/MD file or directory paths")

    # extract-website
    p_web = sub.add_parser("extract-website", help="Fetch URL and extract factors via LLM")
    p_web.add_argument("--index", type=int, required=True, help="Index in papers/website/sources.json")

    # wrap-template
    p_wrap = sub.add_parser("wrap-template", help="Wrap user code into framework .code.py")
    p_wrap.add_argument("--code", required=True, help="User function code file")
    p_wrap.add_argument("--type", required=True, help="Template type: daily_single, cross_section, minute, minute_cross_section, deep_learning")
    p_wrap.add_argument("--lookback", type=int, default=250, help="Lookback days (default: 250)")
    p_wrap.add_argument("--cols", default=None, help="Comma-separated columns to load")
    p_wrap.add_argument("--output", default=None, help="Output .code.py path (default: stdout)")

    # run-test
    p_test = sub.add_parser("run-test", help="Run code.py against 300-stock test data")
    p_test.add_argument("code", help="Factor .code.py file")
    p_test.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds (default: 3600)")

    # export-factor
    p_export = sub.add_parser("export-factor", help="Export factor to literature_reports")
    p_export.add_argument("--code", required=True, help="Factor .code.py file")
    p_export.add_argument("--report", required=True, help="Report name (directory)")
    p_export.add_argument("--factor", required=True, help="Factor name")
    p_export.add_argument("--meta-json", default=None, help="Extra metadata JSON string")
    p_export.add_argument("--description", default=None, help="Factor description")
    p_export.add_argument("--formulation", default=None, help="Factor formulation")
    p_export.add_argument("--source-report-title", default=None, help="Source report title")
    p_export.add_argument("--source-report-path", default=None, help="Source report path")
    p_export.add_argument("--source-excerpt", default=None, help="Source excerpt text")

    # test-and-export
    p_tae = sub.add_parser("test-and-export", help="Wrap + test + export in one shot (auto-detect type/lookback)")
    p_tae.add_argument("--code", required=True, help="User function code file")
    p_tae.add_argument("--report", required=True, help="Report name (directory)")
    p_tae.add_argument("--factor", required=True, help="Factor name")
    p_tae.add_argument("--type", default=None, help="Template type (auto-detected from function names if omitted)")
    p_tae.add_argument("--lookback", type=int, default=None, help="Lookback days (auto-detected from code if omitted)")
    p_tae.add_argument("--cols", default=None, help="Comma-separated columns to load")
    p_tae.add_argument("--meta-json", default=None, help="Extra metadata JSON string")
    p_tae.add_argument("--description", default=None, help="Factor description (auto-fills factor_description in meta)")
    p_tae.add_argument("--formulation", default=None, help="Factor formulation (auto-fills factor_formulation in meta)")
    p_tae.add_argument("--source-report-title", default=None, help="Source report title (auto-fills source_report_title in meta)")
    p_tae.add_argument("--source-report-path", default=None, help="Source report path (auto-fills source_report_path in meta)")
    p_tae.add_argument("--source-excerpt", default=None, help="Source excerpt text (auto-fills source_excerpt in meta)")
    p_tae.add_argument("--timeout", type=int, default=3600, help="Test timeout in seconds (default: 3600)")

    # trigger-full
    p_full = sub.add_parser("trigger-full", help="Trigger full-scale run via FullPipelineExecutor")
    p_full.add_argument("--code", required=True, help="Factor .code.py file (in literature_reports)")

    # run-full
    p_run = sub.add_parser("run-full", help="Run full pipeline directly from .code.py + metadata (independent)")
    p_run.add_argument("--code", required=True, help="Factor .code.py file path")
    p_run.add_argument("--factor-name", required=True, help="Factor name")
    p_run.add_argument("--report-name", required=True, help="Report name (used for output directory structure)")
    p_run.add_argument("--output", default=None, help="Output directory (default: 文献因子_全量/<report>/<factor>/)")
    p_run.add_argument("--type", default=None, help="Factor type (auto-detected if omitted): daily, minute, cross_section, minute_cs, deep_learning")
    p_run.add_argument("--description", default=None, help="Factor description")
    p_run.add_argument("--formulation", default=None, help="Factor formulation")
    p_run.add_argument("--source-excerpt", default=None, help="Source excerpt text")
    p_run.add_argument("--source-report-title", default=None, help="Source report title")
    p_run.add_argument("--meta", default=None, help="JSON file with metadata (alternative to individual --* args)")

    # scan-pending
    p_scan = sub.add_parser("scan-pending", help="Scan for unprocessed papers, websites, and ideas")

    # mark-done
    p_mark = sub.add_parser("mark-done", help="Mark a paper as processed")
    p_mark.add_argument("--name", required=True, help="Paper filename (e.g. '报告名.pdf')")

    # save-extracted
    p_se = sub.add_parser("save-extracted", help="Save factor definitions JSON to extracted_reports/ (read from stdin)")
    p_se.add_argument("--name", required=True, help="Report title (stem, e.g. '基于GRU的因子选股')")

    # add-idea
    p_idea = sub.add_parser("add-idea", help="Add a factor idea (title optional; agent auto-generates if omitted)")
    p_idea.add_argument("--title", default=None, help="Factor idea title (optional; auto-generated from text)")
    p_idea.add_argument("--text", required=True, help="Factor idea description")

    # show-columns
    sub.add_parser("show-columns", help="Show available columns in stock data")

    # retrieve-knowledge
    p_rk = sub.add_parser("retrieve-knowledge", help="Retrieve domain knowledge via RAG (A股规则, 涨停机制, 列定义等)")
    p_rk.add_argument("query", nargs="+", help="Search query (e.g. '涨停阈值 科创板')")
    p_rk.add_argument("--top-k", type=int, default=3, help="Number of results (default: 3)")
    p_rk.add_argument("--min-score", type=float, default=0.01, help="Minimum similarity score (default: 0.01)")
    p_rk.add_argument("--json", action="store_true", help="Output as JSON")

    # wait-full
    p_wait = sub.add_parser("wait-full", help="Wait for all submitted full pipeline tasks to complete")

    # deploy-to-full
    p_deploy = sub.add_parser("deploy-to-full", help="Deploy tested factor to full-scale directory (copy code + patch DATA_DIR + inherit meta, no computation)")
    p_deploy.add_argument("--code", required=True, help="Factor .code.py file (in literature_reports/<report>/<factor>/)")
    p_deploy.add_argument("--type", default=None, help="Factor type (daily, minute, cross_section, minute_cs, deep_learning)")

    # sync-full
    p_sync = sub.add_parser("sync-full", help="Sync deployed factors from 文献因子_全量 to remote")
    p_sync.add_argument("--report", default=None, help="Report name (required unless --all)")
    p_sync.add_argument("--factor", default=None, help="Factor name (optional, syncs whole report if omitted)")
    p_sync.add_argument("--all", action="store_true", help="Sync all deployed factors")

    args = parser.parse_args()

    cmd_map = {
        "extract-pdf": cmd_extract_pdf,
        "extract-website": cmd_extract_website,
        "wrap-template": cmd_wrap_template,
        "run-test": cmd_run_test,
        "export-factor": cmd_export_factor,
        "test-and-export": cmd_test_and_export,
        "trigger-full": cmd_trigger_full,
        "run-full": cmd_run_full,
        "wait-full": cmd_wait_full,
        "deploy-to-full": cmd_deploy_to_full,
        "sync-full": cmd_sync_full,
        "scan-pending": cmd_scan_pending,
        "mark-done": cmd_mark_done,
        "save-extracted": cmd_save_extracted,
        "add-idea": cmd_add_idea,
        "show-columns": cmd_show_columns,
        "retrieve-knowledge": cmd_retrieve_knowledge,
    }
    fn = cmd_map[args.command]
    return fn(args)


if __name__ == "__main__":
    sys.exit(main())
