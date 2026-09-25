#!/usr/bin/env python3
"""
Claude Code /factor skill 辅助脚本。
子命令：extract-pdf, wrap-template, run-test, export-factor, test-and-export, trigger-full, wait-full
"""

import argparse
import json
import os
import re

def _clip(s: str, head: int = 1500, tail: int = 3000) -> str:
    """把长文本截成「首 head + 尾 tail」，中间标注省略量。

    用于失败时回传日志：保留开头（通常含首个报错）和结尾（traceback 收尾），
    避免整段日志（分钟因子可达数万字符）灌进 agent context。
    """
    if not s:
        return ""
    if len(s) <= head + tail:
        return s
    omitted = len(s) - head - tail
    return f"{s[:head]}\n…[中间省略 {omitted} 字符]…\n{s[-tail:]}"


import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = Path("/mnt/d/paper-factor-data")
LITERATURE_REPORTS_DIR = DATA_ROOT / "数据仓库" / "因子产出" / "测试"
SCHEMA_FILE = DATA_ROOT / "schema.json"
# 研报目录：inbox 是投放/处理区，done 是已处理归档（用户手动控制投放数量）
PAPERS_DIR = DATA_ROOT / "papers"
INBOX_DIR = PAPERS_DIR / "inbox"
DONE_DIR = PAPERS_DIR / "done"

def _load_schema() -> dict:
    """读 /mnt/d/paper-factor-data/schema.json（getdata 导入新字段后由 import_new_data.py 自动维护）。"""
    if not SCHEMA_FILE.exists():
        return {"daily": {"columns": {}}, "minute": {"columns": {}}}
    try:
        return json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"daily": {"columns": {}}, "minute": {"columns": {}}}

def _load_ff_schema() -> dict:
    """读 factor_field_schema.json（short_name + note 列含义，getdata 导入时同步更新）。"""
    candidates = [
        DATA_ROOT / "数据仓库" / "行情数据" / "日线" / "测试" / "factor_field_schema.json",
        DATA_ROOT / "数据仓库" / "行情数据" / "日线" / "全量" / "factor_field_schema.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
    return {}

def _desc_for_col(col: str) -> str:
    """取列含义：优先 schema.json description，其次 factor_field_schema short_name，最后空。"""
    schema = _load_schema()
    for group in ("daily", "minute"):
        cc = schema.get(group, {}).get("columns", {})
        if col in cc and cc[col].get("description"):
            return cc[col]["description"]
    ff = _load_ff_schema()
    if col in ff and ff[col].get("short_name"):
        return ff[col]["short_name"]
    return ""

def _detect_data_dir() -> Path:
    """Detect best available data directory."""
    candidates = [
        os.environ.get("FACTOR_DATA_DIR", ""),
        os.environ.get("RDAGENT_FACTOR_DATA_DIR", ""),
        str(DATA_ROOT / "数据仓库" / "行情数据" / "日线" / "测试" / "stock_data" / "daily"),
        str(DATA_ROOT / "数据仓库" / "行情数据" / "日线" / "全量" / "stock_data" / "daily"),
    ]
    for p in candidates:
        if p and Path(p).exists():
            return Path(p).parent.parent  # 返回 stock_data 的父目录
    return DATA_ROOT / "数据仓库" / "行情数据" / "日线" / "测试"


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
    """Extract text from PDFs (pymupdf) or .md files, output JSON (or write .txt files with --outdir)."""
    import fitz  # pymupdf

    def _extract_text(path: Path) -> str:
        if path.suffix == ".md":
            try:
                return path.read_text(encoding="utf-8")
            except Exception:
                return ""
        try:
            doc = fitz.open(path)
            return "\n".join(page.get_text() for page in doc)
        except Exception:
            return ""

    # --outdir 模式：把每份原文写到独立 .txt，打印 {源路径: txt路径} 映射。
    # 主进程预提取后，sub-agent 用 Read 工具直接读 .txt，避免内联 subprocess 一次性管道。
    if getattr(args, "outdir", None):
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        mapping = {}
        for i, p in enumerate(args.paths):
            path = Path(p).resolve()
            if not path.exists():
                continue
            if path.is_dir():
                text = "\n\n".join(
                    _extract_text(pdf_path) for pdf_path in sorted(path.rglob("*.pdf"))
                )
            else:
                text = _extract_text(path)
            stem = re.sub(r"[^\w.-]+", "_", path.stem)[:80] or f"src_{i}"
            txt_path = outdir / f"{i:02d}_{stem}.txt"
            txt_path.write_text(text, encoding="utf-8")
            mapping[str(path)] = str(txt_path)
        print(json.dumps(mapping, ensure_ascii=False))
        return

    result = {}
    for p in args.paths:
        path = Path(p).resolve()
        if not path.exists():
            result[str(path)] = ""
            continue
        if path.is_dir():
            for pdf_path in sorted(path.rglob("*.pdf")):
                result[str(pdf_path)] = _extract_text(pdf_path)
        else:
            result[str(path)] = _extract_text(path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Subcommand: extract-website — 只抓取内容，不调 LLM
# ---------------------------------------------------------------------------
def cmd_extract_website(args):
    """Fetch URL content, output as JSON (no LLM). Claude does the extraction."""

    import warnings as _warnings
    _warnings.filterwarnings("ignore", category=UserWarning, module="requests")

    def _is_content_garbled(text: str) -> bool:
        if len(text) < 100:
            return False
        cjk_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_printable = sum(1 for c in text if c.isprintable())
        if total_printable == 0:
            return True
        return cjk_chars / total_printable < 0.05

    def _fetch_content(url: str, timeout: int = 30, retries: int = 2):
        import requests as _req
        from bs4 import BeautifulSoup as _BS
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        last_error = None
        for attempt in range(1 + retries):
            if attempt > 0:
                import time as _time
                _time.sleep(1)
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
                if len(clean_text) < 50 and attempt < retries:
                    continue
                return page_title, f"标题: {page_title}\n\n来源: {url}\n\n正文:\n{clean_text}"
            except Exception as e:
                last_error = e
                print(f"  ⚠️ 抓取失败(第{attempt+1}次): {e}", flush=True)
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

    sources_json = DATA_ROOT / "papers" / "website" / "sources.json"
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
    load_cols = [c.strip() for c in re.split(r"[,，\s]+", args.cols) if c.strip()] if args.cols else None

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

    # Build meta.json（缺失字段自动从因子名/研报名派生）
    meta = {
        "factor_name": factor_name,
        "source_report": report_name,
        "created_at": __import__("datetime").datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "source_type": "literature_report",
        "factor_description": args.description or f"因子 {factor_name}，来自研报 {report_name}",
        "factor_formulation": args.formulation or f"见因子 {factor_name} 代码实现",
        "source_report_title": args.source_report_title or report_name,
        "source_report_path": args.source_report_path or "",
        "source_excerpt": args.source_excerpt or f"因子 {factor_name}，来自 {report_name}",
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
# 分钟特征：calc_factor_series 既是日线(向量化)也是分钟(向量化)的函数名，需靠代码内容区分。
# 分钟数据带日内时间戳，代码通常含以下分钟专属特征；日线单股数据不含这些（用 index.normalize()
# 把日内时间归一化到日期、或引用分钟专属列 vwap/amount）。不用 groupby(level=0)（日线也可能用）。
_MINUTE_PATTERNS = [
    r"\bvwap\b",
    r"index\s*\.\s*normalize\s*\(",
]


def _looks_like_minute(code: str) -> bool:
    return any(re.search(p, code) for p in _MINUTE_PATTERNS)


def detect_type_from_code(code: str) -> str | None:
    """Detect template type from function definitions in user code."""
    # 特殊函数优先，避免 calc_factor_series 的 daily/minute 歧义
    if re.search(r"\bdef\s+calc_factor_cross_section\s*\(", code):
        return "cross_section"
    if re.search(r"\bdef\s+calc_factor_minute_raw\s*\(", code) or re.search(r"\bdef\s+cross_section_transform\s*\(", code):
        return "minute_cross_section"
    if re.search(r"\bdef\s+calc_factors_one_day\s*\(", code):
        return "minute"
    if re.search(r"\bdef\s+train_model\s*\(", code) or re.search(r"\bdef\s+predict\s*\(", code):
        return "deep_learning"
    if re.search(r"\bdef\s+calc_factor_series\s*\(", code):
        if _looks_like_minute(code):
            return "minute"
        return "daily_single"
    if re.search(r"\bdef\s+calc_factor_single_stock\s*\(", code):
        return "daily_single"
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


def _run_test_in_tmpdir(code_path: Path, timeout: int = 3600, type_key: str = "daily_single") -> dict:
    """Run a .code.py in a temp dir and return result dict + temp dir path."""
    env = os.environ.copy()
    # 根据因子类型设置正确的数据目录
    if type_key in ("minute", "minute_cross_section"):
        minute_test_dir = DATA_ROOT / "数据仓库" / "行情数据" / "分钟线" / "测试"
        if minute_test_dir.exists():
            env["FACTOR_DATA_DIR"] = str(minute_test_dir)
        else:
            env["FACTOR_DATA_DIR"] = str(TEST_DATA_DIR)
        # 分钟 chunk 缓存用模板默认的共享目录（{分钟数据目录}/stock_data/minute_by_date/_minute_chunks）：
        #   - 每个数据集固定一份，跨因子复用，数据未变时跳过预分片（省 ~15s）
        #   - 测试/全量各有自己的目录，天然隔离
        #   - 并发写由模板内的 filelock 串行化；结果文件带 pid 后缀
        # 故此处不再指定 FACTOR_MINUTE_CHUNK_DIR（此前每进程建一个 /tmp 目录，从不清理，
        # 累积到 115 个 / 35GB）。
    else:
        env["FACTOR_DATA_DIR"] = str(TEST_DATA_DIR)
    env["FACTOR_N_WORKERS"] = os.environ.get("FACTOR_N_WORKERS", "2")
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
        # 模板把 parquet 写到 _CODE_DIR（即 code_path 的父目录），文件名由 stem 决定
        _PARQUET_STEM = Path(code_path).stem.removesuffix('.code')
        _alt_path = Path(code_path).parent / f"{_PARQUET_STEM}.parquet"
        # minute_cs 模板将因子名追加到文件名：{stem}_{factor_name}.parquet
        if not _alt_path.exists():
            _glob = sorted(Path(code_path).parent.glob(f"{_PARQUET_STEM}_*.parquet"))
            if _glob:
                _alt_path = _glob[0]
        if _alt_path.exists():
            import pandas as pd
            df = pd.read_parquet(_alt_path)
            non_null_ratio = round(float(df.notna().mean().mean()), 4)
            result_info = {
                "result_exists": True,
                "result_shape": list(df.shape),
                "date_range": [str(df.index.min()), str(df.index.max())] if len(df) > 0 else None,
                "non_null_ratio": non_null_ratio,
            }
            # 复制到 tmpdir 方便后续 export
            import shutil
            shutil.copy2(_alt_path, result_path)
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
        returncode = -2  # 标记 all-NaN 错误，让调用方明确知道不是正常失败

    return {
        "tmpdir": tmpdir,
        "success": success,
        "returncode": returncode,
        "stdout": stdout or "",
        "stderr": stderr or "",
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
    if lookback is None and args.meta_json:
        try:
            meta = json.loads(args.meta_json)
            lookback = meta.get("lookback_days")
        except (json.JSONDecodeError, AttributeError):
            lookback = None
    if lookback is None:
        lookback = detect_lookback_from_code(user_code)

    # Wrap template
    sys.path.insert(0, str(PROJECT_ROOT))
    from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace

    template = getattr(FactorFBWorkspace, TYPE_MAP[type_key])
    load_cols = [c.strip() for c in re.split(r"[,，\s]+", args.cols) if c.strip()] if args.cols else None
    full_code = FactorFBWorkspace._build_factor_code(template, user_code, lookback, load_cols)

    # Write wrapped code to temp file
    wrapped_tmp = Path(tempfile.mktemp(suffix=".code.py", prefix="factor_wrapped_"))
    wrapped_tmp.write_text(full_code, encoding="utf-8")

    # Run test
    test_result = _run_test_in_tmpdir(wrapped_tmp, timeout=args.timeout, type_key=type_key)

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
            # 失败输出不再回传完整 stdout/stderr（分钟因子日志可达数万字符，纯 token 黑洞）。
            # 改为「首 1500 字符 + 尾 3000 字符」，报错（通常在首尾）不丢；中间被截断并标注省略量。
            "stderr_clip": _clip(test_result.get("stderr", ""), head=2000, tail=4000),
            "stderr_tail": test_result["stderr_tail"],
            "stdout_clip": _clip(test_result.get("stdout", ""), head=1500, tail=3000),
            "stdout_tail": test_result["stdout_tail"],
            "returncode": test_result["returncode"],
            **{k: v for k, v in test_result.items() if k in extra_keys},
        }, ensure_ascii=False, indent=2))
        return 0

    # Success → export
    report_name = args.report
    factor_name = args.factor
    reports_base = (LITERATURE_REPORTS_DIR / args.date) if args.date else LITERATURE_REPORTS_DIR
    factor_dir = reports_base / report_name / factor_name
    factor_dir.mkdir(parents=True, exist_ok=True)

    dst_code = factor_dir / f"{factor_name}.code.py"
    shutil.copy2(wrapped_tmp, dst_code)

    # Copy result.parquet from test tmpdir
    src_result = Path(test_result["tmpdir"]) / "result.parquet"
    has_result = src_result.exists()
    if has_result:
        dst_result = factor_dir / f"{factor_name}.parquet"
        shutil.copy2(src_result, dst_result)

    # Build meta.json with accepted=true（缺失字段自动从因子名/研报名派生）
    meta = {
        "factor_name": factor_name,
        "source_report": report_name,
        "created_at": __import__("datetime").datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "source_type": "literature_report",
        "accepted": True,
        "factor_description": args.description or f"因子 {factor_name}，来自研报 {report_name}",
        "factor_formulation": args.formulation or f"见因子 {factor_name} 代码实现",
        "source_report_title": args.source_report_title or report_name,
        "source_report_path": args.source_report_path or "",
        "source_excerpt": args.source_excerpt or f"因子 {factor_name}，来自 {report_name}",
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
    """Run full-scale factor pipeline synchronously: compute → evaluate → Barra → LLM review.

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

    Copies .code.py from 因子产出/测试/<report>/<factor>/ to
    因子产出/全量/<report>/<factor>/, patches DATA_DIR from _1000 → full path,
    and copies/creates meta.json with pipeline_status='deployed'.
    Does NOT run full computation.
    """
    code_path = Path(args.code).resolve()
    if not code_path.exists():
        print(f"ERROR: code file not found: {code_path}", file=sys.stderr)
        return 1

    factor_name = code_path.stem.replace(".code", "")

    # Locate report + factor from directory structure
    # Expected: literature_reports/<date>/<report>/<factor>/<factor>.code.py
    test_factor_dir = code_path.parent
    report_dir = test_factor_dir.parent
    if report_dir.name == "literature_reports":
        print(f"ERROR: code.py must be inside literature_reports/<date>/<report>/<factor>/", file=sys.stderr)
        return 1
    report_name = report_dir.name

    # Target: 因子产出/全量/[<date>/]<report>/<factor>/  （默认当天日期）
    sys.path.insert(0, str(PROJECT_ROOT))
    from rdagent.app.qlib_rd_loop.factor_full_pipeline import FULL_OUTPUT_BASE
    date_str = args.date or __import__("datetime").datetime.now().strftime("%Y-%m-%d")
    full_base = FULL_OUTPUT_BASE / date_str
    full_factor_dir = full_base / report_name / factor_name
    full_factor_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy .code.py, patch DATA_DIR
    code_text = code_path.read_text(encoding="utf-8")
    patched = code_text.replace("factor_implementation_source_data_1000", "factor_implementation_source_data")

    # 注入多级 DATA_DIR 降级链（环境变量 → 全量 → 测试 → .）
    # 使 .code.py 可在任意环境直接运行，无需设置 FACTOR_DATA_DIR
    _simple_dir = r'DATA_DIR = Path\(os\.environ\.get\("FACTOR_DATA_DIR"\) or os\.environ\.get\("RDAGENT_FACTOR_DATA_DIR"\) or "\."\)'
    if re.search(_simple_dir, patched):
        _subdir = "minute_by_date" if "minute_by_date" in patched else "daily"
        _fallback = (
            'DATA_DIR = Path(os.environ.get("FACTOR_DATA_DIR") or os.environ.get("RDAGENT_FACTOR_DATA_DIR") or "")\n'
            f'if not DATA_DIR or not (DATA_DIR/"stock_data"/"{_subdir}").exists():\n'
            f'    DATA_DIR = Path("{DATA_ROOT / "数据仓库" / "行情数据" / "日线" / "全量"}")\n'
            f'    if not (DATA_DIR/"stock_data"/"{_subdir}").exists():\n'
            f'        DATA_DIR = Path("{DATA_ROOT / "数据仓库" / "行情数据" / "日线" / "测试"}")\n'
            f'        if not (DATA_DIR/"stock_data"/"{_subdir}").exists():\n'
            f'            DATA_DIR = Path(".")\n'
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
# (sync-full 已移除)
# ---------------------------------------------------------------------------

    return 0


def _normalize_source(src, index: int) -> dict:
    """Normalize a sources.json entry to dict format (handles both string URLs and dicts)."""
    if isinstance(src, str):
        return {"url": src, "title": "", "source": ""}
    if isinstance(src, dict):
        return {"url": src.get("url", ""), "title": src.get("title", ""), "source": src.get("source", "")}
    return {"url": "", "title": f"未知来源_{index}", "source": ""}


# ---------------------------------------------------------------------------
# Subcommand: save-extracted
# ---------------------------------------------------------------------------
def cmd_save_extracted(args):
    """Save factor definitions JSON to extracted_reports/ (read from stdin)."""
    extracted_base = LITERATURE_REPORTS_DIR.parent / "extracted_reports"
    if args.date:
        extracted_base = extracted_base / args.date
    extracted_base.mkdir(parents=True, exist_ok=True)
    content = sys.stdin.read()
    if not content.strip():
        print("ERROR: no content on stdin", file=sys.stderr)
        return 1
    path = extracted_base / f"{args.name}.extracted.json"
    path.write_text(content, encoding="utf-8")
    print(f"OK: {path}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: write-repro-report — 生成复现报告（成功/未复现及原因）
# ---------------------------------------------------------------------------
def cmd_write_repro_report(args):
    """为一份测试因子报告生成复现报告 Markdown。

    扫描 因子产出/测试/{DATE}/{report}/ 下每个因子子目录，结合
    extracted_reports/{DATE}/{report}.extracted.json 的因子定义，判定：
      - ✅ 成功复现：存在 {factor}.parquet + {factor}.code.py
      - ❌ 缺列未复现：存在 {factor}.missing.json（原因=missing_fields）
      - ⚠️ 未复现/无产出：有 extracted 定义但无上面两者
    输出 因子产出/测试/{DATE}/{report}/复现报告.md，同时打印结构化 JSON。
    """
    date_str = args.date
    report_name = args.report

    report_dir = LITERATURE_REPORTS_DIR / date_str / report_name
    if not report_dir.exists():
        print(f"ERROR: 报告测试目录不存在: {report_dir}", file=sys.stderr)
        return 1

    # 读取因子定义（extracted.json）
    extracted_path = LITERATURE_REPORTS_DIR.parent / "extracted_reports" / date_str / f"{report_name}.extracted.json"
    extracted = {}
    if extracted_path.exists():
        try:
            extracted = json.loads(extracted_path.read_text(encoding="utf-8"))
        except Exception:
            extracted = {}
    # extracted JSON 有两种历史格式：{"factors": [...]} 与裸 [...]，统一成 dict
    if isinstance(extracted, list):
        extracted = {"factors": extracted}
    elif not isinstance(extracted, dict):
        extracted = {}
    def_by_name = {f.get("name", ""): f for f in extracted.get("factors", []) if f.get("name")}

    success = []
    missing = []
    no_output = []
    ignored = []  # 目录里存在、但不属于本批定义的历史残留

    # ⚠️ 以「本批 extracted 定义的因子」为准逐个查，而不是扫目录。
    # 扫目录会把更早运行残留的因子目录也算进来（因子数虚高、孤儿被当成功复现）。
    if not def_by_name:
        print(f"ERROR: 未找到本批因子定义（extracted_reports/{date_str}/{report_name}.extracted.json 缺失或为空），"
              f"无法确定本批范围，拒绝按目录猜测。", file=sys.stderr)
        return 1

    for factor, definition in sorted(def_by_name.items()):
        sub = report_dir / factor
        parquet = sub / f"{factor}.parquet"
        code = sub / f"{factor}.code.py"
        miss = sub / f"{factor}.missing.json"
        if parquet.exists() and code.exists():
            success.append({"name": factor, **definition})
        elif miss.exists():
            reason = "缺字段"
            try:
                m = json.loads(miss.read_text(encoding="utf-8"))
                mf = m.get("missing_fields") or []
                cc = m.get("checked_cols") or []
                if mf:
                    # checked_cols 形如 "中文语义 consensus_eps"，把英文名翻译成中文（老板可读）
                    parts = []
                    for field in mf:
                        cn = None
                        for entry in cc:
                            s = str(entry)
                            # 找 "中文 xxx 英文名" 里中文部分（英文名前的部分）
                            idx = s.find(field)
                            if idx > 0:
                                cand = s[:idx].strip().rstrip("：（()：")
                                if cand:
                                    cn = cand
                                    break
                        if cn:
                            parts.append(f"{cn}({field})")
                        else:
                            parts.append(field)
                    reason = "缺字段: " + ", ".join(parts)
            except Exception:
                pass
            missing.append({"name": factor, "reason": reason, **definition})
        else:
            no_output.append({"name": factor, "reason": "未生成代码（无 parquet / missing.json）", **definition})

    # 记录被忽略的目录（仅提示，不计入报告）
    for sub in sorted(report_dir.iterdir()):
        if sub.is_dir() and sub.name not in def_by_name:
            ignored.append(sub.name)

    def _trunc(s, n=60):
        s = (s or "").strip().replace("\n", " ")
        return s if len(s) <= n else s[: n - 1] + "…"

    lines = []
    lines.append(f"# 复现报告：{report_name}")
    lines.append("")
    lines.append(f"- 日期: {date_str}")
    lines.append(f"- 因子总数: {len(success) + len(missing) + len(no_output)} | "
                 f"成功复现: {len(success)} | 未复现: {len(missing) + len(no_output)}")
    if ignored:
        lines.append(f"- ⚠️ 已忽略 {len(ignored)} 个不属于本批的残留因子目录（未计入上表）: "
                     f"{', '.join(ignored)}")
    lines.append("")

    lines.append("## 成功复现")
    lines.append("")
    if success:
        lines.append("| 因子 | 类型 | lookback | 描述 |")
        lines.append("|------|------|---------|------|")
        for f in success:
            lines.append(f"| {f['name']} | {f.get('type', '')} | {f.get('lookback', '')} | {_trunc(f.get('description'))} |")
    else:
        lines.append("（无）")
    lines.append("")

    lines.append("## 未复现及原因")
    lines.append("")
    if missing or no_output:
        lines.append("| 因子 | 原因 |")
        lines.append("|------|------|")
        for f in missing + no_output:
            lines.append(f"| {f['name']} | {f['reason']} |")
    else:
        lines.append("（无）")
    lines.append("")

    md = "\n".join(lines)
    out_path = report_dir / "复现报告.md"
    out_path.write_text(md, encoding="utf-8")

    result = {
        "date": date_str,
        "report": report_name,
        "total": len(success) + len(missing) + len(no_output),
        "success": len(success),
        "missing": len(missing),
        "no_output": len(no_output),
        "ignored_stale": ignored,
        "report_path": str(out_path),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: show-columns — 显示可用数据列
# ---------------------------------------------------------------------------
def cmd_show_columns(args):
    """Show available columns in stock data parquet files with descriptions.

    --type 决定展示哪套数据的列：
      minute / minute_cross_section → 分钟线列（open/high/low/close/volume/return/factor）
      其他（daily / cross_section / deep_learning / 不传）→ 日线行情列 + 非行情列

    列含义来源（getdata 导入新字段后自动带出新列含义）：
      /mnt/d/paper-factor-data/schema.json（description）→ factor_field_schema.json（short_name）
    """
    import pyarrow.parquet as _pq

    if args.type in ("minute", "minute_cross_section"):
        minute_dir = DATA_ROOT / "数据仓库" / "行情数据" / "分钟线" / "测试" / "stock_data" / "minute_by_date"
        if not minute_dir.exists():
            print("ERROR: 分钟线测试数据不存在", file=sys.stderr)
            return 1
        files = sorted(minute_dir.glob("*.parquet"))
        if not files:
            print("ERROR: 分钟线测试数据为空", file=sys.stderr)
            return 1
        cols = [n for n in _pq.read_schema(files[0]).names if n not in ("datetime", "instrument")]
        print("可用列及含义（分钟线数据）：")
        for c in cols:
            print(f"  {c:30s} {_desc_for_col(c)}")
        return 0

    # 日线行情列 + 非行情列
    cols = []
    daily_dir = TEST_DATA_DIR / "stock_data" / "daily"
    if daily_dir.exists():
        files = sorted(daily_dir.glob("*.parquet"))
        if files:
            cols += [n for n in _pq.read_schema(files[0]).names if n not in ("datetime", "instrument", "trade_date")]
    fund_dir = DATA_ROOT / "数据仓库" / "非行情数据" / "测试" / "stock_data" / "daily"
    if fund_dir.exists():
        files = sorted(fund_dir.glob("*.parquet"))
        if files:
            cols += [n for n in _pq.read_schema(files[0]).names if n not in ("datetime", "instrument", "trade_date")]
    if not cols:
        print("ERROR: 无法找到数据文件")
        return 1
    print("可用列及含义（日线数据 + 非行情数据）：")
    for c in cols:
        print(f"  {c:30s} {_desc_for_col(c)}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: archive-inbox — 把 inbox 里已处理的研报移到 done/
# ---------------------------------------------------------------------------
def cmd_archive_inbox(args):
    """把 papers/inbox/ 下的研报移入 papers/done/{DATE}/，避免 inbox 越堆越多。

    只移动本次实际处理过的（--names 指定），或全部（不传 --names）。
    同时打印移动明细，便于核对。
    """
    date_str = args.date or __import__("datetime").datetime.now().strftime("%Y-%m-%d")
    dst = DONE_DIR / date_str
    dst.mkdir(parents=True, exist_ok=True)

    if not INBOX_DIR.exists():
        print(json.dumps({"moved": 0, "dst": str(dst), "note": "inbox 不存在"}, ensure_ascii=False))
        return 0

    wanted = set(args.names or [])
    moved = []
    overwritten = []
    for p in sorted(list(INBOX_DIR.glob("*.pdf")) + list(INBOX_DIR.glob("*.md"))):
        if wanted and p.name not in wanted and p.stem not in wanted:
            continue
        target = dst / p.name
        # 同名=同一份研报（可能是重跑）→ 直接覆盖旧档，保持 done 一篇一个文件
        if target.exists():
            target.unlink()
            overwritten.append(p.name)
        shutil.move(str(p), str(target))
        moved.append(p.name)

    print(json.dumps({"moved": len(moved), "overwritten": overwritten,
                      "dst": str(dst), "files": moved},
                     ensure_ascii=False, indent=2))
    return 0


# ═══════════════════════════════════════════════════════════════════
# retrieve-domain-knowledge — 领域知识 RAG 检索
# ═══════════════════════════════════════════════════════════════════
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
    p_pdf.add_argument("--outdir", default=None,
                       help="写每个来源的原文到独立 .txt（主进程预提取用），打印 {源路径: txt路径} 映射")

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
    p_export.add_argument("--date", default=None, help="Date subdirectory (YYYY-MM-DD)")

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
    p_tae.add_argument("--date", default=None, help="Date subdirectory (YYYY-MM-DD), e.g. --date 2026-08-09")

    # trigger-full
    p_full = sub.add_parser("trigger-full", help="Trigger full-scale run via FullPipelineExecutor")
    p_full.add_argument("--code", required=True, help="Factor .code.py file (in literature_reports)")

    # run-full
    p_run = sub.add_parser("run-full", help="Run full pipeline directly from .code.py + metadata (independent)")
    p_run.add_argument("--code", required=True, help="Factor .code.py file path")
    p_run.add_argument("--factor-name", required=True, help="Factor name")
    p_run.add_argument("--report-name", required=True, help="Report name (used for output directory structure)")
    p_run.add_argument("--output", default=None, help="Output directory (default: 因子产出/全量/<report>/<factor>/)")
    p_run.add_argument("--type", default=None, help="Factor type (auto-detected if omitted): daily, minute, cross_section, minute_cs, deep_learning")
    p_run.add_argument("--description", default=None, help="Factor description")
    p_run.add_argument("--formulation", default=None, help="Factor formulation")
    p_run.add_argument("--source-excerpt", default=None, help="Source excerpt text")
    p_run.add_argument("--source-report-title", default=None, help="Source report title")
    p_run.add_argument("--meta", default=None, help="JSON file with metadata (alternative to individual --* args)")

    # save-extracted
    p_se = sub.add_parser("save-extracted", help="Save factor definitions JSON to extracted_reports/ (read from stdin)")
    p_se.add_argument("--name", required=True, help="Report title (stem, e.g. '基于GRU的因子选股')")
    p_se.add_argument("--date", default=None, help="Date subdirectory (YYYY-MM-DD)")

    # write-repro-report
    p_rr = sub.add_parser("write-repro-report", help="Generate 复现报告.md for a test report (success / missing reasons)")
    p_rr.add_argument("--date", required=True, help="Date subdirectory (YYYY-MM-DD)")
    p_rr.add_argument("--report", required=True, help="Report name")

    # show-columns
    p_showcols = sub.add_parser("show-columns", help="Show available columns in stock data (--type minute 显示分钟线列)")
    p_showcols.add_argument("--type", default=None, help="Factor type: daily/minute/cross_section/deep_learning（决定展示日线+非行情列 还是 分钟线列）")

    # archive-inbox
    p_arch = sub.add_parser("archive-inbox",
                            help="把 papers/inbox/ 下已处理的研报移入 papers/done/{DATE}/（避免 inbox 越堆越多）")
    p_arch.add_argument("--date", default=None, help="归档子目录名 (YYYY-MM-DD)，默认当天")
    p_arch.add_argument("--names", nargs="*", default=None,
                        help="只归档这些文件名/报告名（默认归档 inbox 全部）")

    # wait-full
    p_wait = sub.add_parser("wait-full", help="Wait for all submitted full pipeline tasks to complete")

    # deploy-to-full
    p_deploy = sub.add_parser("deploy-to-full", help="Deploy tested factor to full-scale directory (copy code + patch DATA_DIR + inherit meta, no computation)")
    p_deploy.add_argument("--code", required=True, help="Factor .code.py file (in literature_reports/<date>/<report>/<factor>/)")
    p_deploy.add_argument("--type", default=None, help="Factor type (daily, minute, cross_section, minute_cs, deep_learning)")
    p_deploy.add_argument("--date", default=None, help="Date subdirectory (YYYY-MM-DD), e.g. --date 2026-08-09")

    # (sync-full 已移除)

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
        "save-extracted": cmd_save_extracted,
        "write-repro-report": cmd_write_repro_report,
        "show-columns": cmd_show_columns,
        "archive-inbox": cmd_archive_inbox,
    }
    fn = cmd_map[args.command]
    return fn(args)


if __name__ == "__main__":
    sys.exit(main())
