#!/usr/bin/env python3
"""
一键启动 Dashboard 网站。
双击 start.sh → 自动打开浏览器 → 动态显示因子数据。
"""

import http.server
import json
import socketserver
import subprocess
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
PORT = 18765

# ── 每日更新路径 ─────────────────────────────────────────
DAILY_UPDATE_CONFIG = ROOT / "git_ignore_folder" / "daily_update_config.json"
DAILY_UPDATE_STATUS = ROOT / "git_ignore_folder" / "daily_update_status.json"
DAILY_UPDATE_DIR = ROOT / "git_ignore_folder" / "factor_outputs" / "文献因子_每日更新"
FULL_OUTPUT_DIR = ROOT / "git_ignore_folder" / "factor_outputs" / "文献因子_全量"

# ── 数据导出 ──────────────────────────────────────────────

def load_factor_meta(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def get_eval_summary(meta):
    ev = meta.get("evaluation", {})
    if not ev:
        return {}
    return {k: ev.get(k) for k in (
        "ic_mean", "ic_ir", "rank_ic_mean", "rank_ic_ir",
        "long_short_mean", "long_short_sharpe", "long_short_max_dd",
        "n_dates", "ic_positive_ratio",
        "D1_mean", "D2_mean", "D3_mean", "D4_mean", "D5_mean",
        "D6_mean", "D7_mean", "D8_mean", "D9_mean", "D10_mean",
    )}

def get_source_tag(meta, factor_name=""):
    sr = meta.get("source_report", "")
    if sr.startswith("idea__") or factor_name.startswith("idea__"):
        return "idea"
    if sr.startswith("website__") or factor_name.startswith("website__"):
        return "website"
    return "paper"

def export_data():
    LIT = ROOT / "git_ignore_folder" / "factor_outputs" / "literature_reports"
    FULL = ROOT / "git_ignore_folder" / "factor_outputs" / "文献因子_全量"

    # 加载每日更新配置，用于在每个因子上标记是否已启用
    du_cfg = {}
    if DAILY_UPDATE_CONFIG.exists():
        try:
            cfg_data = json.loads(DAILY_UPDATE_CONFIG.read_text(encoding="utf-8"))
            du_cfg = {k: True for k in cfg_data.get("enabled", [])}
        except Exception:
            pass

    # 加载每日增量目录中的状态信息
    du_info = {}  # "report/factor" -> { last_date, needs_update, ... }
    if DAILY_UPDATE_DIR.exists():
        try:
            import pandas as pd
            for report_dir in DAILY_UPDATE_DIR.iterdir():
                if not report_dir.is_dir():
                    continue
                for factor_dir in report_dir.iterdir():
                    if not factor_dir.is_dir():
                        continue
                    pq = factor_dir / f"{factor_dir.name}.parquet"
                    if pq.exists():
                        try:
                            df = pd.read_parquet(pq, columns=[])
                            du_info[f"{report_dir.name}/{factor_dir.name}"] = {
                                "last_date": df.index.max().strftime("%Y-%m-%d") if len(df) > 0 else None,
                                "total_dates": len(df),
                            }
                        except Exception:
                            pass
        except Exception:
            pass

    factors = []
    seen = set()

    # literature_reports (测试输出)
    if LIT.exists():
        for rdir in sorted(LIT.iterdir()):
            if not rdir.is_dir():
                continue
            for fdir in sorted(rdir.iterdir()):
                if not fdir.is_dir():
                    continue
                key = (rdir.name, fdir.name)
                seen.add(key)
                meta = load_factor_meta(fdir / f"{fdir.name}.meta.json")
                code_path = fdir / f"{fdir.name}.code.py"
                parquet_path = fdir / f"{fdir.name}.parquet"
                decile_path = fdir / f"{fdir.name}.decile.png"

                full_dir = FULL / rdir.name / fdir.name
                full_meta_path = full_dir / f"{fdir.name}.meta.json"
                has_full = full_dir.exists() and full_meta_path.exists()
                full_meta = load_factor_meta(full_meta_path) if has_full else {}
                ev = get_eval_summary(full_meta) if has_full else get_eval_summary(meta)

                decile_rel = None
                if has_full and (full_dir / f"{fdir.name}.decile.png").exists():
                    decile_rel = f"/api/img/文献因子_全量/{rdir.name}/{fdir.name}"
                elif decile_path.exists():
                    decile_rel = f"/api/img/literature_reports/{rdir.name}/{fdir.name}"

                code_text = ""
                if code_path.exists():
                    code_text = code_path.read_text(encoding="utf-8")

                factors.append({
                    "name": fdir.name,
                    "report": rdir.name,
                    "report_human": rdir.name.replace("_", " "),
                    "source_tag": get_source_tag(meta, fdir.name),
                    "description": meta.get("description", meta.get("factor_description", "")),
                    "formulation": meta.get("formulation", meta.get("factor_formulation", "")),
                    "type": meta.get("type", "daily"),
                    "has_full": has_full,
                    "has_code": code_path.exists(),
                    "has_parquet": parquet_path.exists(),
                    "evaluation": ev,
                    "decile_url": decile_rel,
                    "code": code_text,
                    "daily_update_enabled": f"{rdir.name}/{fdir.name}" in du_cfg,
                    "daily_update_info": du_info.get(f"{rdir.name}/{fdir.name}"),
                })

    # 文献因子_全量 (补齐)
    if FULL.exists():
        for rdir in sorted(FULL.iterdir()):
            if not rdir.is_dir():
                continue
            for fdir in sorted(rdir.iterdir()):
                if not fdir.is_dir():
                    continue
                key = (rdir.name, fdir.name)
                if key in seen:
                    continue
                meta = load_factor_meta(fdir / f"{fdir.name}.meta.json")
                ev = get_eval_summary(meta)
                decile_path = fdir / f"{fdir.name}.decile.png"
                decile_rel = f"/api/img/文献因子_全量/{rdir.name}/{fdir.name}" if decile_path.exists() else None
                code_text = ""
                cp = fdir / f"{fdir.name}.code.py"
                if cp.exists():
                    code_text = cp.read_text(encoding="utf-8")
                factors.append({
                    "name": fdir.name,
                    "report": rdir.name,
                    "report_human": rdir.name.replace("_", " "),
                    "source_tag": get_source_tag(meta, fdir.name),
                    "description": meta.get("description", meta.get("factor_description", "")),
                    "formulation": meta.get("formulation", meta.get("factor_formulation", "")),
                    "type": meta.get("type", "daily"),
                    "has_full": (fdir / f"{fdir.name}.parquet").exists(),
                    "has_code": cp.exists(),
                    "has_parquet": (fdir / f"{fdir.name}.parquet").exists(),
                    "evaluation": ev,
                    "decile_url": decile_rel,
                    "code": code_text,
                    "daily_update_enabled": f"{rdir.name}/{fdir.name}" in du_cfg,
                    "daily_update_info": du_info.get(f"{rdir.name}/{fdir.name}"),
                })

    # papers 素材
    pdfs = []
    inbox = ROOT / "papers" / "inbox"
    if inbox.exists():
        pdfs = sorted([p.name for p in inbox.iterdir() if p.suffix.lower() == ".pdf"])

    ideas = []
    ideas_file = ROOT / "papers" / "ideas" / "ideas.json"
    if ideas_file.exists():
        try:
            ideas = json.loads(ideas_file.read_text(encoding="utf-8"))
            if not isinstance(ideas, list):
                ideas = []
        except Exception:
            ideas = []

    websites = []
    ws_file = ROOT / "papers" / "website" / "sources.json"
    if ws_file.exists():
        try:
            websites = json.loads(ws_file.read_text(encoding="utf-8"))
            if not isinstance(websites, list):
                websites = []
        except Exception:
            websites = []

    return {
        "factors": factors,
        "total": len(factors),
        "pending_full": sum(1 for f in factors if not f["has_full"] and f["has_parquet"]),
        "full_done": sum(1 for f in factors if f["has_full"]),
        "materials": {"pdfs": pdfs, "ideas": ideas, "websites": websites},
    }


# ── 每日更新 API helpers ──────────────────────────────────

def _load_daily_config() -> dict:
    if DAILY_UPDATE_CONFIG.exists():
        return json.loads(DAILY_UPDATE_CONFIG.read_text(encoding="utf-8"))
    return {"enabled": [], "history": []}


def _save_daily_config(cfg: dict):
    DAILY_UPDATE_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    DAILY_UPDATE_CONFIG.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_trade_dates() -> list:
    for p in [
        ROOT / "git_ignore_folder" / "factor_implementation_source_data" / "stock_data" / "daily" / "trade_dates.json",
        ROOT / "git_ignore_folder" / "factor_implementation_source_data" / "stock_data" / "minute_by_date" / "trade_dates.json",
    ]:
        if p.exists():
            return json.loads(p.read_text())
    return []


def _get_daily_update_state() -> dict:
    """获取每日更新的完整状态（config + 每因子状态）"""
    cfg = _load_daily_config()
    trade_dates = _load_trade_dates()
    latest_date = trade_dates[-1] if trade_dates else None

    factors_status = []
    for factor_key in cfg.get("enabled", []):
        parts = factor_key.split("/")
        if len(parts) != 2:
            continue
        report, factor_name = parts

        info = {
            "key": factor_key,
            "report": report,
            "name": factor_name,
            "latest_date": latest_date,
        }

        # 检查增量 parquet
        daily_parquet = DAILY_UPDATE_DIR / report / factor_name / f"{factor_name}.parquet"
        if daily_parquet.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(daily_parquet, columns=[])
                last_date = str(df.index.max().strftime("%Y-%m-%d")) if len(df) > 0 else None
                info["last_date"] = last_date
                info["total_dates"] = len(df)
                info["has_data"] = True
                if latest_date and last_date:
                    info["needs_update"] = latest_date > last_date
            except Exception:
                info["has_data"] = False
        else:
            # 检查全量是否有
            full_parquet = FULL_OUTPUT_DIR / report / factor_name / f"{factor_name}.parquet"
            info["has_data"] = False
            info["has_full"] = full_parquet.exists()

        # meta
        meta_path = DAILY_UPDATE_DIR / report / factor_name / f"{factor_name}.meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                ev = meta.get("evaluation", {})
                info["evaluation"] = {
                    k: ev.get(k) for k in (
                        "ic_mean", "ic_ir", "rank_ic_mean", "rank_ic_ir",
                        "long_short_mean", "long_short_sharpe", "long_short_max_dd",
                        "n_dates", "ic_positive_ratio",
                    )
                }
            except Exception:
                pass
        factors_status.append(info)

    return {
        "enabled": cfg.get("enabled", []),
        "history": cfg.get("history", []),
        "factors": factors_status,
        "latest_date": latest_date,
    }


# ── HTTP Server ─────────────────────────────────────────

MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".png": "image/png",
    ".json": "application/json; charset=utf-8",
}


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    allow_reuse_address = True

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return b""
        return self.rfile.read(length)

    # ── GET ──

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/api/data":
            self._send_json(export_data())
            return

        if path == "/api/daily-update":
            self._send_json(_get_daily_update_state())
            return

        if path == "/api/daily-update/status":
            if DAILY_UPDATE_STATUS.exists():
                try:
                    self._send_json(json.loads(DAILY_UPDATE_STATUS.read_text(encoding="utf-8")))
                except Exception:
                    self._send_json({"running": False})
            else:
                self._send_json({"running": False})
            return

        if path.startswith("/api/img/"):
            rel = path[len("/api/img/"):]
            from urllib.parse import unquote
            rel = unquote(rel)
            parts = rel.split("/")
            if len(parts) >= 3:
                base_dir, report_name, factor_name = parts[0], parts[1], parts[2]
                img_path = ROOT / "git_ignore_folder" / "factor_outputs" / base_dir / report_name / factor_name / f"{factor_name}.decile.png"
                if img_path.exists():
                    body = img_path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", len(body))
                    self.end_headers()
                    self.wfile.write(body)
                    return
            self.send_response(404)
            self.end_headers()
            return

        if path == "/":
            path = "/dashboard.html"

        fpath = ROOT / path.lstrip("/")
        if fpath.exists() and fpath.is_file():
            body = fpath.read_bytes()
            ext = fpath.suffix.lower()
            self.send_response(200)
            self.send_header("Content-Type", MIME.get(ext, "application/octet-stream"))
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"404 Not Found")

    # ── POST ──

    def do_POST(self):
        path = self.path.split("?")[0]

        # 新增思路
        if path == "/api/add-idea":
            try:
                body = json.loads(self._read_body())
                text = body.get("text", "").strip()
                if not text:
                    self._send_json({"ok": False, "error": "内容不能为空"}, 400)
                    return
                title = body.get("title", "").strip() or None  # None = agent 处理时自动生成

                ideas_file = ROOT / "papers" / "ideas" / "ideas.json"
                ideas_file.parent.mkdir(parents=True, exist_ok=True)
                ideas = []
                if ideas_file.exists():
                    try:
                        ideas = json.loads(ideas_file.read_text(encoding="utf-8"))
                        if not isinstance(ideas, list):
                            ideas = []
                    except Exception:
                        ideas = []
                ideas.append({"title": title, "text": text})
                ideas_file.write_text(json.dumps(ideas, indent=2, ensure_ascii=False), encoding="utf-8")
                self._send_json({"ok": True})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 500)
            return

        # 新增网址
        if path == "/api/add-website":
            try:
                body = json.loads(self._read_body())
                url = body.get("url", "").strip()
                if not url:
                    self._send_json({"ok": False, "error": "URL 不能为空"}, 400)
                    return

                ws_dir = ROOT / "papers" / "website"
                ws_dir.mkdir(parents=True, exist_ok=True)
                sources_file = ws_dir / "sources.json"
                sources = []
                if sources_file.exists():
                    try:
                        sources = json.loads(sources_file.read_text(encoding="utf-8"))
                        if not isinstance(sources, list):
                            sources = []
                    except Exception:
                        sources = []

                if url in sources:
                    self._send_json({"ok": False, "error": "该网址已存在"}, 400)
                    return

                sources.append(url)
                sources_file.write_text(json.dumps(sources, indent=2, ensure_ascii=False), encoding="utf-8")

                # 创建占位 .md 文件
                import hashlib
                slug = "website__" + hashlib.md5(url.encode()).hexdigest()[:22]
                md_file = ws_dir / f"{slug}.md"
                if not md_file.exists():
                    md_file.write_text(f"# {url}\n\nSource: {url}\nDate: {datetime.now().strftime('%Y-%m-%d')}\n", encoding="utf-8")

                self._send_json({"ok": True})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 500)
            return

        # 上传 PDF (multipart/form-data)
        if path == "/api/upload-pdf":
            try:
                ct = self.headers.get("Content-Type", "")
                # 解析 boundary
                import re
                m = re.search(r'boundary=(.+)', ct)
                if not m:
                    self._send_json({"ok": False, "error": "缺少 boundary"}, 400)
                    return
                boundary = m.group(1).strip('"').strip("'").encode()

                raw = self._read_body()
                # 找到文件部分的 filename 和内容
                parts = raw.split(b'--' + boundary)
                filename = None
                data = None
                for part in parts:
                    if b'Content-Disposition' not in part:
                        continue
                    # 取 filename
                    fm = re.search(rb'filename="([^"]*)"', part)
                    if not fm:
                        continue
                    filename = fm.group(1).decode("utf-8", errors="replace") or "upload.pdf"
                    # 取内容（header 后面空行之后）
                    header_end = part.find(b'\r\n\r\n')
                    if header_end == -1:
                        continue
                    data = part[header_end + 4:].rstrip(b'\r\n--')
                    break

                if not filename or data is None:
                    self._send_json({"ok": False, "error": "未找到文件"}, 400)
                    return

                inbox = ROOT / "papers" / "inbox"
                inbox.mkdir(parents=True, exist_ok=True)
                save_path = inbox / filename
                if save_path.exists():
                    self._send_json({"ok": False, "error": f"文件 {filename} 已存在"}, 400)
                    return
                save_path.write_bytes(data)
                self._send_json({"ok": True})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 500)
            return

        # ── 每日更新 API ──

        # Toggle: 启用/禁用因子
        if path == "/api/daily-update/toggle":
            try:
                body = json.loads(self._read_body())
                factor_key = body.get("factor", "").strip()
                if not factor_key or "/" not in factor_key:
                    self._send_json({"ok": False, "error": "格式: report/factor_name"}, 400)
                    return
                cfg = _load_daily_config()
                enabled = cfg.get("enabled", [])
                history = cfg.get("history", [])
                ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

                if factor_key in enabled:
                    enabled.remove(factor_key)
                    history.append({"ts": ts, "action": "remove", "factor": factor_key})
                    _save_daily_config(cfg)
                    self._send_json({"ok": True, "action": "removed", "factor": factor_key})
                else:
                    enabled.append(factor_key)
                    history.append({"ts": ts, "action": "add", "factor": factor_key})
                    _save_daily_config(cfg)
                    self._send_json({"ok": True, "action": "added", "factor": factor_key})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 500)
            return

        # Undo: 撤销最后一步
        if path == "/api/daily-update/undo":
            try:
                cfg = _load_daily_config()
                history = cfg.get("history", [])
                if not history:
                    self._send_json({"ok": False, "error": "无操作记录"}, 400)
                    return
                last = history.pop()
                enabled = cfg.get("enabled", [])
                if last["action"] == "add" and last["factor"] in enabled:
                    enabled.remove(last["factor"])
                elif last["action"] == "remove":
                    enabled.append(last["factor"])
                _save_daily_config(cfg)
                self._send_json({"ok": True, "undone": last})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 500)
            return

        # Run: 触发增量更新执行
        if path == "/api/daily-update/run":
            try:
                body = json.loads(self._read_body()) if self.headers.get("Content-Length", "0") != "0" else {}
                dry_run = body.get("dry_run", False)
                skip_eval = body.get("skip_eval", False)
                skip_sync = body.get("skip_sync", False)

                # 检查是否已在运行
                if DAILY_UPDATE_STATUS.exists():
                    try:
                        st = json.loads(DAILY_UPDATE_STATUS.read_text(encoding="utf-8"))
                        if st.get("running"):
                            self._send_json({"ok": False, "error": "已有更新任务运行中"}, 409)
                            return
                    except Exception:
                        pass

                script = ROOT / "scripts" / "daily_update.py"
                if not script.exists():
                    self._send_json({"ok": False, "error": "daily_update.py 不存在"}, 500)
                    return

                cmd = [sys.executable, str(script)]
                if dry_run:
                    cmd.append("--dry-run")
                if skip_eval:
                    cmd.append("--skip-eval")
                if skip_sync:
                    cmd.append("--skip-sync")

                # 后台执行
                log_path = DAILY_UPDATE_STATUS.parent / "daily_update_run.log"
                with open(log_path, "a") as logf:
                    logf.write(f"\n--- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                    subprocess.Popen(
                        cmd,
                        stdout=logf,
                        stderr=subprocess.STDOUT,
                        cwd=str(ROOT),
                    )

                self._send_json({"ok": True, "message": "更新任务已启动"})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 500)
            return

        self._send_json({"ok": False, "error": "未知路径"}, 404)

    # ── OPTIONS (CORS preflight) ──

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, fmt, *args):
        pass  # 静默


def main():
    # 端口复用
    socketserver.TCPServer.allow_reuse_address = True

    # 检查数据目录
    lit_dir = ROOT / "git_ignore_folder" / "factor_outputs" / "literature_reports"
    if not lit_dir.exists():
        print("  ⚠️  未找到因子数据目录")
        print(f"  期望: {lit_dir}")
        print("  请确认项目路径正确\n")

    url = f"http://localhost:{PORT}/dashboard.html"
    print(f"\n  📊 Paper Factor Dashboard")
    print(f"  ─────────────────────────")
    print(f"  浏览器地址 (复制到浏览器打开):")
    print(f"  → {url}")
    print(f"  Ctrl+C 停止服务器\n")

    # 尝试自动打开浏览器
    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
            httpd.serve_forever()
    except OSError as e:
        print(f"  ❌ 端口 {PORT} 被占用，尝试其他端口...")
        for port in range(PORT + 1, PORT + 100):
            try:
                with socketserver.TCPServer(("", port), DashboardHandler) as httpd:
                    url = f"http://localhost:{port}/dashboard.html"
                    print(f"  → {url}")
                    try:
                        webbrowser.open(url)
                    except Exception:
                        pass
                    httpd.serve_forever()
                break
            except OSError:
                continue
        else:
            print("  ❌ 无法找到可用端口")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n  服务器已停止.")
        sys.exit(0)


if __name__ == "__main__":
    main()
