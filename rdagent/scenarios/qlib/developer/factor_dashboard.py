from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


FACTOR_OUTPUT_DIR = Path.cwd() / "git_ignore_folder" / "factor_outputs"
FACTOR_MANIFEST_PATH = FACTOR_OUTPUT_DIR / "manifest.csv"
FACTOR_LEADERBOARD_PATH = FACTOR_OUTPUT_DIR / "leaderboard.csv"
FACTOR_DASHBOARD_DIR = FACTOR_OUTPUT_DIR / "dashboard"
FACTOR_DASHBOARD_PATH = FACTOR_DASHBOARD_DIR / "index.html"
MANIFEST_COLUMNS = [
    "factor_name",
    "display_name",
    "hash",
    "rows",
    "non_null",
    "time_granularity",
    "accepted",
    "ic_score",
    "factor_description",
    "factor_formulation",
    "variables",
    "logic_summary",
    "tags",
    "source_type",
    "source_report_title",
    "source_report_path",
    "review_notes",
    "latest_path",
    "metadata_path",
    "code_path",
    "workspace_path",
    "updated_at",
]


def _ensure_factor_output_dirs() -> None:
    FACTOR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FACTOR_DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _to_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [item.strip() for item in text.split(",") if item.strip()]
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    return [str(value)]


def _read_text(path_str: Any) -> str | None:
    if path_str is None or (isinstance(path_str, float) and pd.isna(path_str)):
        return None
    path = Path(str(path_str))
    if not path.exists() or not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _load_manifest() -> pd.DataFrame:
    if not FACTOR_MANIFEST_PATH.exists():
        return pd.DataFrame()
    try:
        manifest = pd.read_csv(FACTOR_MANIFEST_PATH)
    except Exception:
        return pd.DataFrame()
    if manifest.empty:
        return manifest
    if "accepted" in manifest.columns:
        manifest["accepted"] = manifest["accepted"].apply(_to_bool)
    else:
        manifest["accepted"] = False
    if "ic_score" in manifest.columns:
        manifest["ic_score"] = pd.to_numeric(manifest["ic_score"], errors="coerce")
    else:
        manifest["ic_score"] = pd.NA
    if "updated_at" in manifest.columns:
        manifest["updated_at"] = pd.to_datetime(manifest["updated_at"], errors="coerce")
    return manifest


def rebuild_factor_leaderboard(manifest: pd.DataFrame | None = None) -> None:
    _ensure_factor_output_dirs()
    leaderboard = _load_manifest() if manifest is None else manifest.copy()
    if leaderboard.empty:
        pd.DataFrame(columns=["rank", "factor_name", "ic_score", "logic_summary", "tags"]).to_csv(
            FACTOR_LEADERBOARD_PATH,
            index=False,
        )
        return
    leaderboard = leaderboard.sort_values(
        by=["accepted", "ic_score", "updated_at", "factor_name"],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    leaderboard.insert(0, "rank", leaderboard.index + 1)
    preferred_columns = [col for col in ["rank", "factor_name", "ic_score", "logic_summary", "tags"] if col in leaderboard]
    leaderboard[preferred_columns].to_csv(FACTOR_LEADERBOARD_PATH, index=False)


def _format_metric(value: Any, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _format_datetime(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return "-"
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


def _build_source_label(row: pd.Series) -> str:
    source_type = str(row.get("source_type") or "agent_generated")
    if source_type == "literature_report":
        title = row.get("source_report_title")
        return f"Literature report: {title}" if title else "Literature report"
    return source_type.replace("_", " ")


def _build_factor_record(row: pd.Series) -> dict[str, Any]:
    code = _read_text(row.get("code_path"))
    metadata_text = _read_text(row.get("metadata_path"))
    metadata = None
    if metadata_text:
        try:
            metadata = json.loads(metadata_text)
        except json.JSONDecodeError:
            metadata = None
    return {
        "factor_name": str(row.get("factor_name") or ""),
        "display_name": str(row.get("display_name") or row.get("factor_name") or ""),
        "accepted": _to_bool(row.get("accepted")),
        "ic_score": row.get("ic_score"),
        "logic_summary": str(row.get("logic_summary") or ""),
        "factor_description": str(row.get("factor_description") or ""),
        "factor_formulation": str(row.get("factor_formulation") or ""),
        "variables": str(row.get("variables") or ""),
        "review_notes": str(row.get("review_notes") or ""),
        "tags": _to_list(row.get("tags")),
        "source_label": _build_source_label(row),
        "source_report_title": str(row.get("source_report_title") or ""),
        "source_report_path": str(row.get("source_report_path") or ""),
        "latest_path": str(row.get("latest_path") or ""),
        "code_path": str(row.get("code_path") or ""),
        "metadata_path": str(row.get("metadata_path") or ""),
        "workspace_path": str(row.get("workspace_path") or ""),
        "rows": row.get("rows"),
        "non_null": row.get("non_null"),
        "time_granularity": str(row.get("time_granularity") or ""),
        "updated_at": _format_datetime(row.get("updated_at")),
        "code": code or "# Code snapshot not available.",
        "metadata": metadata,
    }


def _render_tag_list(tags: list[str]) -> str:
    if not tags:
        return '<span class="muted">No tags</span>'
    return "".join(f'<span class="tag">{html.escape(tag)}</span>' for tag in tags)


def _render_path(label: str, path: str) -> str:
    if not path:
        return ""
    escaped = html.escape(path)
    return (
        '<div class="path-row">'
        f'<span class="path-label">{html.escape(label)}</span>'
        f'<code>{escaped}</code>'
        "</div>"
    )


def _render_factor_card(record: dict[str, Any]) -> str:
    metrics = [
        ("IC", _format_metric(record["ic_score"])),
        ("Rows", str(record["rows"]) if record["rows"] is not None else "-"),
        ("Non-null", str(record["non_null"]) if record["non_null"] is not None else "-"),
        ("Granularity", record["time_granularity"] or "-"),
        ("Updated", record["updated_at"]),
        ("Source", record["source_label"]),
    ]
    metric_html = "".join(
        '<div class="metric"><div class="metric-label">{}</div><div class="metric-value">{}</div></div>'.format(
            html.escape(label),
            html.escape(value),
        )
        for label, value in metrics
    )
    delete_enabled = "true" if record["accepted"] else "false"
    source_report = ""
    if record["source_report_path"]:
        source_report = (
            '<div class="info-block"><div class="section-label">Report Source</div>'
            f'<div>{html.escape(record["source_report_title"] or Path(record["source_report_path"]).stem)}</div>'
            f'<code>{html.escape(record["source_report_path"])}</code></div>'
        )
    return f"""
    <article
      class="factor-card"
      data-factor-name="{html.escape(record["factor_name"])}"
      data-source-report-path="{html.escape(record["source_report_path"])}"
      data-source-report-title="{html.escape(record["source_report_title"])}"
      data-source-label="{html.escape(record["source_label"])}"
    >
      <div class="card-head">
        <div>
          <h2>{html.escape(record["display_name"])}</h2>
          <div class="subhead">{html.escape(record["factor_name"])}</div>
        </div>
        <div class="card-actions">
          <span class="status-pill {'accepted' if record['accepted'] else 'rejected'}">
            {'Accepted' if record['accepted'] else 'Not accepted'}
          </span>
          <button class="delete-btn" data-factor-name="{html.escape(record["factor_name"])}" data-delete-enabled="{delete_enabled}">
            Delete Factor
          </button>
        </div>
      </div>
      <div class="metrics">{metric_html}</div>
      <div class="tags">{_render_tag_list(record["tags"])}</div>
      <div class="content-grid">
        <section class="info-block">
          <div class="section-label">Logic Summary</div>
          <p>{html.escape(record["logic_summary"] or record["factor_description"] or "-")}</p>
        </section>
        <section class="info-block">
          <div class="section-label">Formula</div>
          <pre>{html.escape(record["factor_formulation"] or "-")}</pre>
        </section>
        <section class="info-block">
          <div class="section-label">Variables</div>
          <pre>{html.escape(record["variables"] or "-")}</pre>
        </section>
        <section class="info-block">
          <div class="section-label">Review Notes</div>
          <pre>{html.escape(record["review_notes"] or "-")}</pre>
        </section>
      </div>
      {source_report}
      <div class="paths">
        {_render_path("Parquet", record["latest_path"])}
        {_render_path("Code", record["code_path"])}
        {_render_path("Metadata", record["metadata_path"])}
        {_render_path("Workspace", record["workspace_path"])}
      </div>
      <details class="code-block">
        <summary>Factor Code</summary>
        <pre><code>{html.escape(record["code"])}</code></pre>
      </details>
    </article>
    """


def render_factor_dashboard(manifest: pd.DataFrame | None = None) -> str:
    dashboard_df = _load_manifest() if manifest is None else manifest.copy()
    records: list[dict[str, Any]] = []
    if not dashboard_df.empty:
        dashboard_df = dashboard_df.sort_values(
            by=["accepted", "ic_score", "updated_at", "factor_name"],
            ascending=[False, False, False, True],
            na_position="last",
        )
        records = [_build_factor_record(row) for _, row in dashboard_df.iterrows()]
    accepted_count = sum(1 for record in records if record["accepted"])
    body = "".join(_render_factor_card(record) for record in records)
    if not body:
        body = '<div class="empty-state">No exported factors yet.</div>'
    updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Factor Dashboard</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --panel: #fffaf2;
      --panel-strong: #fff;
      --ink: #1d2a34;
      --muted: #6c7a86;
      --line: #dbcdb8;
      --accent: #0d6c63;
      --accent-soft: #d9efe9;
      --danger: #a63636;
      --danger-soft: #fbe2df;
      --accepted: #0f766e;
      --shadow: 0 18px 50px rgba(48, 45, 37, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Hiragino Sans GB", sans-serif;
      background:
        radial-gradient(circle at top right, rgba(13, 108, 99, 0.12), transparent 28%),
        linear-gradient(180deg, #f7f2e8 0%, var(--bg) 55%, #efe6d8 100%);
      color: var(--ink);
    }}
    .shell {{
      max-width: 1380px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(255,255,255,0.88), rgba(255,250,242,0.96));
      border: 1px solid rgba(219, 205, 184, 0.9);
      border-radius: 24px;
      padding: 28px;
      box-shadow: var(--shadow);
      margin-bottom: 24px;
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: 34px;
      line-height: 1.1;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      max-width: 860px;
      line-height: 1.6;
    }}
    .hero-stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .hero-stat {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px 16px;
    }}
    .hero-stat-label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}
    .hero-stat-value {{
      font-size: 24px;
      font-weight: 700;
      margin-top: 6px;
    }}
    .toolbar {{
      display: flex;
      gap: 12px;
      align-items: center;
      margin-bottom: 18px;
      flex-wrap: wrap;
    }}
    .toolbar input {{
      flex: 1 1 280px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 12px 16px;
      background: rgba(255,255,255,0.9);
      font-size: 14px;
    }}
    .toolbar-note {{
      color: var(--muted);
      font-size: 13px;
    }}
    .factor-grid {{
      display: grid;
      gap: 18px;
    }}
    .factor-card {{
      background: rgba(255, 250, 242, 0.94);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 22px;
      box-shadow: var(--shadow);
    }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      margin-bottom: 16px;
    }}
    .card-head h2 {{
      margin: 0;
      font-size: 24px;
    }}
    .subhead {{
      margin-top: 6px;
      color: var(--muted);
      font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
      word-break: break-all;
    }}
    .card-actions {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    .status-pill {{
      display: inline-flex;
      align-items: center;
      padding: 8px 12px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      background: #eef2f5;
      color: #334155;
    }}
    .status-pill.accepted {{
      background: var(--accent-soft);
      color: var(--accepted);
    }}
    .delete-btn {{
      border: none;
      border-radius: 999px;
      padding: 10px 14px;
      background: var(--danger-soft);
      color: var(--danger);
      cursor: pointer;
      font-weight: 600;
    }}
    .delete-btn:disabled {{
      cursor: not-allowed;
      opacity: 0.6;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }}
    .metric {{
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px 14px;
    }}
    .metric-label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}
    .metric-value {{
      margin-top: 6px;
      font-size: 18px;
      font-weight: 700;
      word-break: break-word;
    }}
    .tags {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 16px;
    }}
    .tag {{
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(13, 108, 99, 0.08);
      color: var(--accent);
      font-size: 12px;
      font-weight: 600;
    }}
    .muted {{
      color: var(--muted);
    }}
    .content-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
      margin-bottom: 16px;
    }}
    .info-block {{
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
    }}
    .section-label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .info-block p, .info-block pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.6;
      font-family: inherit;
    }}
    .paths {{
      margin-bottom: 14px;
    }}
    .path-row {{
      display: flex;
      gap: 10px;
      align-items: flex-start;
      margin-bottom: 8px;
      flex-wrap: wrap;
    }}
    .path-label {{
      min-width: 80px;
      color: var(--muted);
      font-size: 13px;
    }}
    .path-row code {{
      white-space: pre-wrap;
      word-break: break-all;
      font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
    }}
    .code-block {{
      background: #1c242c;
      color: #eef5f7;
      border-radius: 18px;
      overflow: hidden;
    }}
    .code-block summary {{
      cursor: pointer;
      padding: 14px 16px;
      font-weight: 600;
      background: #25303b;
    }}
    .code-block pre {{
      margin: 0;
      padding: 18px;
      overflow: auto;
      font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
      line-height: 1.6;
    }}
    .empty-state {{
      border: 1px dashed var(--line);
      border-radius: 20px;
      padding: 40px 24px;
      text-align: center;
      color: var(--muted);
      background: rgba(255,255,255,0.7);
    }}
    .flash {{
      position: sticky;
      top: 14px;
      z-index: 10;
      display: none;
      margin-bottom: 14px;
      padding: 14px 16px;
      border-radius: 14px;
      background: var(--accent-soft);
      color: var(--accent);
      font-weight: 600;
    }}
    .flash.error {{
      background: var(--danger-soft);
      color: var(--danger);
    }}
    @media (max-width: 768px) {{
      .shell {{ padding: 22px 14px 32px; }}
      .hero h1 {{ font-size: 28px; }}
      .card-head {{ flex-direction: column; }}
      .card-actions {{ justify-content: flex-start; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div id="flash" class="flash"></div>
    <section class="hero">
      <h1>Factor Registry Dashboard</h1>
      <p>
        Every accepted factor is persisted immediately and rendered here with the most readable view we have:
        name, logic, formula, code snapshot, IC, source, and storage paths. Open this file directly for browsing,
        or run the lightweight local server if you want the delete buttons to work in-page.
      </p>
      <div class="hero-stats">
        <div class="hero-stat"><div class="hero-stat-label">Accepted Factors</div><div class="hero-stat-value">{accepted_count}</div></div>
        <div class="hero-stat"><div class="hero-stat-label">Total Exported</div><div class="hero-stat-value">{len(records)}</div></div>
        <div class="hero-stat"><div class="hero-stat-label">Last Refreshed</div><div class="hero-stat-value">{html.escape(updated_at)}</div></div>
        <div class="hero-stat"><div class="hero-stat-label">Dashboard File</div><div class="hero-stat-value" style="font-size:14px;">{html.escape(str(FACTOR_DASHBOARD_PATH))}</div></div>
      </div>
    </section>
    <div class="toolbar">
      <input id="searchInput" type="search" placeholder="Search factor name, logic, source, or tags" />
      <div class="toolbar-note">Delete buttons require serving this page with the local dashboard server.</div>
    </div>
    <section id="factorGrid" class="factor-grid">
      {body}
    </section>
    <div id="filteredEmptyState" class="empty-state" style="display:none;">No factors are visible for the current report filter yet.</div>
  </div>
  <script>
    const flash = document.getElementById("flash");
    const searchInput = document.getElementById("searchInput");
    const cards = Array.from(document.querySelectorAll(".factor-card"));
    const filteredEmptyState = document.getElementById("filteredEmptyState");
    const params = new URLSearchParams(window.location.search);
    const reportFilter = (params.get("report") || "").trim().toLowerCase();
    const autoRefreshSeconds = Number(params.get("refresh") || "5");

    function showFlash(message, isError = false) {{
      flash.textContent = message;
      flash.className = isError ? "flash error" : "flash";
      flash.style.display = "block";
      window.setTimeout(() => {{
        flash.style.display = "none";
      }}, 3000);
    }}

    function applyFilters() {{
      const query = searchInput.value.trim().toLowerCase();
      let visibleCount = 0;
      for (const card of cards) {{
        const text = card.textContent.toLowerCase();
        const sourcePath = (card.dataset.sourceReportPath || "").toLowerCase();
        const sourceTitle = (card.dataset.sourceReportTitle || "").toLowerCase();
        const matchesReport = !reportFilter || sourcePath.includes(reportFilter) || sourceTitle.includes(reportFilter);
        const matchesSearch = !query || text.includes(query);
        const visible = matchesReport && matchesSearch;
        card.style.display = visible ? "" : "none";
        if (visible) visibleCount += 1;
      }}
      filteredEmptyState.style.display = visibleCount === 0 ? "" : "none";
    }}

    searchInput.addEventListener("input", applyFilters);

    async function deleteFactor(button) {{
      const factorName = button.dataset.factorName;
      if (!factorName) return;
      if (!window.location.protocol.startsWith("http")) {{
        showFlash("Delete needs the local dashboard server. Open this page through the serve command.", true);
        return;
      }}
      const confirmed = window.confirm(`Delete factor "${{factorName}}" from the local factor registry?`);
      if (!confirmed) return;
      button.disabled = true;
      try {{
        const response = await fetch(`/api/factors/${{encodeURIComponent(factorName)}}`, {{ method: "DELETE" }});
        const payload = await response.json();
        if (!response.ok) {{
          throw new Error(payload.error || "Delete failed");
        }}
        const card = document.querySelector(`.factor-card[data-factor-name="${{CSS.escape(factorName)}}"]`);
        if (card) card.remove();
        showFlash(payload.message || `Deleted ${{factorName}}`);
      }} catch (error) {{
        button.disabled = false;
        showFlash(error.message || "Delete failed", true);
      }}
    }}

    document.querySelectorAll(".delete-btn").forEach((button) => {{
      button.addEventListener("click", () => deleteFactor(button));
    }});

    applyFilters();
    if (autoRefreshSeconds > 0) {{
      window.setInterval(() => window.location.reload(), autoRefreshSeconds * 1000);
    }}
  </script>
</body>
</html>
"""


def refresh_factor_dashboard() -> Path:
    _ensure_factor_output_dirs()
    manifest = _load_manifest()
    rebuild_factor_leaderboard(manifest)
    FACTOR_DASHBOARD_PATH.write_text(render_factor_dashboard(manifest), encoding="utf-8")
    return FACTOR_DASHBOARD_PATH


def delete_factor_record(factor_name: str) -> dict[str, Any]:
    normalized = str(factor_name or "").strip()
    if not normalized:
        raise ValueError("factor_name is required")
    manifest = _load_manifest()
    if manifest.empty or "factor_name" not in manifest.columns:
        raise FileNotFoundError(f"Factor registry is empty; cannot delete {normalized}.")

    matches = manifest["factor_name"].astype(str) == normalized
    if not matches.any():
        raise FileNotFoundError(f"Factor {normalized} was not found in the registry.")

    row = manifest.loc[matches].iloc[-1]
    paths_to_delete = [
        row.get("latest_path"),
        row.get("metadata_path"),
        row.get("code_path"),
    ]
    manifest = manifest.loc[~matches].copy()
    if manifest.empty:
        pd.DataFrame(columns=MANIFEST_COLUMNS).to_csv(FACTOR_MANIFEST_PATH, index=False)
    else:
        manifest = manifest.sort_values("factor_name")
        manifest.to_csv(FACTOR_MANIFEST_PATH, index=False)

    for path_str in paths_to_delete:
        if path_str is None or (isinstance(path_str, float) and pd.isna(path_str)):
            continue
        path = Path(str(path_str))
        if path.exists() and path.is_file():
            path.unlink()

    refresh_factor_dashboard()
    return {
        "factor_name": normalized,
        "message": f"Deleted factor {normalized} and refreshed the dashboard.",
    }
