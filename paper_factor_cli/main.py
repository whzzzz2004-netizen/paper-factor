from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from typing_extensions import Annotated


load_dotenv(".env")

app = typer.Typer(help="Standalone paper_factor: extract and reproduce factors from finance papers.")

DEFAULT_PAPER_REPORT_FOLDER = str(Path.cwd() / "papers" / "inbox")
DEFAULT_FACTOR_PAPER_QUERY = (
    "(cat:q-fin.ST OR cat:q-fin.PM OR cat:q-fin.TR) AND "
    '(all:factor OR all:alpha OR all:predictor OR all:signal OR all:"return prediction" '
    'OR all:"cross-sectional return")'
)

CheckoutOption = Annotated[bool, typer.Option("--checkout/--no-checkout", "-c/-C")]


@contextmanager
def _temporary_env(**updates: object):
    old_values = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _auto_init_workspace(*, download_missing: bool = False) -> None:
    if download_missing:
        from rdagent.app.utils.init_workspace import init_workspace

        init_workspace(force=False)
    else:
        from rdagent.app.utils.init_workspace import validate_workspace_ready

        validate_workspace_ready(require_factor_data=True)


def _is_local_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _launch_factor_dashboard_server(
    host: str,
    port: int,
    *,
    open_browser_tab: bool,
    report_filter: str | None = None,
) -> str:
    dashboard_url = f"http://{host}:{port}"
    if report_filter:
        from urllib.parse import quote

        dashboard_url = f"{dashboard_url}/?report={quote(report_filter)}&refresh=5"
    else:
        dashboard_url = f"{dashboard_url}/?refresh=5"

    if not _is_local_port_open(host, port):
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "rdagent.app.qlib_rd_loop.factor_dashboard",
                "serve",
                "--host",
                host,
                "--port",
                str(port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        for _ in range(20):
            if _is_local_port_open(host, port):
                break
            time.sleep(0.25)
    if open_browser_tab:
        opened = False
        for opener in ("xdg-open", "gio", "gnome-open", "kde-open", "sensible-browser"):
            if not shutil.which(opener):
                continue
            command = [opener, "open", dashboard_url] if opener == "gio" else [opener, dashboard_url]
            try:
                subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                opened = True
                break
            except Exception:
                continue
        if not opened:
            try:
                opened = bool(webbrowser.open_new_tab(dashboard_url))
            except Exception:
                opened = False
        if not opened:
            typer.echo(f"Auto-open did not succeed in this environment. Open this URL manually: {dashboard_url}")
    return dashboard_url


@app.command(name="run")
def run_paper_factor(
    report_folder: str = typer.Option(DEFAULT_PAPER_REPORT_FOLDER, help="Folder containing PDF factor reports."),
    report_file: Optional[str] = typer.Option(
        None,
        help="Specific PDF report to process. If omitted, paper_factor scans the report folder for unprocessed papers.",
    ),
    path: Optional[str] = None,
    all_duration: Optional[str] = None,
    checkout: CheckoutOption = True,
    minimal_mode: bool = typer.Option(
        True,
        "--minimal-mode/--full-mode",
        help="Use the lowest-cost extraction path by skipping report classification and extra hypothesis generation.",
    ),
    auto_fetch: bool = typer.Option(
        False,
        "--auto-fetch/--no-auto-fetch",
        help="Automatically fetch recent arXiv factor papers into the report folder before extraction.",
    ),
    fetch_query: str = typer.Option(DEFAULT_FACTOR_PAPER_QUERY, help="arXiv search query used by --auto-fetch."),
    fetch_max_results: int = typer.Option(20, help="Maximum number of recent arXiv papers to inspect per sync."),
    fetch_download_limit: Optional[int] = typer.Option(
        None,
        help="Maximum number of new PDFs to download during this sync. Defaults to all new matches.",
    ),
    fetch_days_back: int = typer.Option(30, help="Only fetch papers submitted within the last N days."),
    dashboard_host: str = typer.Option("127.0.0.1", help="Host for the factor dashboard server."),
    dashboard_port: int = typer.Option(8765, help="Port for the factor dashboard server."),
    auto_open_dashboard: bool = typer.Option(
        False,
        "--auto-open-dashboard/--no-auto-open-dashboard",
        help="Start the factor dashboard server and open the browser automatically.",
    ),
    llm_max_retry: int = typer.Option(1, "--llm-max-retry", min=1),
    max_factors_per_paper: int = typer.Option(10, "--max-factors-per-paper", min=1, max=10),
    extract_only: bool = typer.Option(
        False,
        "--extract-only/--run-full-pipeline",
        help="Only read report and extract factor info without coding/evaluation/export.",
    ),
) -> None:
    _auto_init_workspace(download_missing=False)
    normalized_report_file = str(Path(report_file).resolve()) if report_file else None

    with _temporary_env(
        MAX_RETRY=str(llm_max_retry),
        LOG_LLM_CHAT_CONTENT="False",
        QLIB_FACTOR_MAX_FACTORS_PER_EXP=str(max_factors_per_paper),
        RDAGENT_PAPER_FACTOR_SKIP_LOW_IC_REPAIR="1",
        RDAGENT_PAPER_FACTOR_FAST="1",
    ):
        try:
            from rdagent.app.qlib_rd_loop.factor_from_report import extract_hypothesis_and_exp_from_reports
            from rdagent.app.qlib_rd_loop.factor_from_report import list_unprocessed_report_paths
            from rdagent.app.qlib_rd_loop.factor_from_report import main as fin_factor_report
            from rdagent.app.qlib_rd_loop.paper_fetcher import sync_latest_factor_papers
        except ModuleNotFoundError as exc:
            missing_name = getattr(exc, "name", None) or str(exc)
            typer.echo(
                "paper_factor cannot start because this Python environment is missing "
                f"the dependency `{missing_name}`. Install this project with its dependencies and rerun."
            )
            raise typer.Exit(code=1) from exc

        if auto_open_dashboard:
            dashboard_url = _launch_factor_dashboard_server(
                dashboard_host,
                dashboard_port,
                open_browser_tab=True,
                report_filter=normalized_report_file,
            )
            typer.echo(f"Factor dashboard: {dashboard_url}")

        if path is not None:
            fin_factor_report(
                report_folder=report_folder,
                path=path,
                all_duration=all_duration,
                checkout=checkout,
                minimal_mode=minimal_mode,
            )
            return

        if normalized_report_file:
            report_path = Path(normalized_report_file)
            if not report_path.exists():
                raise typer.BadParameter(f"Report file does not exist: {normalized_report_file}")
            typer.echo(f"Processing paper: {normalized_report_file}")
            if extract_only:
                _extract_only(extract_hypothesis_and_exp_from_reports, normalized_report_file, minimal_mode)
                return
            fin_factor_report(
                report_folder=report_folder,
                all_duration=all_duration,
                checkout=checkout,
                minimal_mode=minimal_mode,
                report_paths=[normalized_report_file],
            )
            typer.echo("paper_factor finished after processing 1 paper.")
            return

        processed_count = 0
        while True:
            local_pending = list_unprocessed_report_paths(report_folder)
            if local_pending:
                next_report = str(local_pending[0])
                typer.echo(f"Processing local pending paper: {next_report}")
                if extract_only:
                    _extract_only(extract_hypothesis_and_exp_from_reports, next_report, minimal_mode)
                    processed_count += 1
                    continue
                fin_factor_report(
                    report_folder=report_folder,
                    all_duration=all_duration,
                    checkout=checkout,
                    minimal_mode=minimal_mode,
                    report_paths=[next_report],
                )
                processed_count += 1
                continue

            if not auto_fetch:
                break

            try:
                summary = sync_latest_factor_papers(
                    target_dir=report_folder,
                    query=fetch_query,
                    max_results=fetch_max_results,
                    download_limit=1 if fetch_download_limit is None else min(fetch_download_limit, 1),
                    days_back=fetch_days_back,
                )
            except Exception as exc:
                typer.echo(f"Auto-fetch failed, stop fetching new papers: {exc}")
                break

            if summary["downloaded_count"] <= 0:
                typer.echo("No new factor papers to fetch.")
                break

            next_report = summary["downloaded_paths"][0]
            typer.echo(f"Fetched and processing paper: {next_report}")
            fin_factor_report(
                report_folder=report_folder,
                all_duration=all_duration,
                checkout=checkout,
                minimal_mode=minimal_mode,
                report_paths=[next_report],
            )
            processed_count += 1

    typer.echo(f"paper_factor finished after processing {processed_count} paper(s).")


@app.command(name="start")
def start_paper_factor(
    report_folder: str = typer.Option(DEFAULT_PAPER_REPORT_FOLDER, help="Folder containing PDF factor reports."),
    report_file: Optional[str] = typer.Option(
        None,
        help="Specific PDF report to process. If omitted, paper_factor scans the report folder for unprocessed papers.",
    ),
    path: Optional[str] = None,
    all_duration: Optional[str] = None,
    checkout: CheckoutOption = True,
    minimal_mode: bool = typer.Option(
        True,
        "--minimal-mode/--full-mode",
        help="Use the lowest-cost extraction path by skipping report classification and extra hypothesis generation.",
    ),
    auto_fetch: bool = typer.Option(
        False,
        "--auto-fetch/--no-auto-fetch",
        help="Automatically fetch recent arXiv factor papers into the report folder before extraction.",
    ),
    fetch_query: str = typer.Option(DEFAULT_FACTOR_PAPER_QUERY, help="arXiv search query used by --auto-fetch."),
    fetch_max_results: int = typer.Option(20, help="Maximum number of recent arXiv papers to inspect per sync."),
    fetch_download_limit: Optional[int] = typer.Option(
        None,
        help="Maximum number of new PDFs to download during this sync. Defaults to all new matches.",
    ),
    fetch_days_back: int = typer.Option(30, help="Only fetch papers submitted within the last N days."),
    dashboard_host: str = typer.Option("127.0.0.1", help="Host for the factor dashboard server."),
    dashboard_port: int = typer.Option(8765, help="Port for the factor dashboard server."),
    auto_open_dashboard: bool = typer.Option(
        False,
        "--auto-open-dashboard/--no-auto-open-dashboard",
        help="Start the factor dashboard server and open the browser automatically.",
    ),
    llm_max_retry: int = typer.Option(1, "--llm-max-retry", min=1),
) -> None:
    with _temporary_env(
        DS_CODER_COSTEER_ENV_TYPE="docker",
        FACTOR_CoSTEER_EXECUTION_BACKEND="docker",
    ):
        run_paper_factor(
            report_folder=report_folder,
            report_file=report_file,
            path=path,
            all_duration=all_duration,
            checkout=checkout,
            minimal_mode=minimal_mode,
            auto_fetch=auto_fetch,
            fetch_query=fetch_query,
            fetch_max_results=fetch_max_results,
            fetch_download_limit=fetch_download_limit,
            fetch_days_back=fetch_days_back,
            dashboard_host=dashboard_host,
            dashboard_port=dashboard_port,
            auto_open_dashboard=auto_open_dashboard,
            llm_max_retry=llm_max_retry,
            max_factors_per_paper=10,
            extract_only=False,
        )


def _extract_only(extractor, report_file: str, minimal_mode: bool) -> None:
    report_path = Path(report_file)
    exp = extractor(report_file, minimal_mode=minimal_mode)
    preview_path = Path.cwd() / "git_ignore_folder" / "factor_outputs" / "extracted_reports" / f"{report_path.stem}.extracted.json"
    extracted_count = len(exp.sub_tasks) if exp is not None else 0
    typer.echo(f"Extract-only finished. Extracted factors: {extracted_count}")
    typer.echo(f"Preview JSON: {preview_path}")
    if exp is not None and exp.sub_tasks:
        typer.echo("Extracted factor names:")
        for task in exp.sub_tasks:
            typer.echo(f"- {task.factor_name}")


@app.command(name="sync")
def sync_factor_papers(
    report_folder: str = typer.Option(DEFAULT_PAPER_REPORT_FOLDER, help="Folder where downloaded PDF papers are stored."),
    fetch_query: str = typer.Option(DEFAULT_FACTOR_PAPER_QUERY, help="arXiv search query."),
    fetch_max_results: int = typer.Option(20, help="Maximum number of recent arXiv papers to inspect per sync."),
    fetch_download_limit: Optional[int] = typer.Option(None, help="Maximum number of new PDFs to download."),
    fetch_days_back: int = typer.Option(30, help="Only fetch papers submitted within the last N days."),
) -> None:
    from rdagent.app.qlib_rd_loop.paper_fetcher import sync_latest_factor_papers

    summary = sync_latest_factor_papers(
        target_dir=report_folder,
        query=fetch_query,
        max_results=fetch_max_results,
        download_limit=fetch_download_limit,
        days_back=fetch_days_back,
    )
    typer.echo(f"Downloaded {summary['downloaded_count']} new paper(s) into {summary['target_dir']}")
    typer.echo(f"Failed downloads: {summary.get('failed_count', 0)}")
    typer.echo(f"Manifest: {summary['manifest_path']}")


@app.command(name="init")
def init_workspace(force: bool = typer.Option(False, help="Overwrite existing local workspace files.")) -> None:
    from rdagent.app.utils.init_workspace import init_workspace as init_rdagent_workspace

    summary = init_rdagent_workspace(force=force, ingest_factor_improvement=False)
    typer.echo("paper_factor workspace initialized.")
    typer.echo(f"Env: {summary['env']}")
    for item in summary["data"]:
        typer.echo(f"- {item}")


@app.command(name="dashboard")
def dashboard(
    host: str = typer.Option("127.0.0.1", help="Dashboard host."),
    port: int = typer.Option(8765, help="Dashboard port."),
    open_browser: bool = typer.Option(False, "--open-browser/--no-open-browser"),
    report_filter: Optional[str] = typer.Option(None, help="Optional report path filter."),
) -> None:
    url = _launch_factor_dashboard_server(host, port, open_browser_tab=open_browser, report_filter=report_filter)
    typer.echo(f"Factor dashboard: {url}")


def main() -> None:
    app()


def start_main() -> None:
    typer.run(start_paper_factor)


if __name__ == "__main__":
    main()
