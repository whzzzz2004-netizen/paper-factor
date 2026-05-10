from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from rdagent.scenarios.qlib.developer.factor_dashboard import (
    FACTOR_DASHBOARD_PATH,
    delete_factor_record,
    refresh_factor_dashboard,
)


class FactorDashboardHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, path: Path) -> None:
        body = path.read_text(encoding="utf-8").encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        refresh_factor_dashboard()
        if self.path in {"/", "/index.html"}:
            self._send_html(FACTOR_DASHBOARD_PATH)
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_DELETE(self) -> None:  # noqa: N802
        prefix = "/api/factors/"
        if not self.path.startswith(prefix):
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return
        factor_name = unquote(self.path[len(prefix) :]).strip().strip("/")
        try:
            payload = delete_factor_record(factor_name)
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
            return
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._send_json(payload)

    def log_message(self, format: str, *args) -> None:
        return


def build() -> str:
    path = refresh_factor_dashboard()
    return str(path)


def delete_factor(factor_name: str) -> dict:
    return delete_factor_record(factor_name)


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    refresh_factor_dashboard()
    server = ThreadingHTTPServer((host, port), FactorDashboardHandler)
    print(f"Serving factor dashboard at http://{host}:{port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "build": build,
            "delete_factor": delete_factor,
            "serve": serve,
        }
    )
