"""The interactive review dashboard: a localhost-only server.

`review dashboard` runs this. It renders the page from the live store on every load
and applies issue actions (resolve / dismiss / delete) directly, so the dashboard is
read-and-act in one place. Transcripts are rendered on demand. Stdlib only; binds to
127.0.0.1.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from makker.review.dashboard import render_page

_STATUS = {"resolve": "fixed", "dismiss": "dismissed", "reopen": "open"}


def _apply_action(action: str, issue_id: str) -> tuple[bool, str]:
    from makker.review.issues import delete_issue, set_status

    if action == "delete":
        return delete_issue(issue_id), "deleted"
    if action in _STATUS:
        return set_status(issue_id, _STATUS[action]), _STATUS[action]
    return False, f"unknown action {action!r}"


def _fix_prompt(issue_id: str) -> str | None:
    from makker.review.discovery import find_session
    from makker.review.issues import load_issues
    from makker.review.prompt import fix_prompt

    issue = load_issues().get(issue_id)
    if issue is None:
        return None
    paths = [str(s.trace_path) for sid in issue.sessions if (s := find_session(sid))]
    return fix_prompt(issue, transcript_paths=paths or None)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the terminal quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlsplit(self.path)
        if url.path in ("/", "/index.html"):
            self._send(200, render_page().encode("utf-8"), "text/html; charset=utf-8")
            return
        if url.path.startswith("/transcripts/"):
            self._send_transcript(url.path[len("/transcripts/") :].removesuffix(".html"))
            return
        if url.path == "/api/fix":
            sid = parse_qs(url.query).get("id", [""])[0]
            text = _fix_prompt(sid)
            if text is None:
                self._send(404, b"no such issue", "text/plain")
            else:
                self._send(200, text.encode("utf-8"), "text/plain; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain")

    def _send_transcript(self, session_id: str) -> None:
        from makker.review.discovery import find_session, render_trace_html

        session = find_session(session_id)
        if session is None:
            self._send(404, b"no such session", "text/plain")
            return
        self._send(200, render_trace_html(session.trace_path).encode("utf-8"), "text/html")

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/api/action":
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            ok, msg = _apply_action(str(payload.get("action")), str(payload.get("id")))
        except (json.JSONDecodeError, ValueError) as exc:
            self._send(400, str(exc).encode(), "text/plain")
            return
        body = json.dumps({"ok": ok, "message": msg}).encode()
        self._send(200 if ok else 404, body, "application/json")


def serve_dashboard(*, port: int = 8765, open_browser: bool = True) -> None:
    """Serve the dashboard until interrupted (Ctrl-C)."""
    import webbrowser

    httpd = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"Review dashboard at {url} (Ctrl-C to stop).")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()
