#!/usr/bin/env python3
"""Gainsay - browser UI.

A local, private "search engine" face for gainsay.py. Serves a browser-style
UI at http://127.0.0.1:8077 and answers queries by running the SAME engine the CLI
uses: web crawlers (web.py: DuckDuckGo search + page fetch) + a bring-your-own
reference corpus (rag.py, empty by default) + local synthesis (gpt-oss:20b),
with inline [B#]/[W#] citations.

Binds 127.0.0.1 only - private by construction. Only the bare keyword query
touches the network (via web.py, audited); the question context, retrieved
passages, and answer never leave this machine.

Run:   py -3.12 gainsay_web.py            (opens your browser)
       py -3.12 gainsay_web.py --no-open  (don't auto-open)
Env:   GAINSAY_PORT (default 8077), GAINSAY_MODEL (default gpt-oss:20b)
"""
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gainsay  # noqa: E402  -- the engine (web crawlers + library + synthesis)

PORT = int(os.environ.get("GAINSAY_PORT", "8077"))
UI = HERE / "lens_web" / "index.html"

# ---- Favorites: a small SERVER-SIDE store so bookmarks survive a browser cache-clear and
# show in every browser on this machine. (localStorage was per-profile and silently wiped by
# "clear browsing data".) Stays 100% local: the file lives under %LOCALAPPDATA% and the
# server is 127.0.0.1-only.
import urllib.parse  # noqa: E402

_APPDATA = os.environ.get("LOCALAPPDATA") or str(Path.home())
FAV_DIR = Path(_APPDATA) / "gainsay"
FAV_FILE = FAV_DIR / "favorites.json"
FAV_SEED = []  # no default favorites; users add their own
ALLOWED_ORIGINS = {f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}"}


def _fav_clean(items):
    """Server-side allowlist (defense in depth, not just the client): http/https only,
    bounded name/url length, capped count — a javascript:/data:/file: URL can never persist."""
    out = []
    if not isinstance(items, list):
        return out
    for it in items[:60]:
        if not isinstance(it, dict):
            continue
        url = str(it.get("url", "")).strip()[:2048]
        try:
            parts = urllib.parse.urlsplit(url)
        except Exception:
            continue
        if parts.scheme not in ("http", "https") or not parts.netloc:
            continue
        name = (str(it.get("name", "")).strip()[:40]) or parts.hostname or url
        out.append({"name": name, "url": url})
    return out


def _fav_load():
    try:
        data = json.loads(FAV_FILE.read_text(encoding="utf-8"))
        return _fav_clean(data.get("favorites") if isinstance(data, dict) else data)
    except FileNotFoundError:
        return list(FAV_SEED)
    except Exception:
        return []


def _fav_save(items):
    cleaned = _fav_clean(items)
    FAV_DIR.mkdir(parents=True, exist_ok=True)
    tmp = FAV_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"favorites": cleaned}, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(FAV_FILE)  # atomic
    return cleaned


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # quiet console
        pass

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            try:
                self._send(200, UI.read_text(encoding="utf-8"), "text/html; charset=utf-8")
            except Exception as e:
                self._send(500, f"<h1>UI missing</h1><p>{e}</p>", "text/html")
        elif path == "/health":
            self._send(200, json.dumps({"ok": True, "model": gainsay.MODEL}))
        elif path == "/favorites":
            self._send(200, json.dumps({"favorites": _fav_load()}, ensure_ascii=False))
        elif path == "/history":
            self._send(200, json.dumps({**gainsay.lens_history.stats(),
                                        "integrity": gainsay.lens_history.verify()}))
        elif path == "/favicon.ico":
            self._send(204, b"")
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def _sse_event(self, payload: dict) -> None:
        """Write one Server-Sent Event and flush, so the browser sees tokens live."""
        self.wfile.write(b"data: " + json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n\n")
        self.wfile.flush()

    def _sse_stream(self, events):
        """Open an SSE response and pump an iterator of event dicts through it."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for ev in events:
                self._sse_event(ev)
        except Exception as e:
            try:
                self._sse_event({"type": "error", "error": f"{type(e).__name__}: {e}"})
            except Exception:
                pass

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path not in ("/ask", "/ask_stream", "/report_stream", "/favorites", "/history/clear"):
            self._send(404, json.dumps({"error": "not found"}))
            return
        # Refuse cross-site writes: a foreign web page must not POST to this localhost
        # endpoint through the user's browser. Same-origin / no-Origin ok.
        origin = self.headers.get("Origin")
        if origin and origin not in ALLOWED_ORIGINS:
            self._send(403, json.dumps({"error": "cross-origin POST refused"}))
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self._send(400, json.dumps({"error": f"bad request: {e}"}))
            return
        if path == "/favorites":
            self._send(200, json.dumps({"favorites": _fav_save(req.get("favorites"))}, ensure_ascii=False))
            return
        if path == "/history/clear":
            gainsay.lens_history.clear()
            self._send(200, json.dumps({"ok": True, **gainsay.lens_history.stats()}))
            return
        q = (req.get("q") or "").strip()
        if not q:
            self._send(400, json.dumps({"error": "empty query"}))
            return
        if path == "/report_stream":
            import lens_report  # lazy: pulls gainsay which is already loaded
            rkw = dict(
                n_web=int(req.get("web", 5)),
                k_books=int(req.get("books", 4)),
                fetch_top=int(req.get("fetch_top", 3)),
                use_web=bool(req.get("use_web", True)),
                use_books=bool(req.get("use_books", True)),
                do_translate=bool(req.get("translate", False)),
                use_scholar=bool(req.get("scholar", False)),
            )
            self._sse_stream(lens_report.report_stream(q, **rkw))
            return
        kw = dict(
            n_web=int(req.get("web", 5)),
            k_books=int(req.get("books", 4)),
            fetch_top=int(req.get("fetch_top", 3)),
            use_web=bool(req.get("use_web", True)),
            use_books=bool(req.get("use_books", True)),
            do_translate=bool(req.get("translate", False)),
            deep=bool(req.get("deep", False)),
            verify=bool(req.get("verify", False)),
            use_scholar=bool(req.get("scholar", False)),
            history=bool(req.get("history", False)),
        )
        if path == "/ask_stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                for ev in gainsay.ask_stream(q, **kw):
                    self._sse_event(ev)
            except Exception as e:
                try:
                    self._sse_event({"type": "error", "error": f"{type(e).__name__}: {e}"})
                except Exception:
                    pass
            return
        try:
            res = gainsay.ask(q, **kw)
            self._send(200, json.dumps(res, ensure_ascii=False))
        except Exception as e:
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))


def main():
    host = os.environ.get("GAINSAY_HOST", "127.0.0.1")  # 0.0.0.0 to reach over the LAN (scope your firewall accordingly)
    httpd = ThreadingHTTPServer((host, PORT), Handler)
    url = f"http://{host}:{PORT}/"
    print(f"Gainsay UI  ->  {url}   (model: {gainsay.MODEL})")
    print("Ctrl+C to stop.")
    if "--no-open" not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
        httpd.shutdown()


if __name__ == "__main__":
    main()
