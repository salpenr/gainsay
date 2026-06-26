"""Free web search + fetch.

DuckDuckGo's html endpoint gives us keyword-style results without an API key.
`fetch(url)` pulls and cleans a page so the model can read it.

Every search and fetch is recorded to a JSONL audit log so we can see what
content entered the model's context. Indirect-prompt-injection (OWASP
LLM01) is mitigated by oversight, not by refusing access.
"""
from __future__ import annotations

import functools
import html
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
SEARCH_URL = "https://html.duckduckgo.com/html/?q={q}"
# Optional self-hosted SearXNG metasearch backend. Set SEARXNG_URL to its base
# URL to enable it; left unset, this backend is skipped and search falls back to
# DuckDuckGo. SearXNG aggregates many engines (incl. international ones).
SEARXNG_URL = os.environ.get("SEARXNG_URL", "").rstrip("/")
# Optional self-hosted Whoogle (Google front-end) backend. Set WHOOGLE_URL to its
# base URL to enable it; left unset, this backend is skipped.
WHOOGLE_URL = os.environ.get("WHOOGLE_URL", "").rstrip("/")

# ----- Audit log -----------------------------------------------------------
# Every web call writes a JSONL line so we can see what the model fetched,
# when, and on whose behalf.

AUDIT_DIR = Path(os.environ.get("GAINSAY_AUDIT_DIR",
                                os.path.join(os.path.expanduser("~"), ".gainsay", "web-audit")))
AUDIT_ECHO = os.environ.get("WEB_AUDIT_ECHO", "0") not in ("0", "false", "False")


def _audit_path() -> Path:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    return AUDIT_DIR / f"{time.strftime('%Y-%m-%d')}.jsonl"


def _audit_record(fn: str, args: dict, result_summary: str,
                  duration_ms: int | None = None,
                  error: str | None = None) -> None:
    rec = {
        "ts": time.time(),
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fn": fn,
        "args": args,
        "result_summary": result_summary,
        "duration_ms": duration_ms,
        "error": error,
    }
    try:
        with _audit_path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass
    if AUDIT_ECHO:
        arg_preview = ", ".join(f"{k}={v!r}" for k, v in args.items())[:140]
        marker = "[WEB ERR]" if error else "[WEB]"
        print(f"{marker} {fn}({arg_preview}) -> {result_summary[:120]}", flush=True)


def _audited(fn_name: str):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                import inspect
                bound = inspect.signature(fn).bind(*args, **kwargs)
                bound.apply_defaults()
                arg_dict = dict(bound.arguments)
            except Exception:
                arg_dict = {**kwargs, "_args": list(args)}
            t0 = time.time()
            try:
                result = fn(*args, **kwargs)
            except Exception as e:
                _audit_record(fn_name, arg_dict, "",
                              duration_ms=int((time.time() - t0) * 1000),
                              error=f"{type(e).__name__}: {e}")
                raise
            # Build a short summary depending on result type
            if isinstance(result, list):
                summary = f"{len(result)} results"
                if result and isinstance(result[0], dict) and "url" in result[0]:
                    urls = [r.get("url", "")[:80] for r in result[:3]]
                    summary += " | top: " + ", ".join(urls)
            elif isinstance(result, str):
                summary = f"{len(result)} chars"
            else:
                summary = type(result).__name__
            _audit_record(fn_name, arg_dict, summary,
                          duration_ms=int((time.time() - t0) * 1000))
            return result
        return wrapper
    return deco

# DDG's html endpoint emits a block per result. Parse it in two passes:
# first slice on result blocks, then grab title/href/snippet within each.
_RESULT_SPLIT_RE = re.compile(r'class="result__title"', re.IGNORECASE)
_TITLE_RE = re.compile(
    r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)
_SNIPPET_RE = re.compile(
    r'class="result__snippet"[^>]*>(.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)


def _strip_html(s: str) -> str:
    s = _SCRIPT_RE.sub("", s)
    s = _TAG_RE.sub("", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n\s*\n+", "\n\n", s)
    return s.strip()


def _unwrap_ddg(href: str) -> str:
    if href.startswith("//duckduckgo.com/l/?"):
        href = "https:" + href
    if "duckduckgo.com/l/?" in href:
        q = urllib.parse.urlparse(href).query
        params = urllib.parse.parse_qs(q)
        if "uddg" in params:
            return urllib.parse.unquote(params["uddg"][0])
    return href


def _search_ddg(query: str, n: int, _caller: str | None = None) -> list[dict]:
    url = SEARCH_URL.format(q=urllib.parse.quote_plus(query))
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as r:
        body = r.read().decode("utf-8", errors="replace")
    blocks = _RESULT_SPLIT_RE.split(body)[1:]
    out: list[dict] = []
    for block in blocks:
        t = _TITLE_RE.search(block)
        if not t:
            continue
        s = _SNIPPET_RE.search(block)
        out.append({
            "title":   _strip_html(t.group(2)),
            "url":     _unwrap_ddg(t.group(1)),
            "snippet": _strip_html(s.group(1)) if s else "",
            "source":  "ddg",
        })
        if len(out) >= n:
            break
    return out


# Bing and Brave fallback parsers were attempted but both engines block
# static HTTP scrapers (Bing returns a consent interstitial, Brave renders
# results client-side). To get real multi-engine fallback, run a local
# SearXNG (https://github.com/searxng/searxng) Docker container and point
# web.search at its JSON endpoint -- documented but not built.

def _search_bing(query: str, n: int, _caller: str | None = None) -> list[dict]:
    """Bing static-HTML scrape -- DOES NOT WORK in practice (consent
    interstitial returns 0 results). Kept as a stub so the fallback chain
    still iterates without erroring; remove when real Bing access exists."""
    return []


def _search_brave(query: str, n: int, _caller: str | None = None) -> list[dict]:
    """Brave Search static scrape -- DOES NOT WORK (results render via JS,
    static HTML is footer/branding only). Stub so the fallback chain still
    iterates. To enable: install Playwright + replace this with a headless
    fetch, or self-host SearXNG."""
    return []


def _search_whoogle(query: str, n: int, _caller: str | None = None) -> list[dict]:
    """Query the self-hosted Whoogle (Google front-end). Parses its result HTML
    with BeautifulSoup. Returns [] (falls back) if WHOOGLE_URL is unset or
    Whoogle isn't running."""
    if not WHOOGLE_URL:
        return []  # backend not configured -> skip
    url = f"{WHOOGLE_URL}/search?q={urllib.parse.quote_plus(query)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            doc = r.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError):
        return []  # Whoogle down -> let the chain fall back
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []
    soup = BeautifulSoup(doc, "html.parser")
    out, seen = [], set()
    for h3 in soup.find_all("h3"):
        a = h3.find_parent("a")
        if not a:
            continue
        href = a.get("href", "")
        if not href.startswith("http"):
            continue
        if any(s in href for s in ("google.com/", "maps.google", "127.0.0.1", "/search?")):
            continue
        if href in seen:
            continue
        seen.add(href)
        title = h3.get_text(" ", strip=True)
        full = a.get_text(" ", strip=True)
        snippet = full[len(title):].strip() if full.startswith(title) else ""
        out.append({"title": title, "url": href, "snippet": snippet, "source": "whoogle"})
        if len(out) >= n:
            break
    return out


def _search_searxng(query: str, n: int, _caller: str | None = None) -> list[dict]:
    """Query the self-hosted SearXNG metasearch JSON API. Aggregates many
    engines (incl. international), far more reliable than scraping one engine.
    Falls through (returns []) if SEARXNG_URL is unset or SearXNG isn't
    running."""
    if not SEARXNG_URL:
        return []  # backend not configured -> skip
    url = f"{SEARXNG_URL}/search?q={urllib.parse.quote_plus(query)}&format=json"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError):
        return []  # SearXNG down -> let the chain fall back to DDG
    out: list[dict] = []
    for item in data.get("results", []):
        if not item.get("url"):
            continue
        out.append({
            "title":   item.get("title", ""),
            "url":     item.get("url", ""),
            "snippet": item.get("content", "") or "",
            "source":  "searxng",
        })
        if len(out) >= n:
            break
    return out


def _search_ddgs(query: str, n: int, _caller: str | None = None) -> list[dict]:
    """Query DuckDuckGo via the maintained `ddgs` library, which tracks DDG's current
    API + anti-bot far better than scraping html.duckduckgo.com (which DDG now blocks
    outright). Falls through (returns []) if the library is missing or the query fails."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS  # older package name, same API
    except ImportError:
        return []
    try:
        rows = list(DDGS().text(query, max_results=n))
    except Exception:
        return []  # blocked/rate-limited/offline -> let the chain fall back
    out: list[dict] = []
    for r in rows:
        url = r.get("href") or r.get("url") or ""
        if not url:
            continue
        out.append({
            "title":   r.get("title", "") or "",
            "url":     url,
            "snippet": r.get("body", "") or r.get("snippet", "") or "",
            "source":  "ddgs",
        })
        if len(out) >= n:
            break
    return out


@_audited("search")
def search(query: str, n: int = 5, _caller: str | None = None) -> list[dict]:
    """Return up to n results. Tries the optional self-hosted backends
    (Whoogle, SearXNG) when configured, then DuckDuckGo, then the Bing/Brave
    stubs in sequence -- each backend indexes differently; what one misses
    another often catches.

    `_caller` is an optional free-form identity label recorded in the audit
    log; it has no effect on whether the call proceeds.
    """
    last_err: Exception | None = None
    for backend in (_search_whoogle, _search_searxng, _search_ddgs, _search_ddg, _search_bing, _search_brave):
        try:
            results = backend(query, n, _caller=_caller)
        except Exception as e:
            last_err = e
            results = []
        if results:
            return results
    return []


def search_all(query: str, n: int = 5) -> dict:
    """Run BOTH backends and return their results separately. Useful when
    you want to compare or de-dupe across engines, or when one engine's
    indexing differs (DDG is better for small sites, Bing for fresh news)."""
    out = {"ddg": [], "bing": []}
    try:
        out["ddg"] = _search_ddg(query, n)
    except Exception as e:
        out["ddg_error"] = str(e)
    try:
        out["bing"] = _search_bing(query, n)
    except Exception as e:
        out["bing_error"] = str(e)
    return out


def _auto_referer(url: str) -> str | None:
    """Some CDNs 403 unless the request carries a Referer pointing to the parent
    domain. Populate this map with `host -> referer URL` entries if you hit such
    a site; empty by default."""
    p = urllib.parse.urlparse(url)
    h = p.netloc.lower()
    referer_map: dict[str, str] = {}
    return referer_map.get(h)


_META_CHARSET_RE = re.compile(rb'charset=["\']?([\w\-]+)', re.IGNORECASE)


def _decode(raw: bytes, content_type: str) -> str:
    """Decode page bytes to text using the page's declared charset, so
    international pages (GBK / Shift_JIS / cp1251 / ISO-8859-x ...) aren't
    garbled. Order: Content-Type header -> HTML <meta charset> -> utf-8 ->
    charset-normalizer (if installed) -> cp1252 -> latin-1."""
    enc = None
    m = re.search(r"charset=([\w\-]+)", content_type or "", re.IGNORECASE)
    if m:
        enc = m.group(1).strip()
    if not enc:
        mm = _META_CHARSET_RE.search(raw[:2048])
        if mm:
            try:
                enc = mm.group(1).decode("ascii", "ignore").strip()
            except Exception:
                enc = None
    for cand in (enc, "utf-8"):
        if not cand:
            continue
        try:
            return raw.decode(cand)
        except (LookupError, UnicodeDecodeError):
            pass
    try:  # best-effort statistical detection if the lib is around
        from charset_normalizer import from_bytes
        best = from_bytes(raw).best()
        if best:
            return str(best)
    except Exception:
        pass
    for cand in ("cp1252", "latin-1"):
        try:
            return raw.decode(cand)
        except (LookupError, UnicodeDecodeError):
            pass
    return raw.decode("utf-8", "replace")


@_audited("fetch")
def fetch(url: str, max_chars: int = 8000, referer: str | None = None,
          _caller: str | None = None) -> str:
    """Fetch a URL and return readable text. Truncates to max_chars.

    Auto-sets a Referer for any host listed in `_auto_referer`'s map (empty by
    default) so requests to CDNs that require one don't hit their 403 page.

    `_caller` is an optional free-form identity label recorded in the audit
    log; it has no effect on whether the fetch proceeds.
    """
    headers = {"User-Agent": USER_AGENT}
    ref = referer or _auto_referer(url)
    if ref:
        headers["Referer"] = ref
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=20) as r:
        ct = (r.headers.get("Content-Type") or "").lower()
        if "text" not in ct and "html" not in ct and "json" not in ct:
            return f"[non-text content: {ct}]"
        raw = _decode(r.read(), ct)
    text = _strip_html(raw) if "html" in ct else raw
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[...truncated, {len(text) - max_chars} more chars]"
    return text
