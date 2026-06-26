#!/usr/bin/env python3
"""
lens_scholar.py — scholarly source connectors for Gainsay (the [S#] tier).

The genuinely-new capability from the "trustworthy research" stack: citation-grade
sources from free, key-less scholarly APIs, so an answer can be grounded in (and
checked against) the primary literature, not just the open web.

Connectors (all free, no API key):
  - OpenAlex          — primary; rich metadata, reconstructable abstracts, citation
                        counts, ~unlimited with a mailto. Indexes 250M+ works.
  - Semantic Scholar  — graph API; abstracts + citationCount (rate-limited sans key)
  - arXiv             — preprints; Atom feed

Results merge + dedup by DOI/title and rank by citation count (with an abstract
bonus). Each becomes an [S#] source carrying venue + year + DOI so every claim is
traceable to a real paper.

PRIVACY: like web search, ONLY the keyword query leaves the machine (never the
user's question, the retrieved passages, or the answer). Scholarly abstracts are
short, attributed, reputable-API content — treated as trusted-but-cited (not
untrusted like raw web pages), but still defanged defensively before synthesis.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

_UA = "gainsay/1.0 (research)"
_MAILTO = ""
_TIMEOUT = 15


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _openalex_abstract(inv: dict | None) -> str:
    """OpenAlex stores abstracts as an inverted index {word: [positions]}; rebuild it."""
    if not inv:
        return ""
    pos: dict[int, str] = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))[:1500]


def _openalex(query: str, n: int) -> list[dict]:
    params = {"search": query, "per_page": n}
    if _MAILTO:
        params["mailto"] = _MAILTO
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    out = []
    for w in (_get_json(url).get("results") or [])[:n]:
        loc = (w.get("primary_location") or {}).get("source") or {}
        out.append({
            "title": w.get("title") or "",
            "abstract": _openalex_abstract(w.get("abstract_inverted_index")),
            "year": w.get("publication_year"),
            "authors": [(a.get("author") or {}).get("display_name", "")
                        for a in (w.get("authorships") or [])][:5],
            "venue": loc.get("display_name", "") or "",
            "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
            "url": w.get("doi") or w.get("id") or "",
            "citations": w.get("cited_by_count", 0),
            "api": "openalex",
        })
    return out


def _semantic_scholar(query: str, n: int) -> list[dict]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode(
        {"query": query, "limit": n,
         "fields": "title,abstract,year,authors,venue,citationCount,externalIds,url"})
    out = []
    for p in (_get_json(url).get("data") or [])[:n]:
        ext = p.get("externalIds") or {}
        out.append({
            "title": p.get("title") or "",
            "abstract": (p.get("abstract") or "")[:1500],
            "year": p.get("year"),
            "authors": [a.get("name", "") for a in (p.get("authors") or [])][:5],
            "venue": p.get("venue") or "",
            "doi": ext.get("DOI", ""),
            "url": ("https://doi.org/" + ext["DOI"]) if ext.get("DOI") else (p.get("url") or ""),
            "citations": p.get("citationCount"),
            "api": "semanticscholar",
        })
    return out


def _arxiv(query: str, n: int) -> list[dict]:
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": f"all:{query}", "max_results": n})
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        root = ET.fromstring(r.read().decode("utf-8", "replace"))
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in root.findall("a:entry", ns)[:n]:
        out.append({
            "title": " ".join((e.findtext("a:title", "", ns) or "").split()),
            "abstract": " ".join((e.findtext("a:summary", "", ns) or "").split())[:1500],
            "year": (e.findtext("a:published", "", ns) or "")[:4],
            "authors": [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)][:5],
            "venue": "arXiv preprint",
            "doi": "",
            "url": e.findtext("a:id", "", ns) or "",
            "citations": None,
            "api": "arxiv",
        })
    return out


def _norm_key(r: dict) -> str:
    doi = (r.get("doi") or "").lower().strip()
    if doi:
        return doi
    return "".join(ch for ch in (r.get("title", "").lower()) if ch.isalnum())[:60]


def search(query: str, n: int = 5) -> tuple[list[dict], list[str]]:
    """Query the scholarly APIs, merge + dedup by DOI/title, rank by citations.
    Returns (items, apis_that_answered). Each connector is best-effort: a failure
    (timeout, rate-limit, schema drift) drops that source, never the whole search."""
    results: list[dict] = []
    answered: list[str] = []
    for name, fn in (("openalex", _openalex), ("semanticscholar", _semantic_scholar),
                     ("arxiv", _arxiv)):
        try:
            got = fn(query, n)
            if got:
                results += got
                answered.append(name)
        except Exception:
            continue
    # Preserve each API's own RELEVANCE ordering (OpenAlex/arXiv already rank by
    # relevance) and merge in that order. Do NOT re-sort by citation count — that
    # surfaces famous-but-off-topic megahits (a 17k-citation review for an
    # unrelated query). Relevance reranking is the caller's job (lens_rerank);
    # citations stay as displayed metadata. Return a wide candidate pool so the
    # reranker has room to pick.
    seen: set[str] = set()
    merged: list[dict] = []
    for r in results:
        k = _norm_key(r)
        if not k or k in seen:
            continue
        seen.add(k)
        merged.append(r)
    return merged, answered


def format_label(r: dict) -> str:
    """One-line human label: 'Title (First Author et al., Year, Venue) [N cites]'."""
    au = r.get("authors") or []
    who = (au[0] + (" et al." if len(au) > 1 else "")) if au else "unknown"
    bits = [who]
    if r.get("year"):
        bits.append(str(r["year"]))
    if r.get("venue"):
        bits.append(r["venue"])
    tail = f"  [{r['citations']} cites]" if r.get("citations") else ""
    return f"{r.get('title','(untitled)')} ({', '.join(bits)}){tail}"


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "CRISPR-Cas9 off-target effects"
    items, apis = search(q, n=5)
    print(f"Q: {q}\nAPIs answered: {apis}\n")
    for i, r in enumerate(items, 1):
        print(f"[S{i}] {format_label(r)}")
        print(f"      doi={r.get('doi') or '-'}  api={r['api']}")
        print(f"      {(r.get('abstract') or '(no abstract)')[:160]}")
        print()
