#!/usr/bin/env python3
"""
lens_agent.py — agentic query planning for Gainsay (ReAct-lite).

The "Pro" feel of multi-step answer engines is mostly that: they decompose a hard
question into sub-questions, retrieve for each, and synthesize across them
(reasoning-then-action / ReAct). A single-shot retrieval underperforms on compound
questions ("compare X and Y and what's the latest on Z").

This module does ONLY the planning step — one model call that decides whether a
question is compound and, if so, splits it into focused sub-queries. The retrieval
itself stays in gainsay (which owns rag/web/rerank), so there's no circular
dependency and the agent layer is a thin, testable planner.

Guarantees:
  - ONE planner call, structured output (Ollama `format` schema).
  - Gated: simple questions return [question] unchanged (no wasted retrieval).
  - Graceful: any failure returns [question] — deep mode never breaks the base path.

Env:
  GAINSAY_PLAN_MODEL   Ollama model for planning (default: llama3.1:8b)
"""
from __future__ import annotations

import json
import os
import urllib.request

OLLAMA = "http://127.0.0.1:11434/api/chat"
# Same-model default as synthesis to avoid the GPU model-swap tax (see lens_rerank).
PLAN_MODEL = (os.environ.get("GAINSAY_PLAN_MODEL")
              or os.environ.get("GAINSAY_MODEL") or "gpt-oss:20b")
MAX_SUBQUERIES = 4

_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "compound": {"type": "boolean"},
        "subqueries": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["compound", "subqueries"],
}


def decompose(question: str, model: str | None = None,
              max_subqueries: int = MAX_SUBQUERIES) -> tuple[list[str], bool]:
    """Return (subqueries, was_decomposed).

    For a simple, single-intent question: ([question], False).
    For a compound question: (focused sub-queries, True) — each a standalone search
    that retrieves a distinct facet, so the union grounds the full answer."""
    model = model or PLAN_MODEL
    system = (
        "You plan retrieval for a question-answering system. Decide whether the "
        "question is COMPOUND (covers multiple distinct facets, comparisons, or "
        "sub-topics that each need their own search) or SIMPLE (one focused intent). "
        "If compound, split it into 2-4 standalone search sub-queries, each focused "
        "on one facet and phrased so a search engine and a textbook index both work. "
        "If simple, return compound=false and a single-element list with the original "
        "question. Do not invent facets the question doesn't ask for."
    )
    user = f'QUESTION: {question}\n\nReturn JSON: {{"compound": bool, "subqueries": [..]}}.'
    body = json.dumps({
        "model": model, "stream": False, "think": False, "format": _PLAN_SCHEMA,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "options": {"num_ctx": 4096, "temperature": 0.0},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        parsed = json.loads(((data.get("message") or {}).get("content") or "").strip())
        subs = [s.strip() for s in parsed.get("subqueries", []) if isinstance(s, str) and s.strip()]
        compound = bool(parsed.get("compound")) and len(subs) >= 2
    except Exception:
        return [question], False
    if not compound or not subs:
        return [question], False
    return subs[:max_subqueries], True


_OUTLINE_SCHEMA = {
    "type": "object",
    "properties": {
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "query": {"type": "string"}},
                "required": ["title", "query"],
            },
        },
    },
    "required": ["sections"],
}


def outline(question: str, model: str | None = None, n: int = 5) -> list[dict]:
    """Plan a report outline as 3-5 DISTINCT sections [{title, query}] for ANY question
    (unlike decompose, which only splits genuinely compound questions). Each section is a
    different facet/angle with a short title + a focused retrieval query. Falls back to a
    single section on failure so report mode never breaks."""
    model = model or PLAN_MODEL
    fallback = [{"title": question.strip().rstrip("?")[:70], "query": question}]
    system = (
        "You are planning a structured research report. Break the question into 3-5 DISTINCT, "
        "non-overlapping sections that together fully answer it — different facets, angles, or "
        "sub-topics, NOT the same question restated. Each section needs a short title (a few "
        "words) and a focused retrieval query (a standalone search phrase). Order them so the "
        "report reads logically.")
    user = (f"QUESTION: {question}\n\nReturn JSON: "
            '{"sections": [{"title": "...", "query": "..."}, ...]} with 3-5 sections.')
    body = json.dumps({
        "model": model, "stream": False, "think": False, "format": _OUTLINE_SCHEMA,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "options": {"num_ctx": 4096, "temperature": 0.2},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        parsed = json.loads(((data.get("message") or {}).get("content") or "").strip())
        secs = [{"title": (s.get("title") or "").strip(), "query": (s.get("query") or "").strip()}
                for s in parsed.get("sections", [])
                if isinstance(s, dict) and (s.get("query") or s.get("title"))]
        secs = [s for s in secs if s["query"] or s["title"]][:n]
        for s in secs:
            s["query"] = s["query"] or s["title"]
            s["title"] = s["title"] or s["query"]
        return secs if len(secs) >= 2 else fallback
    except Exception:
        return fallback


if __name__ == "__main__":
    for q in [
        "what is the capital of France?",
        "compare solar and wind power for home use, and which is cheaper to install?",
    ]:
        subs, did = decompose(q)
        print(f"\nQ: {q}\n  decomposed={did}")
        for s in subs:
            print("   -", s)
