#!/usr/bin/env python3
"""Gainsay — revision history ("how it changed its mind").

The trust-layer capstone. The disagreement engine shows conflict across SOURCES;
this shows how a single question's ANSWER changed when re-asked — surfacing what
changed, grounded in real structural diffs (sources added/removed, consensus deltas),
never a model-narrated story.

Design decisions:
- OPT-IN, off by default: a persistent on-disk log of questions+answers is a new
  data-at-rest surface; it exists only when the user turns History on.
- The "what changed" note is DETERMINISTIC, no LLM: if sources/consensus didn't move we
  SAY "same evidence, wording differs" rather than let a model invent a reason. No
  untrusted source text reaches it.
- Diff is STRUCTURAL, not a raw-prose diff (paraphrase would make a text diff a sea of
  false red/green). We diff source IDENTITIES and consensus counts.
- TAMPER-EVIDENT, not encrypted: every other local artifact (RAG, reports) is plaintext
  under the same trust boundary, so encrypting only this would be inconsistent theater.
  Each version carries a SHA-256 over its content, so edits are DETECTABLE.
  (Cross-record hash-chaining is noted future hardening, deferred — not silently dropped.)
- Bounded + clearable: cap questions/versions; one-call clear().

Stays 100% local — nothing here ever leaves the machine.
"""
import hashlib
import json
import os
import re
import time
from pathlib import Path

_APPDATA = os.environ.get("LOCALAPPDATA") or str(Path.home())
HIST_DIR = Path(_APPDATA) / "gainsay" / "history"
HIST_FILE = HIST_DIR / "history.json"

MAX_QUESTIONS = 50   # distinct question-keys retained (most-recently-written kept)
MAX_VERSIONS = 5     # versions kept per question

_WS_RE = re.compile(r"\s+")


def key_for(question):
    """Light normalization for 'same question' identity — lowercase, collapse whitespace, strip
    surrounding punctuation. NOT the aggressive stopword stripping used for search (that would
    merge distinct questions); keep content words so the key stays faithful."""
    q = (question or "").strip().lower()
    q = _WS_RE.sub(" ", q)
    return q.strip(" \t\r\n.?!,;:\"'()[]{}")


def _norm_sources(sources):
    """Accept a list of {tag,id,label}; return a clean, diff-stable list, dropping malformed."""
    out = []
    if not isinstance(sources, list):
        return out
    for s in sources:
        if not isinstance(s, dict):
            continue
        sid = str(s.get("id") or s.get("url") or s.get("name") or s.get("label") or "").strip()
        if not sid:
            continue
        out.append({
            "tag": str(s.get("tag") or "?")[:2],
            "id": sid[:400],
            "label": str(s.get("label") or s.get("name") or s.get("title") or s.get("url") or sid)[:200],
        })
    return out


def _consensus(c):
    if not isinstance(c, dict):
        return None
    keys = ("n_claims", "n_contested", "n_unsupported")
    if not any(k in c for k in keys):
        return None
    return {k: int(c.get(k) or 0) for k in keys}


def _canonical(rec):
    """Deterministic bytes over a version's content, excluding its own hash field."""
    body = {k: rec[k] for k in ("seq", "ts", "key", "answer", "model", "sources", "consensus") if k in rec}
    return json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _hash(rec):
    return hashlib.sha256(_canonical(rec)).hexdigest()


def _load():
    try:
        data = json.loads(HIST_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("questions"), dict):
            data.setdefault("next_seq", 1)
            return data
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return {"version": 1, "next_seq": 1, "questions": {}}


def _save(store):
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    tmp = HIST_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(HIST_FILE)  # atomic


def _fmt_delta(then, now):
    d = max(0, int(now - then))
    if d < 90 * 60:
        m = max(1, d // 60)
        return f"{m} minute{'s' if m != 1 else ''}"
    if d < 36 * 3600:
        h = max(1, d // 3600)
        return f"{h} hour{'s' if h != 1 else ''}"
    days = max(1, d // 86400)
    return f"{days} day{'s' if days != 1 else ''}"


def _iso(ts):
    return time.strftime("%b %d, %Y %H:%M", time.localtime(ts))


def _note(prior_ts, now, added, removed, cb, ca, substance):
    """Deterministic 'what changed' — grounded ONLY in structural diffs. No model, no hallucinated
    reason. If nothing structural moved, it says so plainly."""
    parts = [f"You asked this {_fmt_delta(prior_ts, now)} ago."]
    if added or removed:
        bits = []
        if added:
            bits.append(f"+{len(added)} source{'s' if len(added) != 1 else ''}")
        if removed:
            bits.append(f"-{len(removed)} source{'s' if len(removed) != 1 else ''}")
        parts.append("Evidence changed: " + ", ".join(bits) + ".")
    else:
        parts.append("Same sources as before.")
    if cb and ca and cb != ca:
        parts.append(f"Consensus: {cb['n_contested']}→{ca['n_contested']} contested, "
                     f"{cb['n_unsupported']}→{ca['n_unsupported']} unsupported.")
    elif cb and ca:
        parts.append("Consensus unchanged.")
    if not substance:
        parts.append("No new evidence — any differences are wording, not substance.")
    return " ".join(parts)


def revision_for(question, sources, consensus=None):
    """If this question has prior version(s), diff CURRENT (sources, consensus) against the most
    recent one. Returns a revision dict or None. Does NOT record (call record() after)."""
    store = _load()
    entry = store["questions"].get(key_for(question))
    if not entry or not entry.get("versions"):
        return None
    prior = entry["versions"][-1]
    cur = {s["id"]: s for s in _norm_sources(sources)}
    prev = {s["id"]: s for s in (prior.get("sources") or [])}
    added = [cur[i]["label"] for i in cur if i not in prev]
    removed = [prev[i]["label"] for i in prev if i not in cur]
    cb = prior.get("consensus")
    ca = _consensus(consensus)
    consensus_changed = bool(cb and ca and cb != ca)
    substance = bool(added or removed or consensus_changed)
    now = time.time()
    return {
        "prior_ts": prior["ts"],
        "prior_when": _iso(prior["ts"]),
        "ago": _fmt_delta(prior["ts"], now),
        "n_prior": len(entry["versions"]),
        "added": added,
        "removed": removed,
        "consensus_before": cb,
        "consensus_after": ca,
        "substance_changed": substance,
        "prior_answer": prior.get("answer", ""),
        "note": _note(prior["ts"], now, added, removed, cb, ca, substance),
    }


def record(question, answer, sources, consensus=None, model=""):
    """Append a new version for this question. Enforces caps. Returns the stored record."""
    store = _load()
    k = key_for(question)
    if not k:
        return None
    entry = store["questions"].setdefault(k, {"question": (question or "").strip()[:500], "versions": []})
    seq = store.get("next_seq", 1)
    rec = {
        "seq": seq,
        "ts": time.time(),
        "key": k,
        "answer": (answer or "")[:20000],
        "model": str(model or "")[:80],
        "sources": _norm_sources(sources),
        "consensus": _consensus(consensus),
    }
    rec["hash"] = _hash(rec)
    entry["versions"].append(rec)
    store["next_seq"] = seq + 1
    if len(entry["versions"]) > MAX_VERSIONS:
        entry["versions"] = entry["versions"][-MAX_VERSIONS:]
    if len(store["questions"]) > MAX_QUESTIONS:
        kept = sorted(store["questions"].items(),
                      key=lambda kv: (kv[1].get("versions") or [{}])[-1].get("ts", 0),
                      reverse=True)[:MAX_QUESTIONS]
        store["questions"] = dict(kept)
    _save(store)
    return rec


def clear():
    try:
        HIST_FILE.unlink()
    except FileNotFoundError:
        pass
    return True


def verify():
    """Recompute each retained record's hash; report content tampering (tamper-EVIDENT)."""
    store = _load()
    bad, total = [], 0
    for k, entry in store["questions"].items():
        for rec in entry.get("versions", []):
            total += 1
            if rec.get("hash") != _hash(rec):
                bad.append({"key": k, "seq": rec.get("seq")})
    return {"ok": not bad, "checked": total, "tampered": bad}


def stats():
    store = _load()
    return {
        "questions": len(store["questions"]),
        "versions": sum(len(e.get("versions", [])) for e in store["questions"].values()),
    }


if __name__ == "__main__":
    print("Gainsay history store:", HIST_FILE)
    print(stats())
