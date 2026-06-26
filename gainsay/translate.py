#!/usr/bin/env python3
"""Local, private translation for Gainsay (and other local tools).

Translates fetched international web text to English so foreign-language sources
are readable + usable. Fully local: a multilingual Ollama model (default
qwen3:14b - strong on CJK + European) does the work; Argos Translate is used as a
fast offline path for installed pairs (currently es<->en). Nothing leaves the box.

    from translate import translate, looks_english
    en, src_lang, did = translate("Le chat est sur la table.")   # -> ("The cat is on the table.", "French", True)
"""
import json
import os
import re
import urllib.request

OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL = os.environ.get("TRANSLATE_MODEL", "qwen3:14b")

_EN_COMMON = set((
    "the and of to in is that for it with as on are this be by at from or an was "
    "not but have has you we they he she his her its our their will can would i a"
).split())


def looks_english(text: str) -> bool:
    """Cheap heuristic: lots of non-Latin script -> not English; otherwise judge
    by the share of common English words. Good enough to skip needless calls."""
    t = (text or "").strip()
    if not t:
        return True
    s = t[:800]
    nonlatin = sum(1 for c in s if ord(c) > 0x024F and not c.isspace())
    if nonlatin / max(1, len(s)) > 0.12:          # CJK / Cyrillic / Arabic / etc.
        return False
    accented = sum(1 for c in s if 0x00C0 <= ord(c) <= 0x024F)   # à é ñ ü ö ç ...
    words = re.findall(r"[a-zA-ZÀ-ɏ']+", s.lower())
    if len(words) < 5:
        return accented == 0                       # short Latin text: foreign only if accented
    eng = sum(1 for w in words if w in _EN_COMMON)
    ratio = eng / len(words)
    if accented / max(1, len(s)) > 0.015 and ratio < 0.25:
        return False                               # accents + few English words -> foreign
    return ratio > 0.16


def _argos(text: str):
    """Try Argos for an installed pair (offline). Returns English text or None."""
    try:
        import argostranslate.translate as at
        langs = {l.code: l for l in at.get_installed_languages()}
        en = langs.get("en")
        for code, lang in langs.items():
            if code == "en":
                continue
            tr = lang.get_translation(en)
            if tr:
                out = tr.translate(text)
                # crude: accept only if it actually changed something
                if out and out.strip() != text.strip():
                    return out, code
    except Exception:
        pass
    return None


def _model_translate(text: str, target: str = "English"):
    sys_msg = (f"You are a translator. Translate the user's text into {target}. "
               "First output one line 'LANG: <source language name>', then the "
               "translation only. Do not add commentary, notes, or the original.")
    body = json.dumps({
        "model": MODEL, "stream": False, "think": False,
        "messages": [{"role": "system", "content": sys_msg},
                     {"role": "user", "content": text[:6000]}],
        "options": {"temperature": 0.2, "num_ctx": 8192},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read())
    out = ((data.get("message") or {}).get("content", "") or "").strip()
    lang = "auto"
    m = re.match(r"\s*LANG:\s*(.+)", out)
    if m:
        lang = m.group(1).splitlines()[0].strip()
        out = out[m.end():].lstrip("\n ")
    return out, lang


def translate(text: str, target: str = "English"):
    """Return (english_text, source_language, was_translated).

    If the text already looks English, returns it unchanged. Otherwise tries the
    local model (covers all languages); language label comes from the model."""
    t = (text or "").strip()
    if not t or looks_english(t):
        return t, "English", False
    try:
        out, lang = _model_translate(t, target)
        if out:
            return out, lang, True
    except Exception:
        pass
    return t, "unknown", False   # translation unavailable -> hand back original


if __name__ == "__main__":
    import sys
    txt = " ".join(sys.argv[1:]) or "Le chat noir dort sur la table de la cuisine."
    en, lang, did = translate(txt)
    print(f"[{lang}{' -> translated' if did else ''}]\n{en}")
