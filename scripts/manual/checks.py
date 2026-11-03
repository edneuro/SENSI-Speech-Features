"""Checks the build enforces (spec item 9); errors fail the build."""
from __future__ import annotations

import re
import urllib.parse
import urllib.request
from pathlib import Path

ISC_STEMS = ("1_FB_P", "1_M", "2_FB_P", "2_M", "3_M")
MAX_BYTES = 30 * 1024 * 1024
_ATTR = re.compile(r'(?:href|src)="([^"]+)"')
_ID = re.compile(r'(?:id|name)="([^"]+)"')


def run(out_dir: Path, manifest: dict, docs: dict, order) -> list[str]:
    errs: list[str] = []
    placed = [f["key"] for lv in manifest["levels"] for s in lv["sections"] for f in s["features"]]
    placed += list(manifest.get("chapter_docs", {}))
    for k in docs:
        n = placed.count(k)
        if n != 1:
            errs.append(f"docs/features/{k}.md is placed {n} times (must be exactly once)")
    for k in set(placed) - set(docs):
        errs.append(f"manifest names {k}, which has no docs/features/{k}.md")

    pages = sorted(out_dir.rglob("*.html"))
    ids = {p: set(_ID.findall(p.read_text(encoding="utf-8"))) for p in pages}
    for p in pages:
        text = p.read_text(encoding="utf-8")
        for ref in _ATTR.findall(text):
            if re.match(r"^(https?:|mailto:|data:|javascript:)", ref):
                continue
            if "\\" in ref:
                errs.append(f"{p.relative_to(out_dir).as_posix()}: backslash in link {ref}")
            path, _, frag = ref.partition("#")
            target = (p.parent / urllib.parse.unquote(path)).resolve() if path else p
            if not target.exists():
                errs.append(f"{p.relative_to(out_dir).as_posix()}: broken link {ref}")
                continue
            if frag and target.suffix == ".html":
                t_ids = ids.get(target) if target in ids else set(_ID.findall(target.read_text(encoding="utf-8")))
                if frag not in t_ids:
                    errs.append(f"{p.relative_to(out_dir).as_posix()}: missing anchor {ref}")
        for stem in ISC_STEMS:
            if re.search(rf"\b{re.escape(stem)}\b", text):
                errs.append(f"{p.relative_to(out_dir).as_posix()}: mentions ISC stimulus {stem}")
    for f in out_dir.rglob("*"):
        if f.is_file() and any(f.name.startswith(s + "_") or f.stem == s for s in ISC_STEMS):
            errs.append(f"ISC stimulus file shipped: {f.relative_to(out_dir).as_posix()}")
    size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    if size > MAX_BYTES:
        errs.append(f"manual is {size / 2**20:.1f} MB (limit {MAX_BYTES / 2**20:.0f} MB)")
    return errs


def external_links(out_dir: Path) -> list[str]:
    """Request every external link once; report the ones that fail."""
    urls = sorted({u for p in out_dir.rglob("*.html") for u in _ATTR.findall(p.read_text(encoding="utf-8"))
                   if u.startswith(("http://", "https://"))})
    bad = []

    def doi_registered(u: str) -> bool:
        """Publishers (Elsevier especially) refuse scripted requests; for a DOI
        link, ask the DOI registry whether it exists instead."""
        import json
        doi = u.split("doi.org/", 1)[1]
        try:
            with urllib.request.urlopen(f"https://doi.org/api/handles/{urllib.parse.quote(doi)}", timeout=12) as r:
                return json.load(r).get("responseCode") == 1
        except Exception:
            return False

    for u in urls:
        if "doi.org/" in u:
            if not doi_registered(u):
                bad.append(f"DOI not registered: {u}")
            continue
        req = urllib.request.Request(u, method="GET", headers={"User-Agent": "Mozilla/5.0 (SENSI SpeechFeatures manual link check)"})
        try:
            with urllib.request.urlopen(req, timeout=12) as r:
                if r.status >= 400:
                    bad.append(f"{r.status} {u}")
        except urllib.error.HTTPError as e:
            if e.code not in (401, 403, 405, 429):       # paywalls / bot walls: the link itself is fine
                bad.append(f"{e.code} {u}")
        except Exception as e:
            bad.append(f"ERR {type(e).__name__} {u}")
    return [f"{len(urls)} external links checked, {len(bad)} failed"] + bad
