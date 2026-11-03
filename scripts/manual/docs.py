"""Read docs/features/*.md (the files the app's ⓘ pop-ups render) into page parts.

The docs are never edited by the build. Their headings are mapped onto the
manual's standard parts (spec item 11); the YAML front matter is authoritative
for name, levels, toggles, output columns, .mat variables, libraries, status.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from markdown_it import MarkdownIt

from sensi.paths import DOCS_FEATURES_DIR, PROJECT_ROOT

GITHUB_BLOB = "https://github.com/edneuro/SENSI-Speech-Features/blob/main/"

# doc heading -> (part id, manual title). Unlisted headings become extra parts
# in document order (their own title, id from a slug).
HEADING_MAP = {
    "meaning": ("what", "What it is"),
    "math": ("computed", "How it's computed"),
    "math / linguistics": ("computed", "How it's computed"),
    "what it is / how it's computed": ("computed", "How it's computed"),
    "implementation in sensi speechfeatures": ("code", "In the code"),
    "library": ("library", "Library"),
    "eeg rationale": ("eeg", "EEG rationale"),
    "papers": ("papers", "Papers"),
    "datasets": ("datasets", "Datasets"),
}
# The order parts appear on a page; generated parts are filled in by site.py.
PART_ORDER = ["what", "see", "output", "matlab", "computed", "code", "library",
              "eeg", "papers", "datasets"]
GENERATED_TITLES = {"see": "See it", "output": "In your output", "matlab": "MATLAB"}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "part"


def github_slug(text: str) -> str:
    """The anchor GitHub gives a heading — what the docs' own cross-links use."""
    t = re.sub(r"<[^>]+>", "", text).strip().lower()
    t = re.sub(r"[^\w\- ]", "", t)
    return t.replace(" ", "-")


def _heading_ids(fragment: str) -> str:
    """id="…" on h3/h4 inside a part (GitHub-style), so in-doc links resolve."""
    def add(m):
        return f'<{m.group(1)} id="{github_slug(m.group(2))}">{m.group(2)}</{m.group(1)}>'
    return re.sub(r"<(h[3-4])>(.*?)</\1>", add, fragment, flags=re.S)


@dataclass
class Part:
    id: str
    title: str
    html: str
    aliases: list[str] = field(default_factory=list)   # GitHub anchors of the doc headings


@dataclass
class FeatureDoc:
    key: str
    path: Path
    meta: dict
    parts: dict[str, Part] = field(default_factory=dict)    # id -> Part (doc-derived)
    extra_order: list[str] = field(default_factory=list)    # extra part ids, doc order
    summary: str = ""                                       # first sentence, plain text
    preamble: str = ""                                      # text before the first ## (a note)

    @property
    def name(self) -> str:
        return self.meta.get("name") or self.key


def _md() -> MarkdownIt:
    return MarkdownIt("commonmark", {"html": True, "typographer": False}).enable("table")


def _split(body: str) -> list[tuple[str, str]]:
    """[(heading, markdown)] for each '## ' section; link reference definitions
    (`[X]: #anchor`, usually at the very end) are appended to every section so
    each renders its citations on its own."""
    refs = "\n".join(l for l in body.splitlines() if re.match(r"^\[[^\]]+\]:\s*\S", l))
    out, cur, buf, pre = [], None, [], []
    for line in body.splitlines():
        m = re.match(r"^## (.+?)\s*$", line)
        if m:
            if cur is not None:
                out.append((cur, "\n".join(buf)))
            cur, buf = m.group(1), []
        elif not re.match(r"^\[[^\]]+\]:\s*\S", line):
            (buf if cur is not None else pre).append(line)
    if "\n".join(pre).strip():
        out.insert(0, ("", "\n".join(pre)))
    if cur is not None:
        out.append((cur, "\n".join(buf)))
    return [(h, f"{t}\n\n{refs}\n") for h, t in out]


def _is_empty(md_text: str) -> bool:
    t = re.sub(r"<!--.*?-->", "", md_text, flags=re.S)
    t = "\n".join(l for l in t.splitlines() if not re.match(r"^\[[^\]]+\]:\s*\S", l))
    return not t.strip()


def rewrite_href(href: str, *, here_key: str, page_of: dict[str, str]) -> str:
    """Make a doc link work in the manual.

    `other.md#x` -> that feature's page (page_of: key -> url relative to the
    features root); repo-relative files -> GitHub; '#x' stays."""
    if re.match(r"^[a-z]+:|^#|^mailto:", href):
        return href
    path, _, frag = href.partition("#")
    if path.endswith(".md") and "/" not in path:
        key = path[:-3]
        if key in page_of:
            return page_of[key] + (f"#{frag}" if frag else "")
    target = (DOCS_FEATURES_DIR / path).resolve()
    try:
        rel = target.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return href
    return GITHUB_BLOB + rel + (f"#{frag}" if frag else "")


def load_feature_doc(path: Path) -> FeatureDoc:
    text = path.read_text(encoding="utf-8")
    _, fm, body = text.split("---", 2)
    meta = yaml.safe_load(fm) or {}
    doc = FeatureDoc(key=meta.get("key") or path.stem, path=path, meta=meta)
    md = _md()
    for heading, section in _split(body):
        if _is_empty(section):
            continue
        if heading == "":                         # the note before the first heading
            doc.preamble = _heading_ids(md.render(section))
            continue
        pid, title = HEADING_MAP.get(heading.strip().lower(), (None, heading.strip()))
        if pid is None:
            pid = "x-" + slug(heading)
            doc.extra_order.append(pid)
        rendered = _heading_ids(md.render(section))
        alias = github_slug(heading)
        if pid in doc.parts:                      # two doc headings → one part
            doc.parts[pid].html += rendered
            doc.parts[pid].aliases.append(alias)
        else:
            doc.parts[pid] = Part(pid, title, rendered, [alias])
    first = doc.parts.get("what")
    if first:
        para = re.search(r"<p>(.*?)</p>", first.html, flags=re.S)
        plain = re.sub(r"<[^>]+>", "", para.group(1) if para else first.html)
        m = re.search(r"(.+?[.!?])(\s|$)", " ".join(plain.split()))
        summ = (m.group(1) if m else " ".join(plain.split())[:200]).strip()
        doc.summary = summ[:-1] + "." if summ.endswith(":") else summ
    return doc


def load_all() -> dict[str, FeatureDoc]:
    return {d.key: d for d in (load_feature_doc(p) for p in sorted(DOCS_FEATURES_DIR.glob("*.md")))}
