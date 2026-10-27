"""Assemble the User Manual: pages, outline, references, search index, assets."""
from __future__ import annotations

import datetime as dt
import html
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader
from markdown_it import MarkdownIt
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound

from sensi.paths import ASSETS_DIR, PROJECT_ROOT

from manual import docs as D

MANUAL_SRC = PROJECT_ROOT / "dev" / "manual"
CHAPTERS_DIR = MANUAL_SRC / "chapters"
STATIC_DIR = MANUAL_SRC / "static"
TEMPLATES_DIR = MANUAL_SRC / "templates"
MANIFEST = MANUAL_SRC / "features.yaml"
# docs placed inside a chapter: the heading id they live under there
CHAPTER_DOC_ANCHOR = {"llm_credentials": "huggingface-token"}
MARKER = ".sensi-user-manual"      # written into the output folder; only such a folder is ever emptied
EXAMPLES_DIR = MANUAL_SRC / "examples"
OUTPUTS_DIR = MANUAL_SRC / "example_outputs"


# --------------------------------------------------------------------------- pages
@dataclass
class Page:
    id: str
    url: str                 # relative to the manual root, forward slashes
    title: str
    num: str = ""
    kind: str = "chapter"    # chapter | level | feature | section-index | references
    children: list["Page"] = field(default_factory=list)
    parent: "Page | None" = None
    data: dict = field(default_factory=dict)

    @property
    def root(self) -> str:
        return "../" * self.url.count("/")


@dataclass
class BuildReport:
    out_dir: Path
    pages: int = 0
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        lines = [f"[manual] {self.pages} pages → {self.out_dir}"]
        lines += [f"  note: {n}" for n in self.notes]
        lines += [f"  ERROR: {e}" for e in self.errors]
        lines.append("[manual] OK" if self.ok else f"[manual] FAILED ({len(self.errors)} errors)")
        return "\n".join(lines)


def md_renderer() -> MarkdownIt:
    def hl(code, lang, _attrs):
        try:
            lexer = get_lexer_by_name(lang or "text")
        except ClassNotFound:
            lexer = get_lexer_by_name("text")
        return highlight(code, lexer, HtmlFormatter(nowrap=True))
    return MarkdownIt("commonmark", {"html": True, "highlight": hl}).enable("table")


def _chapter_sources() -> list[tuple[dict, str]]:
    out = []
    for p in sorted(CHAPTERS_DIR.glob("*.md")):
        text = p.read_text(encoding="utf-8")
        if not text.startswith("---"):
            raise ValueError(f"chapter {p.name} has no front matter (--- id/title ---)")
        _, fm, body = text.split("---", 2)
        meta = yaml.safe_load(fm)
        meta.setdefault("id", p.stem.split("-", 1)[-1])
        out.append((meta, body))
    return out


def build_tree(manifest: dict, docs: dict[str, D.FeatureDoc]) -> tuple[Page, list[Page]]:
    """The whole manual as a tree; returns (root, reading order)."""
    root = Page("root", "index.html", "SensiExtractor User Manual", kind="root")
    chapters = _chapter_sources()
    n = 0

    def add(parent, page):
        page.parent = parent
        parent.children.append(page)
        return page

    for meta, body in chapters:
        if meta.get("after_features"):
            continue
        n += 1
        url = "index.html" if meta["id"] == "introduction" else f"chapters/{meta['id']}.html"
        add(root, Page(meta["id"], url, meta["title"], str(n), "chapter", data={"meta": meta, "body": body}))
    n += 1
    feats = add(root, Page("features", "features/index.html", "Features", str(n), "features"))
    for li, level in enumerate(manifest["levels"], 1):
        lv = add(feats, Page(f"level-{level['id']}", f"features/{level['id']}/index.html", level["title"],
                             f"{n}.{li}", "level", data={"level": level}))
        fi = 0
        for section in level["sections"]:
            for f in section["features"]:
                fi += 1
                doc = docs[f["key"]]
                add(lv, Page(f["key"], f"features/{level['id']}/{f['key']}.html", doc.name,
                             f"{n}.{li}.{fi}", "feature",
                             data={"level": level, "section": section, "entry": f, "doc": doc}))
    for meta, body in chapters:
        if not meta.get("after_features"):
            continue
        n += 1
        add(root, Page(meta["id"], f"chapters/{meta['id']}.html", meta["title"], str(n), "chapter",
                       data={"meta": meta, "body": body}))
    n += 1
    add(root, Page("references", "references.html", "References", str(n), "references"))

    order: list[Page] = []

    def walk(p):
        for c in p.children:
            order.append(c)
            walk(c)
    walk(root)
    return root, order


def outline_html(root: Page, current: Page) -> str:
    chain = set()
    p = current
    while p is not None:
        chain.add(p.id)
        p = p.parent

    def item(p: Page, depth: int) -> str:
        has = bool(p.children)
        is_open = has and (p.id in chain or (depth == 0 and p.id == "features"))
        cls = ["chapter"] if depth == 0 else []
        if is_open:
            cls.append("open")
        tag = ""
        if p.kind == "feature" and p.data["entry"].get("not_computed"):
            tag = '<span class="tag" title="Not computed in this manual">◇</span>'
        num = f'<span class="num">{p.num}</span>' if p.num and p.kind != "feature" else ""
        a_cls = ' class="current"' if p is current else ""
        toggle = (f'<button class="toggle" type="button" aria-expanded="{str(is_open).lower()}" '
                  f'aria-label="Show or hide {html.escape(p.title)}">▶</button>') if has else \
            '<span class="toggle blank"></span>'
        kids = "".join(item(c, depth + 1) for c in p.children) if has else ""
        sub = f"<ul>{kids}</ul>" if has else ""
        return (f'<li class="{" ".join(cls)}"><div class="row">{toggle}'
                f'<a href="{current.root}{p.url}"{a_cls}>{num}{html.escape(p.title)}{tag}</a></div>{sub}</li>')
    return "<ul>" + "".join(item(c, 0) for c in root.children) + "</ul>"


# ----------------------------------------------------------------- link rewriting
_HREF = re.compile(r'(href|src)="([^"]+)"')


def rewrite_links(fragment: str, page: Page, pages_by_key: dict[str, Page], chapter_links=None) -> str:
    page_of = {k: page.root + p.url for k, p in pages_by_key.items()}
    page_of.update({k: page.root + u for k, u in (chapter_links or {}).items()})

    def sub(m):
        attr, val = m.group(1), m.group(2)
        if attr == "href":
            return f'{attr}="{D.rewrite_href(html.unescape(val), here_key=page.id, page_of=page_of)}"'
        return m.group(0)
    return _HREF.sub(sub, fragment)


# --------------------------------------------------------------------- build
def _git(*args) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def build_site(out_dir: Path, *, check_links: bool = False) -> BuildReport:
    from manual import pages as P          # page bodies (chapters, features, references)
    from manual import checks

    report = BuildReport(out_dir)
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    docs = D.load_all()
    root, order = build_tree(manifest, docs)
    pages_by_key = {p.id: p for p in order if p.kind == "feature"}

    # Empty the folder rather than delete it (a viewer or server may hold it open) —
    # but only a folder this builder made (marker) or an empty one.
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()) and not (out_dir / MARKER).exists():
        raise SystemExit(f"refusing to empty {out_dir}: not a User Manual build folder (no {MARKER})")
    for child in out_dir.iterdir():
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    (out_dir / "assets").mkdir(parents=True)
    (out_dir / MARKER).write_text("Built by scripts/build_manual.py; this folder is emptied on every build.\n", encoding="utf-8")
    for name in ("theme.css", "layout.css", "manual.js"):
        shutil.copyfile(STATIC_DIR / name, out_dir / "assets" / name)
    shutil.copyfile(ASSETS_DIR / "icon.svg", out_dir / "assets" / "icon.svg")
    (out_dir / "assets" / "code.css").write_text(
        HtmlFormatter(style="friendly").get_style_defs("pre code"), encoding="utf-8")

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=False,
                      keep_trailing_newline=True)
    base = env.get_template("base.html.j2")
    from sensi import __version__
    build = {"date": dt.date.today().isoformat(), "version": __version__}
    ctx = P.Context(out_dir=out_dir, root=root, order=order, docs=docs, manifest=manifest,
                    pages_by_key=pages_by_key, report=report, md=md_renderer(),
                    chapter_links={k: f"chapters/{ch}.html#{CHAPTER_DOC_ANCHOR.get(k, k)}"
                                   for k, ch in manifest.get("chapter_docs", {}).items()})
    index = []
    for i, page in enumerate(order):
        body, toc, meta, entries = P.render(page, ctx)
        # doc-derived parts are link-rewritten inside pages.render (not generated HTML)
        crumbs = []
        a = page.parent
        while a is not None and a.kind != "root":
            crumbs.insert(0, {"title": a.title, "url": a.url})
            a = a.parent
        prev = order[i - 1] if i > 0 else None
        nxt = order[i + 1] if i + 1 < len(order) else None
        out = base.render(title=page.title, root=page.root, outline_html=outline_html(root, page),
                          body=body, toc=toc, meta=meta, crumbs=crumbs,
                          prev={"title": prev.title, "url": prev.url} if prev else None,
                          next={"title": nxt.title, "url": nxt.url} if nxt else None,
                          build=build)
        dest = out_dir / page.url
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(out, encoding="utf-8", newline="\n")
        index.extend(entries)
        report.pages += 1
    (out_dir / "assets" / "search-index.js").write_text(
        "window.MANUAL_INDEX = " + json.dumps(index, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8")
    report.errors += checks.run(out_dir, manifest, docs, order)
    if check_links:
        report.notes += checks.external_links(out_dir)
    return report
