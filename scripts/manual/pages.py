"""Page bodies: chapters, feature pages, level overviews, references."""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from pathlib import Path

from manual import docs as D
from manual import generated as G
from manual.site import EXAMPLES_DIR, OUTPUTS_DIR, Page, rewrite_links


@dataclass
class Context:
    out_dir: Path
    root: Page
    order: list
    docs: dict
    manifest: dict
    pages_by_key: dict
    report: object
    md: object
    citations: dict = field(default_factory=dict)   # url -> {"label","html","pages":[(title,url)]}


def _plain(s: str, n: int = 400) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", html.unescape(s)).split())[:n]


def _section(pid: str, num: str, title: str, inner: str, aliases=()) -> str:
    anchors = "".join(f'<a id="{a}"></a>' for a in aliases if a != pid)
    return f'<section id="{pid}">{anchors}<h2><span class="num">{num}</span>{html.escape(title)}</h2>{inner}</section>'


def _entry(t, p, u, k="", b="", w=0):
    return {"t": t, "p": p, "u": u, "k": k, "b": b, "w": w}


def render(page: Page, ctx: Context):
    if page.kind == "feature":
        return _feature(page, ctx)
    if page.kind == "level":
        return _level(page, ctx)
    if page.kind == "features":
        return _features_index(page, ctx)
    if page.kind == "references":
        return _references(page, ctx)
    return _chapter(page, ctx)


# ------------------------------------------------------------------ chapters
def _chapter(page: Page, ctx: Context):
    meta, src = page.data["meta"], page.data["body"]
    # {{feature:key}} -> link to that feature page; {{doc:key}} -> the doc's parts inline
    def feat_link(m):
        key = m.group(1)
        p = ctx.pages_by_key.get(key)
        if not p:
            ctx.report.errors.append(f"chapter {meta['id']}: unknown feature {key}")
            return key
        return f'<a href="{page.root}{p.url}">{html.escape(p.title)}</a>'
    src = re.sub(r"\{\{feature:([\w]+)\}\}", feat_link, src)

    def doc_inline(m):
        key = m.group(1)
        d = ctx.docs[key]
        inner = "".join(p.html for p in d.parts.values())
        return rewrite_links(inner, page, ctx.pages_by_key)
    src = re.sub(r"\{\{doc:([\w]+)\}\}", doc_inline, src)
    src = re.sub(r"\{\{include:([^}]+)\}\}", lambda m: _include(m.group(1), ctx), src)
    src = re.sub(r"\{\{file:([^}|]+)(?:\|([^}]*))?\}\}", lambda m: _file_block(m.group(1), m.group(2), ctx), src)
    src = re.sub(r"\{\{screen:([\w-]+)\|([^}]*)\}\}", lambda m: _screen(m.group(1), m.group(2), page, ctx), src)
    src = src.replace("{{outputs_table}}", _outputs_table(page, ctx))
    src = src.replace("{{root}}", page.root)
    body_html = ctx.md.render(src)
    toc, n = [], 0

    def h2(m):
        nonlocal n
        n += 1
        title = re.sub(r"<[^>]+>", "", m.group(1))
        pid = D.slug(title)
        toc.append({"id": pid, "title": title})
        return f'<h2 id="{pid}"><span class="num">{page.num}.{n}</span>{m.group(1)}</h2>'
    body_html = re.sub(r"<h2>(.*?)</h2>", h2, body_html)
    lede = f'<p class="lede">{html.escape(meta["lede"])}</p>' if meta.get("lede") else ""
    body = f'<h1><span class="num">{page.num}</span>{html.escape(page.title)}</h1>{lede}{body_html}'
    entries = [_entry(page.title, "Chapter", page.url, meta.get("keywords", ""), _plain(body_html), 6)]
    entries += [_entry(t["title"], page.title, f"{page.url}#{t['id']}", "", "", 2) for t in toc]
    return body, toc, [], entries


def _include(rel: str, ctx: Context) -> str:
    """Another repo markdown file, inline (title line dropped; its relative links → GitHub)."""
    from sensi.paths import PROJECT_ROOT
    p = PROJECT_ROOT / rel.strip()
    if not p.exists():
        ctx.report.errors.append(f"include: {rel} does not exist")
        return ""
    text = p.read_text(encoding="utf-8")
    text = re.sub(r"\A# .*\n", "", text)

    def link(m):
        label, href = m.group(1), m.group(2)
        if re.match(r"^[a-z]+:|^#", href):
            return m.group(0)
        target = (p.parent / href.split("#")[0]).resolve()
        try:
            return f"[{label}]({D.GITHUB_BLOB}{target.relative_to(PROJECT_ROOT).as_posix()})"
        except ValueError:
            return m.group(0)
    return re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", link, text)


def _file_block(rel: str, lines: str | None, ctx: Context) -> str:
    """A real repo file (or a line range "a-b") as a fenced code block."""
    from sensi.paths import PROJECT_ROOT
    p = PROJECT_ROOT / rel.strip()
    if not p.exists():
        ctx.report.errors.append(f"file: {rel} does not exist")
        return ""
    content = p.read_text(encoding="utf-8").splitlines()
    if lines:
        a, b = (int(x) for x in lines.split("-"))
        content = content[a - 1:b]
    lang = {".yaml": "yaml", ".yml": "yaml", ".sh": "bash", ".bat": "bat", ".py": "python"}.get(p.suffix, "text")
    return f"\n```{lang}\n" + "\n".join(content) + "\n```\n"


def _screen(name: str, caption: str, page: Page, ctx: Context) -> str:
    import shutil
    from manual.screenshots import SCREENS_DIR
    matches = sorted(SCREENS_DIR.glob(f"*-{name}.png"))
    if not matches:
        ctx.report.errors.append(f"screen: no screenshot for '{name}' (run --refresh-screenshots)")
        return ""
    dest = ctx.out_dir / "assets" / "screens" / matches[0].name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(matches[0], dest)
    return (f'\n<figure class="card"><img class="shot" src="{page.root}assets/screens/{matches[0].name}" '
            f'alt="{html.escape(caption)}" loading="lazy"><figcaption>{html.escape(caption)} '
            '(screenshot of the real app)</figcaption></figure>\n')


def _outputs_table(page: Page, ctx: Context) -> str:
    """Every output file → the feature pages whose columns/variables it carries."""
    files: dict[str, list] = {}
    for lv in ctx.manifest["levels"]:
        for s in lv["sections"]:
            for f in s["features"]:
                for o in G.outputs(f):
                    where = o.format(stem="<stim>").replace("documents_summary.csv", "documents_summary_<timestamp>.csv")
                    files.setdefault(where, []).append(f["key"])
    rows = []
    for where in sorted(files):
        links = ", ".join(f'<a href="{page.root}{ctx.pages_by_key[k].url}">{html.escape(ctx.pages_by_key[k].title)}</a>'
                          for k in dict.fromkeys(files[where]))
        rows.append(f"<tr><td><span class='file'>{html.escape(where)}</span></td><td>{links}</td></tr>")
    return ('\n<table><thead><tr><th>File</th><th>Features in it (see each page for its columns)</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>\n')


# ------------------------------------------------------------------ features
def _features_index(page: Page, ctx: Context):
    cards = []
    for lv in page.children:
        n = len(lv.children)
        cards.append(f'<div class="card"><a href="{page.root}{lv.url}">{lv.num} {html.escape(lv.title)}</a>'
                     f'<div class="caption">{html.escape(lv.data["level"]["blurb"])}</div>'
                     f'<div class="caption">{n} feature pages</div></div>')
    gui = ctx.manifest["gui_pages"]
    body = (f'<h1><span class="num">{page.num}</span>Features</h1>'
            '<p class="lede">Every feature SensiExtractor produces, grouped the way the app groups them: one '
            'chapter per output level (the app’s feature pages), one section per card on that page.</p>'
            f'<div class="grid-cards">{"".join(cards)}</div>'
            '<h2 id="how-to-read">How to read a feature page</h2>'
            '<p>Each feature page has the same parts, listed on the right as you read:</p>'
            '<ul><li><b>What it is</b> — the plain meaning, and the exact column or variable names.</li>'
            '<li><b>See it</b> — a figure drawn from SensiExtractor’s real output on a public-domain example '
            '(LJ Speech recordings, or two invented sentences run without audio). Where a setting changes the '
            'result, you can switch between real runs that differ only in that setting.</li>'
            '<li><b>In your output</b> — which file it lands in, and an excerpt of that file.</li>'
            '<li><b>MATLAB</b> — how to load it, inspect it and plot it.</li>'
            '<li><b>How it’s computed</b>, <b>In the code</b>, <b>Library</b> — the method, the function that '
            'does it (linked to GitHub) and the libraries used.</li>'
            '<li><b>EEG rationale</b> and <b>Papers</b> — why it matters for EEG, with links to the papers.</li></ul>'
            '<p>Pages marked ◇ in the outline describe features that cannot be computed on the manual’s '
            'examples; they explain why instead of showing a figure.</p>'
            f'<p>In the app, each level is a page in the sidebar: {", ".join(gui.values())}.</p>')
    toc = [{"id": "how-to-read", "title": "How to read a feature page"}]
    return body, toc, [], [_entry("Features", "Chapter", page.url, "features levels", "", 5)]


def _level(page: Page, ctx: Context):
    level = page.data["level"]
    gui = ctx.manifest["gui_pages"][level["id"]]
    secs, toc = [], []
    kids = {c.id: c for c in page.children}
    for i, section in enumerate(level["sections"], 1):
        sid = D.slug(section["title"])
        toc.append({"id": sid, "title": section["title"]})
        items = []
        for f in section["features"]:
            p = kids[f["key"]]
            d = ctx.docs[f["key"]]
            tag = ' <span class="chip status">not computed here</span>' if f.get("not_computed") else ""
            items.append(f'<div class="card"><a href="{page.root}{p.url}">{p.num} {html.escape(p.title)}</a>{tag}'
                         f'<div class="caption">{html.escape(d.summary)}</div></div>')
        secs.append(f'<section id="{sid}"><h2><span class="num">{page.num}.{i}</span>{html.escape(section["title"])}</h2>'
                    f'<p class="caption">App: <b>{html.escape(gui)}</b> page → <b>{html.escape(section["title"])}</b> card.</p>'
                    f'<div class="grid-cards">{"".join(items)}</div></section>')
    body = (f'<h1><span class="num">{page.num}</span>{html.escape(page.title)}</h1>'
            f'<p class="lede">{html.escape(level["blurb"])}</p>{"".join(secs)}')
    return body, toc, [], [_entry(page.title, "Features", page.url, level["id"], level["blurb"], 4)]


def _feature(page: Page, ctx: Context):
    doc: D.FeatureDoc = page.data["doc"]
    entry = dict(page.data["entry"])
    level, section = page.data["level"], page.data["section"]
    entry["_title"] = doc.name
    entry["_level"] = level["id"]
    meta_fm = doc.meta
    errs = G.check_names(entry, OUTPUTS_DIR)
    ctx.report.errors += errs

    # header
    gui = ctx.manifest["gui_pages"][level["id"]]
    chips = [f'<span class="chip">App: <b>{html.escape(gui)}</b> → <b>{html.escape(section["title"])}</b></span>']
    for o in G.outputs(entry):
        chips.append(f'<span class="chip">Output: <b>{html.escape(o.format(stem="<stim>"))}</b></span>')
    if meta_fm.get("libraries"):
        chips.append(f'<span class="chip">Libraries: <b>{html.escape(", ".join(meta_fm["libraries"]))}</b></span>')
    if entry.get("not_computed"):
        chips.append('<span class="chip status">Not computed in this manual</span>')
    if meta_fm.get("status") not in (None, "implemented"):
        chips.append(f'<span class="chip status">Status: {html.escape(str(meta_fm["status"]))}</span>')
    head = (f'<h1><span class="num">{page.num}</span>{html.escape(doc.name)}</h1>'
            f'<p class="lede">{html.escape(html.unescape(doc.summary))}</p><div class="facts">{"".join(chips)}</div>')

    # parts
    parts: list[tuple[str, str, str]] = []
    def doc_part(pid):
        p = doc.parts.get(pid)
        return rewrite_links(p.html, page, ctx.pages_by_key) if p else None

    order = []
    for pid in D.PART_ORDER:
        order.append(pid)
        if pid == "computed":
            order += doc.extra_order
    if "computed" not in doc.parts:
        pass
    for pid in order:
        try:
            if pid == "see":
                inner = G.see_it(entry, page.root, ctx.out_dir, OUTPUTS_DIR, EXAMPLES_DIR)
                title = "See it"
            elif pid == "output":
                inner, title = G.in_output(entry, OUTPUTS_DIR, meta_fm), "In your output"
            elif pid == "matlab":
                inner, title = G.matlab(entry), "MATLAB"
            elif pid == "code":
                raw = doc.parts.get("code")
                if not raw:
                    continue
                links, cerrs = G.code_links(raw.html)
                ctx.report.errors += [f"{doc.key}: {e}" for e in cerrs]
                inner, title = links + doc_part("code"), raw.title
            else:
                inner = doc_part(pid)
                if inner is None:
                    continue
                title = doc.parts[pid].title
        except Exception as exc:   # report, keep building the rest
            ctx.report.errors.append(f"{doc.key}: {pid}: {exc}")
            continue
        parts.append((pid, title, inner))
    body = head + "".join(_section(pid, f"{page.num}.{i}", t, h, doc.parts[pid].aliases if pid in doc.parts else ())
                          for i, (pid, t, h) in enumerate(parts, 1))
    toc = [{"id": pid, "title": t} for pid, t, _ in parts]

    # citations for the References page
    if "papers" in doc.parts:
        for li in re.findall(r"<li>(.*?)</li>", doc.parts["papers"].html, flags=re.S):
            urls = re.findall(r'href="(https?://[^"]+)"', li)
            if not urls:
                continue
            label = re.search(r"<strong>(?:<a[^>]*>)?\[?([^<\]]+)\]?", li)
            c = ctx.citations.setdefault(urls[0], {"label": (label.group(1) if label else urls[0]).strip(),
                                                   "html": re.sub(r'<a name="[^"]*"></a>', "", li), "pages": []})
            c["pages"].append((doc.name, page.url))

    meta = [f'<b>Key</b> <code>{html.escape(doc.key)}</code>',
            f'<a href="{D.GITHUB_BLOB}docs/features/{doc.key}.md">This page’s source text</a> (also the app’s ⓘ pop-up)']
    cols = ", ".join(entry.get("show", []))
    entries = [_entry(doc.name, f"{level['title']} › {section['title']}", page.url,
                      f"{doc.key} {cols} {' '.join(meta_fm.get('output_columns') or [])} {' '.join(meta_fm.get('mat_variables') or [])}",
                      doc.summary, 5)]
    entries += [_entry(f"{doc.name} — {t}", f"{level['title']} › {section['title']}", f"{page.url}#{pid}", "",
                       _plain(h, 200), 0) for pid, t, h in parts if pid in ("computed", "eeg", "papers")]
    return body, toc, meta, entries


# ---------------------------------------------------------------- references
def _references(page: Page, ctx: Context):
    items = []
    for url, c in sorted(ctx.citations.items(), key=lambda kv: kv[1]["label"].lower()):
        seen, links = set(), []
        for title, u in c["pages"]:
            if u in seen:
                continue
            seen.add(u)
            links.append(f'<a href="{page.root}{u}#papers">{html.escape(title)}</a>')
        body = re.sub(r'<a href="#[^"]*">([^<]*)</a>', r"\1", c["html"])
        items.append(f'<li>{body}<div class="caption">Cited on: {", ".join(links)}</div></li>')
    body = (f'<h1><span class="num">{page.num}</span>References</h1>'
            f'<p class="lede">Every paper cited in the feature pages ({len(items)}), with links back to the pages '
            'that cite it.</p>'
            f'<section id="papers"><ul class="refs">{"".join(items)}</ul></section>')
    return body, [], [], [_entry("References", "Chapter", page.url, "papers citations bibliography", "", 3)]
