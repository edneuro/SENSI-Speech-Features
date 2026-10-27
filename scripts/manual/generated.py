"""The generated parts of a feature page: See it, In your output, MATLAB, code links.

All values come from the real example outputs (dev/manual/example_outputs/);
every column or variable a page names is checked against the file it reads.
"""
from __future__ import annotations

import ast
import html
import re
import shutil
from pathlib import Path

import numpy as np

from sensi.paths import PROJECT_ROOT

from manual import figures as F
from manual.docs import GITHUB_BLOB

DEFAULT_STEM = "LJ002-0321"
UNITS = {"Pitch": "semitones re. speaker median", "Volume": "dB"}


class ManualError(RuntimeError):
    pass


def stems(entry) -> list[str]:
    ex = entry.get("example", DEFAULT_STEM)
    return list(ex) if isinstance(ex, list) else [ex]


def outputs(entry) -> list[str]:
    o = entry.get("output") or []
    return [o] if isinstance(o, str) else list(o)


def _path(outputs_root: Path, variant: str, pattern: str, stem: str) -> Path:
    return outputs_root / variant / pattern.format(stem=stem)


def check_names(entry, outputs_root: Path) -> list[str]:
    """Every name the page shows must exist in the file it reads."""
    errs = []
    if entry.get("not_computed"):
        return errs
    for pattern in outputs(entry):
        for stem in stems(entry):
            p = _path(outputs_root, "base", pattern, stem)
            if not p.exists():
                errs.append(f"{entry['key']}: missing example output {p.relative_to(PROJECT_ROOT).as_posix()}")
                continue
            if p.suffix == ".csv":
                names = set(F.read_csv(p).columns)
                want = entry.get("show", [])
            else:
                names = set(F.read_mat(p))
                want = entry.get("mat_show") or entry.get("show", [])
            for n in want:
                if n not in names:
                    errs.append(f"{entry['key']}: '{n}' is not in {p.name}")
    return errs


# ---------------------------------------------------------------------- See it
def _figure_card(src_rel: str, caption: str, provenance: str) -> str:
    return (f'<figure><img src="{src_rel}" alt="{html.escape(caption)}" loading="lazy"></figure>'
            f'<figcaption>{caption}</figcaption><div class="provenance">{provenance}</div>')


def _draw(kind, entry, variant, outputs_root, examples_dir, out_svg) -> None:
    stem = stems(entry)[0]
    pat = outputs(entry)[0] if outputs(entry) else None
    p = _path(outputs_root, variant, pat, stem) if pat else None
    title = entry.get("_title", "")
    y = entry.get("y")
    if kind == "word_bars":
        F.word_bars(F.read_csv(p), y, out_svg, title)
    elif kind == "predict_compare":
        F.predict_compare({s: F.read_csv(_path(outputs_root, variant, pat, s)) for s in stems(entry)}, y, out_svg, title)
    elif kind == "norms":
        F.norms(F.read_csv(p), entry["show"], out_svg, title)
    elif kind == "word_timeline":
        F.word_timeline(F.read_csv(p), stem, examples_dir, out_svg, title,
                        show_gaps="Word_Gap" in entry["show"])
    elif kind == "word_prosody":
        unit = "Hz" if variant == "pitch_hz" and y == "Pitch" else UNITS.get(y, "")
        F.word_prosody(F.read_csv(p), y, out_svg, title, unit)
    elif kind in ("syllable_timeline",):
        F.syllable_timeline(F.read_csv(p), stem, examples_dir, out_svg, title,
                            pcenter="P_Center" in entry["show"])
    elif kind == "syllable_prosody":
        F.syllable_prosody(F.read_csv(p), out_svg, title)
    elif kind == "phoneme_timeline":
        F.phoneme_timeline(F.read_csv(p), out_svg, title)
    elif kind == "phoneme_surprisal":
        F.phoneme_timeline(F.read_csv(p), out_svg, title, ycol="phoneme_surprisal")
    elif kind == "timeseries":
        F.timeseries(F.read_mat(p), y, out_svg, title)
    elif kind == "image":
        F.image(F.read_mat(p), y, out_svg, title)
    elif kind == "embedding":
        F.embedding(F.read_mat(p), out_svg, title)
    elif kind == "embedding_layers":
        F.embedding_layers(F.read_mat(p), out_svg, title)
    elif kind == "sentence_similarity":
        F.sentence_similarity(F.read_mat(p), out_svg, title)
    elif kind == "categories":
        F.categories(F.read_csv(p), out_svg, title)
    elif kind == "doc_bars":
        df = F.read_csv(p)
        F.doc_bars(df, entry["show"], out_svg, title)
    elif kind == "dependency":
        F.dependency_svg(F.read_csv(p), out_svg, title)
    else:
        raise ManualError(f"{entry['key']}: unknown figure kind '{kind}'")


VARIANT_LABEL = {"base": "Default settings", "env_hilbert": "env_method: hilbert",
                 "mel_music": "mel_preset: music", "pitch_hz": "pitch_units: hz"}
SVG_KINDS = {"word_bars", "predict_compare", "norms", "word_timeline", "word_prosody", "syllable_timeline",
             "syllable_prosody", "phoneme_timeline", "phoneme_surprisal", "timeseries", "image", "embedding",
             "embedding_layers", "sentence_similarity", "categories", "doc_bars", "dependency"}


def see_it(entry, page_root: str, out_dir: Path, outputs_root: Path, examples_dir: Path) -> str:
    key = entry["key"]
    if entry.get("not_computed"):
        cols = ", ".join(f"<code>{html.escape(c)}</code>" for c in entry.get("show", []))
        return (f'<div class="card notcomputed"><div class="card-title">◇ Not computed in this manual</div>'
                f'<p>{html.escape(entry["not_computed"])}</p>'
                f'{"<p>Columns this feature writes when it runs: " + cols + "</p>" if cols else ""}</div>')
    kind = entry.get("figure", "table")
    stem = stems(entry)[0]
    src_note = " and ".join(f"<span class='file'>{o.format(stem=s)}</span>" for o in outputs(entry)[:1] for s in stems(entry))
    prov = f"Read from {src_note}, written by SensiExtractor on the example below. Nothing on this figure is typed in."
    audio = ""
    wav = examples_dir / "audio" / f"{stem}.wav"
    if wav.exists() and entry.get("_level") != "document":
        dest = out_dir / "assets" / "audio" / wav.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copyfile(wav, dest)
        txt = examples_dir / "transcripts" / f"{stem}.txt"
        said = html.escape(" ".join(txt.read_text(encoding="utf-8").split())) if txt.exists() else "(audio-only clip)"
        audio = (f'<div class="caption">Example <b>{stem}</b> (LJ Speech, public domain): “{said}”</div>'
                 f'<audio controls preload="none" src="{page_root}assets/audio/{wav.name}"></audio>')
    elif entry.get("_level") != "document":
        lines = []
        for s in stems(entry):
            txt = examples_dir / "transcripts" / f"{s}.txt"
            if txt.exists():
                lines.append(f"<b>{s}.txt</b>: “{html.escape(' '.join(txt.read_text(encoding='utf-8').split()))}”")
        audio = f'<div class="caption">Example text (invented, run without audio): {"; ".join(lines)}</div>'
    else:
        audio = ('<div class="caption">One row per example stimulus: the five LJ Speech clips and the two invented '
                 'text-only sentences (no audio, so their audio-based columns are empty).</div>')

    if kind in SVG_KINDS:
        variants = ["base"] + list(entry.get("variants") or [])
        panels, buttons = [], []
        for i, v in enumerate(variants):
            rel = f"assets/figures/{key}{'' if v == 'base' else '__' + v}.svg"
            _draw(kind, entry, v, outputs_root, examples_dir, out_dir / rel)
            panels.append(f'<div class="tabpanel{" on" if i == 0 else ""}">'
                          + _figure_card(page_root + rel, f"{entry['_title']} — {VARIANT_LABEL.get(v, v)}", prov) + "</div>")
            buttons.append(f'<button type="button" class="{"on" if i == 0 else ""}">{VARIANT_LABEL.get(v, v)}</button>')
        if len(variants) > 1:
            body = (f'<div data-tabs><div class="tabs" role="tablist">{"".join(buttons)}</div>{"".join(panels)}</div>'
                    '<div class="caption">Switch between real runs that differ only in this one setting.</div>')
        else:
            body = panels[0].replace('class="tabpanel on"', "")
        return f'<div class="card">{audio}{body}</div>'
    # HTML visuals from the data
    df = F.read_csv(_path(outputs_root, "base", outputs(entry)[0], stem))
    if kind == "char_offsets":
        vis = F.char_offsets_html(df)
    else:
        vis = F.strip_html(df, entry["show"], kind="tags" if kind == "tags" else "table")
    return f'<div class="card">{audio}{vis}<div class="provenance">{prov}</div></div>'


# ------------------------------------------------------------- In your output
_MAT_CLASS = {"float64": "double", "float32": "single", "int64": "int64", "uint8": "uint8", "object": "cell"}


def _excerpt_csv(p: Path, show: list[str], stem: str) -> str:
    df = F.read_csv(p)
    sub = F.first_sentence(df).head(12) if "sentence_id" in df and len(df) > 12 else df.head(12)
    keys = {"token", "stimulus_id", "sentence_id", "syllable_id", "phoneme_id"}
    cols = [c for c in show if c in df.columns]
    head = "".join(f'<th class="{"" if c in keys else "hl"}">{html.escape(c)}</th>' for c in cols)

    def cell(v):
        if isinstance(v, float):
            return "" if v != v else (f"{v:.4g}")
        return html.escape(str(v))
    rows = "".join("<tr>" + "".join(f'<td class="{"" if c in keys else "hl"}">{cell(r[c])}</td>' for c in cols) + "</tr>"
                   for _, r in sub.iterrows())
    more = f"{len(df)} rows × {len(df.columns)} columns in the file; showing the feature's columns"
    return (f'<table class="excerpt"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>'
            f'<div class="caption">{more}{" for the first sentence" if len(sub) < len(df) else ""}. '
            'An empty cell means no value (e.g. punctuation, or a word the measure doesn’t cover).</div>')


def _excerpt_mat(p: Path, show: list[str]) -> str:
    m = F.read_mat(p)
    rows = []
    for n in show:
        v = m[n]
        cls = "struct" if v.dtype.names else _MAT_CLASS.get(str(v.dtype), str(v.dtype))
        size = " × ".join(str(s) for s in v.shape)
        rows.append(f"<tr><td><code>{html.escape(n)}</code></td><td>{size}</td><td>{cls}</td></tr>")
    return ('<table class="excerpt vars"><thead><tr><th>variable</th><th>size</th><th>MATLAB class</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def in_output(entry, outputs_root: Path, front: dict) -> str:
    if entry.get("not_computed"):
        cols = front.get("output_columns") or entry.get("show") or []
        return ("<p>When it runs, this feature writes these columns: "
                + ", ".join(f"<code>{html.escape(c)}</code>" for c in cols) + ".</p>")
    parts = []
    stem = stems(entry)[-1] if entry.get("figure") == "predict_compare" else stems(entry)[0]
    for pattern in outputs(entry):
        p = _path(outputs_root, "base", pattern, stem)
        where = pattern.format(stem="<stim>").replace("documents_summary.csv", "documents_summary_<timestamp>.csv")
        if p.suffix == ".csv":
            body = _excerpt_csv(p, entry["show"], stem)
        else:
            body = _excerpt_mat(p, entry.get("mat_show") or entry["show"])
        parts.append(f'<div class="card"><div class="card-title">File <span class="file">{html.escape(where)}</span>'
                     f'</div>{body}<div class="provenance">Excerpt of the real file for <b>{stem}</b>.</div></div>')
    return "".join(parts)


# ---------------------------------------------------------------------- MATLAB
def _plot_line(kind, entry, var):
    y = entry.get("y")
    if kind in ("word_bars", "predict_compare"):
        return (f"w = ~strcmp(T.upos, 'PUNCT');            % words only\n"
                f"bar(T.{y}(w)); xticks(1:nnz(w)); xticklabels(T.token(w)); ylabel('{y}')")
    if kind == "norms":
        cols = [c for c in entry["show"] if c != "token"][:3]
        return "w = ~strcmp(T.upos, 'PUNCT');\n" + "\n".join(
            f"subplot({len(cols)},1,{i + 1}); bar(T.{c}(w)); ylabel('{c}')" for i, c in enumerate(cols)) + \
            "\nxticks(1:nnz(w)); xticklabels(T.token(w))"
    if kind == "word_timeline":
        return ("ok = ~isnan(T.Onset);                    % aligned words\n"
                "for i = find(ok)'\n"
                "    patch([T.Onset(i) T.End(i) T.End(i) T.Onset(i)], [0 0 1 1], [0 .7 .82], 'FaceAlpha', .3, 'EdgeColor', 'none');\n"
                "    text(mean([T.Onset(i) T.End(i)]), 1.05, T.token{i}, 'HorizontalAlignment', 'center');\n"
                "end\nxlabel('Time (s)')")
    if kind == "word_prosody":
        return (f"ok = ~isnan(T.{y}_Mean);\n"
                f"plot([T.Onset(ok) T.End(ok)]', [T.{y}_Mean(ok) T.{y}_Mean(ok)]', 'LineWidth', 2)\n"
                f"xlabel('Time (s)'); ylabel('{y}_Mean')")
    if kind in ("syllable_timeline", "syllable_prosody", "phoneme_timeline", "phoneme_surprisal"):
        col = "phoneme_surprisal" if kind == "phoneme_surprisal" else ("Pitch_Mean" if kind == "syllable_prosody" else "Duration")
        return (f"ok = ~isnan(T.Onset);\n"
                f"stem((T.Onset(ok) + T.End(ok)) / 2, T.{col}(ok)); xlabel('Time (s)'); ylabel('{col}')")
    if kind == "doc_bars":
        c = [x for x in entry["show"] if x != "stimulus_id"][0]
        return f"barh(T.{c}); yticks(1:height(T)); yticklabels(T.stimulus_id); xlabel('{c}')"
    if kind == "categories":
        return ("cats = startsWith(T.Properties.VariableNames, 'Sem_') & ~strcmp(T.Properties.VariableNames, 'Sem_Categories');\n"
                "imagesc(T{:, cats}'); yticks(1:nnz(cats)); yticklabels(T.Properties.VariableNames(cats))\n"
                "xticks(1:height(T)); xticklabels(T.token)")
    return None


def matlab(entry) -> str:
    if entry.get("not_computed"):
        return "<p>Nothing to load: this feature is not written in this manual's example outputs.</p>"
    kind = entry.get("figure", "table")
    stem = stems(entry)[0]
    blocks = []
    for pattern in outputs(entry):
        rel = pattern.format(stem=stem)
        if rel.endswith(".csv"):
            cols = ", ".join(f"'{c}'" for c in entry["show"])
            code = (f"% Load the {rel.split('/')[0]}-level table (one row per "
                    f"{'stimulus' if 'documents_summary' in rel else rel.split('/')[0]})\n"
                    f"T = readtable('{rel}', 'TextType', 'string');\n"
                    f"T.Properties.VariableNames             % every column in the file\n"
                    f"T(1:10, {{{cols}}})                       % this feature's columns\n")
            if "documents_summary" in rel:
                code = code.replace(f"'{rel}'", "'document/documents_summary_<timestamp>.csv'")
            line = _plot_line(kind, entry, None)
            if line:
                code += "\n% Plot\n" + line.replace("T.token{i}", "T.token(i)") + "\n"
            code = code.replace("strcmp(T.upos, 'PUNCT')", "T.upos == \"PUNCT\"")
        else:
            show = entry.get("mat_show") or entry["show"]
            code = (f"S = load('{rel}');\n"
                    f"whos('-file', '{rel}')                % variables, sizes, classes\n")
            y = entry.get("y")
            if kind == "timeseries":
                code += f"\nplot(S.t, S.{y}); xlabel('Time (s)'); ylabel('{y}')   % one value per time step, fs = S.fs Hz\n"
            elif kind == "image":
                code += (f"\nimagesc(S.t, 1:size(S.{y}, 2), S.{y}'); axis xy\n"
                         f"xlabel('Time (s)'); ylabel('{'mel band' if y == 'logmel' else 'coefficient'}'); colorbar\n")
                if y == "logmel":
                    code += "% band centre frequencies (Hz): S.mel_center_freqs\n"
            elif kind == "embedding":
                code += ("\nE = S.embeddings(S.is_valid == 1, :);    % words x 300 (single)\n"
                         "words = S.tokens(S.is_valid == 1);       % cell array of the matching words\n"
                         "En = E ./ vecnorm(E, 2, 2);               % unit length\n"
                         "imagesc(En * En'); axis square; colorbar  % cosine similarity between words\n"
                         "xticks(1:numel(words)); xticklabels(words); yticks(1:numel(words)); yticklabels(words)\n")
            elif kind == "embedding_layers":
                code += ("\nsize(S.embeddings)                         % words x 12 layers x 768 (single)\n"
                         "L = 9;                                     % pick a layer\n"
                         "E = squeeze(S.embeddings(S.is_valid == 1, L, :));\n"
                         "En = E ./ vecnorm(E, 2, 2);\n"
                         "imagesc(En * En'); axis square; colorbar   % cosine similarity at layer L\n")
            elif kind == "sentence_similarity":
                code += ("\nsize(S.embeddings)                         % sentences x 12 layers x 768\n"
                         "a = squeeze(S.embeddings(1, :, :)); b = squeeze(S.embeddings(2, :, :));\n"
                         "c = sum(a .* b, 2) ./ (vecnorm(a, 2, 2) .* vecnorm(b, 2, 2));\n"
                         "plot(1:12, c, 'o-'); xlabel('BERT layer'); ylabel('cosine(sentence 1, sentence 2)')\n")
            elif kind == "phoneme_surprisal":
                code += ("\nstem(S.t, S.phoneme_surprisal, 'Marker', 'none')   % impulse at each phoneme onset\n"
                         "xlabel('Time (s)'); ylabel('phoneme surprisal (bits)')\n")
        blocks.append((rel, code))
    tabs = "".join(f'<button type="button" class="{"on" if i == 0 else ""}">{html.escape(r.split("/")[-1].replace(stem, "<stim>"))}</button>'
                   for i, (r, _) in enumerate(blocks))
    panels = "".join(f'<div class="tabpanel{" on" if i == 0 else ""}"><pre data-copy><code>{html.escape(c)}</code></pre></div>'
                     for i, (_, c) in enumerate(blocks))
    head = f'<div class="tabs" role="tablist">{tabs}</div>' if len(blocks) > 1 else ""
    return (f'<div class="card" data-tabs>{head}{panels}'
            '<div class="caption">Run from the output folder. Variable and column names were checked against the '
            'real files when this manual was built; the code itself was not run in MATLAB.</div></div>')


# ------------------------------------------------------------------ code links
_REF = re.compile(r"(sensi/[\w/]+\.py)(?::([A-Za-z_]\w*))?")


def _def_line(path: Path, name: str) -> int | None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            return node.lineno
    return None


def code_links(code_part_html: str) -> tuple[str, list[str]]:
    """A card linking every `sensi/...py[:name]` the doc's implementation part names."""
    seen, items, errs = set(), [], []
    text = re.sub(r"<[^>]+>", " ", code_part_html)
    for m in _REF.finditer(text):
        path, name = m.group(1), m.group(2)
        if (path, name) in seen:
            continue
        seen.add((path, name))
        f = PROJECT_ROOT / path
        if not f.exists():
            errs.append(f"code reference {path} does not exist")
            continue
        line = _def_line(f, name) if name else None
        if name and line is None:
            errs.append(f"code reference {path}:{name} — no such function or class")
            url, label = GITHUB_BLOB + path, path
        else:
            url = GITHUB_BLOB + path + (f"#L{line}" if line else "")
            label = f"{path} → {name}()" if name else path
        items.append(f'<li><a href="{url}"><code>{html.escape(label)}</code></a></li>')
    if not items:
        return "", errs
    return (f'<div class="card"><div class="card-title">Open in GitHub</div><ul>{"".join(items)}</ul></div>', errs)
