"""Figures for the manual's "See it" parts, drawn only from real output files.

Each kind takes the manifest entry, reads the example output(s) it names, and
writes an SVG (matplotlib) or returns an HTML visual. Nothing here computes a
feature value: values are read from what SENSI SpeechFeatures wrote. Derived views
(e.g. cosine similarity between stored embedding vectors) are labelled as such.
"""
from __future__ import annotations

import html
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.io as sio  # noqa: E402

ACCENT, ACCENT_DARK, GREY, GREY_LIGHT, TEXT = "#00B2D0", "#0093AB", "#8A8F98", "#D8DCE0", "#1F2328"
SECOND = "#C77A00"   # the app's amber, for a second series

plt.rcParams.update({
    "svg.fonttype": "none", "svg.hashsalt": "sensi-manual",
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 10, "axes.edgecolor": GREY_LIGHT, "axes.labelcolor": TEXT, "axes.titlesize": 10.5,
    "axes.titleweight": "semibold", "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": "#5B6370", "ytick.color": "#5B6370", "axes.grid": True, "grid.color": "#EEF0F2",
    "grid.linewidth": .8, "legend.frameon": False, "figure.dpi": 100,
})


class FigureError(RuntimeError):
    pass


# ------------------------------------------------------------------ reading
def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FigureError(f"missing example output {path.as_posix()}")
    return pd.read_csv(path)


def read_mat(path: Path) -> dict:
    if not path.exists():
        raise FigureError(f"missing example output {path.as_posix()}")
    return {k: v for k, v in sio.loadmat(str(path)).items() if not k.startswith("__")}


def wav(stem: str, examples_dir: Path):
    import soundfile as sf
    p = examples_dir / "audio" / f"{stem}.wav"
    if not p.exists():
        return None, None
    y, sr = sf.read(str(p), always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


def first_sentence(df: pd.DataFrame) -> pd.DataFrame:
    if "sentence_id" in df:
        return df[df["sentence_id"] == df["sentence_id"].min()]
    return df


def _save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight", metadata={"Date": None, "Creator": None})
    plt.close(fig)


def _words(df):
    return df[df["upos"] != "PUNCT"] if "upos" in df else df


# ------------------------------------------------------------------ SVG kinds
def word_bars(df, y, out, title):
    d = _words(first_sentence(df)).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.4, 2.9))
    vals = d[y].astype(float)
    ax.bar(range(len(d)), vals.fillna(0), color=[ACCENT if v == v else GREY_LIGHT for v in vals], width=.72)
    for i, v in enumerate(vals):
        if v != v:
            ax.text(i, .05, "NaN", ha="center", va="bottom", fontsize=7.5, color=GREY, rotation=90)
    ax.set_xticks(range(len(d)), d["token"], rotation=40, ha="right")
    ax.set_ylabel(y); ax.set_title(title, loc="left"); ax.grid(axis="x", visible=False)
    _save(fig, out)


def _plain(v, _pos=None):
    """0.001 rather than 10^-3 on a log axis."""
    return f"{v:.10f}".rstrip("0").rstrip(".") if v < 1 else f"{v:g}"


def predict_compare(dfs: dict, y, out, title, log=False):
    fig, axes = plt.subplots(1, len(dfs), figsize=(8.4, 3.1 if log else 2.8), sharey=True)
    for ax, (stem, df) in zip(np.atleast_1d(axes), dfs.items()):
        d = _words(df).reset_index(drop=True)
        vals = d[y].astype(float)
        colors = [GREY_LIGHT] * len(d)
        if len(d):
            colors[-1] = ACCENT if stem == "predictable" else SECOND
        if log:
            floor = 10 ** np.floor(np.log10(np.nanmin([np.nanmin(df[y].astype(float)) for df in dfs.values()])))
            ax.bar(range(len(d)), (vals - floor).clip(lower=0).fillna(0), bottom=floor, color=colors, width=.7)
            ax.set_yscale("log"); ax.set_ylim(floor, 1.6)
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_plain))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            for i, v in enumerate(vals):
                ax.text(i, floor * 1.15 if v != v else v * 1.15, "NaN" if v != v else f"{v:.2g}",
                        ha="center", va="bottom", fontsize=7, color="#5B6370")
        else:
            ax.bar(range(len(d)), vals.fillna(0), color=colors, width=.7)
            for i, v in enumerate(vals):
                ax.text(i, (0 if v != v else v), "NaN" if v != v else (f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}"),
                        ha="center", va="bottom", fontsize=7.5, color="#5B6370")
        ax.set_xticks(range(len(d)), d["token"], rotation=30, ha="right")
        ax.set_title(f"{stem}.txt", loc="left"); ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].set_ylabel(y + ("  (log scale)" if log else ""))
    fig.suptitle(title, x=0.01, ha="left", fontsize=10.5, fontweight="semibold")
    fig.tight_layout()
    _save(fig, out)


def norms(df, cols, out, title):
    d = _words(first_sentence(df)).reset_index(drop=True)
    cols = [c for c in cols if c != "token"]
    fig, axes = plt.subplots(len(cols), 1, figsize=(8.4, 1.05 * len(cols) + .6), sharex=True)
    for ax, c in zip(axes, cols):
        v = d[c].astype(float)
        ax.bar(range(len(d)), v.fillna(0), color=[ACCENT if x == x else GREY_LIGHT for x in v], width=.7)
        ax.set_ylabel(c, rotation=0, ha="right", va="center", fontsize=8.5)
        ax.grid(axis="x", visible=False); ax.tick_params(axis="y", labelsize=7.5)
    axes[-1].set_xticks(range(len(d)), d["token"], rotation=40, ha="right")
    axes[0].set_title(title + "  (grey = word not in that norm table → NaN)", loc="left")
    _save(fig, out)


def word_timeline(df, stem, examples_dir, out, title, show_gaps=False):
    d = df[df["Onset"].notna()].reset_index(drop=True)
    y, sr = wav(stem, examples_dir)
    t_end = float(d["End"].max()) + .2
    fig, ax = plt.subplots(figsize=(8.6, 2.6))
    if y is not None:
        t = np.arange(len(y)) / sr
        m = t <= t_end
        ax.plot(t[m], y[m] / (np.abs(y).max() or 1), color=GREY_LIGHT, lw=.5)
    for i, r in d.iterrows():
        ax.axvspan(r["Onset"], r["End"], ymin=.06, ymax=.94, color=ACCENT if i % 2 == 0 else ACCENT_DARK, alpha=.18, lw=0)
        ax.axvline(r["Onset"], ymin=.06, ymax=.94, color=ACCENT_DARK, lw=.6, alpha=.7)
        ax.text((r["Onset"] + r["End"]) / 2, 1.08 if i % 2 == 0 else -1.18, r["token"], ha="center", fontsize=7.5,
                color=TEXT)
    if show_gaps and "Word_Gap" in d:
        for i in range(1, len(d)):
            g = d.loc[i, "Onset"] - d.loc[i - 1, "End"]
            if g > 0.04:
                ax.annotate("", xy=(d.loc[i, "Onset"], 0), xytext=(d.loc[i - 1, "End"], 0),
                            arrowprops=dict(arrowstyle="<->", color=SECOND, lw=1))
    ax.set_ylim(-1.35, 1.3); ax.set_yticks([]); ax.set_xlim(0, t_end)
    ax.set_xlabel("Time (s)"); ax.set_title(title, loc="left"); ax.grid(False)
    _save(fig, out)


def word_prosody(df, prefix, out, title, unit):
    d = df[df["Onset"].notna() & df[f"{prefix}_Mean"].notna()].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.6, 2.9))
    for _, r in d.iterrows():
        ax.fill_between([r["Onset"], r["End"]], r[f"{prefix}_Min"], r[f"{prefix}_Max"], color=ACCENT, alpha=.16, lw=0)
        ax.plot([r["Onset"], r["End"]], [r[f"{prefix}_Mean"]] * 2, color=ACCENT_DARK, lw=2)
        ax.text((r["Onset"] + r["End"]) / 2, r[f"{prefix}_Max"], r["token"], ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("Time (s)"); ax.set_ylabel(f"{prefix} ({unit})")
    ax.set_title(title + f"  — line: {prefix}_Mean; band: {prefix}_Min to {prefix}_Max (5th–95th percentile)", loc="left")
    _save(fig, out)


def syllable_timeline(df, stem, examples_dir, out, title, pcenter=False, phonemes=None, focus=None):
    d = df[df["Onset"].notna()].reset_index(drop=True)
    d = d[d["sentence_id"] == d["sentence_id"].min()]
    if focus:                                  # zoom onto these consecutive words (first sentence)
        tok = d["token"].astype(str).str.lower().tolist()
        k = tok.index(focus[0].lower())
        last = len(tok) - 1 - tok[::-1].index(focus[-1].lower())
        d = d.iloc[k:last + 1]
        title += "  — zoomed on “" + " ".join(focus) + "”"
    y, sr = wav(stem, examples_dir)
    t_end = float(d["End"].max()) + .15
    fig, ax = plt.subplots(figsize=(8.6, 2.6))
    if y is not None:
        t = np.arange(len(y)) / sr
        m = t <= t_end
        ax.plot(t[m], y[m] / (np.abs(y).max() or 1), color=GREY_LIGHT, lw=.5)
    for i, (_, r) in enumerate(d.iterrows()):
        ax.axvspan(r["Onset"], r["End"], ymin=.08, ymax=.92, color=ACCENT if i % 2 == 0 else ACCENT_DARK, alpha=.18, lw=0)
        ax.axvline(r["Onset"], ymin=.08, ymax=.92, color=ACCENT_DARK, lw=.6, alpha=.7)
        ax.text((r["Onset"] + r["End"]) / 2, 1.1 if i % 2 == 0 else -1.22, str(r["Label"]), ha="center", fontsize=7.5)
        if pcenter and r.get("P_Center") == r.get("P_Center"):
            ax.plot([r["P_Center"]] * 2, [-.85, .85], color=SECOND, lw=1.6)
    if pcenter and phonemes is not None:
        v = phonemes[phonemes["Stress"].notna() & phonemes["Onset"].notna()]
        v = v[(v["Onset"] >= d["Onset"].min()) & (v["Onset"] <= d["End"].max())]
        for x in v["Onset"]:
            ax.plot([x, x], [-.85, .85], color=TEXT, lw=.9, ls=(0, (2, 2)))
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], color=ACCENT_DARK, lw=.8, label="syllable Onset"),
                           Line2D([], [], color=TEXT, lw=.9, ls=(0, (2, 2)), label="vowel onset (from the phoneme CSV)"),
                           Line2D([], [], color=SECOND, lw=1.6, label="P_Center")],
                  loc="upper center", bbox_to_anchor=(.5, -.28), ncol=3, fontsize=8)
    ax.set_ylim(-1.4, 1.35); ax.set_yticks([]); ax.set_xlim(float(d["Onset"].min()) - .05, t_end)
    ax.set_xlabel("Time (s)"); ax.set_title(title, loc="left"); ax.grid(False)
    _save(fig, out)


def syllable_prosody(df, out, title, units):
    d = df[df["Onset"].notna()].reset_index(drop=True)
    d = d[d["sentence_id"] == d["sentence_id"].min()]
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 4.2), sharex=True)
    for ax, col, color, name in ((axes[0], "Pitch_Mean", ACCENT_DARK, "Pitch"), (axes[1], "Volume_Mean", SECOND, "Volume")):
        v = d[col].astype(float)
        for (_, r), x in zip(d.iterrows(), v):
            if x == x:
                ax.plot([r["Onset"], r["End"]], [x, x], color=color, lw=2.4, solid_capstyle="butt")
        top = np.nanmax(v) if v.notna().any() else 0
        span = (np.nanmax(v) - np.nanmin(v)) if v.notna().any() else 1
        for (_, r), x in zip(d.iterrows(), v):
            if x == x:
                ax.text((r["Onset"] + r["End"]) / 2, x + .06 * (span or 1), str(r["Label"]), ha="center",
                        va="bottom", fontsize=6.5, color=TEXT)
        ax.set_ylabel(f"{col}\n({units.get(name, '')})", fontsize=8.5)
        ax.set_title(f"{col} — one bar per syllable, from its Onset to its End", loc="left", fontsize=9.5)
        ax.set_ylim(np.nanmin(v) - .1 * (span or 1), top + .3 * (span or 1))
    axes[1].set_xlabel("Time (s)")
    fig.suptitle(title, x=.01, ha="left", fontsize=10.5, fontweight="semibold"); fig.tight_layout()
    _save(fig, out)


def phoneme_timeline(df, out, title, ycol=None):
    d = df[df["Onset"].notna()].reset_index(drop=True)
    d = d[d["sentence_id"] == d["sentence_id"].min()].head(28)
    fig, ax = plt.subplots(figsize=(8.6, 2.6 if ycol is None else 3.0))
    for i, (_, r) in enumerate(d.iterrows()):
        h = 1 if ycol is None else (r[ycol] if r[ycol] == r[ycol] else 0)
        ax.bar(r["Onset"], h, width=r["End"] - r["Onset"], align="edge",
               color=(SECOND if r.get("Stress") in (1, 2, 1.0, 2.0) else ACCENT) if ycol is None else ACCENT,
               alpha=.35 if ycol is None else .75, edgecolor="white")
        ax.text(r["Onset"] + (r["End"] - r["Onset"]) / 2, h + .03 * (1 if ycol is None else max(1, d[ycol].max())),
                str(r["IPA"]), ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Time (s)")
    if ycol is None:
        ax.set_yticks([]); ax.set_ylim(0, 1.35); ax.grid(False)
        title += "  (amber = vowel carrying stress 1 or 2)"
    else:
        ax.set_ylabel(ycol)
    ax.set_title(title, loc="left")
    _save(fig, out)


def timeseries(m, y, out, title, t_max=None):
    t = m["t"].ravel(); v = m[y].ravel()
    if t_max:
        k = t <= t_max; t, v = t[k], v[k]
    fig, ax = plt.subplots(figsize=(8.6, 2.6))
    ax.plot(t, v, color=ACCENT_DARK, lw=1.1)
    ax.fill_between(t, v, v.min(), color=ACCENT, alpha=.12, lw=0)
    ax.set_xlabel("Time (s)"); ax.set_ylabel(y)
    ax.set_title(title + f"  — {len(m['t'])} samples at fs = {float(m['fs'].squeeze()):g} Hz", loc="left")
    _save(fig, out)


def image(m, y, out, title):
    a = m[y]; t = m["t"].ravel()
    fig, ax = plt.subplots(figsize=(8.6, 2.8))
    extent = [t[0], t[-1], .5, a.shape[1] + .5]
    im = ax.imshow(a.T, aspect="auto", origin="lower", extent=extent, cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Time (s)")
    if y == "logmel" and "mel_center_freqs" in m:
        f = m["mel_center_freqs"].ravel()
        idx = np.linspace(0, len(f) - 1, min(6, len(f))).astype(int)
        ax.set_yticks(idx + 1, [f"{f[i]:.0f}" for i in idx]); ax.set_ylabel("Band centre (Hz)")
    else:
        ax.set_ylabel("Coefficient")
    ax.grid(False); fig.colorbar(im, ax=ax, pad=.01, label=y)
    ax.set_title(title + f"  — matrix {a.shape[0]} × {a.shape[1]}", loc="left")
    _save(fig, out)


def _cos(E):
    n = np.linalg.norm(E, axis=1, keepdims=True); n[n == 0] = 1
    E = E / n
    return E @ E.T


def embedding(m, out, title):
    E = np.asarray(m["embeddings"], dtype=float)
    toks = [str(np.asarray(x).ravel()[0]) if np.asarray(x).size else "" for x in m["tokens"].ravel()]
    valid = m["is_valid"].ravel().astype(bool)
    Ev, tv = E[valid], [t for t, ok in zip(toks, valid) if ok]
    k = min(14, len(tv))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.4), gridspec_kw={"width_ratios": [1.5, 1]})
    a1.imshow(Ev[:k, :60], aspect="auto", cmap="RdBu_r", vmin=-np.abs(Ev[:k, :60]).max(), vmax=np.abs(Ev[:k, :60]).max())
    a1.set_yticks(range(k), tv[:k], fontsize=7.5); a1.set_xlabel(f"dimension (first 60 of {E.shape[1]})")
    a1.set_title("Stored vectors", loc="left"); a1.grid(False)
    C = _cos(Ev[:k])
    a2.imshow(C, cmap="viridis", vmin=-.2, vmax=1)
    a2.set_xticks(range(k), tv[:k], rotation=90, fontsize=6.5); a2.set_yticks(range(k), tv[:k], fontsize=6.5)
    a2.set_title("Cosine similarity (derived)", loc="left"); a2.grid(False)
    fig.suptitle(title, x=.01, ha="left", fontsize=10.5, fontweight="semibold"); fig.tight_layout()
    _save(fig, out)


def embedding_layers(m, out, title):
    E = np.asarray(m["embeddings"], dtype=float)          # words x layers x dims
    toks = [str(np.asarray(x).ravel()[0]) if np.asarray(x).size else "" for x in m["tokens"].ravel()]
    valid = m["is_valid"].ravel().astype(bool)
    Ev, tv = E[valid], [t for t, ok in zip(toks, valid) if ok]
    k = min(12, len(tv))
    layers = [0, E.shape[1] // 2, E.shape[1] - 1]
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.3))
    for ax, L in zip(axes, layers):
        ax.imshow(_cos(Ev[:k, L, :]), cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"layer {L + 1}", loc="left"); ax.grid(False)
        ax.set_xticks(range(k), tv[:k], rotation=90, fontsize=6.3); ax.set_yticks(range(k), tv[:k], fontsize=6.3)
    fig.suptitle(title + " — cosine similarity between words (derived from stored vectors)", x=.01, ha="left",
                 fontsize=10.5, fontweight="semibold")
    fig.tight_layout()
    _save(fig, out)


def sentence_similarity(m, out, title):
    E = np.asarray(m["embeddings"], dtype=float)          # sentences x layers x dims
    if E.shape[0] < 2:
        raise FigureError("sentence_similarity needs an example with at least two sentences")
    sims = [float(_cos(E[:2, L, :])[0, 1]) for L in range(E.shape[1])]
    fig, ax = plt.subplots(figsize=(8.4, 2.6))
    ax.plot(range(1, len(sims) + 1), sims, "o-", color=ACCENT_DARK, ms=4)
    ax.set_xticks(range(1, len(sims) + 1)); ax.set_xlabel("BERT layer"); ax.set_ylabel("cosine(sentence 1, sentence 2)")
    ax.set_title(title + " — similarity of the two sentences at each layer (derived)", loc="left")
    _save(fig, out)


def categories(df, out, title):
    d = first_sentence(_words(df)).reset_index(drop=True)
    cats = [c for c in d.columns if c.startswith("Sem_") and c != "Sem_Categories"]
    M = d[cats].fillna(0).astype(float).to_numpy().T
    fig, ax = plt.subplots(figsize=(8.6, .28 * len(cats) + 1.2))
    ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=1.4)
    ax.set_yticks(range(len(cats)), [c.replace("Sem_Empath_", "") for c in cats], fontsize=7.5)
    ax.set_xticks(range(len(d)), d["token"], rotation=50, ha="right", fontsize=7.5); ax.grid(False)
    ax.set_title(title + "  (filled = the word is in that Empath category)", loc="left")
    _save(fig, out)


def doc_bars(df, cols, out, title, units=None):
    d = df.copy()
    cols = [c for c in cols if c != "stimulus_id"]
    n = len(cols)
    ncol = 2 if n > 1 else 1
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(8.8, 2.0 * nrow + .3), squeeze=False)
    for ax, c in zip(axes.ravel(), cols):
        v = pd.to_numeric(d[c], errors="coerce")
        ax.barh(range(len(d)), v.fillna(0), color=[ACCENT if x == x else GREY_LIGHT for x in v], height=.65)
        ax.set_yticks(range(len(d)), d["stimulus_id"], fontsize=7.5); ax.invert_yaxis()
        note = "  — NaN for every example (texts too short for this measure)" if v.isna().all() else ""
        u = (units or {}).get(c)
        ax.set_title(c + (f"  ({u})" if u else "") + note, loc="left", fontsize=9.5); ax.grid(axis="y", visible=False)
        for i, x in enumerate(v):
            ax.text(x if x == x else 0, i, " NaN" if x != x else f" {x:.3g}", va="center", fontsize=7, color="#5B6370")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(title, x=.01, ha="left", fontsize=10.5, fontweight="semibold"); fig.tight_layout()
    _save(fig, out)


# ------------------------------------------------------------------ HTML kinds
def strip_html(df, cols, kind="table") -> str:
    """The first sentence laid out word by word, with each column's value under it."""
    d = first_sentence(df).head(16)
    key = next(k for k in ("token", "sentence_id", "stimulus_id") if k in d)
    cols = [c for c in cols if c != key]
    head = "".join(f"<th>{html.escape(str(t))}</th>" for t in d[key])
    rows = []
    for c in cols:
        cells = []
        for v in d[c]:
            sval = "" if (isinstance(v, float) and v != v) else str(v)
            if kind == "tags" and c == "upos":
                cells.append(f'<td><span class="chip"><b>{html.escape(sval)}</b></span></td>')
            else:
                empty = '<span style="color:var(--text-muted)">∅</span>'
                cells.append(f"<td>{html.escape(sval) if sval else empty}</td>")
        rows.append(f'<tr><th class="hl">{html.escape(c)}</th>{"".join(cells)}</tr>')
    return (f'<table class="excerpt strip"><thead><tr><th>{key}</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def char_offsets_html(df) -> str:
    d = first_sentence(df)
    sent = str(d["sentence"].iloc[0])
    spans = []
    for i, (_, r) in enumerate(d.iterrows()):
        s, e = int(r["start"]), int(r["end"])
        spans.append((s, e, i))
    out, pos = [], 0
    for s, e, i in spans:
        out.append(html.escape(sent[pos:s]))
        out.append(f'<span style="background:var(--accent-soft);border-bottom:2px solid var(--accent);'
                   f'padding:0 1px" title="start={s} end={e}">{html.escape(sent[s:e])}'
                   f'<sub style="color:var(--text-muted);font-size:10px">{s}–{e}</sub></span>')
        pos = e
    out.append(html.escape(sent[pos:]))
    return f'<p style="font-family:var(--font-code);font-size:14px;line-height:2.4">{"".join(out)}</p>'


def dependency_svg(df, out: Path, title) -> None:
    d = first_sentence(df).reset_index(drop=True)
    n = len(d)
    # space tokens by their length so long words don't collide
    w = np.array([max(len(str(t)), 3) for t in d["token"]], dtype=float) * .13 + .35
    xs = np.cumsum(w) - w / 2
    fig, ax = plt.subplots(figsize=(max(8.6, xs[-1] * .62), 3.6))
    tid = dict(zip(d["token_id"], xs))            # head_token_id refers to the global token_id
    top = 2.0
    for x, (_, r) in zip(xs, d.iterrows()):
        ax.text(x, 0, r["token"], ha="center", va="top", fontsize=8.5)
        ax.text(x, -.42, r["upos"], ha="center", va="top", fontsize=6.5, color=GREY)
        h = r["head_token_id"]
        if r["dep_rel"] == "ROOT" or h not in tid:
            ax.annotate("ROOT", xy=(x, .12), xytext=(x, 1.7), ha="center", fontsize=7, color=SECOND,
                        arrowprops=dict(arrowstyle="->", color=SECOND, lw=1))
            continue
        hx = tid[h]
        span = abs(hx - x)
        height = .35 + .28 * span
        mid = (hx + x) / 2
        ts = np.linspace(0, 1, 30)
        px = hx + (x - hx) * ts
        py = .12 + 4 * height * ts * (1 - ts)
        ax.plot(px, py, color=ACCENT_DARK, lw=.9)
        ax.annotate("", xy=(x, .12), xytext=(px[-3], py[-3]), arrowprops=dict(arrowstyle="->", color=ACCENT_DARK, lw=.9))
        ax.text(mid, .12 + height + .03, r["dep_rel"], ha="center", fontsize=6.5, color=ACCENT_DARK)
        top = max(top, .12 + height + .35)
    ax.set_xlim(-.4, xs[-1] + w[-1] / 2); ax.set_ylim(-.8, top); ax.axis("off")
    ax.set_title(title, loc="left")
    _save(fig, out)


# ------------------------------------------------------------ annotated text
def _token_xs(tokens, gap_after=()):
    """x centre and width of each token, spaced by its length (a wider gap after `gap_after` indices)."""
    w = np.array([max(len(str(s)), 2) for s in tokens], dtype=float) * .14 + .42
    extra = np.array([.6 if i - 1 in gap_after else 0 for i in range(len(tokens))])
    left = np.cumsum(np.r_[0, w[:-1]]) + np.cumsum(extra)
    return left + w / 2, w


def _bracket(ax, x0, x1, y, label, color, h=.12):
    ax.plot([x0, x0, x1, x1], [y - h, y, y, y - h], color=color, lw=1.3)
    ax.text((x0 + x1) / 2, y + .06, label, ha="center", va="bottom", fontsize=8, color=color, fontweight="semibold")


def structure(df, out, title):
    """The whole (short) example: brackets for document, paragraphs and sentences; ids under each token."""
    d = df.reset_index(drop=True)
    breaks = [i for i in range(len(d) - 1) if d.loc[i + 1, "paragraph_id"] != d.loc[i, "paragraph_id"]]
    xs, w = _token_xs(d["token"], breaks)
    colors = [ACCENT_DARK, SECOND, "#5E5BC0", "#2E8B57"]
    fig, ax = plt.subplots(figsize=(max(8.6, (xs[-1] + w[-1]) * .7), 3.4))
    for i, r in d.iterrows():
        c = colors[(int(r["sentence_id"]) - 1) % len(colors)]
        ax.text(xs[i], 0, r["token"], ha="center", va="center", fontsize=10, color=TEXT,
                bbox=dict(boxstyle="round,pad=.25", fc="white", ec=c, lw=1))
        ax.text(xs[i], -.55, str(r["token_id"]), ha="center", va="center", fontsize=9, color=TEXT)
        ax.text(xs[i], -.95, str(r["token_in_sentence_id"]), ha="center", va="center", fontsize=9, color=c)
    lx = -.25
    ax.text(lx, -.55, "token_id", ha="right", va="center", fontsize=8, color=GREY)
    ax.text(lx, -.95, "token_in_sentence_id", ha="right", va="center", fontsize=8, color=GREY)
    for sid, g in d.groupby("sentence_id"):
        i0, i1 = g.index.min(), g.index.max()
        _bracket(ax, xs[i0] - w[i0] / 2 + .05, xs[i1] + w[i1] / 2 - .05, .42, f"sentence_id = {sid}",
                 colors[(int(sid) - 1) % len(colors)])
    for pid, g in d.groupby("paragraph_id"):
        i0, i1 = g.index.min(), g.index.max()
        _bracket(ax, xs[i0] - w[i0] / 2, xs[i1] + w[i1] / 2, .95, f"paragraph_id = {pid}", GREY)
    _bracket(ax, xs[0] - w[0] / 2 - .1, xs[-1] + w[-1] / 2 + .1, 1.45, f"doc_id = {d.loc[0, 'doc_id']}", TEXT)
    for i in breaks:
        ax.text((xs[i] + w[i] / 2 + xs[i + 1] - w[i + 1] / 2) / 2, 0, "¶", ha="center", va="center", fontsize=11, color=GREY)
    ax.set_xlim(lx - 1.9, xs[-1] + w[-1] / 2 + .3); ax.set_ylim(-1.2, 1.85); ax.axis("off")
    ax.set_title(title, loc="left")
    _save(fig, out)


def lemmas(df, out, title):
    """Each token above its lemma; an arrow (highlighted) where the lemma differs."""
    d = df.reset_index(drop=True)
    xs, w = _token_xs(d["token"])
    fig, ax = plt.subplots(figsize=(max(8.6, (xs[-1] + w[-1]) * .72), 2.5))
    for i, r in d.iterrows():
        changed = str(r["token"]).lower() != str(r["lemma"]).lower()
        c = SECOND if changed else GREY
        ax.text(xs[i], 1, r["token"], ha="center", va="center", fontsize=10, color=TEXT)
        ax.annotate("", xy=(xs[i], .3), xytext=(xs[i], .78),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.4 if changed else .8))
        ax.text(xs[i], .1, r["lemma"], ha="center", va="center", fontsize=10, color=c,
                fontweight="semibold" if changed else "normal")
    ax.text(-.3, 1, "token", ha="right", va="center", fontsize=8, color=GREY)
    ax.text(-.3, .1, "lemma", ha="right", va="center", fontsize=8, color=GREY)
    ax.set_xlim(-1.6, xs[-1] + w[-1] / 2 + .2); ax.set_ylim(-.2, 1.3); ax.axis("off")
    ax.set_title(title + "  (amber = the lemma differs from the token)", loc="left")
    _save(fig, out)


def morph(df, out, title):
    """Each word with its part of speech and its morphological features listed under it,
    wrapped onto as many rows as the figure width needs."""
    d = _words(df).reset_index(drop=True)
    feats = [[] if not isinstance(f, str) else f.split("|") for f in d["feats"]]
    w = np.array([max([len(str(t))] + [len(x) for x in fs]) for t, fs in zip(d["token"], feats)], dtype=float) * .1 + .35
    limit = 12.0
    rows, cur, used = [], [], 0.0
    for i in range(len(d)):
        if cur and used + w[i] > limit:
            rows.append(cur); cur, used = [], 0.0
        cur.append(i); used += w[i]
    if cur:
        rows.append(cur)
    nmax = max((len(f) for f in feats), default=0)
    row_h = 1.3 + .34 * max(nmax, 1)
    fig, ax = plt.subplots(figsize=(8.6, .9 + .62 * row_h * len(rows)))
    for r, idx in enumerate(rows):
        y0 = -r * row_h
        x = 0.0
        for i in idx:
            xc = x + w[i] / 2
            ax.text(xc, y0, d.loc[i, "token"], ha="center", va="center", fontsize=10, color=TEXT)
            ax.text(xc, y0 - .42, d.loc[i, "upos"], ha="center", va="center", fontsize=8, color="white",
                    bbox=dict(boxstyle="round,pad=.25", fc=ACCENT_DARK, ec="none"))
            if not feats[i]:
                ax.text(xc, y0 - .85, "(none)", ha="center", va="center", fontsize=8, color=GREY)
            for k, f in enumerate(feats[i]):
                ax.text(xc, y0 - .85 - .34 * k, f, ha="center", va="center", fontsize=8, color=SECOND)
            x += w[i]
    ax.set_xlim(0, limit); ax.set_ylim(-(len(rows) - 1) * row_h - .85 - .34 * nmax, .4); ax.axis("off")
    ax.set_title(title + "  (word · upos · each feats entry)", loc="left")
    _save(fig, out)


def sentence_timeline(df, stem, examples_dir, out, title):
    """Each sentence's Onset–End span over the recording."""
    d = df[df["Onset"].notna()].reset_index(drop=True)
    y, sr = wav(stem, examples_dir)
    t_end = float(d["End"].max()) + .3
    fig, ax = plt.subplots(figsize=(8.6, 2.6))
    if y is not None:
        t = np.arange(len(y)) / sr
        m = t <= t_end
        ax.plot(t[m], y[m] / (np.abs(y).max() or 1), color=GREY_LIGHT, lw=.5)
    for i, r in d.iterrows():
        c = ACCENT_DARK if i % 2 == 0 else SECOND
        ax.axvspan(r["Onset"], r["End"], ymin=.1, ymax=.9, color=c, alpha=.14, lw=0)
        ax.annotate("", xy=(r["End"], 1.12), xytext=(r["Onset"], 1.12), arrowprops=dict(arrowstyle="<->", color=c, lw=1.2))
        ax.text((r["Onset"] + r["End"]) / 2, 1.2, f"sentence_id = {r['sentence_id']}", ha="center", va="bottom",
                fontsize=8, color=c, fontweight="semibold")
        ax.text(r["Onset"], -1.25, f"Onset {r['Onset']:.2f}", ha="left", fontsize=7, color=c)
        ax.text(r["End"], -1.45, f"End {r['End']:.2f}", ha="right", fontsize=7, color=c)
    ax.set_ylim(-1.6, 1.55); ax.set_yticks([]); ax.set_xlim(0, t_end)
    ax.set_xlabel("Time (s)"); ax.set_title(title, loc="left"); ax.grid(False)
    _save(fig, out)
