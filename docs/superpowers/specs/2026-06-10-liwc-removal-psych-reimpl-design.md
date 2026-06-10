# LIWC removal + license-clean composite-psych reimplementation

**Date:** 2026-06-10
**Status:** in implementation

## Why

The LIWC-22 EULA (<https://www.liwc.app/help/eula>) forbids using LIWC
dictionary material outside LIWC's own software. SensiExtractor must
therefore stop offering LIWC as a shipped backend. LIWC touches two
features:

1. **Word-level semantic indicators** — LIWC is one of two backends.
   **Empath** (free, already implemented) fully covers this job.
2. **Document-level composite psychological scores** (Authentic /
   Analytic / Tone / Clout) — `extractors/liwc_summary.py` applies the
   *published* Pennebaker (2014) formulas, but their inputs are LIWC
   category proportions read from a licensed `.dic`.

**Empath cannot recycle the composite scores** — its categories are
content/topic (joy, money…), not the grammatical function-word
categories (% articles, % prepositions, % pronouns-by-person,
% auxiliaries…) the formulas need. Those grammatical categories *are*
exactly what spaCy already produces in the pipeline. Tone (valence) is
the only semantic input; a free sentiment lexicon (VADER, MIT,
self-contained) supplies it.

## Decisions (confirmed with user)

- Composite psych: **disable the LIWC-backed version AND reimplement
  license-clean** in this batch.
- Word-level LIWC backend: **keep the radio but disabled** (greyed +
  EULA tooltip); Empath becomes the default. LIWC stays reachable only
  by a license holder via explicit CLI `--sem-dict path/to/your.dic`.
- Keep LIWC code + docs (disable, don't delete). Docs explain the EULA
  reason and link where to obtain LIWC
  (<https://liwcsoftware.onfastspring.com/>).

## New extractor — `extractors/psych_composite.py`

`compute_psych_composite(token_df, *, valence_lexicon=None) -> dict`
returning `{authentic, analytic, tone, clout}` as 0–100 floats (empty /
wordless → all NaN). Same four Pennebaker (2014) linear formulas already
in `liwc_summary.py`, but category proportions are sourced from the
spaCy-parsed token DataFrame (`upos`, `lemma`, `feats`, `dep_rel`)
instead of a `.dic`:

| Pennebaker category | Source from token_df |
|---|---|
| article | token ∈ {a, an, the} |
| prep | `upos == ADP` |
| pronoun / ppron / ipron | `upos == PRON`; personal = token in closed pronoun sets, else impersonal |
| i / we / you / shehe / they | token ∈ fixed English pronoun sets (incl. object/possessive/contraction forms) |
| auxverb / adverb / conj | `upos` ∈ AUX / ADV / {CCONJ,SCONJ} |
| negate | token ∈ negation set, or `dep_rel == neg`, or `Polarity=Neg` in feats |
| discrep | token ∈ {should, would, could, ought, …} |
| focuspresent / focusfuture | `Tense=Pres` in feats / token ∈ {will, shall, gonna} |
| posemo / negemo | `valence_lexicon[token]` sign (VADER lexicon by default) |

VADER loaded lazily (like `empath`/`liwc`); tests inject a fake
`valence_lexicon` so the suite never requires the package. Robust to a
df carrying only `token` + `upos`. These are **approximations of
approximations** — flagged ⚠️ in the doc.

## Wiring changes

- `utils/doc_summary.py` — psych block calls `compute_psych_composite(token_df)`;
  no longer gated on `dictionary_used == 'LIWC'` or `liwc_dictionary_path`
  (param kept, ignored, for call-site compat).
- `gui/planner/registry.py` — rename stage `doc_liwc` → `doc_psych`;
  cluster "doc LIWC composite" → "doc composite psych"; drop
  `available_when=("semantic.backend","liwc")`.
- `gui/planner/emit.py` + `main.py` — rename `no_doc_liwc_*` →
  `no_doc_psych_*`, keyed on `doc_psych`.
- `extractors/semantic_domain.py` — `detect_backend()` no longer
  auto-selects LIWC (always returns `empath`/None); `_LiwcScorer` kept
  for explicit-path opt-in.
- `gui/section_word_features.py` — disable the LIWC radio (tooltip +
  note linking where to obtain LIWC); default backend Empath.
- `gui/section_doc_features.py` — composite-psych card always available;
  subtitle describes the spaCy+VADER method; drop the backend gate.
- `utils/preferences.py` — **v10 → v11**: default `semantic.backend`
  → `"empath"`; migration coerces stored `"liwc"` → `"empath"`.
- `requirements.txt` — add `vaderSentiment`; annotate `liwc` as
  retained for license-holder CLI opt-in only.

## Docs

- `docs/features/doc_psych_composite.md` — rewrite: license-clean
  spaCy+VADER method, libraries `[spacy, vaderSentiment]`, EULA note on
  the retired LIWC path, Pennebaker citations retained (formulas are
  open-access).
- `docs/features/sem_indicators.md` — note LIWC backend disabled in-app
  (EULA), Empath is the default, where to obtain LIWC + price list.

## Tests

- `tests/extractors/test_psych_composite.py` (TDD) — keys/floats, empty
  → NaN, tone↑ with positive words, analytic↑ for article/prep-heavy,
  clout↑ with you/we vs i, robust to token+upos-only df, 0–100 clamp.
- Update `test_semantic_domain.py` detect_backend test; planner
  `test_stages.py` / `test_registry.py` stage rename; add v11 migration
  test.
