---
key: g2p_oov
name: Out-of-dictionary words (MFA G2P)
levels: [word]
gui_toggles: [g2p_oov]
output_columns: [ARPABET, Stress, IPA, Pron_Source]
libraries: [montreal-forced-aligner]
status: implemented
depth: deep
---

## Meaning
An **alignment option**, not a feature column: it changes how MFA handles
a word that is missing from its pronunciation dictionary. It lives at
`alignment.g2p_oov` in `config/features.yaml`, on the GUI's Sub-word page
("Out-of-dictionary words"), and as `--g2p-oov` on the CLI.

**Default: off.** Turning it on adds phone identities that were never
observed, so nothing gains them unasked.

### The problem it addresses

MFA is *lexicon-driven*: to align a word it must first look the word up
in `english_us_arpa.dict` to know which phones to search the audio for.
Words the dictionary has never heard of — unusual proper nouns, coinages,
names — have no entry. MFA still aligns them, but against its unknown-word
model, whose single phone is `spn` ("spoken noise"). You get a real
interval with no phonetic content: one `spn` phoneme row, `Stress` empty,
`IPA` empty, `Pron_Source = 'oov_unresolved'`.

That is honest but lossy. If a stimulus contains a character named
*Zorblax*, every sub-word analysis simply has a hole where that word was.

### What grapheme-to-phoneme (G2P) conversion does

G2P predicts a word's pronunciation **from its spelling**. Given the
letters `z-o-r-b-l-a-x`, a G2P model trained on the same dictionary's
spelling↔pronunciation pairs proposes the phone sequence an English
speaker would most plausibly produce: `Z AO1 R B L AE2 K S`. MFA ships
pretrained G2P models per language, and `mfa align` accepts
`--g2p_model_path`, documented as *"Path to G2P model to use for OOV
items"* ([MFA G2P]). With the option on, SensiExtractor passes the
vendored `english_us_arpa` G2P model on that flag, so out-of-dictionary
words get a generated pronunciation and are aligned with real phone
boundaries like any other word.

The affected rows are then marked `Pron_Source = 'g2p'` — see
[`phoneme`](phoneme.md#pron_source--where-the-pronunciation-came-from).

### ⚠️ The pronunciation is modeled, not observed

This is the whole reason the option is opt-in. A `g2p` phone sequence is
a **hypothesis derived from orthography**, not a transcription of what the
speaker did. English spelling is an unreliable guide to pronunciation, and
proper nouns — exactly the population this option exists for — are the
worst case: surnames and place names routinely violate the regularities a
G2P model learned. MFA's own documentation says of generated
pronunciations, "It is important to check the pronunciations manually"
([MFA generate]).

The *boundaries* MFA then finds are real (they come from the acoustic
model searching the audio). The *identities* are not. Practical
consequence: `g2p` rows are fine for onset/duration and phone-count
analyses, and questionable for anything that treats the phone label as
ground truth. `Pron_Source` exists so you can partition on this rather
than guess.

Verified on a real stimulus: with the option on, *Zorblax* received
`Z AO1 R B L AE2 K S`, no word was left phone-less, and
**in-dictionary phones were untouched** — diffing the two phone tiers,
the only change outside the OOV word was its shared boundary with the
following word moving 0.33 → 0.35 s. Every other interval was identical.

### ⚠️ It does **not** tell you whether a word was actually spoken

This is the most important thing on the page, because the option's name
invites the opposite reading.

MFA is a **forced** aligner ([McAuliffe 2017]). It takes the transcript as
ground truth and must place every word in it somewhere on the timeline. It
has no "this word is absent from the audio" output — not for OOV words,
not for any word. A word that appears in the transcript but was never
uttered gets squeezed into a sliver of time, stealing it from its
neighbours. **That limitation is pre-existing and applies to every word in
every run**, in-dictionary or not; this option neither introduces nor
fixes it.

G2P specifically does not adjudicate the question. Verified: because the
test audio genuinely says *"Bella"*, the G2P run confidently aligned
`Z AO1 R B L AE2 K S` across the interval where *Bella* was spoken. G2P
recovers pronunciations for *unusual words that were spoken*; it has no
opinion on whether they were.

Detecting genuine transcript/audio mismatch needs per-word alignment
likelihood or an ASR cross-check. MFA 3.4.0's `align` CLI exposes no such
output (checked: no likelihood/confidence flag, and no
`alignment_analysis.csv` is produced), so that work is parked as the
*Correctness scoring* roadmap item and is out of scope here.

### Turning it on or off re-runs alignment — once

The setting changes MFA's output, so a cached TextGrid produced under the
other setting is stale. Alignment persistence handles this: the freshness
sidecar written next to each TextGrid records an `mfa_g2p` key, and its
`SIDECAR_VERSION` was bumped 1 → 2 when the key was added. So:

- The first batch run after upgrading re-aligns every stimulus **once**
  (the sidecar version changed), then caches normally.
- Flipping the toggle thereafter re-aligns **once** and caches again.
- MFA still runs at most once per document per dispatch — the option
  changes the *arguments*, never the number of alignment passes.

### ⚠️ Known gap: a stale TextGrid on the per-document path is trusted

The batch path is safe. `aligner.ensure_alignments` — the pre-alignment
pass both `run_word_level_batch` and `run_doc_level_batch` call, and
therefore the path every GUI run, `run.py` run and folder-mode CLI run
takes — compares the sidecar's `mfa_g2p` against the requested setting and
re-aligns when they differ.

The **per-document** path does not. `aligner.align_with_phones` checks
only whether `<stem>.TextGrid` *exists* under `persist_dir`; if it does,
it parses it and never consults the sidecar. So if a TextGrid produced
with the opposite `g2p` setting is already sitting there, flipping this
option **changes nothing and reports nothing**. That path is reached when
an output folder is configured but the batch pre-alignment pass did not
run — for example when the pre-pass failed and fell back to per-document
alignment — and by any direct caller of `align_with_phones(...,
persist_dir=...)`.

**If you flip this option and the phones do not change, delete
`<output_folder>/Montreal Forced Alignment/<stem>.TextGrid` (and its
`.json`) and re-run, or use the folder/batch path, which re-aligns
correctly on its own.** A pure single-document CLI run (`--audio` +
`--transcript` with no output folder) has no persistence directory at all,
runs MFA every time, and honours the flag.

## Usage
Turn it **on** when your stimuli contain unusual proper nouns, invented
names or domain jargon *and* you need sub-word coverage of those words —
phoneme/syllable rows, `P_Center`, phoneme surprisal. Filter on
`Pron_Source == 'g2p'` afterwards to see exactly which words it affected,
and spot-check their `ARPABET` by ear before treating the labels as data.

Leave it **off** — the default — when you want every phone label in the
output to be dictionary-attested. The out-of-dictionary words then appear
as single `oov_unresolved` rows carrying correct timing and no phone
identity, which is a clean, self-labelling gap you can exclude in one
filter.

Either way you are told which words were affected: any word that ends up
with no phonemes at all is reported to stderr and the run log,

```
[phonemes] WARNING: 2 words had no phonemes (likely OOV): zorblax, mciver
```

and if you request G2P on a machine whose G2P model is missing, the run
warns and proceeds without it rather than failing:

```
[mfa] WARNING: g2p requested but no G2P model at …; out-of-dictionary
words will align to spn. Run: python utils/download_models.py
```

## Implementation in SensiExtractor
Sources: `extractors/aligner.py` (the flag, the sidecar, the model path),
`extractors/syllabify.py` (`Pron_Source`), `utils/download_models.py`
(fetching the model).

- **The model.** `mfa model download g2p english_us_arpa`, ~11 MB, into
  the already-vendored, git-ignored `mfa_data/pretrained_models/g2p/`.
  `utils/download_models.py` fetches it alongside the acoustic model and
  dictionary, so a fresh checkout that runs the setup script has it.
  `aligner.g2p_model_available()` is the existence check.
- **The flag.** `_run_mfa(corpus_dir, output_dir, *, g2p=False)` appends
  `--g2p_model_path <…/english_us_arpa.zip>` only when `g2p` is true
  **and** the model file exists; otherwise it warns and runs unchanged.
- **Cache correctness.** `_freshness_dict` includes `'mfa_g2p': bool(g2p)`
  and `SIDECAR_VERSION` is 2; `_is_fresh` compares it. `g2p` is also part
  of `align_with_phones`'s in-process memo key, so one process cannot
  serve an alignment cached under the opposite setting.
  ⚠️ `g2p` is deliberately **keyword-only with default `False`** on
  `align_with_phones`, `ensure_alignments`, `_write_sidecar`, `_is_fresh`
  and `_run_mfa`. Every pre-existing call site passes its arguments
  positionally, so making the parameter positional would have broken all
  of them at once.
- **Plumbing.** `alignment.g2p_oov` (prefs) → `gui/planner/emit.py` sets
  `ns.g2p_oov` → `main._prealign_batch` and `main.run` pass it to
  `ensure_alignments` / `align_with_phones`. The CLI flag is `--g2p-oov`;
  `run.py` reads the same pref. The option is **not** a per-column toggle
  and drops no columns — it changes the aligner's behaviour, so it sits in
  its own `alignment` prefs section rather than under `features`.
  ⚠️ It nonetheless lives in `features.yaml` rather than `run.yaml`
  (`utils/config_yaml.py` routes only `shared` to `run.yaml`). That is the
  right home: it is a "what to extract" decision, and it means the GUI's
  **Reset settings** turns it back off — the correct behaviour for an
  opt-in that adds modeled data.
- **`Pron_Source` derivation** is described in
  [`phoneme`](phoneme.md#implementation-in-sensiextractor). Briefly: MFA
  does not mark G2P-derived words in the TextGrid, so provenance is
  inferred from whether the **MFA words-tier form** is present in
  `english_us_arpa.dict`. Consequence worth knowing: `Pron_Source` reports
  `g2p` on the basis of that dictionary lookup, so a word absent from the
  dictionary is labelled `g2p` whenever it came back with real phones.
  With the option off it cannot have real phones — it can only be `spn` →
  `oov_unresolved` — so the two values do not collide in practice.

## Library
No new dependency. MFA is already required for all alignment; the G2P
model is a data asset MFA downloads, and `--g2p_model_path` is part of
`mfa align`'s existing interface.

## Papers
- <a name="mfa-g2p"></a>**[MFA G2P]** *Montreal Forced Aligner user guide — `mfa align` command reference.* [montreal-forced-aligner.readthedocs.io](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/alignment.html) — documents `--g2p_model_path`: "Path to G2P model to use for OOV items."
- <a name="mfa-generate"></a>**[MFA generate]** *Montreal Forced Aligner user guide — Generate pronunciations for words (`mfa g2p`).* [montreal-forced-aligner.readthedocs.io](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/dictionary_generating.html) — G2P generation of pronunciations for OOV items, and the caution that generated pronunciations should be checked manually.
- <a name="mcauliffe-2017"></a>**[McAuliffe 2017]** McAuliffe, Socolof, Mihuc, Wagner, Sonderegger. *Montreal Forced Aligner: trainable text-speech alignment using Kaldi.* Interspeech 2017. [doi:10.21437/Interspeech.2017-1386](https://doi.org/10.21437/Interspeech.2017-1386) — the aligner, and the forced-alignment assumption that the transcript is what was said.

[MFA G2P]: #mfa-g2p
[MFA generate]: #mfa-generate
[McAuliffe 2017]: #mcauliffe-2017

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
