---
key: feat_phoneme_ipa
name: IPA phoneme symbol
levels: [phoneme]
gui_toggles: [feat_phoneme_ipa]
output_columns: [IPA]
libraries: []
status: implemented
depth: deep
---

## Meaning
One IPA segment per row of `phoneme/<stim>_phonemes.csv`, giving the same
phone as the [`ARPABET`](phoneme.md) column in the notation most
phoneticians and most non-English-speech literature actually read.
`ARPABET` `EY` becomes `eɪ`; `R` becomes `ɹ`; `AH` with `Stress = 0`
becomes `ə`.

This is a **relabelling, not a new measurement**. `IPA` and `ARPABET`
carry identical information and are perfectly redundant as regressors —
pick one. The column exists so that a collaborator who does not read
ARPABET can inspect and share the phoneme CSV, and so phone inventories
can be joined against non-ARPABET corpora.

The syllable CSV carries a dash-joined counterpart; see
[`syllable_labels`](syllable_labels.md).

### ⚠️ `IPA` is not a function of `ARPABET` alone — it reads `Stress` too

Two ARPABET vowels correspond to **two different IPA vowels depending on
the stress digit**, so you cannot reconstruct this column from `ARPABET`
by itself, and a lookup table keyed on `ARPABET` alone will be wrong on
the commonest vowel in English:

| `ARPABET` | `Stress` | `IPA` | Name |
|---|---|---|---|
| `AH` | `0` | `ə` | schwa |
| `AH` | `1` or `2` | `ʌ` | STRUT vowel |
| `ER` | `0` | `ɚ` | unstressed r-coloured vowel |
| `ER` | `1` or `2` | `ɝ` | stressed r-coloured vowel |

Both pairs are notational conventions of English IPA practice rather than
claims about two acoustically distinct sounds, and both follow directly
from how the CMU dictionary encodes stress. ARPABET has **no separate
symbol for schwa** — a reduced vowel is written `AH0`. This is checkable
in the dictionary this pipeline ships and aligns against
(`mfa_data/pretrained_models/dictionary/english_us_arpa.dict`):

```
about    AH0 B AW1 T        # AH0 = /ə/,  AW1 = /aʊ/
sofa     S OW1 F AH0        # final schwa, again AH0
butter   B AH1 T ER0        # AH1 = /ʌ/ (STRUT),  ER0 = /ɚ/
hurt     HH ER1 T           # ER1 = /ɝ/
```

For `ER`, the stressed/unstressed split mirrors the standard ⟨ɜ⟩/⟨ə⟩
convention: "in some English adaptations of IPA, ⟨ɝ⟩ and ⟨ɚ⟩ signify
stressed and unstressed, respectively, rather than a difference in
phonetic quality" ([R-coloured vowel]).

Every other ARPABET code is stress-independent: `EY` is `eɪ` whether it
carries `0`, `1` or `2`.

### No stress marks are emitted

IPA transcription normally prefixes a stressed syllable with `ˈ`
(primary) or `ˌ` (secondary). This column emits **neither**, for two
reasons:

1. The [`Stress`](phoneme.md) column already carries that information, in
   a form you can filter and group on numerically.
2. `ˈ`/`ˌ` are *syllable*-level diacritics, and this column holds exactly
   one *segment* per row. There is no correct place to put them.

If you need conventional IPA output, reassemble it yourself from
`(IPA, Stress, syllable_id)` — the pieces are all present.

## The full table

39 phones, reproduced from `extractors/arpabet_ipa.py`, which is the
single source of truth. Example words are given with the **dominant**
(highest pronunciation-probability) entry the shipped MFA dictionary
lists for them, so every row is independently checkable. `english_us_arpa.dict`
carries a pronunciation-probability weight per entry when a word has more
than one — several common words do — and citing a low-probability variant
as "the" pronunciation would undercut a page whose whole point is
dictionary verification, so the weight is checked for every example below.

### Vowels (15)

| `ARPABET` | `IPA` | Example | Dictionary entry |
|---|---|---|---|
| `AA` | `ɑ` | odd | `AA1 D` |
| `AE` | `æ` | at | `AE1 T` |
| `AH` | `ʌ` (`ə` when `Stress == 0`) | butter / about | `B AH1 T ER0` / `AH0 B AW1 T` |
| `AO` | `ɔ` | caught | `K AO1 T` |
| `AW` | `aʊ` | cow | `K AW1` |
| `AY` | `aɪ` | hide | `HH AY1 D` |
| `EH` | `ɛ` | Ed | `EH1 D` |
| `ER` | `ɝ` (`ɚ` when `Stress == 0`) | hurt / letter | `HH ER1 T` / `L EH1 T ER0` |
| `EY` | `eɪ` | ate | `EY1 T` |
| `IH` | `ɪ` | big | `B IH1 G` |
| `IY` | `i` | eat | `IY1 T` |
| `OW` | `oʊ` | oat | `OW1 T` |
| `OY` | `ɔɪ` | toy | `T OY1` |
| `UH` | `ʊ` | hood | `HH UH1 D` |
| `UW` | `u` | two | `T UW1` |

CMU's canonical example word for `AO` is *ought*, but the MFA dictionary
this pipeline ships lists `ought` as `AA1` (and `AA1 T` at weight `0.99`,
its dominant entry) — it reflects the **cot–caught merger** common in
American English, so *ought* is not a usable `AO` example here. *caught*
is used above instead, and is the least clean of the four dictionary
examples on this page even so: `english_us_arpa.dict` carries three
weighted entries for it — `K AO1 T` at `0.24`, `K AA1` at `0.01`, and
`K AA1 T` at `0.99` (the dominant, merged form). `K AO1 T` is quoted above
because it is the entry that actually contains `AO`; the dominant entry
for the word does not. Worth knowing generally: `AA` and `AO` are not
cleanly separated by this lexicon.

### Consonants (24)

| `ARPABET` | `IPA` | Example | Dictionary entry |
|---|---|---|---|
| `B` | `b` | be | `B IY0` |
| `CH` | `tʃ` | cheese | `CH IY1 Z` |
| `D` | `d` | dee | `D IY1` |
| `DH` | `ð` | thee | `DH IY1` |
| `F` | `f` | fee | `F IY1` |
| `G` | `ɡ` | green | `G R IY1 N` |
| `HH` | `h` | he | `HH IY1` |
| `JH` | `dʒ` | gee | `JH IY1` |
| `K` | `k` | key | `K IY1` |
| `L` | `l` | lee | `L IY1` |
| `M` | `m` | me | `M IY1` |
| `N` | `n` | knee | `N IY1` |
| `NG` | `ŋ` | ping | `P IH1 NG` |
| `P` | `p` | pee | `P IY1` |
| `R` | `ɹ` | read | `R IY1 D` |
| `S` | `s` | sea | `S IY1` |
| `SH` | `ʃ` | she | `SH IY1` |
| `T` | `t` | tea | `T IY1` |
| `TH` | `θ` | theta | `TH EY1 T AH0` |
| `V` | `v` | vee | `V IY1` |
| `W` | `w` | we | `W IY1` |
| `Y` | `j` | yield | `Y IY1 L D` |
| `Z` | `z` | zee | `Z IY1` |
| `ZH` | `ʒ` | seizure | `S IY1 ZH ER0` |

*read*'s dictionary entry is itself a small illustration of the homograph
problem discussed in [`g2p_oov`](g2p_oov.md) (the same *read*
present/past example MFA's own G2P documentation uses):
`english_us_arpa.dict` carries `R IY1 D` (present tense, weight `0.87`)
alongside `R EH1 D` (past tense, weight `0.99`, the dominant entry) and two
lower-weight partial forms. `R IY1 D` is quoted above because it is the
entry that illustrates plain `R` → `ɹ`, not because it is dominant — a
reader who needs "the" pronunciation of *read* should not assume the
present tense without sentence context, which is exactly the ambiguity a
spelling-only lookup (dictionary **or** G2P) cannot resolve on its own.

Three conventions worth naming, because they surprise people:

- `R` → `ɹ` (a **lowercase turned r**, the alveolar approximant), not
  `r` (the alveolar trill). English `r` is not a trill.
- `G` → `ɡ` is the IPA **script g** (U+0261), not the ASCII letter `g`
  (U+0067). String comparisons against a hand-typed `"g"` will fail.
- `IY` → `i` and `UW` → `u` are bare, without the length mark `ː` some
  transcription traditions add.

### What is *not* in the table

`spn` — MFA's unknown-word phone — and any code outside the 39 above map
to the **empty string**, never to a guess. A wrong IPA symbol entering
the data silently is worse than a blank, so `arpabet_ipa.to_ipa` refuses
to approximate. `spn` rows therefore have empty `IPA`, empty `Stress`,
and `Pron_Source = 'oov_unresolved'`; see [`phoneme`](phoneme.md).

## Implementation in SensiExtractor
Source: `extractors/arpabet_ipa.py` — pure lookup, no I/O, no pandas, no
third-party dependency.

```python
ARPABET_TO_IPA: dict[str, str]                       # 39 bare codes
def to_ipa(arpabet: str, stress: int | None) -> str  # one segment
def syllable_to_ipa(pairs) -> str                    # dash-joined
```

`ARPABET_TO_IPA` lists `AH` and `ER` at their **stressed** values; a
small `_UNSTRESSED` override dict is consulted first when `stress == 0`.
Keeping the stress-dependent rows in a two-entry override rather than
expanding the table to 45 stress-qualified keys means the table stays a
readable 1:1 phone inventory, and the two exceptions are visible instead
of buried.

Three robustness choices, each there to prevent a silent wrong value:

- **A fused label is tolerated.** `to_ipa("EY2", None)` works: the
  trailing digit is stripped, and the recovered digit is used as the
  stress, so `to_ipa("AH0", None)` correctly returns `ə`. This matters
  because some callers still hold MFA's raw label.
- **A stress-dependent vowel with `stress=None` returns the stressed
  form**, never `""`. ⚠️ This is a deliberate choice of the less-wrong
  default rather than a phonetically motivated one: `AH` with unknown
  stress is more often `ʌ` than `ə` in a *stressed-position* lookup, and
  in the pipeline the case never arises (`Stress` is always known when
  `IPA` is computed), so it is defensive only.
- **Unknown codes return `""`.** No fallback, no approximation.

`syllable_to_ipa` **drops** unmappable segments rather than emitting
empty fields, so a syllable containing noise yields `d-i`, not `d--i`.

`extractors/syllabify.py:syllabify_pairs` is the only production caller.
It splits MFA's label once per phone with `_split_label` and passes the
`(code, stress)` pair to `to_ipa` for the phoneme row and to
`syllable_to_ipa` for the syllable row — so the two CSVs can never
disagree.

### Encoding

The `.mat` and `.csv` writers emit UTF-8, and `pandas.read_csv` /
`scipy.io.loadmat` round-trip these symbols unchanged. ⚠️ On Windows,
`print()`ing an IPA character to a cp1252 console raises
`UnicodeEncodeError`; if you need to inspect the column from a terminal,
set `PYTHONIOENCODING=utf-8` or read the CSV in a notebook. No pipeline
code prints IPA, for exactly this reason.

## Library
None. Deliberately: a G2P/IPA library (e.g. `phonemizer`, `eng-to-ipa`)
would add a dependency, and in some cases a second pronunciation model,
to do a 39-row dictionary lookup. Since the input is already ARPABET from
MFA, there is nothing to *model* — only to rename.

## Papers
No research paper underpins a notation table; the sources below are the
reference documentation the table was built and checked against.

- <a name="cmudict"></a>**[CMUdict]** *The CMU Pronouncing Dictionary.* Carnegie Mellon University Speech Group. [speech.cs.cmu.edu/cgi-bin/cmudict](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) · [`cmudict.phones`](https://github.com/cmusphinx/cmudict/blob/master/cmudict.phones) (the 39-phone inventory with vowel/consonant classes) · [`cmudict.symbols`](https://github.com/cmusphinx/cmudict/blob/master/cmudict.symbols) (the same inventory with `0`/`1`/`2` appearing on **vowels only**). Source of the ARPABET side of the table and of the stress-digit convention. ⚠️ CMU's own documentation gives example words, not IPA — the IPA column here is the standard correspondence for those example words, cross-checked row by row against [ARPABET] and against the dictionary entries quoted above.
- <a name="arpabet"></a>**[ARPABET]** *ARPABET.* Wikipedia. [en.wikipedia.org/wiki/ARPABET](https://en.wikipedia.org/wiki/ARPABET) — secondary source carrying the ARPABET↔IPA table in the form used here; verified to agree with `extractors/arpabet_ipa.py` on all 39 phones, including `R`→`ɹ` and `G`→`ɡ`, and on the `0`/`1`/`2` stress convention.
- <a name="r-coloured-vowel"></a>**[R-coloured vowel]** *R-colored vowel.* Wikipedia. [en.wikipedia.org/wiki/R-colored_vowel](https://en.wikipedia.org/wiki/R-colored_vowel) — source of the quoted ⟨ɝ⟩/⟨ɚ⟩ stressed-vs-unstressed convention.

[CMUdict]: #cmudict
[ARPABET]: #arpabet
[R-coloured vowel]: #r-coloured-vowel

## Datasets
<!-- auto-populated from docs/datasets/*.md once the dataset registry exists -->
