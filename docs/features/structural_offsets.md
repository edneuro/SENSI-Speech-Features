---
key: structural_offsets
name: Structural character offsets
levels: [word]
gui_toggles: [feat_struct_char_start, feat_struct_char_end]
output_columns: [start, end]
libraries: [spaCy]
status: implemented
depth: stub
---

## Meaning
1-indexed, inclusive character offsets of each token within the
**sentence string** (the `sentence` column).

| Column | Meaning |
|---|---|
| `start` | Position of the token's first character within its sentence. First character of the sentence is `1`. |
| `end`   | Position of the token's last character within its sentence. Inclusive. |

> ⚠️ Easy to confuse with `Onset` / `End` (capitalized), which are the
> acoustic word boundaries in **seconds** from `extractors/aligner.py`.
> These character offsets are lowercase `start` / `end` and are in
> **characters within the sentence**, not seconds.

## Usage
- Slice the `sentence` string with `sentence[start-1:end]` to get the
  token's surface form (matching the `token` column).
- Useful for joining downstream annotations that are also indexed in
  sentence-relative character coordinates.

## Implementation in SensiExtractor
Source: `extractors/text_parser.py:parse`.

Computed as `token.idx - sent.start_char + 1` and
`char_start + len(token.text) - 1` during the spaCy sentence walk. The
`+1` converts spaCy's native 0-indexed offsets into the 1-indexed
inclusive convention used throughout the SensiExtractor outputs.
