"""Post-change validation against the pre-v20 baseline.

Run AFTER re-running the pipeline over validation/data/. Compares the new
validation/output/ against validation/output_baseline_pre_v20/ and asserts the
spec's expected deltas -- and, crucially, that nothing else moved.

P_Center gets two checks, not one, because a naive bit-identical comparison
against the regenerated pipeline output is unachievable by construction (see
docs in the two check functions below):

  (a) _check_pcenter_code_isolation: baseline TextGrid (the OLD alignment) +
      the NEW syllabify_pairs/add_pcenter code -> compared bit-exactly to the
      baseline syllable CSV. Same alignment in, so any mismatch here is a
      genuine code regression in the pcenter rewire.
  (b) _check_pcenter_pipeline_tolerance: the regenerated (NEW alignment, NEW
      code) syllable CSV vs baseline, joined on a renumbering-immune key
      (token_in_sentence_id is legitimately renumbered by the SPACE-row
      removal in this same batch), with a small tolerance for MFA re-align
      jitter (SIDECAR_VERSION bump forces exactly one full re-align, and MFA
      is not bit-reproducible at frame granularity -- 10ms frame shift).

Usage:  python validation/verify_phoneme_batch.py
Exit 0 = all checks passed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
_REPO_ROOT = ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

NEW = ROOT / "output"
OLD = ROOT / "output_baseline_pre_v20"
OLD_MFA_DIR = OLD / "Montreal Forced Alignment"

# Measured on the baseline; see spec section 10.
EXPECTED_PHONEME_DELTA = {"3_M": -6, "LJ003-0007": -5}
EXPECTED_SYLLABLE_DELTA = {"3_M": -2, "LJ003-0007": -2}

# Frame-level jitter tolerance for MFA-derived timing columns after a forced
# full re-align (SIDECAR_VERSION bump). Kaldi's default frame shift is 10ms;
# empirically observed max delta across all 10 stimuli is 0.02s at the
# phoneme level. 0.03s gives headroom without hiding a real regression.
MFA_JITTER_TOL_S = 0.03

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    cond = bool(cond)
    if not cond:
        failures.append(msg)
    else:
        print(f"  OK: {msg}")


def _stems() -> list[str]:
    return sorted(p.name.replace("_features.csv", "")
                  for p in (OLD / "word").glob("*_features.csv"))


def _renumber_map(old_word_df: pd.DataFrame) -> dict[tuple, int]:
    """(doc_id, sentence_id, old token_in_sentence_id) -> new token_in_sentence_id.

    The word/syllable/phoneme parser change in this batch drops SPACE rows
    from the word CSV, which renumbers every later token_in_sentence_id in
    that sentence. Reproduce that renumbering here (drop SPACE, reindex 1..N
    per (doc_id, sentence_id) in original order) so comparisons can join old
    and new rows correctly instead of accidentally pairing different words.
    """
    df = old_word_df.loc[old_word_df["upos"] != "SPACE"].copy()
    df["new_tisid"] = df.groupby(["doc_id", "sentence_id"]).cumcount() + 1
    return {
        (doc, sent, old_id): new_id
        for doc, sent, old_id, new_id in zip(
            df["doc_id"], df["sentence_id"], df["token_in_sentence_id"], df["new_tisid"])
    }


def _check_pcenter_code_isolation(stem: str) -> None:
    """(a) Strict: baseline TextGrid + NEW code, compared bit-exactly.

    Isolates the code change from MFA's own re-alignment jitter: this feeds
    the OLD alignment (never re-run) through the NEW syllabify_pairs/add_pcenter,
    so the join key (old token_in_sentence_id, unrenumbered) matches the
    baseline exactly and any P_Center difference is a real regression in the
    pcenter rewire from the fused Label to the Stress column.
    """
    tg_path = OLD_MFA_DIR / f"{stem}.TextGrid"
    if not tg_path.exists():
        check(False, f"{stem}: baseline TextGrid missing at {tg_path}")
        return
    from utils.textgrid_parser import parse_phones
    from extractors.syllabify import syllabify_pairs
    from extractors.pcenter import add_pcenter

    phone_df = parse_phones(str(tg_path)).rename(
        columns={"phone": "phone", "onset": "onset", "offset": "offset"})
    old_w = pd.read_csv(OLD / "word" / f"{stem}_features.csv")
    word_df = old_w.loc[old_w["upos"] != "SPACE",
                         ["doc_id", "sentence_id", "token_in_sentence_id",
                          "token", "Onset", "End"]]

    syll_df, phon_df = syllabify_pairs(word_df, phone_df)
    syll_df = add_pcenter(syll_df, phon_df, k=0.3)

    old_s = pd.read_csv(OLD / "syllable" / f"{stem}_syllables.csv")
    key = ["doc_id", "sentence_id", "token_in_sentence_id", "syllable_id"]
    merged = syll_df.merge(old_s[key + ["P_Center"]], on=key,
                            how="inner", suffixes=("_isolated", "_baseline"))
    # 3_M / LJ003-0007 legitimately dedup 2 syllables each even under the OLD
    # alignment (the duplicated-contraction fix is alignment-independent), so
    # the isolated run's row count is intentionally lower for those two.
    expected_rows = len(old_s) + EXPECTED_SYLLABLE_DELTA.get(stem, 0)
    check(len(merged) == expected_rows,
          f"{stem}: code-isolation join produced {len(merged)} rows, "
          f"expected {expected_rows} ({len(old_s)} baseline + "
          f"{EXPECTED_SYLLABLE_DELTA.get(stem, 0)} expected dedup delta)")
    bad = merged[merged["P_Center_isolated"] != merged["P_Center_baseline"]]
    check(bad.empty,
          f"{stem}: code-isolation P_Center bit-identical "
          f"({len(merged)} syllables, same baseline alignment, new pcenter code)")


def _check_pcenter_pipeline_tolerance(stem: str, remap: dict[tuple, int]) -> None:
    """(b) Loose: full pipeline (NEW alignment + NEW code) vs baseline.

    Renumbering-immune key (see _renumber_map) + a small tolerance for MFA
    re-align jitter. Prints the worst delta per stimulus so a future, larger
    regression is still visible even though exact equality is not asserted.
    """
    old_s = pd.read_csv(OLD / "syllable" / f"{stem}_syllables.csv").copy()
    new_s = pd.read_csv(NEW / "syllable" / f"{stem}_syllables.csv")
    old_s["token_in_sentence_id"] = [
        remap.get((doc, sent, tid))
        for doc, sent, tid in zip(old_s["doc_id"], old_s["sentence_id"],
                                   old_s["token_in_sentence_id"])
    ]
    key = ["doc_id", "sentence_id", "token_in_sentence_id", "syllable_id"]
    merged = new_s.merge(old_s[key + ["P_Center"]], on=key, how="inner",
                          suffixes=("_new", "_baseline"))
    check(len(merged) > 0, f"{stem}: pipeline P_Center comparison found no matching rows")
    if len(merged) == 0:
        return
    delta = (merged["P_Center_new"] - merged["P_Center_baseline"]).abs()
    worst = round(float(delta.max()), 6)  # avoid float-subtraction noise at the 0.03s boundary
    n_diff = int((delta > 1e-9).sum())
    print(f"  INFO: {stem}: P_Center pipeline delta -- {n_diff}/{len(merged)} "
          f"syllables differ, worst |delta|={worst:.4f}s")
    check(worst <= MFA_JITTER_TOL_S,
          f"{stem}: P_Center pipeline delta {worst:.4f}s exceeds "
          f"{MFA_JITTER_TOL_S}s tolerance (MFA re-align jitter budget)")


def _check_word_timing_tolerance(stem: str) -> None:
    """Onset/End/Word_Duration are MFA-derived and will show frame-level
    jitter after the SIDECAR_VERSION-forced full re-align; use a tolerance,
    not exact equality. Zipf_Freq is NOT alignment-derived and stays exact.
    """
    old_w = pd.read_csv(OLD / "word" / f"{stem}_features.csv")
    new_w = pd.read_csv(NEW / "word" / f"{stem}_features.csv")
    o = old_w.loc[old_w["upos"] != "SPACE"].reset_index(drop=True)
    n = new_w.reset_index(drop=True)
    if len(o) != len(n):
        check(False, f"{stem}: word row count mismatch after dropping SPACE "
                      f"({len(o)} vs {len(n)}); cannot compare columns positionally")
        return
    for col in ("Onset", "End", "Word_Duration"):
        if col in o.columns and col in n.columns:
            delta = (o[col] - n[col]).abs()
            worst = round(float(delta.max()), 6)  # avoid float-subtraction noise at the 0.03s boundary
            print(f"  INFO: {stem}: {col} pipeline delta -- worst |delta|={worst:.4f}s")
            check(worst <= MFA_JITTER_TOL_S,
                  f"{stem}: {col} delta {worst:.4f}s exceeds {MFA_JITTER_TOL_S}s "
                  f"tolerance (MFA re-align jitter budget)")
    if "Zipf_Freq" in o.columns and "Zipf_Freq" in n.columns:
        check(o["Zipf_Freq"].equals(n["Zipf_Freq"]),
              f"{stem}: Zipf_Freq changed in the word CSV (not alignment-derived; "
              f"must be exact)")


for stem in _stems():
    print(f"== {stem} ==")
    old_p = pd.read_csv(OLD / "phoneme" / f"{stem}_phonemes.csv")
    new_p = pd.read_csv(NEW / "phoneme" / f"{stem}_phonemes.csv")
    old_s = pd.read_csv(OLD / "syllable" / f"{stem}_syllables.csv")
    new_s = pd.read_csv(NEW / "syllable" / f"{stem}_syllables.csv")
    old_w_full = pd.read_csv(OLD / "word" / f"{stem}_features.csv")

    # 1a/1b. P_Center: strict code-isolation + loose pipeline tolerance.
    if "P_Center" in old_s.columns:
        check("P_Center" in new_s.columns, f"{stem}: P_Center column vanished")
        if "P_Center" in new_s.columns:
            _check_pcenter_code_isolation(stem)
            remap = _renumber_map(old_w_full)
            _check_pcenter_pipeline_tolerance(stem, remap)

    # 2. Row-count deltas are exactly the deduplicated copies.
    check(len(new_p) - len(old_p) == EXPECTED_PHONEME_DELTA.get(stem, 0),
          f"{stem}: phoneme rows {len(old_p)} -> {len(new_p)}, expected delta "
          f"{EXPECTED_PHONEME_DELTA.get(stem, 0)}")
    check(len(new_s) - len(old_s) == EXPECTED_SYLLABLE_DELTA.get(stem, 0),
          f"{stem}: syllable rows {len(old_s)} -> {len(new_s)}, expected delta "
          f"{EXPECTED_SYLLABLE_DELTA.get(stem, 0)}")

    # 3. No duplicate acoustic intervals remain.
    dup = int(new_p.duplicated(subset=["Onset", "End", "ARPABET"], keep="first").sum())
    check(dup == 0, f"{stem}: {dup} duplicate phoneme intervals remain")

    # 4. Stress is empty on consonants, populated on vowels.
    vowels = new_p["IPA"].isin(list("ɑæʌɔɛ iɪ"
                                    "uʊəɚɝ")) | \
        new_p["ARPABET"].isin(["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER",
                               "EY", "IH", "IY", "OW", "OY", "UH", "UW"])
    check(bool(new_p.loc[vowels, "Stress"].notna().all()),
          f"{stem}: some vowel rows have empty Stress")
    check(bool(new_p.loc[~vowels & (new_p["ARPABET"] != "spn"), "Stress"].isna().all()),
          f"{stem}: some consonant rows have a Stress value")

    # 4b. Pron_Source sanity: with g2p off, a corpus of ordinary English read
    #     speech should be entirely 'dictionary'. Any 'g2p' here means the
    #     provenance check is judging the wrong string (the classic failure is
    #     using the spaCy token, so every number reads as g2p).
    check(set(new_p["Pron_Source"]) <= {"dictionary", "oov_unresolved"},
          f"{stem}: unexpected Pron_Source values "
          f"{sorted(set(new_p['Pron_Source']))} with g2p off")

    # 5. IPA populated for every real phone; Label column gone.
    real = new_p["ARPABET"] != "spn"
    check(bool(new_p.loc[real, "IPA"].astype(str).str.len().gt(0).all()),
          f"{stem}: some non-spn rows have empty IPA")
    check("Label" not in new_p.columns, f"{stem}: Label column still present")

    # 6. Word CSV: no whitespace rows, contiguous ids, heads resolve.
    new_w = pd.read_csv(NEW / "word" / f"{stem}_features.csv")
    check(bool((new_w["upos"] != "SPACE").all()), f"{stem}: SPACE rows still emitted")
    check(list(new_w["token_id"]) == list(range(1, len(new_w) + 1)),
          f"{stem}: token_id is not contiguous")
    check(set(new_w["head_token_id"]) <= set(new_w["token_id"]),
          f"{stem}: head_token_id points outside token_id")
    check(bool(new_w["paragraph_id"].notna().all()),
          f"{stem}: paragraph_id has gaps")

    # 7. Onset/End/Word_Duration: tolerant (MFA re-align jitter). Zipf_Freq:
    #    exact (not alignment-derived).
    _check_word_timing_tolerance(stem)

print(f"\nchecked {len(_stems())} stimuli")
if failures:
    print(f"\n{len(failures)} FAILURES:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("all checks passed")
