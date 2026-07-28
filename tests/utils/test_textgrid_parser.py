"""TextGrid parsing (utils/textgrid_parser.py)."""
import textwrap

from utils.textgrid_parser import parse_phones, parse_words


def _write_textgrid(tmp_path, intervals_words, intervals_phones):
    """Minimal long-form TextGrid with a words tier and a phones tier."""
    def _tier(name, intervals):
        body = "".join(
            f'        intervals [{i}]:\n'
            f'            xmin = {s}\n'
            f'            xmax = {e}\n'
            f'            text = "{t}"\n'
            for i, (s, e, t) in enumerate(intervals, start=1)
        )
        return (
            f'    item [{{n}}]:\n'
            f'        class = "IntervalTier"\n'
            f'        name = "{name}"\n'
            f'        xmin = 0\n'
            f'        xmax = 1\n'
            f'        intervals: size = {len(intervals)}\n'
        ) + body

    header = textwrap.dedent('''\
        File type = "ooTextFile"
        Object class = "TextGrid"

        xmin = 0
        xmax = 1
        tiers? <exists>
        size = 2
        item []:
        ''')
    text = (header
            + _tier("words", intervals_words).format(n=1)
            + _tier("phones", intervals_phones).format(n=2))
    path = tmp_path / "oov.TextGrid"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_spn_is_kept_so_oov_words_are_not_silently_dropped(tmp_path):
    tg = _write_textgrid(
        tmp_path,
        [(0.0, 0.33, "zorblax"), (0.33, 0.5, "has")],
        [(0.0, 0.33, "spn"), (0.33, 0.5, "HH")],
    )
    phones = parse_phones(tg)
    assert list(phones["phone"]) == ["spn", "HH"]


def test_real_silence_labels_are_still_filtered(tmp_path):
    # Covers all four members of the silence set {'', 'SIL', 'sil', 'sp'}
    # (Task 2 rewrote it to include uppercase 'SIL'; nothing previously
    # exercised that spelling surviving the filter).
    tg = _write_textgrid(
        tmp_path,
        [(0.0, 0.6, "has")],
        [(0.0, 0.1, "sil"), (0.1, 0.2, "sp"), (0.2, 0.3, ""),
         (0.3, 0.4, "SIL"), (0.4, 0.6, "HH")],
    )
    phones = parse_phones(tg)
    assert list(phones["phone"]) == ["HH"]


def test_words_tier_unchanged(tmp_path):
    tg = _write_textgrid(
        tmp_path,
        [(0.0, 0.33, "zorblax"), (0.33, 0.5, "has")],
        [(0.0, 0.33, "spn"), (0.33, 0.5, "HH")],
    )
    words = parse_words(tg)
    assert list(words["word"]) == ["zorblax", "has"]
