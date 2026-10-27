"""The User Manual builder (scripts/build_manual.py): a full build from the
committed example outputs passes every check, and the guards that keep it
honest (named columns must exist; not-computed features show no figure) hold."""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from manual import generated as G  # noqa: E402
from manual.site import OUTPUTS_DIR, build_site  # noqa: E402


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("manual")
    report = build_site(out)
    return out, report


def test_full_build_passes_every_check(built):
    out, report = built
    assert report.errors == []
    assert (out / "index.html").exists()
    assert report.pages >= 55


def test_every_page_uses_relative_forward_slash_links(built):
    out, _ = built
    for page in out.rglob("*.html"):
        for ref in re.findall(r'(?:href|src)="([^"]+)"', page.read_text(encoding="utf-8")):
            if ref.startswith(("http", "#", "mailto:")):
                continue
            assert "\\" not in ref and not ref.startswith("/"), (page.name, ref)


def test_search_index_covers_columns_and_feature_keys(built):
    out, _ = built
    js = (out / "assets" / "search-index.js").read_text(encoding="utf-8")
    index = json.loads(js[js.index("["):js.rindex("]") + 1])
    keys = " ".join(e["k"] for e in index)
    assert "gpt2_Surprisal" in keys and "acoustic_envelope" in keys
    assert all(e["u"] and not e["u"].startswith("/") for e in index)


def test_no_page_fetches_local_files(built):
    out, _ = built
    js = (out / "assets" / "manual.js").read_text(encoding="utf-8")
    assert "fetch(" not in js and "XMLHttpRequest" not in js      # must work from file://


def test_a_named_column_missing_from_the_output_is_an_error():
    entry = {"key": "freq_subtlex", "output": "word/{stem}_features.csv",
             "show": ["token", "Not_A_Column"]}
    errs = G.check_names(entry, OUTPUTS_DIR)
    assert errs and "freq_subtlex" in errs[0] and "Not_A_Column" in errs[0]


def test_not_computed_feature_shows_its_reason_and_no_figure(tmp_path):
    entry = {"key": "edu_zipf", "not_computed": "No lookup table exists.", "show": ["Edu_Zipf_K2"]}
    html = G.see_it(entry, "../../", tmp_path, OUTPUTS_DIR, ROOT / "dev" / "manual" / "examples")
    assert "No lookup table exists." in html and "<img" not in html
    assert "Edu_Zipf_K2" in html
