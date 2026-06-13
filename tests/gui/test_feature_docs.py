"""Tests for gui.feature_docs loader and the info-icon wiring."""
from __future__ import annotations

from gui import feature_docs


def test_has_feature_doc_finds_shallow_example():
    assert feature_docs.has_feature_doc("feat_struct_sentence_id")


def test_has_feature_doc_finds_deep_example():
    assert feature_docs.has_feature_doc("master_bert_word")


def test_has_feature_doc_returns_false_for_unknown_key():
    assert not feature_docs.has_feature_doc("definitely_not_a_real_toggle_key")


def test_load_feature_doc_parses_frontmatter_and_body():
    doc = feature_docs.load_feature_doc("master_bert_word")
    assert doc is not None
    assert doc.frontmatter["key"] == "master_bert_word"
    assert doc.frontmatter["name"] == "BERT contextual word embeddings"
    assert "transformers" in doc.frontmatter["libraries"]
    # Body should not contain the frontmatter delimiters
    assert not doc.body.startswith("---")
    assert "EEG rationale" in doc.body
    # Implementation section is the whole reason this field exists
    assert "Implementation in SensiExtractor" in doc.body


def test_load_feature_doc_returns_none_for_unknown_key():
    assert feature_docs.load_feature_doc("definitely_not_a_real_toggle_key") is None


def test_load_shallow_doc_has_stub_depth():
    doc = feature_docs.load_feature_doc("feat_struct_sentence_id")
    assert doc is not None
    assert doc.frontmatter["depth"] == "stub"


def test_toggle_row_renders_info_icon_when_doc_exists(qtbot):
    from gui.widgets import ToggleRow
    from gui.info_popup import InfoIcon
    row = ToggleRow(label="Sentence ID", info_key="feat_struct_sentence_id")
    qtbot.addWidget(row)
    icons = row.findChildren(InfoIcon)
    assert len(icons) == 1


def test_toggle_row_no_info_icon_when_doc_missing(qtbot):
    from gui.widgets import ToggleRow
    from gui.info_popup import InfoIcon
    row = ToggleRow(label="Bogus", info_key="definitely_not_a_real_toggle_key")
    qtbot.addWidget(row)
    icons = row.findChildren(InfoIcon)
    assert len(icons) == 0


def test_toggle_row_no_info_icon_when_info_key_none(qtbot):
    """Cluster master toggles pass info_key=None and must not render an icon
    even if a doc happens to exist for some unrelated key."""
    from gui.widgets import ToggleRow
    from gui.info_popup import InfoIcon
    row = ToggleRow(label="Enable all structural columns")
    qtbot.addWidget(row)
    icons = row.findChildren(InfoIcon)
    assert len(icons) == 0


def test_feature_info_dialog_renders_doc_for_known_key(qtbot):
    from gui.info_popup import FeatureInfoDialog
    dlg = FeatureInfoDialog("master_bert_word")
    qtbot.addWidget(dlg)
    text = dlg.browser.toMarkdown()
    assert "BERT" in text or "bert" in text.lower()
    # Title reflects the human-readable name from frontmatter, not the raw key
    assert "BERT contextual word embeddings" in dlg.windowTitle()


def test_feature_info_dialog_handles_missing_doc(qtbot):
    from gui.info_popup import FeatureInfoDialog
    dlg = FeatureInfoDialog("definitely_not_a_real_toggle_key")
    qtbot.addWidget(dlg)
    # Falls back to the raw key in the title and an explanatory message
    text = dlg.browser.toPlainText()
    assert "No documentation file found" in text


def test_shared_doc_map_resolves_multiple_toggle_keys_to_same_file():
    """All five structural ID toggles share structural_ids.md — picked one
    pair to confirm the mapping is wired both directions."""
    doc_a = feature_docs.load_feature_doc("feat_struct_doc_id")
    doc_b = feature_docs.load_feature_doc("feat_struct_sentence_id")
    doc_c = feature_docs.load_feature_doc("feat_struct_token_id")
    assert doc_a is not None and doc_b is not None and doc_c is not None
    # Same underlying file → identical body content
    assert doc_a.body == doc_b.body == doc_c.body
    # Frontmatter `key` reflects the shared doc, not the toggle key
    assert doc_a.frontmatter["key"] == "structural_ids"


def test_offsets_doc_resolves_for_both_offset_toggles():
    a = feature_docs.load_feature_doc("feat_struct_char_start")
    b = feature_docs.load_feature_doc("feat_struct_char_end")
    assert a is not None and b is not None
    assert a.body == b.body


def test_load_doc_file_by_filename():
    doc = feature_docs.load_doc_file("prosody_pitch.md")
    assert doc is not None
    assert doc.key == "prosody_pitch"
    # works without the .md suffix too
    assert feature_docs.load_doc_file("prosody_pitch") is not None


def test_load_doc_file_unknown_returns_none():
    assert feature_docs.load_doc_file("not_a_real_doc.md") is None


def test_resolve_doc_link_variants():
    # existing doc, with and without fragment / directory parts
    assert feature_docs.resolve_doc_link("prosody_pitch.md") == "prosody_pitch.md"
    assert feature_docs.resolve_doc_link("prosody_pitch.md#meaning") == "prosody_pitch.md"
    assert feature_docs.resolve_doc_link("../features/phoneme.md") == "phoneme.md"
    # not feature docs
    assert feature_docs.resolve_doc_link("https://example.com") is None
    assert feature_docs.resolve_doc_link("../../gui/widgets.py") is None
    assert feature_docs.resolve_doc_link("#meaning") is None
    assert feature_docs.resolve_doc_link("definitely_missing.md") is None


def test_dialog_md_link_navigates_in_place(qtbot):
    from gui.info_popup import FeatureInfoDialog
    from PySide6.QtCore import QUrl
    dlg = FeatureInfoDialog("feat_prosody_pitch_mean")   # → prosody_pitch.md
    qtbot.addWidget(dlg)
    # isHidden() reflects the explicit flag even though the dialog isn't shown.
    assert dlg._back_btn.isHidden()        # no history yet

    dlg._on_anchor_clicked(QUrl("prosody_volume.md"))
    assert dlg._doc.key == "prosody_volume"   # navigated to the linked doc
    assert not dlg._back_btn.isHidden()       # back becomes available

    dlg._go_back()
    assert dlg._doc.key == "feat_prosody_pitch_mean"   # original toggle's doc
    assert dlg._back_btn.isHidden()


def test_dialog_external_link_opens_in_system_browser(qtbot, monkeypatch):
    from gui.info_popup import FeatureInfoDialog
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QDesktopServices
    dlg = FeatureInfoDialog("feat_prosody_pitch_mean")
    qtbot.addWidget(dlg)

    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl",
                        lambda url: opened.append(url.toString()) or True)
    dlg._on_anchor_clicked(QUrl("https://example.com/page"))
    assert opened == ["https://example.com/page"]
    # external click must NOT replace the current doc
    assert dlg._doc.key == "feat_prosody_pitch_mean"


def test_dialog_unresolvable_link_does_not_blank(qtbot):
    from gui.info_popup import FeatureInfoDialog
    from PySide6.QtCore import QUrl
    dlg = FeatureInfoDialog("feat_prosody_pitch_mean")
    qtbot.addWidget(dlg)
    before = dlg.browser.toMarkdown()
    dlg._on_anchor_clicked(QUrl("../../gui/widgets.py"))   # source-file link
    assert dlg.browser.toMarkdown() == before               # unchanged, not blank


def test_info_icon_opens_dialog_on_click(qtbot, monkeypatch):
    from gui.info_popup import InfoIcon, FeatureInfoDialog
    icon = InfoIcon("master_bert_word")
    qtbot.addWidget(icon)
    opened = []

    def fake_exec(self):
        opened.append(self._toggle_key)
        return 0

    monkeypatch.setattr(FeatureInfoDialog, "exec", fake_exec)
    icon.click()
    assert opened == ["master_bert_word"]


def test_acoustic_ts_toggles_route_to_docs():
    from gui.feature_docs import _doc_path
    assert _doc_path("feat_acoustic_ts_envelope").name == "acoustic_envelope.md"
    assert _doc_path("feat_acoustic_ts_onset_env").name == "acoustic_onset_envelope.md"
    assert _doc_path("feat_acoustic_ts_logmel").name == "acoustic_logmel.md"
    assert _doc_path("feat_acoustic_ts_mfcc").name == "acoustic_mfcc.md"
