"""Every feature page uses the section-switch card (Task 5 of the output
switches batch): no "Enable all …"/"Emit …" master rows, section-only settings
grey with their card, shared settings and credentials stay editable."""

import pytest
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QAbstractButton, QLabel

from gui.section_acoustic_features import AcousticFeaturesSection
from gui.section_doc_features import DocFeaturesSection
from gui.section_sentence_features import SentenceFeaturesSection
from gui.section_subword_features import SubwordFeaturesSection
from gui.section_word_features import WordFeaturesSection
from gui.widgets import Card, ClusterCard, SectionSwitch

PAGES = (WordFeaturesSection, DocFeaturesSection, SubwordFeaturesSection,
         SentenceFeaturesSection, AcousticFeaturesSection)


def _card_titled(page, title: str):
    for card in page.findChildren(Card):
        lbl = getattr(card, "title_label", None) or card.header.title_label
        if lbl.text() == title:
            return card
    raise AssertionError(f"no card titled {title!r}")


@pytest.mark.parametrize("page_cls", PAGES)
def test_no_enable_all_wording(qtbot, page_cls):
    page = page_cls()
    qtbot.addWidget(page)
    texts = [w.text() for w in page.findChildren(QLabel)]
    texts += [w.text() for w in page.findChildren(QAbstractButton)]
    for t in texts:
        assert "Enable all" not in t, t
        assert "Emit " not in t, t


def test_predictability_models_greyed_when_left_out(qtbot, monkeypatch):
    from extractors import predictability
    monkeypatch.setattr(predictability, "has_hf_token", lambda: False)
    page = WordFeaturesSection()
    qtbot.addWidget(page)
    card = _card_titled(page, "Predictability")
    card.master_toggle().set_checked(False)

    from utils.preferences import load_preferences
    assert load_preferences()["features"]["master_predictability"] is False
    for cb in page.model_checkboxes.values():
        assert not cb.isEnabled()
    for spin in page.context_spins.values():
        assert not spin.isEnabled()
    for cb in page.metric_checkboxes.values():
        assert not cb.isEnabled()
    assert page.hf_token_line.isEnabled()
    assert page.hf_token_save_btn.isEnabled()
    assert card.tag_text() == "Left out of outputs"


def test_pitch_settings_stay_editable(qtbot):
    page = WordFeaturesSection()
    qtbot.addWidget(page)
    card = _card_titled(page, "Prosody")
    card.master_toggle().set_checked(False)
    assert not page.toggle("feat_prosody_pitch_mean").check.isEnabled()
    assert page.pitch_floor_spin.isEnabled()
    assert page.pitch_ceiling_spin.isEnabled()
    assert page.pitch_units_combo.isEnabled()


def test_g2p_card_has_no_switch(qtbot):
    page = SubwordFeaturesSection()
    qtbot.addWidget(page)
    card = _card_titled(page, "Out-of-dictionary words")
    assert not isinstance(card, ClusterCard)
    assert card.findChildren(SectionSwitch) == []


def test_acoustic_card_switch_is_level_acoustic(qtbot):
    from utils.preferences import load_preferences
    page = AcousticFeaturesSection()
    qtbot.addWidget(page)
    card = _card_titled(page, "Acoustic features (audio-only)")
    switch = card.master_toggle()
    start = load_preferences()["output"]["level_acoustic"]
    switch.set_checked(not start)
    assert load_preferences()["output"]["level_acoustic"] is (not start)
    switch.set_checked(start)
    assert load_preferences()["output"]["level_acoustic"] is start


def test_embedding_ticks_write_feature_keys_only(qtbot, monkeypatch):
    import gui.section_sentence_features as sent_mod
    import gui.section_word_features as word_mod
    from utils.preferences import load_preferences, save_preferences
    word = WordFeaturesSection()
    qtbot.addWidget(word)
    sent = SentenceFeaturesSection()
    qtbot.addWidget(sent)
    payloads: list[dict] = []

    def spy(update):
        payloads.append(update)
        save_preferences(update)

    monkeypatch.setattr(word_mod, "save_preferences", spy)
    monkeypatch.setattr(sent_mod, "save_preferences", spy)
    for page, key in ((word, "master_word2vec"), (word, "master_bert_word"),
                      (sent, "master_bert_sentence")):
        payloads.clear()
        row = page._toggles[key]
        row.set_checked(not row.is_checked())
        assert payloads == [{"features": {key: row.is_checked()}}], key
        assert load_preferences()["features"][key] is row.is_checked()


def test_semantic_categories_stay_editable(qtbot):
    from PySide6.QtWidgets import QRadioButton
    page = WordFeaturesSection()
    qtbot.addWidget(page)
    card = _card_titled(page, "Semantic indicators")
    card.master_toggle().set_checked(False)
    from utils.preferences import load_preferences
    assert load_preferences()["features"]["master_semantic_indicators"] is False
    assert page.configure_cats_button.isEnabled()
    radios = card.findChildren(QRadioButton)
    assert radios
    for rb in radios:
        if rb.text() == "LIWC":
            assert not rb.isEnabled()   # permanently disabled (license)
        else:
            assert rb.isEnabled(), rb.text()
    assert not page.toggle("sem_indicators").check.isEnabled()


def test_phoneme_surprisal_mat_tick(qtbot):
    from utils.preferences import load_preferences
    page = SubwordFeaturesSection()
    qtbot.addWidget(page)
    row = page.toggle("feat_phoneme_surprisal_mat")
    assert "_phoneme_surprisal.mat" in row.label.text()
    from gui.info_popup import InfoIcon
    assert row.findChildren(InfoIcon), "the .mat tick should have an info icon"
    row.set_checked(True)
    assert load_preferences()["features"]["feat_phoneme_surprisal_mat"] is True
    card = _card_titled(page, "Phoneme surprisal")
    assert not any("Run page" in lbl.text() for lbl in card.findChildren(QLabel))
    card.master_toggle().set_checked(False)
    assert not row.check.isEnabled()
    assert not page._psr_sr_spin.isEnabled()


@pytest.mark.parametrize("page_cls,title,key,section", [
    (WordFeaturesSection, "Semantic indicators", "master_semantic_indicators", "features"),
    (WordFeaturesSection, "Semantic similarity", "master_semantic_similarity", "features"),
    (DocFeaturesSection, "Aggregates", "master_doc_aggregates", "output"),
    (DocFeaturesSection, "Lexical diversity", "master_doc_lex_diversity", "output"),
    (DocFeaturesSection, "Readability", "master_doc_readability", "output"),
    (DocFeaturesSection, "Category proportions", "master_doc_category_proportions", "output"),
    (SubwordFeaturesSection, "P-center", "master_syllable_pcenter", "features"),
    (SentenceFeaturesSection, "Semantic similarity", "master_sentence_similarity", "features"),
])
def test_new_section_switches_write_their_key(qtbot, page_cls, title, key, section):
    from utils.preferences import load_preferences
    page = page_cls()
    qtbot.addWidget(page)
    card = _card_titled(page, title)
    assert isinstance(card, ClusterCard)
    card.master_toggle().set_checked(False)
    assert load_preferences()[section][key] is False
    card.master_toggle().set_checked(True)
    assert load_preferences()[section][key] is True


def test_radio_backend_does_not_count_as_a_selected_feature(qtbot):
    from utils.preferences import save_preferences
    save_preferences({"features": {"sem_indicators": False,
                                   "master_semantic_indicators": True}})
    page = WordFeaturesSection()
    qtbot.addWidget(page)
    card = _card_titled(page, "Semantic indicators")
    assert card.tag_text() == "Included — nothing selected"
    page.toggle("sem_indicators").set_checked(True)
    assert card.tag_text() == "Included"
