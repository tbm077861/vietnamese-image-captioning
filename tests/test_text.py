import unicodedata

import pytest

from vic.text import CaptionTokenizer, Vocabulary, detokenize, normalize_caption, to_syllables
from vic.text.vocab import SPECIAL_TOKENS


def test_normalize_composes_unicode_and_lowercases():
    decomposed = unicodedata.normalize("NFD", "Cầu Thủ")
    assert decomposed != "Cầu Thủ"
    assert normalize_caption(decomposed) == "cầu thủ"


def test_normalize_strips_punctuation_and_whitespace():
    assert (
        normalize_caption("  Hai   người, đang chơi\tbóng đá!  ") == "hai người đang chơi bóng đá"
    )
    assert normalize_caption("Xin chào.", strip_punctuation=False) == "xin chào."


def test_normalize_unifies_tone_placement():
    assert normalize_caption("hoà bình, thuỷ thủ, khoẻ") == "hòa bình thủy thủ khỏe"
    # The mark stays where it is when the rhyme continues or after "qu".
    assert normalize_caption("hoàn toàn quý thuyền") == "hoàn toàn quý thuyền"


def test_normalize_fixes_lookalike_characters_and_underscores():
    assert normalize_caption("Ðường phố") == "đường phố"
    assert normalize_caption("sân_bóng") == "sân bóng"


def test_syllable_tokenizer():
    tokenizer = CaptionTokenizer(level="syllable")
    assert tokenizer.tokenize("Một cậu bé đang ném bóng.") == [
        "một", "cậu", "bé", "đang", "ném", "bóng",
    ]  # fmt: skip
    assert tokenizer.tokenize("   ") == []


def test_word_tokenizer_joins_syllables_and_round_trips():
    pytest.importorskip("underthesea")
    tokenizer = CaptionTokenizer(level="word")
    text = "Hai cầu thủ bóng đá đang tranh bóng trên sân cỏ"
    tokens = tokenizer.tokenize(text)
    assert any("_" in token for token in tokens)
    assert len(tokens) < len(text.split())
    assert detokenize(tokens) == text.lower()
    assert to_syllables(tokens) == text.lower().split()


def test_unknown_level_is_rejected():
    with pytest.raises(ValueError):
        CaptionTokenizer(level="char")


def test_vocabulary_build_is_deterministic_and_applies_min_freq():
    captions = [["một", "người", "đang", "chạy"], ["một", "người", "đứng"], ["một", "con", "chó"]]
    vocab = Vocabulary.build(captions, min_freq=2)
    assert tuple(vocab.itos[:4]) == SPECIAL_TOKENS
    assert vocab.itos[4:] == ["một", "người"]
    assert Vocabulary.build(reversed(captions), min_freq=2).itos == vocab.itos
    assert len(Vocabulary.build(captions, max_size=6)) == 6


def test_vocabulary_encode_decode_round_trip(tmp_path):
    vocab = Vocabulary.build([["một", "người", "đang", "chạy"]])
    ids = vocab.encode(["một", "người", "bay"])
    assert ids[0] == vocab.bos_id and ids[-1] == vocab.eos_id
    assert ids[3] == vocab.unk_id
    assert vocab.decode(ids) == ["một", "người", "<unk>"]
    # Decoding stops at the end-of-sentence id and ignores padding.
    word_id = vocab.stoi["một"]
    assert vocab.decode([vocab.bos_id, word_id, vocab.eos_id, word_id, vocab.pad_id]) == ["một"]

    path = tmp_path / "vocab.json"
    vocab.save(path)
    assert Vocabulary.load(path).itos == vocab.itos
