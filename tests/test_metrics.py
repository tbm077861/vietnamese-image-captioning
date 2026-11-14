import pytest

from vic.metrics import cider_d, compute_metrics, corpus_bleu, meteor_exact
from vic.metrics.meteor import meteor_exact_sentence

REFERENCES = {
    "1": ["hai cầu thủ đang tranh bóng trên sân", "các cầu thủ bóng đá đang thi đấu trên sân cỏ"],
    "2": ["một người đàn ông đang đánh quần vợt", "người đàn ông cầm vợt chuẩn bị đánh bóng"],
    "3": ["cậu bé đang ném quả bóng chày", "một cậu bé mặc áo đỏ đang chơi bóng chày"],
}
HYPOTHESES = {
    "1": "hai cầu thủ đang thi đấu trên sân",
    "2": "một người đàn ông đang cầm vợt",
    "3": "một cậu bé đang chơi bóng chày",
}


def _tokenised():
    refs = {key: [caption.split() for caption in value] for key, value in REFERENCES.items()}
    hyps = {key: value.split() for key, value in HYPOTHESES.items()}
    return refs, hyps


def test_bleu_and_cider_match_the_coco_reference_implementation():
    # Expected values were produced with pycocoevalcap (Bleu(4), Cider()) on the same data.
    refs, hyps = _tokenised()
    scores = compute_metrics(refs, hyps)
    assert scores["BLEU-1"] == pytest.approx(0.955563, abs=1e-6)
    assert scores["BLEU-2"] == pytest.approx(0.930077, abs=1e-6)
    assert scores["BLEU-3"] == pytest.approx(0.852680, abs=1e-6)
    assert scores["BLEU-4"] == pytest.approx(0.723113, abs=1e-6)
    assert scores["CIDEr"] == pytest.approx(4.396419, abs=1e-6)
    assert 0.0 < scores["METEOR"] < 1.0


def test_bleu_perfect_and_disjoint():
    refs = [[["a", "b", "c", "d", "e"]], [["f", "g", "h", "i"]]]
    assert corpus_bleu(refs, [["a", "b", "c", "d", "e"], ["f", "g", "h", "i"]]) == pytest.approx(
        [1.0] * 4
    )
    assert corpus_bleu(refs, [["x", "y"], ["z"]]) == [0.0] * 4


def test_bleu_brevity_penalty_and_clipping():
    # "a a a a" against "a b c d": only one "a" may count.
    assert corpus_bleu([[["a", "b", "c", "d"]]], [["a", "a", "a", "a"]])[0] == pytest.approx(0.25)
    # A two-token perfect prefix of a four-token reference: precision 1, penalty exp(1 - 4/2).
    assert corpus_bleu([[["a", "b", "c", "d"]]], [["a", "b"]])[0] == pytest.approx(0.3678794)


def test_cider_rewards_the_matching_caption():
    refs = [[["a", "b", "c", "d"]], [["e", "f", "g", "h"]], [["i", "j", "k", "l"]]]
    good, per_image = cider_d(refs, [ref[0] for ref in refs])
    swapped, _ = cider_d(refs, [refs[1][0], refs[2][0], refs[0][0]])
    assert good == pytest.approx(10.0)
    assert per_image == pytest.approx([10.0] * 3)
    assert swapped == pytest.approx(0.0)


def test_meteor_exact_properties():
    sentence = ["một", "cậu", "bé", "đang", "chơi"]
    assert meteor_exact_sentence(sentence, sentence) == pytest.approx(1.0)
    assert meteor_exact_sentence(["x", "y"], sentence) == 0.0
    in_order = meteor_exact_sentence(["một", "cậu", "bé"], sentence)
    shuffled = meteor_exact_sentence(["bé", "một", "cậu"], sentence)
    assert 0.0 < shuffled < in_order < 1.0
    assert meteor_exact([[sentence, ["x"]]], [sentence]) == pytest.approx(1.0)


def test_compute_metrics_requires_references_for_every_hypothesis():
    with pytest.raises(KeyError):
        compute_metrics({"1": [["a"]]}, {"2": ["a"]})
