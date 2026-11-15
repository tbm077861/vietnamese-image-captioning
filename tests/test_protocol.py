"""Tests for evaluation file names and decoding labels."""

from vic.engine.protocol import (
    CHOSEN_DECODING,
    PROTOCOL_DECODING,
    decoding_label,
    evaluation_filename,
    is_protocol_decoding,
)


def test_protocol_files_keep_their_historical_names():
    for beam in (1, 3, 5):
        decoding = {**PROTOCOL_DECODING, "beam_size": beam}
        assert evaluation_filename("val", decoding) == f"eval_val_beam{beam}.json"
        assert is_protocol_decoding(decoding)
    assert is_protocol_decoding(None)


def test_other_decodings_get_distinct_names():
    assert evaluation_filename("val", CHOSEN_DECODING) == "eval_val_beam3_norepeat3.json"
    assert not is_protocol_decoding(CHOSEN_DECODING)
    full = {"beam_size": 5, "length_penalty": 1.0, "beam_stop": "best", "no_repeat_ngram_size": 2}
    assert evaluation_filename("test", full) == "eval_test_beam5_lp1_stopbest_norepeat2.json"
    # Greedy ignores the beam options.
    greedy = {"beam_size": 1, "length_penalty": 1.3, "beam_stop": "best"}
    assert evaluation_filename("val", greedy) == "eval_val_beam1.json"
    assert decoding_label(CHOSEN_DECODING) == "beam 3, LP 0,7, dừng first, chặn lặp 3-gram"
    assert decoding_label({"beam_size": 1}) == "greedy, không chặn lặp"
