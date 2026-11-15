"""Tests for the decoder, decoding algorithms and the image encoder wrapper."""

import pytest
import torch

from vic.models import Captioner, ImageEncoder, beam_search, build_model, greedy_decode

MODEL_CONFIG = {
    "encoder": {"name": "resnet50", "image_size": 64, "trainable": False},
    "decoder": {"type": "lstm", "embed_dim": 16, "hidden_dim": 24, "num_layers": 1, "dropout": 0.0},
}
DECODER_CONFIGS = {
    "lstm": MODEL_CONFIG["decoder"],
    "lstm_attention": {
        "type": "lstm_attention", "embed_dim": 16, "hidden_dim": 24, "attention_dim": 20,
        "dropout": 0.0,
    },
}  # fmt: skip
VOCAB_SIZE, PAD, BOS, EOS = 12, 0, 1, 2


def _decoder_only(decoder_type: str = "lstm"):
    torch.manual_seed(0)
    config = {**MODEL_CONFIG, "decoder": DECODER_CONFIGS[decoder_type]}
    return build_model(config, VOCAB_SIZE, PAD, with_encoder=False).eval()


all_decoders = pytest.mark.parametrize("decoder_type", sorted(DECODER_CONFIGS))


@all_decoders
def test_forward_shape_and_half_precision_features(decoder_type):
    model = _decoder_only(decoder_type)
    features = torch.randn(3, 4, 2048).half()  # cached features are float16
    input_ids = torch.randint(3, VOCAB_SIZE, (3, 5))
    logits = model(input_ids, features=features)
    assert logits.shape == (3, 5, VOCAB_SIZE) and logits.dtype == torch.float32
    with pytest.raises(RuntimeError):
        model(input_ids, images=torch.zeros(3, 3, 64, 64))  # built without an encoder


@all_decoders
def test_teacher_forcing_matches_step_by_step_decoding(decoder_type):
    model = _decoder_only(decoder_type)
    features = torch.randn(2, 4, 2048)
    input_ids = torch.randint(3, VOCAB_SIZE, (2, 6))
    full = model(input_ids, features=features)
    state = model.decoder.init_state(features)
    for position in range(input_ids.size(1)):
        logits, state = model.decoder.step(input_ids[:, position], state)
        assert torch.allclose(logits, full[:, position], atol=1e-5)


@all_decoders
def test_greedy_and_beam_search_agree_with_beam_one_and_respect_limits(decoder_type):
    model = _decoder_only(decoder_type)
    features = torch.randn(4, 4, 2048)
    greedy = greedy_decode(model.decoder, features, BOS, EOS, max_length=7)
    assert len(greedy) == 4 and all(len(seq) <= 7 and EOS not in seq for seq in greedy)
    for index in range(4):
        beam = beam_search(
            model.decoder, features[index : index + 1], BOS, EOS, beam_size=1, max_length=7
        )
        assert beam == greedy[index]
    wide = beam_search(model.decoder, features[:1], BOS, EOS, beam_size=4, max_length=7)
    assert len(wide) <= 7 and EOS not in wide
    with pytest.raises(ValueError):
        beam_search(model.decoder, features, BOS, EOS)


@all_decoders
def test_suppressed_tokens_are_never_generated(decoder_type):
    model = _decoder_only(decoder_type)
    features = torch.randn(3, 4, 2048)
    banned = (PAD, BOS, *range(3, VOCAB_SIZE - 1))  # leave only EOS and the last token
    for sequence in greedy_decode(model.decoder, features, BOS, EOS, 6, suppress_ids=banned):
        assert set(sequence) <= {VOCAB_SIZE - 1}
    beam = beam_search(model.decoder, features[:1], BOS, EOS, 3, 6, suppress_ids=banned)
    assert set(beam) <= {VOCAB_SIZE - 1}


@all_decoders
def test_reorder_state_matches_decoding_the_selected_images(decoder_type):
    """Reordering the state is the same as having decoded the selected images directly."""
    model = _decoder_only(decoder_type)
    features = torch.randn(3, 4, 2048)
    index = torch.tensor([2, 2, 0])
    tokens = torch.tensor([3, 4, 5])
    state = model.decoder.reorder_state(model.decoder.init_state(features), index)
    logits, _ = model.decoder.step(tokens, state)
    expected, _ = model.decoder.step(tokens, model.decoder.init_state(features[index]))
    assert torch.allclose(logits, expected, atol=1e-6)


@all_decoders
def test_decoders_train_under_dropout_and_backpropagate(decoder_type):
    torch.manual_seed(0)
    config = {**MODEL_CONFIG, "decoder": {**DECODER_CONFIGS[decoder_type], "dropout": 0.5}}
    model = build_model(config, VOCAB_SIZE, PAD, with_encoder=False).train()
    logits = model(torch.randint(3, VOCAB_SIZE, (2, 5)), features=torch.randn(2, 4, 2048))
    logits.sum().backward()
    assert all(p.grad is not None for p in model.decoder.parameters())


def test_attention_weights_are_a_distribution_over_regions():
    decoder = _decoder_only("lstm_attention").decoder
    hidden, _, regions, region_keys = decoder.init_state(torch.randn(2, 9, 2048))
    alpha = decoder.attention(hidden, region_keys)
    assert alpha.shape == (2, 9) and regions.shape == (2, 9, 20)
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(2), atol=1e-5) and bool((alpha >= 0).all())


class _ScriptedDecoder(torch.nn.Module):
    """Deterministic decoder whose next-token log-probabilities depend only on the last token."""

    def __init__(self, table: dict[int, dict[int, float]], vocab_size: int = 6) -> None:
        super().__init__()
        self.table = torch.full((vocab_size, vocab_size), -1e9)
        for previous, row in table.items():
            for token, probability in row.items():
                self.table[previous, token] = torch.log(torch.tensor(probability))

    def init_state(self, features):
        return None

    def step(self, tokens, state):
        return self.table[tokens], state

    @staticmethod
    def reorder_state(state, index):
        return state


def test_no_repeat_ngram_blocks_loops_in_greedy_and_beam_search():
    # 3 -> 4 -> 3 -> 4 ... forever unless the repeated 2-gram is blocked; then 5 ends the caption.
    looping = _ScriptedDecoder(
        {BOS: {3: 1.0}, 3: {4: 0.9, 5: 0.1}, 4: {3: 0.9, 5: 0.1}, 5: {EOS: 1.0}}
    )
    features = torch.zeros(1, 1, 1)
    assert greedy_decode(looping, features, BOS, EOS, max_length=8) == [[3, 4] * 4]
    blocked = greedy_decode(looping, features, BOS, EOS, max_length=8, no_repeat_ngram_size=2)
    assert blocked == [[3, 4, 3, 5]]
    for stop in ("first", "best"):
        beam = beam_search(looping, features, BOS, EOS, 2, 8, no_repeat_ngram_size=2, stop=stop)
        assert beam and EOS not in beam
        ngrams = list(zip(beam, beam[1:], strict=False))
        assert len(ngrams) == len(set(ngrams))


def test_beam_stop_best_finds_the_longer_caption_with_the_better_normalised_score():
    # [3] ends first (probability 0.51) and [4, 5] ends next (0.245): with a beam of two, "first"
    # stops there. The long caption [4, 5, 6, 7, 8] also has probability 0.245 but, normalised
    # by its length, beats both; only "best" keeps searching long enough to finish it.
    table = {
        BOS: {3: 0.51, 4: 0.49},
        3: {EOS: 1.0},
        4: {5: 1.0},
        5: {EOS: 0.5, 6: 0.5},
        6: {7: 1.0},
        7: {8: 1.0},
        8: {EOS: 1.0},
    }
    decoder = _ScriptedDecoder(table, vocab_size=9)
    features = torch.zeros(1, 1, 1)
    first = beam_search(decoder, features, BOS, EOS, 2, 8, length_penalty=1.0, stop="first")
    best = beam_search(decoder, features, BOS, EOS, 2, 8, length_penalty=1.0, stop="best")
    assert first == [3] and best == [4, 5, 6, 7, 8]
    # Without length normalisation the most probable caption wins under both rules.
    for stop in ("first", "best"):
        assert beam_search(decoder, features, BOS, EOS, 2, 8, 0.0, stop=stop) == [3]
    with pytest.raises(ValueError, match="beam stop"):
        beam_search(decoder, features, BOS, EOS, stop="sometimes")


def test_reorder_state_selects_batch_entries():
    model = _decoder_only()
    state = model.decoder.init_state(torch.randn(3, 4, 2048))
    hidden, cell = model.decoder.reorder_state(state, torch.tensor([2, 2, 0]))
    assert torch.equal(hidden[:, 0], state[0][:, 2]) and torch.equal(cell[:, 2], state[1][:, 0])


def test_frozen_encoder_stays_in_eval_mode_and_outputs_a_grid():
    encoder = ImageEncoder("resnet50", pretrained=False, trainable=False)
    model = Captioner(_decoder_only().decoder, encoder)
    model.train()
    assert model.decoder.training and not encoder.training
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    features = model.encode(torch.zeros(2, 3, 64, 64))
    assert features.shape == (2, 4, 2048)  # 64 px input -> 2 x 2 grid
    assert not features.requires_grad
    with pytest.raises(ValueError):
        ImageEncoder("unknown")


@pytest.mark.parametrize("name", ["clip_vit_b16", "siglip_b16"])
def test_vision_transformer_encoders_output_the_patch_grid(name):
    encoder = ImageEncoder(name, pretrained=False, trainable=False)  # random weights, no download
    assert encoder.feature_dim == 768 and not encoder.training
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    features = encoder(torch.zeros(2, 3, 64, 64))  # 64 px / 16 px patches -> 4 x 4 grid
    assert features.shape == (2, 16, 768) and not features.requires_grad


@all_decoders
def test_feature_norm_makes_decoders_insensitive_to_feature_scale(decoder_type):
    torch.manual_seed(0)
    decoder_config = {**DECODER_CONFIGS[decoder_type], "feature_norm": True}
    config = {"encoder": {"name": "clip_vit_b16"}, "decoder": decoder_config}
    model = build_model(config, VOCAB_SIZE, PAD, with_encoder=False).eval()
    features = torch.randn(2, 5, 768)
    input_ids = torch.randint(3, VOCAB_SIZE, (2, 4))
    logits = model(input_ids, features=features)
    assert logits.shape == (2, 4, VOCAB_SIZE)
    assert torch.allclose(logits, model(input_ids, features=features * 50), atol=1e-3)
