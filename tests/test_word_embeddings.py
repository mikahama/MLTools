"""Numerical regressions for vocabulary and unseen-word vector composition."""

import pytest
import torch

from MLTools.LM import word_embeddings as embeddings


def analyze(word):
    return {
        "kudo": ["kudo+N+Sg"],
        "kudoso": ["kudo+N+Sg+Ine"],
        "foreign": ["unknown+N"],
    }.get(word, [])


@pytest.fixture(params=["fasttext", "morph"])
def model(request, tmp_path):
    model_type = request.param
    config = {
        "model_type": model_type, "dimension": 3, "bucket_size": 17,
        "min_n": 3, "max_n": 3, "lowercase": True, "device": "cpu",
    }
    morph_vocab = {"lemmas": {"kudo": 0}, "tags": {"<none>": 0, "N": 1, "Sg": 2, "Ine": 3}}
    network = embeddings._SkipGram(2, 3, model_type, 17, 1, 4)
    with torch.no_grad():
        network.input.weight.copy_(torch.tensor([[11., 13., 17.], [19., 23., 29.]]))
        network.subwords.weight.copy_(torch.arange(51).reshape(17, 3) / 10)
        if model_type == "morph":
            network.lemmas.weight.copy_(torch.tensor([[2., 3., 5.]]))
            network.tags.weight.copy_(torch.tensor([[0., 0., 0.], [7., 11., 13.],
                                                   [17., 19., 23.], [29., 31., 37.]]))
    return embeddings.WordEmbeddings(network, ["kudo", "other"], config, tmp_path, analyze, morph_vocab)


def subword_mean(model, word):
    grams = embeddings._ngrams(word, model.config["min_n"], model.config["max_n"]) or [word]
    ids = [embeddings._stable_hash(gram) % model.config["bucket_size"] for gram in grams]
    return model.network.subwords.weight.detach()[ids].mean(dim=0)


def expected_features(model, word):
    result = subword_mean(model, word)
    if model.config["model_type"] == "morph" and word in {"kudo", "kudoso"}:
        tags = [1, 2, 3] if word == "kudoso" else [1, 2]
        result = result + model.network.lemmas.weight.detach()[0]
        result = result + model.network.tags.weight.detach()[tags].mean(dim=0)
    return result


def test_oov_is_independent_of_first_vocabulary_embedding(model):
    before = model.embed("kudoso")
    torch.testing.assert_close(before, expected_features(model, "kudoso"))
    known_before = model.embed("kudo")
    with torch.no_grad():
        model.network.input.weight[0].add_(1000)
    torch.testing.assert_close(model.embed("kudoso"), before)
    # Index zero is still a real word and must retain its input contribution.
    torch.testing.assert_close(model.embed("kudo"), known_before + 1000)
    assert before.device.type == "cpu"
    assert not before.requires_grad


def test_mixed_batch_preserves_known_words_and_oov_features(model):
    words = ["kudo", "kudoso", "other", "unanalysed", "kudoso"]
    expected = torch.stack([
        expected_features(model, word)
        + (model.network.input.weight.detach()[model.word_to_id[word]] if word in model.word_to_id else 0)
        for word in words
    ])
    torch.testing.assert_close(model._vectors(words), expected)


def test_oov_uses_its_own_subwords(model):
    for word in ["kudoso", "unanalysed", "foreign"]:
        torch.testing.assert_close(model.embed(word), expected_features(model, word))
    assert not torch.allclose(model.embed("unanalysed"), model.embed("foreign"))
    before = model.embed("kudoso")
    with torch.no_grad():
        model.network.subwords.weight.add_(3)
    torch.testing.assert_close(model.embed("kudoso"), before + 3)


def test_short_oov_uses_subword_fallback(model):
    model.config.update(min_n=6, max_n=6)
    torch.testing.assert_close(model.embed("x"), subword_mean(model, "x"))


def test_oov_keeps_available_morphology(model):
    before = model.embed("kudoso")
    if model.config["model_type"] != "morph":
        assert model.network.lemmas is None
        torch.testing.assert_close(before, subword_mean(model, "kudoso"))
        return
    with torch.no_grad():
        model.network.lemmas.weight[0].add_(6)
    torch.testing.assert_close(model.embed("kudoso"), before + 6)
    with torch.no_grad():
        model.network.tags.weight[3].add_(9)
    torch.testing.assert_close(model.embed("kudoso"), before + 6 + 3)
    model.analyzer = None
    torch.testing.assert_close(model.embed("kudoso"), subword_mean(model, "kudoso"))


@pytest.mark.parametrize("model_type", ["word2vec", "fasttext", "morph"])
def test_train_and_reload_preserve_inference(tmp_path, model_type):
    model = embeddings.train(
        tmp_path, text="kudo other kudo other", model_type=model_type,
        analyzer=analyze if model_type == "morph" else None,
        dimension=3, bucket_size=17, epochs=1, batch_size=8, device="cpu",
    )
    loaded = embeddings.load_model(tmp_path, device="cpu", analyzer=analyze)
    torch.testing.assert_close(loaded._vectors(model.vocabulary), model._vectors(model.vocabulary))
    if model_type == "word2vec":
        torch.testing.assert_close(model.embed("kudo"), model.network.input.weight.detach()[0])
        for candidate in [model, loaded]:
            with pytest.raises(KeyError, match="Unknown word: 'kudoso'"):
                candidate.embed("kudoso")
            with pytest.raises(KeyError, match="Unknown word: 'kudoso'"):
                candidate.similarity("kudo", "kudoso")
    else:
        before = model.embed("kudoso")
        torch.testing.assert_close(loaded.embed("KUDOSO"), before)
        with torch.no_grad():
            loaded.network.input.weight[0].add_(1000)
        torch.testing.assert_close(loaded.embed("kudoso"), before)
