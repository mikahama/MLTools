"""Friendly image-to-text (captioning) models built on PyTorch.

Each corpus item is ``[image_path, description]``.  Image dimensions are
sampled from the training files and a practical common training size is chosen
automatically, so differently sized camera images can be mixed directly.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from . import seq2seq

Pair = Tuple[str, str]

# Kept as an alias so image and text data split in the same, predictable way.
split = seq2seq.split


def _image_pairs(corpus: Sequence[Sequence[str]], *, check_files: bool = True) -> List[Pair]:
    pairs = seq2seq._pairs(corpus)
    if check_files:
        missing = [source for source, _ in pairs if not Path(source).is_file()]
        if missing:
            example = missing[0]
            raise FileNotFoundError(f"Could not find {len(missing)} image file(s); for example: {example}")
    return pairs


def infer_input_size(image_paths: Sequence[Union[str, Path]], *, sample_size: int = 100) -> Tuple[int, int]:
    """Choose a sensible ``(width, height)`` from image dimensions.

    The median aspect ratio is retained.  The median long edge is rounded to a
    modest, model-friendly bucket (128, 224, 320, or 384 pixels), and both
    dimensions are divisible by 32 for common CNN encoders.
    """
    paths = list(image_paths)
    if not paths:
        raise ValueError("At least one image path is needed to infer an input size.")
    # Evenly spaced samples avoid opening thousands of files for very large sets.
    if len(paths) > sample_size:
        step = (len(paths) - 1) / (sample_size - 1)
        paths = [paths[round(index * step)] for index in range(sample_size)]
    dimensions = []
    for path in paths:
        with Image.open(path) as image:
            dimensions.append(image.size)
    median_width = statistics.median(width for width, _ in dimensions)
    median_height = statistics.median(height for _, height in dimensions)
    long_edge = max(median_width, median_height)
    if long_edge <= 160:
        chosen_long = 128
    elif long_edge <= 320:
        chosen_long = 224
    elif long_edge <= 800:
        chosen_long = 320
    else:
        chosen_long = 384
    scale = chosen_long / long_edge
    width = max(32, round(median_width * scale / 32) * 32)
    height = max(32, round(median_height * scale / 32) * 32)
    return int(width), int(height)


def _image_tensor(path: str, size: Tuple[int, int]) -> Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        width, height = image.size
        pixels = torch.frombuffer(memoryview(image.tobytes()), dtype=torch.uint8).clone()
    tensor = pixels.reshape(height, width, 3).permute(2, 0, 1).float().div_(255)
    # The normalization lets pretrained ImageNet encoders work out of the box.
    return (tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class _ImageTextData(Dataset):
    def __init__(self, pairs: Sequence[Pair], vocabulary: seq2seq.Vocabulary, size: Tuple[int, int], tokenizer: seq2seq.Tokenize) -> None:
        self.pairs, self.vocabulary, self.size, self.tokenizer = list(pairs), vocabulary, size, tokenizer

    def __len__(self) -> int: return len(self.pairs)

    def __getitem__(self, index: int):
        path, caption = self.pairs[index]
        return _image_tensor(path, self.size), torch.tensor(self.vocabulary.encode(caption, self.tokenizer, add_bos=True))


def _collate(items, pad_id: int):
    images, captions = zip(*items)
    return torch.stack(images), nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=pad_id)


class _SmallCNN(nn.Module):
    """A compact encoder that is useful for small, domain-specific datasets."""
    def __init__(self) -> None:
        super().__init__()
        channels = [3, 32, 64, 128, 256]
        layers: List[nn.Module] = []
        for incoming, outgoing in zip(channels, channels[1:]):
            layers += [nn.Conv2d(incoming, outgoing, 3, stride=2, padding=1), nn.BatchNorm2d(outgoing), nn.ReLU(inplace=True)]
        self.layers = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.feature_size = 256

    def forward(self, images: Tensor) -> Tensor: return self.layers(images)


def _encoder(model_type: str, pretrained: bool) -> Tuple[nn.Module, int]:
    if model_type == "cnn": return _SmallCNN(), 256
    try:
        from torchvision import models
    except ImportError as error:
        raise ImportError("ResNet and MobileNet require torchvision. Install MLTools with its standard dependencies.") from error
    if model_type == "resnet18":
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        return nn.Sequential(*list(base.children())[:-1], nn.Flatten()), base.fc.in_features
    if model_type == "resnet50":
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        return nn.Sequential(*list(base.children())[:-1], nn.Flatten()), base.fc.in_features
    if model_type == "mobilenet_v3_small":
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        return nn.Sequential(base.features, base.avgpool, nn.Flatten()), base.classifier[0].in_features
    raise ValueError("model_type must be 'cnn', 'resnet18', 'resnet50', or 'mobilenet_v3_small'.")


class _Captioner(nn.Module):
    def __init__(self, encoder: nn.Module, feature_size: int, vocabulary_size: int, embedding_size: int, hidden_size: int, layers: int, dropout: float, pad_id: int) -> None:
        super().__init__()
        self.encoder, self.layers, self.hidden_size = encoder, layers, hidden_size
        self.initial_state = nn.Linear(feature_size, hidden_size * layers)
        self.embedding = nn.Embedding(vocabulary_size, embedding_size, padding_idx=pad_id)
        self.decoder = nn.GRU(embedding_size, hidden_size, layers, batch_first=True, dropout=dropout if layers > 1 else 0)
        self.output = nn.Linear(hidden_size, vocabulary_size)

    def initial_hidden(self, images: Tensor) -> Tensor:
        features = self.encoder(images)
        return torch.tanh(self.initial_state(features)).reshape(images.size(0), self.layers, self.hidden_size).transpose(0, 1).contiguous()

    def forward(self, images: Tensor, caption_input: Tensor) -> Tensor:
        output, _ = self.decoder(self.embedding(caption_input), self.initial_hidden(images))
        return self.output(output)

    def decode_step(self, previous: Tensor, hidden: Tensor) -> Tuple[Tensor, Tensor]:
        output, hidden = self.decoder(self.embedding(previous).unsqueeze(1), hidden)
        return self.output(output[:, 0]), hidden


@dataclass
class _Config:
    model_type: str = "cnn"
    pretrained: bool = False
    input_size: Tuple[int, int] = (224, 224)
    embedding_size: int = 256
    hidden_size: int = 512
    layers: int = 2
    dropout: float = 0.2
    min_frequency: int = 1


def _network(config: _Config, vocabulary: seq2seq.Vocabulary, *, load_pretrained: bool = True) -> nn.Module:
    encoder, feature_size = _encoder(config.model_type, config.pretrained and load_pretrained)
    return _Captioner(encoder, feature_size, len(vocabulary.itos), config.embedding_size, config.hidden_size, config.layers, config.dropout, vocabulary.pad_id)


class Image2TextModel:
    """A trained captioning model returned by :func:`train`."""
    def __init__(self, network: nn.Module, vocabulary: seq2seq.Vocabulary, config: _Config, folder: Union[str, Path], tokenizer: seq2seq.Tokenize = seq2seq._default_tokenizer, detokenizer: seq2seq.Detokenize = seq2seq._default_detokenizer, device: Optional[Union[str, torch.device]] = None) -> None:
        self.network, self.vocabulary, self.config = network, vocabulary, config
        self.folder, self.tokenizer, self.detokenizer = Path(folder), tokenizer, detokenizer
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.network.to(self.device); self.epoch = 0

    @property
    def input_size(self) -> Tuple[int, int]:
        """The automatically selected ``(width, height)`` used for all images."""
        return tuple(self.config.input_size)

    def _loader(self, data: Sequence[Sequence[str]], batch_size: int, shuffle: bool) -> DataLoader:
        dataset = _ImageTextData(_image_pairs(data), self.vocabulary, self.input_size, self.tokenizer)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda items: _collate(items, self.vocabulary.pad_id))

    def _loss(self, images: Tensor, captions: Tensor) -> Tensor:
        logits = self.network(images, captions[:, :-1])
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), captions[:, 1:].reshape(-1), ignore_index=self.vocabulary.pad_id)

    def _validation_loss(self, data: Sequence[Sequence[str]], batch_size: int) -> float:
        self.network.eval(); total = count = 0
        with torch.no_grad():
            for images, captions in self._loader(data, batch_size, False):
                loss = self._loss(images.to(self.device), captions.to(self.device))
                total += loss.item() * len(images); count += len(images)
        return total / max(count, 1)

    def _save(self, optimizer: torch.optim.Optimizer, validation_loss: float, *, best: bool) -> None:
        state = {"state_dict": self.network.state_dict(), "optimizer": optimizer.state_dict(), "epoch": self.epoch, "validation_loss": validation_loss, "config": asdict(self.config), "vocabulary": self.vocabulary.itos}
        torch.save(state, self.folder / "model_latest.pt")
        torch.save(state, self.folder / f"checkpoint_epoch_{self.epoch:03d}.pt")
        if best: torch.save(state, self.folder / "model_best.pt")

    def fit(self, train_data: Sequence[Sequence[str]], validation_data: Sequence[Sequence[str]], *, epochs: int = 20, batch_size: int = 16, learning_rate: float = 0.001, patience: int = 3, min_delta: float = 0.0, optimizer_state: Optional[dict] = None) -> "Image2TextModel":
        """Train for up to ``epochs`` additional epochs with early stopping."""
        if not validation_data: raise ValueError("validation_data cannot be empty because it is used for early stopping.")
        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        if optimizer_state: optimizer.load_state_dict(optimizer_state)
        best_loss, stalled = float("inf"), 0
        for _ in range(epochs):
            self.epoch += 1; self.network.train(); running = 0.0
            progress = tqdm(self._loader(train_data, batch_size, True), desc=f"Epoch {self.epoch}", unit="batch")
            for images, captions in progress:
                optimizer.zero_grad(); loss = self._loss(images.to(self.device), captions.to(self.device))
                loss.backward(); torch.nn.utils.clip_grad_norm_(self.network.parameters(), 5.0); optimizer.step()
                running += loss.item(); progress.set_postfix(train_loss=f"{running / max(progress.n, 1):.3f}")
            validation_loss = self._validation_loss(validation_data, batch_size)
            improved = validation_loss < best_loss - min_delta
            best_loss, stalled = (validation_loss, 0) if improved else (best_loss, stalled + 1)
            self._save(optimizer, validation_loss, best=improved)
            print(f"Epoch {self.epoch}: validation loss {validation_loss:.4f}" + (" (best)" if improved else ""))
            if stalled >= patience:
                print(f"Early stopping after {self.epoch} epochs (no validation improvement for {patience} epochs).")
                break
        return self

    def _decode_one(self, path: Union[str, Path], n_best: int, max_length: int, beam_size: int) -> List[Tuple[str, float]]:
        image = _image_tensor(str(path), self.input_size).unsqueeze(0).to(self.device)
        self.network.eval()
        with torch.no_grad():
            initial_hidden = self.network.initial_hidden(image)  # type: ignore[attr-defined]
            beams: List[Tuple[List[int], float, bool, Tensor]] = [([self.vocabulary.bos_id], 0.0, False, initial_hidden)]
            for _ in range(max_length):
                candidates = []
                for prefix, score, ended, hidden in beams:
                    if ended:
                        candidates.append((prefix, score, ended, hidden)); continue
                    logits, next_hidden = self.network.decode_step(torch.tensor([prefix[-1]], device=self.device), hidden)  # type: ignore[attr-defined]
                    scores, token_ids = torch.topk(F.log_softmax(logits[0], dim=-1), min(beam_size, logits.size(-1)))
                    candidates.extend((prefix + [token.item()], score + value.item(), token.item() == self.vocabulary.eos_id, next_hidden) for value, token in zip(scores, token_ids))
                beams = sorted(candidates, key=lambda item: item[1] / max((len(item[0]) - 1) ** 0.7, 1), reverse=True)[:beam_size]
                if all(ended for _, _, ended, _ in beams): break
        return [(self.detokenizer(self.vocabulary.decode(ids)), score) for ids, score, _, _ in beams[:n_best]]

    def predict(self, image: Union[str, Path, Sequence[Union[str, Path]]], *, n_best: int = 1, return_scores: bool = False, max_length: int = 100, beam_size: Optional[int] = None):
        """Create a caption for one image path or a list of image paths."""
        if n_best < 1 or max_length < 1: raise ValueError("n_best and max_length must be at least 1.")
        one = isinstance(image, (str, Path)); paths = [image] if one else list(image)
        beam_size = beam_size or max(n_best, 1)
        results = [self._decode_one(path, n_best, max_length, beam_size) for path in paths]
        if n_best == 1: output = [items[0] if return_scores else items[0][0] for items in results]
        else: output = [items if return_scores else [caption for caption, _ in items] for items in results]
        return output[0] if one else output

    def evaluate(self, test_data: Sequence[Sequence[str]]) -> Dict[str, float]:
        """Print and return token precision, recall, F1, BLEU, and WER."""
        pairs = _image_pairs(test_data)
        hypotheses = [self.tokenizer(caption) for caption in self.predict([path for path, _ in pairs])]
        references = [self.tokenizer(caption) for _, caption in pairs]
        from collections import Counter
        overlap = sum(sum((Counter(hypothesis) & Counter(reference)).values()) for hypothesis, reference in zip(hypotheses, references))
        predicted, reference = sum(map(len, hypotheses)), sum(map(len, references))
        precision = overlap / predicted if predicted else 0.0; recall = overlap / reference if reference else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics = {"precision": precision, "recall": recall, "f1": f1, "bleu": seq2seq._corpus_bleu(references, hypotheses), "wer": sum(seq2seq._edit_distance(reference, hypothesis) for reference, hypothesis in zip(references, hypotheses)) / max(reference, 1)}
        print("Evaluation result\n" + "\n".join(f"  {name.upper():9} {value:.4f}" for name, value in metrics.items()))
        return metrics


def train(output_folder: Union[str, Path], train_data: Sequence[Sequence[str]], validation_data: Sequence[Sequence[str]], *, model_type: str = "cnn", pretrained: bool = False, input_size: Optional[Tuple[int, int]] = None, embedding_size: int = 256, hidden_size: int = 512, layers: int = 2, dropout: float = 0.2, min_frequency: int = 1, epochs: int = 20, batch_size: int = 16, learning_rate: float = 0.001, patience: int = 3, min_delta: float = 0.0, tokenizer: seq2seq.Tokenize = seq2seq._default_tokenizer, detokenizer: seq2seq.Detokenize = seq2seq._default_detokenizer, device: Optional[Union[str, torch.device]] = None) -> Image2TextModel:
    """Train an image-captioning model and save it in a new empty folder.

    The corpus contains ``[image_path, caption]`` pairs.  When ``input_size``
    is omitted, it is inferred from the training images and printed.
    """
    folder = Path(output_folder)
    if folder.exists() and any(folder.iterdir()): raise ValueError(f"Output folder must be empty: {folder}. Use continue_train for an existing model.")
    folder.mkdir(parents=True, exist_ok=True)
    pairs = _image_pairs(train_data)
    size = tuple(input_size) if input_size else infer_input_size([path for path, _ in pairs])
    if len(size) != 2 or min(size) < 32: raise ValueError("input_size must be a (width, height) pair of at least 32 pixels.")
    print(f"Using image input size {size[0]}×{size[1]} pixels.")
    config = _Config(model_type.lower(), pretrained, size, embedding_size, hidden_size, layers, dropout, min_frequency)
    vocabulary = seq2seq.Vocabulary.build([caption for _, caption in pairs], tokenizer, min_frequency)
    model = Image2TextModel(_network(config, vocabulary), vocabulary, config, folder, tokenizer, detokenizer, device)
    (folder / "training_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    return model.fit(pairs, validation_data, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, patience=patience, min_delta=min_delta)


def load_model(path: Union[str, Path], *, device: Optional[Union[str, torch.device]] = None, tokenizer: seq2seq.Tokenize = seq2seq._default_tokenizer, detokenizer: seq2seq.Detokenize = seq2seq._default_detokenizer) -> Image2TextModel:
    """Load the latest checkpoint from a model folder, or a checkpoint file."""
    path = Path(path); checkpoint_path = path / "model_latest.pt" if path.is_dir() else path
    if not checkpoint_path.is_file(): raise FileNotFoundError(f"No model checkpoint found at {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
    config_data = checkpoint["config"]; config_data["input_size"] = tuple(config_data["input_size"])
    config = _Config(**config_data); vocabulary = seq2seq.Vocabulary(checkpoint["vocabulary"])
    model = Image2TextModel(_network(config, vocabulary, load_pretrained=False), vocabulary, config, checkpoint_path.parent, tokenizer, detokenizer, device)
    model.network.load_state_dict(checkpoint["state_dict"]); model.epoch = checkpoint.get("epoch", 0)
    return model


def continue_train(output_folder: Union[str, Path], train_data: Sequence[Sequence[str]], validation_data: Sequence[Sequence[str]], **kwargs) -> Image2TextModel:
    """Resume image-captioning training from ``model_latest.pt``."""
    folder = Path(output_folder); checkpoint = torch.load(folder / "model_latest.pt", map_location=kwargs.get("device", "cpu"))
    model = load_model(folder, device=kwargs.pop("device", None), tokenizer=kwargs.pop("tokenizer", seq2seq._default_tokenizer), detokenizer=kwargs.pop("detokenizer", seq2seq._default_detokenizer))
    return model.fit(train_data, validation_data, optimizer_state=checkpoint.get("optimizer"), **kwargs)
