"""Approachable automatic speech recognition (audio to transcription).

Train from ``[audio_path, transcription]`` pairs::

    model = audio2text.train("speech-model", train, validation)
    model.predict("recording.wav")

MLTools automatically converts recordings to mono and selects a common sample
rate from the training data.  The bundled models learn character-level CTC
transcription, so no language-specific tokenizer is required.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torchaudio
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from . import seq2seq

Pair = Tuple[str, str]
split = seq2seq.split


def _audio_pairs(corpus: Sequence[Sequence[str]], *, check_files: bool = True) -> List[Pair]:
    pairs = seq2seq._pairs(corpus)
    if check_files:
        missing = [path for path, _ in pairs if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Could not find {len(missing)} audio file(s); for example: {missing[0]}")
    return pairs


def infer_sample_rate(audio_paths: Sequence[Union[str, Path]], *, sample_size: int = 100) -> int:
    """Choose the median recording sample rate, rounded to a standard rate."""
    paths = list(audio_paths)
    if not paths: raise ValueError("At least one audio path is needed to infer a sample rate.")
    if len(paths) > sample_size:
        step = (len(paths) - 1) / (sample_size - 1)
        paths = [paths[round(index * step)] for index in range(sample_size)]
    rates = [torchaudio.info(str(path)).sample_rate for path in paths]
    median = statistics.median(rates)
    standards = [8000, 16000, 22050, 24000, 32000, 44100, 48000]
    return min(standards, key=lambda rate: abs(rate - median))


class _Characters:
    """A character vocabulary for CTC; index zero is the required blank token."""
    def __init__(self, characters: List[str]) -> None:
        self.characters = characters
        self.ids = {character: index for index, character in enumerate(characters)}

    @classmethod
    def build(cls, texts: Sequence[str], min_frequency: int = 1) -> "_Characters":
        counts = Counter(character for text in texts for character in text)
        return cls(["<blank>", "<unknown>"] + sorted(character for character, count in counts.items() if count >= min_frequency))

    @property
    def blank_id(self) -> int: return 0
    @property
    def unknown_id(self) -> int: return 1

    def encode(self, text: str) -> List[int]: return [self.ids.get(character, self.unknown_id) for character in text]

    def decode_ctc(self, ids: Sequence[int]) -> str:
        output, previous = [], None
        for index in ids:
            if index != self.blank_id and index != previous and 0 <= index < len(self.characters): output.append(self.characters[index])
            previous = index
        return "".join(output).replace("<unknown>", "?")


def _features(path: str, sample_rate: int, mel: nn.Module) -> Tensor:
    waveform, original_rate = torchaudio.load(path)
    waveform = waveform.mean(dim=0, keepdim=True)  # stereo and multichannel recordings become mono.
    if original_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, original_rate, sample_rate)
    return mel(waveform).squeeze(0).transpose(0, 1).clamp_min(1e-5).log()


class _SpeechData(Dataset):
    def __init__(self, pairs: Sequence[Pair], characters: _Characters, sample_rate: int, n_mels: int) -> None:
        self.pairs, self.characters = list(pairs), characters
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, n_fft=400, hop_length=160)
        self.sample_rate = sample_rate

    def __len__(self) -> int: return len(self.pairs)
    def __getitem__(self, index: int):
        path, text = self.pairs[index]
        encoded = self.characters.encode(text)
        if not encoded: raise ValueError(f"Transcription cannot be empty: {path}")
        return _features(path, self.sample_rate, self.mel), torch.tensor(encoded, dtype=torch.long)


def _collate(items):
    features, targets = zip(*items)
    lengths = torch.tensor([len(item) for item in features], dtype=torch.long)
    target_lengths = torch.tensor([len(item) for item in targets], dtype=torch.long)
    return nn.utils.rnn.pad_sequence(features, batch_first=True), lengths, torch.cat(targets), target_lengths


class _RNNRecognizer(nn.Module):
    def __init__(self, n_mels: int, vocabulary_size: int, hidden_size: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.rnn = nn.GRU(n_mels, hidden_size, layers, batch_first=True, bidirectional=True, dropout=dropout if layers > 1 else 0)
        self.output = nn.Linear(hidden_size * 2, vocabulary_size)

    def forward(self, features: Tensor, lengths: Tensor) -> Tensor:
        packed = pack_padded_sequence(features, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=features.size(1))
        return self.output(output)


class _TransformerRecognizer(nn.Module):
    def __init__(self, n_mels: int, vocabulary_size: int, hidden_size: int, layers: int, dropout: float) -> None:
        super().__init__()
        if hidden_size % 4: raise ValueError("Transformer hidden_size must be divisible by 4.")
        self.input = nn.Linear(n_mels, hidden_size)
        self.position = nn.Embedding(20000, hidden_size)
        layer = nn.TransformerEncoderLayer(hidden_size, nhead=4, dim_feedforward=hidden_size * 4, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.output = nn.Linear(hidden_size, vocabulary_size)

    def forward(self, features: Tensor, lengths: Tensor) -> Tensor:
        if features.size(1) > self.position.num_embeddings: raise ValueError("Recording is too long (over 20,000 feature frames).")
        positions = torch.arange(features.size(1), device=features.device).unsqueeze(0)
        padding = torch.arange(features.size(1), device=features.device).unsqueeze(0) >= lengths.to(features.device).unsqueeze(1)
        return self.output(self.encoder(self.input(features) + self.position(positions), src_key_padding_mask=padding))


@dataclass
class _Config:
    model_type: str = "rnn"
    sample_rate: int = 16000
    n_mels: int = 80
    hidden_size: int = 256
    layers: int = 3
    dropout: float = 0.2
    min_frequency: int = 1


def _network(config: _Config, characters: _Characters) -> nn.Module:
    if config.model_type == "rnn": return _RNNRecognizer(config.n_mels, len(characters.characters), config.hidden_size, config.layers, config.dropout)
    if config.model_type == "transformer": return _TransformerRecognizer(config.n_mels, len(characters.characters), config.hidden_size, config.layers, config.dropout)
    raise ValueError("model_type must be 'rnn' (the default) or 'transformer'.")


class Audio2TextModel:
    """A trained speech-recognition model returned by :func:`train`."""
    def __init__(self, network: nn.Module, characters: _Characters, config: _Config, folder: Union[str, Path], device: Optional[Union[str, torch.device]] = None) -> None:
        self.network, self.characters, self.config, self.folder = network, characters, config, Path(folder)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.network.to(self.device); self.epoch = 0
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=config.sample_rate, n_mels=config.n_mels, n_fft=400, hop_length=160)

    @property
    def sample_rate(self) -> int:
        """The common sample rate selected for training and prediction."""
        return self.config.sample_rate

    def _loader(self, data: Sequence[Sequence[str]], batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(_SpeechData(_audio_pairs(data), self.characters, self.config.sample_rate, self.config.n_mels), batch_size=batch_size, shuffle=shuffle, collate_fn=_collate)

    def _loss(self, features: Tensor, lengths: Tensor, targets: Tensor, target_lengths: Tensor) -> Tensor:
        logits = self.network(features, lengths)
        return nn.CTCLoss(blank=self.characters.blank_id, zero_infinity=True)(F.log_softmax(logits, -1).transpose(0, 1), targets, lengths, target_lengths)

    def _validation_loss(self, data: Sequence[Sequence[str]], batch_size: int) -> float:
        self.network.eval(); total = count = 0
        with torch.no_grad():
            for features, lengths, targets, target_lengths in self._loader(data, batch_size, False):
                loss = self._loss(features.to(self.device), lengths, targets.to(self.device), target_lengths)
                total += loss.item() * len(features); count += len(features)
        return total / max(count, 1)

    def _save(self, optimizer: torch.optim.Optimizer, validation_loss: float, *, best: bool) -> None:
        state = {"state_dict": self.network.state_dict(), "optimizer": optimizer.state_dict(), "epoch": self.epoch, "validation_loss": validation_loss, "config": asdict(self.config), "characters": self.characters.characters}
        torch.save(state, self.folder / "model_latest.pt")
        torch.save(state, self.folder / f"checkpoint_epoch_{self.epoch:03d}.pt")
        if best: torch.save(state, self.folder / "model_best.pt")

    def fit(self, train_data: Sequence[Sequence[str]], validation_data: Sequence[Sequence[str]], *, epochs: int = 20, batch_size: int = 16, learning_rate: float = 0.001, patience: int = 3, min_delta: float = 0.0, optimizer_state: Optional[dict] = None) -> "Audio2TextModel":
        if not validation_data: raise ValueError("validation_data cannot be empty because it is used for early stopping.")
        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        if optimizer_state: optimizer.load_state_dict(optimizer_state)
        best_loss, stalled = float("inf"), 0
        for _ in range(epochs):
            self.epoch += 1; self.network.train(); running = 0.0
            progress = tqdm(self._loader(train_data, batch_size, True), desc=f"Epoch {self.epoch}", unit="batch")
            for features, lengths, targets, target_lengths in progress:
                optimizer.zero_grad(); loss = self._loss(features.to(self.device), lengths, targets.to(self.device), target_lengths)
                loss.backward(); torch.nn.utils.clip_grad_norm_(self.network.parameters(), 5.0); optimizer.step()
                running += loss.item(); progress.set_postfix(ctc_loss=f"{running / max(progress.n, 1):.3f}")
            validation_loss = self._validation_loss(validation_data, batch_size)
            improved = validation_loss < best_loss - min_delta
            best_loss, stalled = (validation_loss, 0) if improved else (best_loss, stalled + 1)
            self._save(optimizer, validation_loss, best=improved)
            print(f"Epoch {self.epoch}: validation CTC loss {validation_loss:.4f}" + (" (best)" if improved else ""))
            if stalled >= patience:
                print(f"Early stopping after {self.epoch} epochs (no validation improvement for {patience} epochs).")
                break
        return self

    def _predict_one(self, path: Union[str, Path], return_score: bool) -> Union[str, Tuple[str, float]]:
        features = _features(str(path), self.config.sample_rate, self.mel).unsqueeze(0).to(self.device)
        self.network.eval()
        with torch.no_grad():
            logits = self.network(features, torch.tensor([features.size(1)]))[0]
            log_probabilities = F.log_softmax(logits, -1)
            scores, ids = log_probabilities.max(-1)
        text = self.characters.decode_ctc(ids.tolist())
        return (text, scores.mean().item()) if return_score else text

    def predict(self, audio: Union[str, Path, Sequence[Union[str, Path]]], *, return_scores: bool = False):
        """Transcribe one audio path or a list of audio paths with greedy CTC decoding."""
        one = isinstance(audio, (str, Path)); paths = [audio] if one else list(audio)
        if not paths: raise ValueError("predict needs at least one audio path.")
        results = [self._predict_one(path, return_scores) for path in paths]
        return results[0] if one else results

    def evaluate(self, test_data: Sequence[Sequence[str]]) -> Dict[str, float]:
        """Print and return word error rate (WER) and character error rate (CER)."""
        pairs = _audio_pairs(test_data)
        predictions = self.predict([path for path, _ in pairs])
        references = [text for _, text in pairs]
        word_total = sum(len(text.split()) for text in references)
        character_total = sum(len(text) for text in references)
        wer = sum(seq2seq._edit_distance(reference.split(), prediction.split()) for reference, prediction in zip(references, predictions)) / max(word_total, 1)
        cer = sum(seq2seq._edit_distance(list(reference), list(prediction)) for reference, prediction in zip(references, predictions)) / max(character_total, 1)
        metrics = {"wer": wer, "cer": cer}
        print("Evaluation result\n" + "\n".join(f"  {name.upper():5} {value:.4f}" for name, value in metrics.items()))
        return metrics


def train(output_folder: Union[str, Path], train_data: Sequence[Sequence[str]], validation_data: Sequence[Sequence[str]], *, model_type: str = "rnn", sample_rate: Optional[int] = None, n_mels: int = 80, hidden_size: int = 256, layers: int = 3, dropout: float = 0.2, min_frequency: int = 1, epochs: int = 20, batch_size: int = 16, learning_rate: float = 0.001, patience: int = 3, min_delta: float = 0.0, device: Optional[Union[str, torch.device]] = None) -> Audio2TextModel:
    """Train a speech recognizer from ``[audio_path, transcription]`` pairs."""
    folder = Path(output_folder)
    if folder.exists() and any(folder.iterdir()): raise ValueError(f"Output folder must be empty: {folder}. Use continue_train for an existing model.")
    folder.mkdir(parents=True, exist_ok=True)
    pairs = _audio_pairs(train_data)
    selected_rate = sample_rate or infer_sample_rate([path for path, _ in pairs])
    if selected_rate < 4000 or n_mels < 8: raise ValueError("sample_rate must be at least 4000 and n_mels at least 8.")
    print(f"Using {selected_rate} Hz audio and {n_mels} log-mel features.")
    config = _Config(model_type.lower(), selected_rate, n_mels, hidden_size, layers, dropout, min_frequency)
    characters = _Characters.build([text for _, text in pairs], min_frequency)
    model = Audio2TextModel(_network(config, characters), characters, config, folder, device)
    (folder / "training_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    return model.fit(pairs, validation_data, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, patience=patience, min_delta=min_delta)


def load_model(path: Union[str, Path], *, device: Optional[Union[str, torch.device]] = None) -> Audio2TextModel:
    """Load the newest model in a folder, or a supplied checkpoint file."""
    path = Path(path); checkpoint_path = path / "model_latest.pt" if path.is_dir() else path
    if not checkpoint_path.is_file(): raise FileNotFoundError(f"No model checkpoint found at {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
    config = _Config(**checkpoint["config"]); characters = _Characters(checkpoint["characters"])
    model = Audio2TextModel(_network(config, characters), characters, config, checkpoint_path.parent, device)
    model.network.load_state_dict(checkpoint["state_dict"]); model.epoch = checkpoint.get("epoch", 0)
    return model


def continue_train(output_folder: Union[str, Path], train_data: Sequence[Sequence[str]], validation_data: Sequence[Sequence[str]], **kwargs) -> Audio2TextModel:
    """Resume speech-recognition training from a model folder."""
    folder = Path(output_folder); checkpoint = torch.load(folder / "model_latest.pt", map_location=kwargs.get("device", "cpu"))
    model = load_model(folder, device=kwargs.pop("device", None))
    return model.fit(train_data, validation_data, optimizer_state=checkpoint.get("optimizer"), **kwargs)
