"""Simple text-to-speech training and synthesis.

Train from ``[audio_path, text]`` pairs.  A compact attention-based acoustic
model predicts mel spectrograms, then Griffin--Lim converts them to audible
waveforms.  This is a useful, easy-to-understand baseline for a single voice
or a consistent small speech dataset; it is not a replacement for a large
pretrained neural vocoder.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torchaudio
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from . import audio2text, seq2seq

Pair = Tuple[str, str]
split = seq2seq.split


def _pairs(corpus: Sequence[Sequence[str]], *, check_files: bool = True) -> List[Pair]:
    pairs = seq2seq._pairs(corpus)
    if check_files:
        missing = [path for path, _ in pairs if not Path(path).is_file()]
        if missing: raise FileNotFoundError(f"Could not find {len(missing)} audio file(s); for example: {missing[0]}")
    return pairs


class _Characters:
    def __init__(self, characters: List[str]) -> None:
        self.characters, self.ids = characters, {character: index for index, character in enumerate(characters)}

    @classmethod
    def build(cls, texts: Sequence[str], min_frequency: int) -> "_Characters":
        from collections import Counter
        counts = Counter(character for text in texts for character in text)
        return cls(["<pad>", "<unknown>"] + sorted(character for character, count in counts.items() if count >= min_frequency))

    @property
    def pad_id(self) -> int: return 0
    @property
    def unknown_id(self) -> int: return 1
    def encode(self, text: str) -> List[int]: return [self.ids.get(character, self.unknown_id) for character in text]


def _mel_features(path: str, sample_rate: int, transform: nn.Module) -> Tensor:
    waveform, original_rate = torchaudio.load(path)
    waveform = waveform.mean(dim=0, keepdim=True)
    if original_rate != sample_rate: waveform = torchaudio.functional.resample(waveform, original_rate, sample_rate)
    return transform(waveform).squeeze(0).transpose(0, 1).clamp_min(1e-5).log()


class _SpeechData(Dataset):
    def __init__(self, pairs: Sequence[Pair], characters: _Characters, sample_rate: int, n_mels: int, n_fft: int, hop_length: int) -> None:
        self.pairs, self.characters, self.sample_rate = list(pairs), characters, sample_rate
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=2.0)

    def __len__(self) -> int: return len(self.pairs)
    def __getitem__(self, index: int):
        path, text = self.pairs[index]
        encoded = self.characters.encode(text)
        if not encoded: raise ValueError(f"Text cannot be empty: {path}")
        return torch.tensor(encoded), _mel_features(path, self.sample_rate, self.mel)


def _collate(items, pad_id: int):
    texts, mels = zip(*items)
    text_lengths = torch.tensor([len(text) for text in texts], dtype=torch.long)
    mel_lengths = torch.tensor([len(mel) for mel in mels], dtype=torch.long)
    return nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=pad_id), text_lengths, nn.utils.rnn.pad_sequence(mels, batch_first=True), mel_lengths


class _Attention(nn.Module):
    def __init__(self, encoder_size: int, decoder_size: int) -> None:
        super().__init__()
        self.key, self.query, self.energy = nn.Linear(encoder_size, decoder_size, False), nn.Linear(decoder_size, decoder_size, False), nn.Linear(decoder_size, 1, False)

    def forward(self, decoder_state: Tensor, encoder_values: Tensor, mask: Tensor) -> Tensor:
        scores = self.energy(torch.tanh(self.key(encoder_values) + self.query(decoder_state).unsqueeze(1))).squeeze(-1)
        return torch.bmm(F.softmax(scores.masked_fill(~mask, -1e9), 1).unsqueeze(1), encoder_values).squeeze(1)


class _TacotronLite(nn.Module):
    """A small attention encoder-decoder that predicts one mel frame at a time."""
    def __init__(self, vocabulary_size: int, pad_id: int, n_mels: int, embedding_size: int, hidden_size: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.layers, self.hidden_size = layers, hidden_size
        self.embedding = nn.Embedding(vocabulary_size, embedding_size, padding_idx=pad_id)
        self.encoder = nn.GRU(embedding_size, hidden_size, layers, batch_first=True, bidirectional=True, dropout=dropout if layers > 1 else 0)
        self.bridge = nn.Linear(hidden_size * 2, hidden_size)
        self.attention = _Attention(hidden_size * 2, hidden_size)
        self.decoder = nn.GRU(n_mels + hidden_size * 2, hidden_size, layers, batch_first=True, dropout=dropout if layers > 1 else 0)
        self.mel_output = nn.Linear(hidden_size + hidden_size * 2, n_mels)
        self.stop_output = nn.Linear(hidden_size + hidden_size * 2, 1)

    def encode(self, texts: Tensor, lengths: Tensor):
        packed = pack_padded_sequence(self.embedding(texts), lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_values, hidden = self.encoder(packed)
        values, _ = pad_packed_sequence(packed_values, batch_first=True, total_length=texts.size(1))
        hidden = torch.tanh(self.bridge(torch.cat((hidden[0::2], hidden[1::2]), -1)))
        return values, hidden, texts.ne(0)

    def decode_step(self, previous_mel: Tensor, hidden: Tensor, values: Tensor, mask: Tensor):
        context = self.attention(hidden[-1], values, mask)
        decoded, hidden = self.decoder(torch.cat((previous_mel, context), -1).unsqueeze(1), hidden)
        joined = torch.cat((decoded[:, 0], context), -1)
        return self.mel_output(joined), self.stop_output(joined).squeeze(-1), hidden

    def forward(self, texts: Tensor, lengths: Tensor, mel_inputs: Tensor):
        values, hidden, mask = self.encode(texts, lengths)
        frames, stops = [], []
        for frame in mel_inputs.unbind(1):
            mel, stop, hidden = self.decode_step(frame, hidden, values, mask)
            frames.append(mel); stops.append(stop)
        return torch.stack(frames, 1), torch.stack(stops, 1)


@dataclass
class _Config:
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    embedding_size: int = 256
    hidden_size: int = 512
    layers: int = 2
    dropout: float = 0.2
    min_frequency: int = 1


class GeneratedAudio:
    """Generated mono audio, with a convenient :meth:`save` method."""
    def __init__(self, waveform: Tensor, sample_rate: int) -> None:
        self.waveform, self.sample_rate = waveform.cpu(), sample_rate

    def save(self, path: Union[str, Path]) -> Path:
        """Save this audio as a WAV file and return its path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(path), self.waveform, self.sample_rate)
        return path


class Text2AudioModel:
    """A trained text-to-speech model returned by :func:`train`."""
    def __init__(self, network: _TacotronLite, characters: _Characters, config: _Config, folder: Union[str, Path], device: Optional[Union[str, torch.device]] = None) -> None:
        self.network, self.characters, self.config, self.folder = network, characters, config, Path(folder)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.network.to(self.device); self.epoch = 0
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=config.sample_rate, n_mels=config.n_mels, n_fft=config.n_fft, hop_length=config.hop_length, power=2.0)
        self.inverse_mel = torchaudio.transforms.InverseMelScale(n_stft=config.n_fft // 2 + 1, n_mels=config.n_mels, sample_rate=config.sample_rate)
        self.griffin_lim = torchaudio.transforms.GriffinLim(n_fft=config.n_fft, hop_length=config.hop_length, power=2.0)

    @property
    def sample_rate(self) -> int: return self.config.sample_rate

    def _loader(self, data: Sequence[Sequence[str]], batch_size: int, shuffle: bool) -> DataLoader:
        dataset = _SpeechData(_pairs(data), self.characters, self.config.sample_rate, self.config.n_mels, self.config.n_fft, self.config.hop_length)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda items: _collate(items, self.characters.pad_id))

    def _loss(self, texts: Tensor, text_lengths: Tensor, target_mels: Tensor, mel_lengths: Tensor) -> Tensor:
        # A zero frame starts the autoregressive decoder; teacher forcing uses prior true frames.
        inputs = torch.cat((torch.zeros_like(target_mels[:, :1]), target_mels[:, :-1]), 1)
        prediction, stops = self.network(texts, text_lengths, inputs)
        mask = torch.arange(target_mels.size(1), device=target_mels.device).unsqueeze(0) < mel_lengths.to(target_mels.device).unsqueeze(1)
        mel_loss = ((prediction - target_mels).square() * mask.unsqueeze(-1)).sum() / (mask.sum() * target_mels.size(-1)).clamp_min(1)
        stop_targets = torch.zeros_like(stops)
        stop_targets[torch.arange(len(mel_lengths), device=stops.device), mel_lengths.to(stops.device) - 1] = 1
        stop_loss = F.binary_cross_entropy_with_logits(stops[mask], stop_targets[mask])
        return mel_loss + stop_loss

    def _validation_loss(self, data: Sequence[Sequence[str]], batch_size: int) -> float:
        self.network.eval(); total = count = 0
        with torch.no_grad():
            for texts, text_lengths, mels, mel_lengths in self._loader(data, batch_size, False):
                loss = self._loss(texts.to(self.device), text_lengths, mels.to(self.device), mel_lengths)
                total += loss.item() * len(texts); count += len(texts)
        return total / max(count, 1)

    def _save(self, optimizer: torch.optim.Optimizer, validation_loss: float, *, best: bool) -> None:
        state = {"state_dict": self.network.state_dict(), "optimizer": optimizer.state_dict(), "epoch": self.epoch, "validation_loss": validation_loss, "config": asdict(self.config), "characters": self.characters.characters}
        torch.save(state, self.folder / "model_latest.pt")
        torch.save(state, self.folder / f"checkpoint_epoch_{self.epoch:03d}.pt")
        if best: torch.save(state, self.folder / "model_best.pt")

    def fit(self, train_data: Sequence[Sequence[str]], validation_data: Sequence[Sequence[str]], *, epochs: int = 20, batch_size: int = 8, learning_rate: float = 0.001, patience: int = 3, min_delta: float = 0.0, optimizer_state: Optional[dict] = None) -> "Text2AudioModel":
        if not validation_data: raise ValueError("validation_data cannot be empty because it is used for early stopping.")
        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        if optimizer_state: optimizer.load_state_dict(optimizer_state)
        best_loss, stalled = float("inf"), 0
        for _ in range(epochs):
            self.epoch += 1; self.network.train(); running = 0.0
            progress = tqdm(self._loader(train_data, batch_size, True), desc=f"Epoch {self.epoch}", unit="batch")
            for texts, text_lengths, mels, mel_lengths in progress:
                optimizer.zero_grad(); loss = self._loss(texts.to(self.device), text_lengths, mels.to(self.device), mel_lengths)
                loss.backward(); torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0); optimizer.step()
                running += loss.item(); progress.set_postfix(loss=f"{running / max(progress.n, 1):.3f}")
            validation_loss = self._validation_loss(validation_data, batch_size)
            improved = validation_loss < best_loss - min_delta
            best_loss, stalled = (validation_loss, 0) if improved else (best_loss, stalled + 1)
            self._save(optimizer, validation_loss, best=improved)
            print(f"Epoch {self.epoch}: validation loss {validation_loss:.4f}" + (" (best)" if improved else ""))
            if stalled >= patience:
                print(f"Early stopping after {self.epoch} epochs (no validation improvement for {patience} epochs).")
                break
        return self

    def _predict_mel(self, text: str, max_frames: int) -> Tensor:
        encoded = self.characters.encode(text)
        if not encoded: raise ValueError("Text cannot be empty.")
        texts = torch.tensor([encoded], device=self.device); lengths = torch.tensor([len(encoded)])
        self.network.eval()
        with torch.no_grad():
            values, hidden, mask = self.network.encode(texts, lengths)
            previous = torch.zeros(1, self.config.n_mels, device=self.device)
            frames = []
            for index in range(max_frames):
                frame, stop, hidden = self.network.decode_step(previous, hidden, values, mask)
                frames.append(frame); previous = frame
                # Do not stop too early: a few frames are necessary for audible output.
                if index >= 10 and torch.sigmoid(stop).item() > 0.5: break
        # InverseMelScale expects [mel_bins, frames] for one recording.
        return torch.stack(frames, dim=1)[0].transpose(0, 1).exp().cpu()

    def synthesize(self, text: Union[str, Sequence[str]], *, output_path: Optional[Union[str, Path]] = None, max_frames: int = 1000):
        """Create speech from one text or a list of texts.

        Results are :class:`GeneratedAudio` objects.  Set ``output_path`` for
        one text to save it immediately, or call ``result.save("speech.wav")``.
        """
        texts, one = ([text], True) if isinstance(text, str) else (list(text), False)
        if not texts or not all(isinstance(item, str) for item in texts): raise TypeError("synthesize expects one text or a non-empty sequence of texts.")
        if output_path is not None and not one: raise ValueError("output_path can only be used when synthesizing one text.")
        results = []
        for item in texts:
            mel = self._predict_mel(item, max_frames)
            waveform = self.griffin_lim(self.inverse_mel(mel)).unsqueeze(0)
            results.append(GeneratedAudio(waveform, self.sample_rate))
        if output_path is not None: results[0].save(output_path)
        return results[0] if one else results

    def evaluate(self, test_data: Sequence[Sequence[str]], *, max_items: Optional[int] = None) -> Dict[str, float]:
        """Print and return average validation-style mel loss on held-out speech."""
        pairs = _pairs(test_data)
        if max_items is not None: pairs = pairs[:max_items]
        loss = self._validation_loss(pairs, batch_size=1)
        metrics = {"mel_loss": loss}
        print(f"Evaluation result\n  MEL_LOSS  {loss:.4f}")
        return metrics


def _network(config: _Config, characters: _Characters) -> _TacotronLite:
    return _TacotronLite(len(characters.characters), characters.pad_id, config.n_mels, config.embedding_size, config.hidden_size, config.layers, config.dropout)


def train(output_folder: Union[str, Path], train_data: Sequence[Sequence[str]], validation_data: Sequence[Sequence[str]], *, sample_rate: Optional[int] = None, n_mels: int = 80, n_fft: int = 1024, hop_length: int = 256, embedding_size: int = 256, hidden_size: int = 512, layers: int = 2, dropout: float = 0.2, min_frequency: int = 1, epochs: int = 20, batch_size: int = 8, learning_rate: float = 0.001, patience: int = 3, min_delta: float = 0.0, device: Optional[Union[str, torch.device]] = None) -> Text2AudioModel:
    """Train a simple text-to-speech model from ``[audio_path, text]`` pairs."""
    folder = Path(output_folder)
    if folder.exists() and any(folder.iterdir()): raise ValueError(f"Output folder must be empty: {folder}. Use continue_train for an existing model.")
    folder.mkdir(parents=True, exist_ok=True)
    pairs = _pairs(train_data)
    selected_rate = sample_rate or audio2text.infer_sample_rate([path for path, _ in pairs])
    if n_mels < 8 or n_fft < 64 or hop_length < 16: raise ValueError("n_mels must be at least 8, n_fft at least 64, and hop_length at least 16.")
    print(f"Using {selected_rate} Hz audio and {n_mels} mel features.")
    config = _Config(selected_rate, n_mels, n_fft, hop_length, embedding_size, hidden_size, layers, dropout, min_frequency)
    characters = _Characters.build([text for _, text in pairs], min_frequency)
    model = Text2AudioModel(_network(config, characters), characters, config, folder, device)
    (folder / "training_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    return model.fit(pairs, validation_data, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, patience=patience, min_delta=min_delta)


def load_model(path: Union[str, Path], *, device: Optional[Union[str, torch.device]] = None) -> Text2AudioModel:
    """Load the latest model from a folder or a selected checkpoint file."""
    path = Path(path); checkpoint_path = path / "model_latest.pt" if path.is_dir() else path
    if not checkpoint_path.is_file(): raise FileNotFoundError(f"No model checkpoint found at {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
    config = _Config(**checkpoint["config"]); characters = _Characters(checkpoint["characters"])
    model = Text2AudioModel(_network(config, characters), characters, config, checkpoint_path.parent, device)
    model.network.load_state_dict(checkpoint["state_dict"]); model.epoch = checkpoint.get("epoch", 0)
    return model


def continue_train(output_folder: Union[str, Path], train_data: Sequence[Sequence[str]], validation_data: Sequence[Sequence[str]], **kwargs) -> Text2AudioModel:
    """Resume text-to-speech training from an existing model folder."""
    folder = Path(output_folder); checkpoint = torch.load(folder / "model_latest.pt", map_location=kwargs.get("device", "cpu"))
    model = load_model(folder, device=kwargs.pop("device", None))
    return model.fit(train_data, validation_data, optimizer_state=checkpoint.get("optimizer"), **kwargs)
