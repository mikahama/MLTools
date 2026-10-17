"""A small, friendly text-to-text modelling API built on PyTorch.

Typical use::

    from MLTools import seq2seq
    train, validation, test = seq2seq.split(corpus)
    model = seq2seq.train("translation-model", train, validation)
    model.predict("Kissa juoksee")

Text is tokenised on whitespace by default.  Supply ``tokenizer`` and
``detokenizer`` to :func:`train` when that is not appropriate for your data.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

Pair = Tuple[str, str]
Tokenize = Callable[[str], List[str]]
Detokenize = Callable[[List[str]], str]

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"


def _default_tokenizer(text: str) -> List[str]:
    return text.strip().split()


def _default_detokenizer(tokens: List[str]) -> str:
    return " ".join(tokens)


def _pairs(corpus: Sequence[Sequence[str]]) -> List[Pair]:
    result: List[Pair] = []
    for index, item in enumerate(corpus):
        if len(item) != 2:
            raise ValueError(f"Corpus item {index} must contain exactly [source, target].")
        source, target = item
        if not isinstance(source, str) or not isinstance(target, str):
            raise TypeError(f"Corpus item {index} must contain two strings.")
        result.append((source, target))
    if not result:
        raise ValueError("The corpus must contain at least one source-target pair.")
    return result


def split(
    corpus: Sequence[Sequence[str]],
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    *,
    seed: Optional[int] = 42,
    shuffle: bool = True,
) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    """Split paired text data into train, validation, and test sets.

    Ratios must add up to one.  A fixed default seed makes the split
    reproducible; pass ``seed=None`` for a fresh random split.
    """
    records = _pairs(corpus)
    if any(r < 0 for r in (train_ratio, validation_ratio, test_ratio)) or not math.isclose(
        train_ratio + validation_ratio + test_ratio, 1.0, abs_tol=1e-8
    ):
        raise ValueError("train_ratio, validation_ratio, and test_ratio must be non-negative and add up to 1.")
    records = records.copy()
    if shuffle:
        random.Random(seed).shuffle(records)
    train_end = int(len(records) * train_ratio)
    validation_end = train_end + int(len(records) * validation_ratio)
    as_lists = lambda xs: [[source, target] for source, target in xs]
    return as_lists(records[:train_end]), as_lists(records[train_end:validation_end]), as_lists(records[validation_end:])


class Vocabulary:
    def __init__(self, tokens: Optional[List[str]] = None) -> None:
        self.itos = tokens or [PAD, BOS, EOS, UNK]
        self.stoi = {token: i for i, token in enumerate(self.itos)}

    @classmethod
    def build(cls, texts: Sequence[str], tokenizer: Tokenize, min_frequency: int = 1) -> "Vocabulary":
        counts: Counter[str] = Counter(token for text in texts for token in tokenizer(text))
        words = sorted(token for token, count in counts.items() if count >= min_frequency and token not in {PAD, BOS, EOS, UNK})
        return cls([PAD, BOS, EOS, UNK] + words)

    @property
    def pad_id(self) -> int: return 0
    @property
    def bos_id(self) -> int: return 1
    @property
    def eos_id(self) -> int: return 2
    @property
    def unk_id(self) -> int: return 3

    def encode(self, text: str, tokenizer: Tokenize, *, add_bos: bool = False, add_eos: bool = True) -> List[int]:
        ids = [self.stoi.get(token, self.unk_id) for token in tokenizer(text)]
        return ([self.bos_id] if add_bos else []) + ids + ([self.eos_id] if add_eos else [])

    def decode(self, ids: Sequence[int]) -> List[str]:
        special = {self.pad_id, self.bos_id, self.eos_id}
        return [self.itos[token] if 0 <= token < len(self.itos) else UNK for token in ids if token not in special]


class _TextPairs(Dataset):
    def __init__(self, pairs: Sequence[Pair], src_vocab: Vocabulary, tgt_vocab: Vocabulary, tokenizer: Tokenize) -> None:
        self.items = [(src_vocab.encode(s, tokenizer), tgt_vocab.encode(t, tokenizer, add_bos=True)) for s, t in pairs]

    def __len__(self) -> int: return len(self.items)
    def __getitem__(self, index: int): return self.items[index]


def _collate(items, pad_id: int):
    sources, targets = zip(*items)
    src = nn.utils.rnn.pad_sequence([torch.tensor(x) for x in sources], batch_first=True, padding_value=pad_id)
    tgt = nn.utils.rnn.pad_sequence([torch.tensor(x) for x in targets], batch_first=True, padding_value=pad_id)
    return src, torch.tensor([len(x) for x in sources]), tgt


class _Attention(nn.Module):
    def __init__(self, encoder_size: int, decoder_size: int) -> None:
        super().__init__()
        self.keys = nn.Linear(encoder_size, decoder_size, bias=False)
        self.query = nn.Linear(decoder_size, decoder_size, bias=False)
        self.energy = nn.Linear(decoder_size, 1, bias=False)

    def forward(self, query: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        scores = self.energy(torch.tanh(self.keys(values) + self.query(query).unsqueeze(1))).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        return torch.bmm(F.softmax(scores, dim=1).unsqueeze(1), values).squeeze(1)


class _RNNSeq2Seq(nn.Module):
    def __init__(self, src_size: int, tgt_size: int, embedding_size: int, hidden_size: int, layers: int, dropout: float, pad_id: int) -> None:
        super().__init__()
        self.pad_id, self.hidden_size = pad_id, hidden_size
        self.src_embedding = nn.Embedding(src_size, embedding_size, padding_idx=pad_id)
        self.tgt_embedding = nn.Embedding(tgt_size, embedding_size, padding_idx=pad_id)
        self.encoder = nn.GRU(embedding_size, hidden_size, layers, batch_first=True, bidirectional=True, dropout=dropout if layers > 1 else 0)
        self.bridge = nn.Linear(hidden_size * 2, hidden_size)
        self.attention = _Attention(hidden_size * 2, hidden_size)
        self.decoder = nn.GRU(embedding_size + hidden_size * 2, hidden_size, layers, batch_first=True, dropout=dropout if layers > 1 else 0)
        self.output = nn.Linear(hidden_size * 3, tgt_size)

    def encode(self, source: Tensor, lengths: Tensor):
        packed = pack_padded_sequence(self.src_embedding(source), lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.encoder(packed)
        values, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=source.size(1))
        # A bidirectional state is combined for every decoder layer.
        hidden = torch.tanh(self.bridge(torch.cat((hidden[0::2], hidden[1::2]), dim=-1)))
        return values, hidden, source.ne(self.pad_id)

    def decode_step(self, previous: Tensor, hidden: Tensor, values: Tensor, source_mask: Tensor):
        context = self.attention(hidden[-1], values, source_mask)
        embedded = self.tgt_embedding(previous).unsqueeze(1)
        output, hidden = self.decoder(torch.cat((embedded, context.unsqueeze(1)), dim=-1), hidden)
        logits = self.output(torch.cat((output.squeeze(1), context), dim=-1))
        return logits, hidden

    def forward(self, source: Tensor, lengths: Tensor, target_input: Tensor) -> Tensor:
        values, hidden, mask = self.encode(source, lengths)
        outputs = []
        for i in range(target_input.size(1)):
            logits, hidden = self.decode_step(target_input[:, i], hidden, values, mask)
            outputs.append(logits)
        return torch.stack(outputs, dim=1)


class _TransformerSeq2Seq(nn.Module):
    def __init__(self, src_size: int, tgt_size: int, embedding_size: int, hidden_size: int, layers: int, dropout: float, pad_id: int) -> None:
        super().__init__()
        self.pad_id, self.embedding_size = pad_id, embedding_size
        self.src_embedding = nn.Embedding(src_size, embedding_size, padding_idx=pad_id)
        self.tgt_embedding = nn.Embedding(tgt_size, embedding_size, padding_idx=pad_id)
        self.position = nn.Embedding(2048, embedding_size)
        heads = 8 if embedding_size % 8 == 0 else 4 if embedding_size % 4 == 0 else 1
        self.transformer = nn.Transformer(d_model=embedding_size, nhead=heads, num_encoder_layers=layers, num_decoder_layers=layers, dim_feedforward=hidden_size * 4, dropout=dropout, batch_first=True)
        self.output = nn.Linear(embedding_size, tgt_size)

    def _embed(self, ids: Tensor, embedding: nn.Embedding) -> Tensor:
        positions = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        return embedding(ids) * math.sqrt(self.embedding_size) + self.position(positions)

    def forward(self, source: Tensor, lengths: Tensor, target_input: Tensor) -> Tensor:
        del lengths
        target_mask = nn.Transformer.generate_square_subsequent_mask(target_input.size(1), device=source.device)
        decoded = self.transformer(self._embed(source, self.src_embedding), self._embed(target_input, self.tgt_embedding), tgt_mask=target_mask, src_key_padding_mask=source.eq(self.pad_id), tgt_key_padding_mask=target_input.eq(self.pad_id), memory_key_padding_mask=source.eq(self.pad_id))
        return self.output(decoded)


@dataclass
class _Config:
    model_type: str = "rnn"
    embedding_size: int = 256
    hidden_size: int = 512
    layers: int = 2
    dropout: float = 0.2
    min_frequency: int = 1


class Seq2SeqModel:
    """A trained text-to-text model returned by :func:`train`."""
    def __init__(self, network: nn.Module, src_vocab: Vocabulary, tgt_vocab: Vocabulary, config: _Config, folder: Union[str, Path], tokenizer: Tokenize = _default_tokenizer, detokenizer: Detokenize = _default_detokenizer, device: Optional[Union[str, torch.device]] = None) -> None:
        self.network, self.src_vocab, self.tgt_vocab, self.config = network, src_vocab, tgt_vocab, config
        self.folder, self.tokenizer, self.detokenizer = Path(folder), tokenizer, detokenizer
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.network.to(self.device)
        self.epoch = 0

    def _loader(self, pairs: Sequence[Sequence[str]], batch_size: int, shuffle: bool) -> DataLoader:
        dataset = _TextPairs(_pairs(pairs), self.src_vocab, self.tgt_vocab, self.tokenizer)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: _collate(x, self.src_vocab.pad_id))

    def _loss(self, source: Tensor, lengths: Tensor, target: Tensor) -> Tensor:
        logits = self.network(source, lengths, target[:, :-1])
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), target[:, 1:].reshape(-1), ignore_index=self.tgt_vocab.pad_id)

    def _validation_loss(self, pairs: Sequence[Sequence[str]], batch_size: int) -> float:
        self.network.eval(); total = count = 0
        with torch.no_grad():
            for source, lengths, target in self._loader(pairs, batch_size, False):
                loss = self._loss(source.to(self.device), lengths, target.to(self.device))
                total += loss.item() * source.size(0); count += source.size(0)
        return total / max(count, 1)

    def _save(self, optimizer: torch.optim.Optimizer, val_loss: float, *, best: bool = False) -> Path:
        state = {"state_dict": self.network.state_dict(), "optimizer": optimizer.state_dict(), "epoch": self.epoch, "validation_loss": val_loss, "config": asdict(self.config), "src_vocab": self.src_vocab.itos, "tgt_vocab": self.tgt_vocab.itos}
        latest = self.folder / "model_latest.pt"
        torch.save(state, latest)
        checkpoint = self.folder / f"checkpoint_epoch_{self.epoch:03d}.pt"
        torch.save(state, checkpoint)
        if best:
            torch.save(state, self.folder / "model_best.pt")
        return latest

    def fit(self, train_data: Sequence[Sequence[str]], validation_data: Sequence[Sequence[str]], *, epochs: int = 20, batch_size: int = 32, learning_rate: float = 0.001, patience: int = 3, min_delta: float = 0.0, optimizer_state: Optional[dict] = None) -> "Seq2SeqModel":
        """Train for up to ``epochs`` more epochs, with validation early stopping."""
        if not validation_data:
            raise ValueError("validation_data cannot be empty because it is used for early stopping.")
        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate, betas=(0.9, 0.998))
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        best_loss, stalled = float("inf"), 0
        loader = self._loader(train_data, batch_size, True)
        for _ in range(epochs):
            self.epoch += 1; self.network.train(); running = 0.0
            progress = tqdm(loader, desc=f"Epoch {self.epoch}", unit="batch")
            for source, lengths, target in progress:
                optimizer.zero_grad()
                loss = self._loss(source.to(self.device), lengths, target.to(self.device))
                loss.backward(); torch.nn.utils.clip_grad_norm_(self.network.parameters(), 5.0); optimizer.step()
                running += loss.item(); progress.set_postfix(train_loss=f"{running / (progress.n or 1):.3f}")
            val_loss = self._validation_loss(validation_data, batch_size)
            improved = val_loss < best_loss - min_delta
            if improved: best_loss, stalled = val_loss, 0
            else: stalled += 1
            self._save(optimizer, val_loss, best=improved)
            print(f"Epoch {self.epoch}: validation loss {val_loss:.4f}" + (" (best)" if improved else ""))
            if stalled >= patience:
                print(f"Early stopping after {self.epoch} epochs (no validation improvement for {patience} epochs).")
                break
        return self

    def _next_logits(self, source: Tensor, lengths: Tensor, prefix: List[int]) -> Tensor:
        if self.config.model_type == "rnn":
            values, hidden, mask = self.network.encode(source, lengths)  # type: ignore[attr-defined]
            logits = None
            for token in prefix:
                logits, hidden = self.network.decode_step(torch.tensor([token], device=self.device), hidden, values, mask)  # type: ignore[attr-defined]
            return logits[0]  # type: ignore[index]
        target = torch.tensor([prefix], device=self.device)
        return self.network(source, lengths, target)[0, -1]

    def _decode_one(self, text: str, n_best: int, max_length: int, beam_size: int) -> List[Tuple[str, float]]:
        source_ids = self.src_vocab.encode(text, self.tokenizer)
        source = torch.tensor([source_ids], device=self.device); lengths = torch.tensor([len(source_ids)])
        beams: List[Tuple[List[int], float, bool]] = [([self.tgt_vocab.bos_id], 0.0, False)]
        self.network.eval()
        with torch.no_grad():
            for _ in range(max_length):
                candidates = []
                for prefix, score, ended in beams:
                    if ended: candidates.append((prefix, score, ended)); continue
                    log_probs = F.log_softmax(self._next_logits(source, lengths, prefix), dim=-1)
                    values, ids = torch.topk(log_probs, min(beam_size, log_probs.numel()))
                    candidates += [(prefix + [token.item()], score + value.item(), token.item() == self.tgt_vocab.eos_id) for value, token in zip(values, ids)]
                beams = sorted(candidates, key=lambda x: x[1] / max((len(x[0]) - 1) ** 0.7, 1), reverse=True)[:beam_size]
                if all(done for _, _, done in beams): break
        return [(self.detokenizer(self.tgt_vocab.decode(ids)), score) for ids, score, _ in beams[:n_best]]

    def predict(self, text: Union[str, Sequence[str]], *, n_best: int = 1, return_scores: bool = False, max_length: int = 100, beam_size: Optional[int] = None):
        """Translate one string or a list of strings.

        ``n_best`` returns alternatives.  Set ``return_scores=True`` to return
        ``(text, log_probability)`` pairs.  Beam search is used automatically
        when more than one result is requested.
        """
        if n_best < 1 or max_length < 1: raise ValueError("n_best and max_length must be at least 1.")
        texts, one = ([text], True) if isinstance(text, str) else (list(text), False)
        if not all(isinstance(item, str) for item in texts): raise TypeError("predict expects a string or a sequence of strings.")
        beam_size = beam_size or max(1, n_best)
        decoded = [self._decode_one(item, n_best, max_length, beam_size) for item in texts]
        if n_best == 1:
            result = [pairs[0] if return_scores else pairs[0][0] for pairs in decoded]
        else:
            result = [pairs if return_scores else [sentence for sentence, _ in pairs] for pairs in decoded]
        return result[0] if one else result

    def evaluate(self, test_data: Sequence[Sequence[str]]) -> Dict[str, float]:
        """Print and return token precision, recall, F1, BLEU, and WER."""
        pairs = _pairs(test_data)
        predictions = self.predict([source for source, _ in pairs])
        references = [self.tokenizer(target) for _, target in pairs]
        hypotheses = [self.tokenizer(prediction) for prediction in predictions]
        overlap = sum(sum((Counter(h) & Counter(r)).values()) for h, r in zip(hypotheses, references))
        predicted = sum(map(len, hypotheses)); reference = sum(map(len, references))
        precision = overlap / predicted if predicted else 0.0; recall = overlap / reference if reference else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        wer = sum(_edit_distance(r, h) for r, h in zip(references, hypotheses)) / max(reference, 1)
        bleu = _corpus_bleu(references, hypotheses)
        metrics = {"precision": precision, "recall": recall, "f1": f1, "bleu": bleu, "wer": wer}
        print("Evaluation result\n" + "\n".join(f"  {key.upper():9} {value:.4f}" for key, value in metrics.items()))
        return metrics


def _edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, token in enumerate(reference, 1):
        current = [i]
        for j, other in enumerate(hypothesis, 1): current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (token != other)))
        previous = current
    return previous[-1]


def _corpus_bleu(references: Sequence[List[str]], hypotheses: Sequence[List[str]]) -> float:
    precisions = []
    for order in range(1, 5):
        matched = total = 0
        for reference, hypothesis in zip(references, hypotheses):
            ref_counts = Counter(tuple(reference[i:i + order]) for i in range(max(len(reference) - order + 1, 0)))
            hyp_counts = Counter(tuple(hypothesis[i:i + order]) for i in range(max(len(hypothesis) - order + 1, 0)))
            matched += sum((ref_counts & hyp_counts).values()); total += sum(hyp_counts.values())
        precisions.append((matched + 1) / (total + 1))
    ref_length, hyp_length = sum(map(len, references)), sum(map(len, hypotheses))
    brevity = math.exp(min(0.0, 1 - ref_length / max(hyp_length, 1)))
    return brevity * math.exp(sum(math.log(p) for p in precisions) / 4)


def _network(config: _Config, src_vocab: Vocabulary, tgt_vocab: Vocabulary) -> nn.Module:
    if config.model_type == "rnn": return _RNNSeq2Seq(len(src_vocab.itos), len(tgt_vocab.itos), config.embedding_size, config.hidden_size, config.layers, config.dropout, src_vocab.pad_id)
    if config.model_type == "transformer": return _TransformerSeq2Seq(len(src_vocab.itos), len(tgt_vocab.itos), config.embedding_size, config.hidden_size, config.layers, config.dropout, src_vocab.pad_id)
    raise ValueError("model_type must be 'rnn' (the default) or 'transformer'.")


def train(output_folder: Union[str, Path], train_data: Sequence[Sequence[str]], validation_data: Sequence[Sequence[str]], *, model_type: str = "rnn", embedding_size: int = 256, hidden_size: int = 512, layers: int = 2, dropout: float = 0.2, min_frequency: int = 1, epochs: int = 20, batch_size: int = 32, learning_rate: float = 0.001, patience: int = 3, min_delta: float = 0.0, tokenizer: Tokenize = _default_tokenizer, detokenizer: Detokenize = _default_detokenizer, device: Optional[Union[str, torch.device]] = None) -> Seq2SeqModel:
    """Train a text-to-text model and save checkpoints in a new empty folder."""
    folder = Path(output_folder)
    if folder.exists() and any(folder.iterdir()): raise ValueError(f"Output folder must be empty: {folder}. Use continue_train for an existing model.")
    folder.mkdir(parents=True, exist_ok=True)
    records = _pairs(train_data)
    config = _Config(model_type.lower(), embedding_size, hidden_size, layers, dropout, min_frequency)
    src_vocab = Vocabulary.build([source for source, _ in records], tokenizer, min_frequency)
    tgt_vocab = Vocabulary.build([target for _, target in records], tokenizer, min_frequency)
    model = Seq2SeqModel(_network(config, src_vocab, tgt_vocab), src_vocab, tgt_vocab, config, folder, tokenizer, detokenizer, device)
    (folder / "training_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    return model.fit(records, validation_data, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, patience=patience, min_delta=min_delta)


def load_model(path: Union[str, Path], *, device: Optional[Union[str, torch.device]] = None, tokenizer: Tokenize = _default_tokenizer, detokenizer: Detokenize = _default_detokenizer) -> Seq2SeqModel:
    """Load ``model_latest.pt`` from a folder, or load a checkpoint file."""
    source = Path(path); checkpoint_path = source / "model_latest.pt" if source.is_dir() else source
    if not checkpoint_path.is_file(): raise FileNotFoundError(f"No model checkpoint found at {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
    config = _Config(**checkpoint["config"]); src_vocab, tgt_vocab = Vocabulary(checkpoint["src_vocab"]), Vocabulary(checkpoint["tgt_vocab"])
    model = Seq2SeqModel(_network(config, src_vocab, tgt_vocab), src_vocab, tgt_vocab, config, checkpoint_path.parent, tokenizer, detokenizer, device)
    model.network.load_state_dict(checkpoint["state_dict"]); model.epoch = checkpoint.get("epoch", 0)
    return model


def continue_train(output_folder: Union[str, Path], train_data: Sequence[Sequence[str]], validation_data: Sequence[Sequence[str]], **kwargs) -> Seq2SeqModel:
    """Resume from the latest checkpoint in ``output_folder``."""
    folder = Path(output_folder); checkpoint_path = folder / "model_latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location=kwargs.get("device", "cpu"))
    model = load_model(folder, device=kwargs.pop("device", None), tokenizer=kwargs.pop("tokenizer", _default_tokenizer), detokenizer=kwargs.pop("detokenizer", _default_detokenizer))
    return model.fit(train_data, validation_data, optimizer_state=checkpoint.get("optimizer"), **kwargs)
