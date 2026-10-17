"""Friendly Hugging Face BERT-style foundation models.

This module supports BERT, RoBERTa, and compatible encoder-only Hugging Face
models.  It deliberately excludes decoder-only (GPT-style) models.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .word_embeddings import _affinity_propagation, _dbscan, _read_files

_MANIFEST = "mltools_foundation.json"
_DEFAULTS = {"bert": "bert-base-uncased", "roberta": "roberta-base"}


def _transformers():
    try:
        import transformers
    except ImportError as exc:
        raise ImportError(
            "Foundation models need Hugging Face Transformers. Install MLTools with "
            "`pip install transformers`, then try again."
        ) from exc
    return transformers


def _device(device: Optional[Union[str, torch.device]]) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


class _Texts(Dataset):
    def __init__(self, texts: Sequence[str]) -> None: self.texts = list(texts)
    def __len__(self) -> int: return len(self.texts)
    def __getitem__(self, index: int) -> str: return self.texts[index]


class FoundationModel:
    """An encoder-only Hugging Face model with sentence-embedding helpers."""
    def __init__(self, encoder, tokenizer, *, folder: Optional[Union[str, Path]] = None,
                 classifier=None, labels: Optional[List[str]] = None, task: Optional[str] = None,
                 device: Optional[Union[str, torch.device]] = None) -> None:
        self.encoder, self.tokenizer = encoder, tokenizer
        self.classifier, self.labels, self.task = classifier, labels, task
        self.folder = Path(folder) if folder is not None else None
        self.device = _device(device)
        self.encoder.to(self.device).eval()
        if self.classifier is not None: self.classifier.to(self.device).eval()

    def _encode(self, texts: Sequence[str], max_length: int = 256):
        if not all(isinstance(text, str) for text in texts):
            raise TypeError("texts must be strings.")
        return self.tokenizer(list(texts), padding=True, truncation=True, max_length=max_length,
                              return_tensors="pt").to(self.device)

    def embed(self, text: Union[str, Sequence[str]], *, batch_size: int = 32, max_length: int = 256):
        """Return mean-pooled sentence embedding(s) as CPU tensors."""
        one = isinstance(text, str)
        texts = [text] if one else list(text)
        if not texts: return torch.empty((0, self.encoder.config.hidden_size))
        vectors = []
        self.encoder.eval()
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                inputs = self._encode(texts[start:start + batch_size], max_length)
                hidden = self.encoder(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                vectors.append((hidden * mask).sum(1) / mask.sum(1).clamp_min(1))
        result = torch.cat(vectors).cpu()
        return result[0] if one else result

    def similarity(self, first: str, second: str) -> float:
        """Return cosine similarity between two sentences."""
        vectors = self.embed([first, second])
        return float(F.cosine_similarity(vectors[0], vectors[1], dim=0))

    def cluster(self, texts: Sequence[str], *, method: str = "affinity_propagation", preference: Optional[float] = None,
                damping: float = 0.7, max_iter: int = 200, eps: float = 0.25,
                min_samples: int = 2) -> Dict[int, List[str]]:
        """Cluster sentences with built-in affinity propagation or DBSCAN."""
        if not texts: return {}
        vectors = F.normalize(self.embed(list(texts)), dim=1)
        if method == "affinity_propagation":
            labels = _affinity_propagation(vectors @ vectors.T, preference, damping, max_iter)
        elif method == "dbscan":
            labels = _dbscan(1 - vectors @ vectors.T, eps, min_samples)
        else:
            raise ValueError("method must be 'affinity_propagation' or 'dbscan'.")
        clusters: Dict[int, List[str]] = defaultdict(list)
        for label, value in zip(labels, texts): clusters[label].append(value)
        return dict(clusters)

    def fine_tune(self, folder: Union[str, Path], data: Sequence[Sequence[object]], task: str = "classification", *,
                  epochs: int = 3, batch_size: int = 8, learning_rate: float = 2e-5,
                  max_length: int = 256) -> "FoundationModel":
        """Fine-tune a classifier or regressor and save it in ``folder``.

        For ``classification``, data is ``[[text, label], ...]`` where labels
        are short strings or integers.  For ``regression``, use
        ``[[text, number], ...]``.  Pair classification also works with
        ``[[first_text, second_text, label], ...]``.
        """
        transformers = _transformers()
        task = task.lower().replace("-", "_")
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'.")
        if not data: raise ValueError("data cannot be empty.")
        rows = [list(row) for row in data]
        if any(len(row) not in {2, 3} for row in rows):
            raise ValueError("Each item must be [text, label] or [text, second_text, label].")
        paired = len(rows[0]) == 3
        if any((len(row) == 3) != paired for row in rows): raise ValueError("Use one consistent data format.")
        texts = [row[0] for row in rows]; second = [row[1] for row in rows] if paired else None
        targets = [row[-1] for row in rows]
        if not all(isinstance(value, str) for value in texts) or (second and not all(isinstance(value, str) for value in second)):
            raise TypeError("Training texts must be strings.")
        if task == "classification":
            labels = sorted({str(value) for value in targets})
            if len(labels) < 2: raise ValueError("Classification needs at least two labels.")
            label_ids = {label: index for index, label in enumerate(labels)}
            targets_tensor = torch.tensor([label_ids[str(value)] for value in targets])
            problem_type = "single_label_classification"
        else:
            labels = None; targets_tensor = torch.tensor([float(value) for value in targets]); problem_type = "regression"
        source = str(self.folder) if self.folder is not None else self.encoder.config._name_or_path
        classifier = transformers.AutoModelForSequenceClassification.from_pretrained(
            source, num_labels=len(labels) if labels else 1, problem_type=problem_type,
            ignore_mismatched_sizes=True,
        ).to(self.device)
        classifier.train(); optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate)
        order = list(range(len(rows)))
        for epoch in range(epochs):
            random.shuffle(order)
            progress = tqdm(range(0, len(order), batch_size), desc=f"Fine-tuning epoch {epoch + 1}/{epochs}", unit="batch")
            for start in progress:
                indices = order[start:start + batch_size]
                first = [texts[i] for i in indices]
                second_batch = [second[i] for i in indices] if second else None
                inputs = self.tokenizer(first, second_batch, padding=True, truncation=True, max_length=max_length,
                                        return_tensors="pt").to(self.device)
                labels_batch = targets_tensor[indices].to(self.device)
                loss = classifier(**inputs, labels=labels_batch).loss
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                progress.set_postfix(loss=f"{loss.item():.3f}")
        output = Path(folder); output.mkdir(parents=True, exist_ok=True)
        classifier.save_pretrained(output); self.tokenizer.save_pretrained(output)
        _write_manifest(output, task, labels)
        # Update this object as well as returning it, so both concise styles
        # work: ``model.fine_tune(...); model.predict(...)`` and assignment.
        self.encoder, self.classifier = classifier.base_model, classifier
        self.folder, self.labels, self.task = output, labels, task
        self.encoder.to(self.device).eval(); self.classifier.to(self.device).eval()
        return self

    def predict(self, text: Union[str, Sequence[str]], *, max_length: int = 256):
        """Classify text after :meth:`fine_tune`; accepts one text or many."""
        if self.classifier is None or self.task is None:
            raise RuntimeError("predict is available after fine_tune(..., task='classification' or 'regression').")
        one = isinstance(text, str); texts = [text] if one else list(text)
        self.classifier.eval()
        with torch.no_grad(): logits = self.classifier(**self._encode(texts, max_length)).logits.cpu()
        if self.task == "regression":
            result = logits.squeeze(-1).tolist()
        else:
            result = [self.labels[index] for index in logits.argmax(dim=1).tolist()]
        return result[0] if one else result


def _write_manifest(folder: Path, task: Optional[str] = None, labels: Optional[List[str]] = None) -> None:
    (folder / _MANIFEST).write_text(json.dumps({"task": task, "labels": labels}, indent=2), encoding="utf-8")


def train(folder: Union[str, Path], *, text: Optional[str] = None,
          files: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None, model_type: str = "bert",
          model_name: Optional[str] = None, epochs: int = 1, batch_size: int = 8, learning_rate: float = 5e-5,
          max_length: int = 128, device: Optional[Union[str, torch.device]] = None) -> FoundationModel:
    """Continue masked-language-model training on a text corpus and save it.

    ``model_type`` selects ``"bert"`` (default) or ``"roberta"``. Pass
    ``model_name`` for any compatible Hugging Face encoder model or local path.
    Provide ``text``, ``files``, or both; folders are scanned recursively.
    """
    if text is None and files is None: raise ValueError("Provide text, files, or both.")
    if text is not None and not isinstance(text, str): raise TypeError("text must be a string.")
    if epochs < 1 or batch_size < 1 or max_length < 8: raise ValueError("epochs and batch_size must be positive; max_length must be at least 8.")
    model_type = model_type.lower().replace("-", "")
    if model_name is None:
        if model_type not in _DEFAULTS: raise ValueError("model_type must be 'bert' or 'roberta', or provide model_name.")
        model_name = _DEFAULTS[model_type]
    corpus = "\n".join(part for part in (text or "", _read_files(files) if files is not None else "") if part)
    # A one-line corpus is common. Split it into manageable spans so training
    # uses all of it instead of silently retaining only the first max_length
    # tokens during tokenizer truncation.
    span = max_length * 4
    texts = [line[index:index + span] for line in corpus.splitlines() if line.strip()
             for index in range(0, len(line), span)]
    if not texts: raise ValueError("The supplied corpus contains no text.")
    transformers = _transformers(); target_device = _device(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mlm = transformers.AutoModelForMaskedLM.from_pretrained(model_name).to(target_device)
    optimizer = torch.optim.AdamW(mlm.parameters(), lr=learning_rate)
    loader = DataLoader(_Texts(texts), batch_size=batch_size, shuffle=True)
    mlm.train()
    for epoch in range(epochs):
        progress = tqdm(loader, desc=f"Foundation epoch {epoch + 1}/{epochs}", unit="batch")
        for batch in progress:
            inputs = tokenizer(list(batch), padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(target_device)
            labels = inputs["input_ids"].clone()
            probability = torch.full(labels.shape, 0.15, device=target_device)
            special = [tokenizer.get_special_tokens_mask(row.tolist(), already_has_special_tokens=True) for row in labels]
            probability.masked_fill_(torch.tensor(special, dtype=torch.bool, device=target_device), 0)
            masked = torch.bernoulli(probability).bool(); labels[~masked] = -100
            replacement = torch.bernoulli(torch.full(labels.shape, 0.8, device=target_device)).bool() & masked
            inputs["input_ids"][replacement] = tokenizer.mask_token_id
            random_words = torch.randint(len(tokenizer), labels.shape, device=target_device)
            random_mask = masked & ~replacement & torch.bernoulli(torch.full(labels.shape, 0.5, device=target_device)).bool()
            inputs["input_ids"][random_mask] = random_words[random_mask]
            loss = mlm(**inputs, labels=labels).loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            progress.set_postfix(loss=f"{loss.item():.3f}")
    output = Path(folder); output.mkdir(parents=True, exist_ok=True)
    mlm.save_pretrained(output); tokenizer.save_pretrained(output); _write_manifest(output)
    encoder = transformers.AutoModel.from_pretrained(output)
    return FoundationModel(encoder, tokenizer, folder=output, device=target_device)


def load_model(folder_or_model: Union[str, Path], *, device: Optional[Union[str, torch.device]] = None) -> FoundationModel:
    """Load a saved MLTools model, a local Hugging Face model, or a Hub model ID."""
    transformers = _transformers(); source = str(folder_or_model); path = Path(source)
    tokenizer = transformers.AutoTokenizer.from_pretrained(source, use_fast=True)
    manifest = path / _MANIFEST
    metadata = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {"task": None, "labels": None}
    if metadata["task"]:
        classifier = transformers.AutoModelForSequenceClassification.from_pretrained(source)
        return FoundationModel(classifier.base_model, tokenizer, folder=path if path.exists() else None,
                               classifier=classifier, labels=metadata["labels"], task=metadata["task"], device=device)
    encoder = transformers.AutoModel.from_pretrained(source)
    return FoundationModel(encoder, tokenizer, folder=path if path.exists() else None, device=device)
