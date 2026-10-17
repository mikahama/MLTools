"""Friendly Hugging Face wrappers for GPT-style causal language models."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .foundation_models import _device, _transformers
from .word_embeddings import _affinity_propagation, _dbscan, _read_files

_MANIFEST = "mltools_gpt.json"


class _TokenBlocks(Dataset):
    def __init__(self, token_ids: torch.Tensor, block_size: int) -> None:
        self.blocks = [token_ids[index:index + block_size] for index in range(0, len(token_ids) - 1, block_size)
                       if len(token_ids[index:index + block_size]) > 1]
    def __len__(self) -> int: return len(self.blocks)
    def __getitem__(self, index: int) -> torch.Tensor: return self.blocks[index]


def _collate(blocks: Sequence[torch.Tensor], pad_id: int):
    inputs = torch.nn.utils.rnn.pad_sequence(blocks, batch_first=True, padding_value=pad_id)
    labels = inputs.clone(); labels[labels == pad_id] = -100
    return {"input_ids": inputs, "attention_mask": inputs.ne(pad_id).long(), "labels": labels}


class GPT:
    """A GPT-style causal language model returned by :func:`train`."""
    def __init__(self, model, tokenizer, *, folder: Optional[Union[str, Path]] = None,
                 device: Optional[Union[str, torch.device]] = None) -> None:
        self.model, self.tokenizer = model, tokenizer
        self.folder = Path(folder) if folder is not None else None
        self.device = _device(device)
        self.model.to(self.device).eval()

    def _encode(self, texts: Sequence[str], max_length: int = 256):
        if not all(isinstance(text, str) for text in texts): raise TypeError("texts must be strings.")
        return self.tokenizer(list(texts), padding=True, truncation=True, max_length=max_length,
                              return_tensors="pt").to(self.device)

    def embed(self, text: Union[str, Sequence[str]], *, batch_size: int = 16, max_length: int = 256):
        """Return mean-pooled final-layer representations for one or more texts."""
        one = isinstance(text, str); texts = [text] if one else list(text)
        if not texts: return torch.empty((0, getattr(self.model.config, "hidden_size", getattr(self.model.config, "n_embd", 0))))
        results = []; self.model.eval()
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                inputs = self._encode(texts[start:start + batch_size], max_length)
                output = self.model(**inputs, output_hidden_states=True, use_cache=False)
                hidden = output.hidden_states[-1]; mask = inputs["attention_mask"].unsqueeze(-1)
                results.append((hidden * mask).sum(1) / mask.sum(1).clamp_min(1))
        values = torch.cat(results).cpu()
        return values[0] if one else values

    def similarity(self, first: str, second: str) -> float:
        """Return cosine similarity between two texts."""
        vectors = self.embed([first, second])
        return float(F.cosine_similarity(vectors[0], vectors[1], dim=0))

    def cluster(self, texts: Sequence[str], *, method: str = "affinity_propagation", preference: Optional[float] = None,
                damping: float = 0.7, max_iter: int = 200, eps: float = 0.25,
                min_samples: int = 2) -> Dict[int, List[str]]:
        """Cluster texts with built-in affinity propagation or DBSCAN."""
        if not texts: return {}
        vectors = F.normalize(self.embed(list(texts)), dim=1)
        if method == "affinity_propagation": labels = _affinity_propagation(vectors @ vectors.T, preference, damping, max_iter)
        elif method == "dbscan": labels = _dbscan(1 - vectors @ vectors.T, eps, min_samples)
        else: raise ValueError("method must be 'affinity_propagation' or 'dbscan'.")
        groups: Dict[int, List[str]] = defaultdict(list)
        for label, text in zip(labels, texts): groups[label].append(text)
        return dict(groups)

    def predict(self, prompt: Union[str, Sequence[str]], *, max_new_tokens: int = 80, temperature: float = 0.8,
                top_p: float = 0.95, do_sample: bool = True) -> Union[str, List[str]]:
        """Continue a prompt (or prompts) with generated text."""
        if max_new_tokens < 1: raise ValueError("max_new_tokens must be at least 1.")
        one = isinstance(prompt, str); prompts = [prompt] if one else list(prompt)
        if not prompts: return []
        positions = getattr(self.model.config, "max_position_embeddings", getattr(self.model.config, "n_positions", 1024))
        inputs = self._encode(prompts, max_length=max(1, positions - max_new_tokens))
        self.model.eval()
        options = {"max_new_tokens": max_new_tokens, "do_sample": do_sample, "pad_token_id": self.tokenizer.pad_token_id}
        if do_sample: options.update(temperature=temperature, top_p=top_p)
        with torch.no_grad():
            generated = self.model.generate(**inputs, **options)
        # Return the generated answer rather than repeating the user's prompt.
        answers = self.tokenizer.batch_decode(generated[:, inputs["input_ids"].size(1):], skip_special_tokens=True)
        return answers[0] if one else answers

    def fine_tune(self, folder: Union[str, Path], *, text: Optional[str] = None,
                  files: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
                  epochs: int = 1, batch_size: int = 2, learning_rate: float = 5e-5,
                  block_size: int = 512) -> "GPT":
        """Continue causal-language-model training on new text and save it.

        Supply ``text``, ``files`` (including folders), or both.  The object is
        updated in place and also returned.
        """
        corpus = _corpus(text, files)
        _train_causal(self.model, self.tokenizer, corpus, epochs, batch_size, learning_rate, block_size, self.device)
        output = Path(folder); output.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output); self.tokenizer.save_pretrained(output)
        (output / _MANIFEST).write_text(json.dumps({"base_model": self.model.config._name_or_path}, indent=2), encoding="utf-8")
        self.folder = output; self.model.eval()
        return self


def _corpus(text: Optional[str], files: Optional[Union[str, Path, Sequence[Union[str, Path]]]]) -> str:
    if text is None and files is None: raise ValueError("Provide text, files, or both.")
    if text is not None and not isinstance(text, str): raise TypeError("text must be a string.")
    corpus = "\n".join(part for part in (text or "", _read_files(files) if files is not None else "") if part)
    if not corpus.strip(): raise ValueError("The supplied corpus contains no text.")
    return corpus


def _train_causal(model, tokenizer, corpus: str, epochs: int, batch_size: int, learning_rate: float,
                  block_size: int, device: torch.device) -> None:
    if epochs < 1 or batch_size < 1 or block_size < 8: raise ValueError("epochs and batch_size must be positive; block_size must be at least 8.")
    # An EOS token prevents unrelated file/text boundaries being interpreted as one sentence.
    ids = tokenizer(corpus + tokenizer.eos_token, return_tensors="pt", add_special_tokens=False).input_ids[0]
    positions = getattr(model.config, "max_position_embeddings", getattr(model.config, "n_positions", block_size))
    dataset = _TokenBlocks(ids, min(block_size, positions))
    if not len(dataset): raise ValueError("The corpus is too short to form a training block.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda blocks: _collate(blocks, tokenizer.pad_token_id))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate); model.train()
    for epoch in range(epochs):
        progress = tqdm(loader, desc=f"GPT epoch {epoch + 1}/{epochs}", unit="batch")
        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            progress.set_postfix(loss=f"{loss.item():.3f}")


def train(folder: Union[str, Path], *, text: Optional[str] = None,
          files: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
          model_name: str = "gpt2", epochs: int = 1, batch_size: int = 2,
          learning_rate: float = 5e-5, block_size: int = 512,
          device: Optional[Union[str, torch.device]] = None) -> GPT:
    """Fine-tune GPT-2 (or another Hugging Face causal model) on a corpus."""
    corpus = _corpus(text, files)
    transformers = _transformers(); target_device = _device(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # GPT-2 has no padding token. Using EOS is Hugging Face's standard choice.
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name).to(target_device)
    _train_causal(model, tokenizer, corpus, epochs, batch_size, learning_rate, block_size, target_device)
    output = Path(folder); output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output); tokenizer.save_pretrained(output)
    (output / _MANIFEST).write_text(json.dumps({"base_model": model_name}, indent=2), encoding="utf-8")
    return GPT(model, tokenizer, folder=output, device=target_device)


def load_model(folder_or_model: Union[str, Path], *, device: Optional[Union[str, torch.device]] = None) -> GPT:
    """Load a saved model, local Hugging Face causal model, or Hub model ID."""
    transformers = _transformers(); source = str(folder_or_model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(source, use_fast=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(source)
    return GPT(model, tokenizer, folder=Path(source) if Path(source).exists() else None, device=device)
