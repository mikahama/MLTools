"""Small, approachable word2vec and fastText-style word embeddings.

The implementation intentionally only depends on PyTorch, which MLTools already
uses.  It provides skip-gram training with negative sampling; ``fasttext`` adds
character n-gram vectors so it can embed words not seen during training.
``morph`` additionally composes each word from its morphological analyses
(lemma + tags), e.g. from an FST, so that all inflected forms of a lemma share a
vector even when their spelling differs (consonant gradation, stem changes).
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm.auto import tqdm

_TOKEN = re.compile(r"\b\w+(?:['’]\w+)?\b", re.UNICODE)
_STATE_FILE = "embeddings.pt"
_INFO_FILE = "metadata.json"


def _tokenize(text: str, lowercase: bool = True) -> List[str]:
    """Turn ordinary prose into words, keeping Unicode letters and numbers."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    return _TOKEN.findall(text.lower() if lowercase else text)


def _read_files(files: Union[str, Path, Sequence[Union[str, Path]]]) -> str:
    paths = [files] if isinstance(files, (str, Path)) else files
    if not paths:
        return ""
    pieces: List[str] = []
    for supplied in paths:
        path = Path(supplied)
        if not path.exists():
            raise FileNotFoundError(f"Corpus path does not exist: {path}")
        candidates = (p for p in path.rglob("*") if p.is_file()) if path.is_dir() else (path,)
        for candidate in candidates:
            # Ignore any hidden file and all content below a hidden directory.
            if any(part.startswith(".") for part in candidate.relative_to(path.parent).parts):
                continue
            try:
                pieces.append(candidate.read_text(encoding="utf-8", errors="ignore"))
            except OSError as exc:
                raise OSError(f"Could not read corpus file {candidate}: {exc}") from exc
    return "\n".join(pieces)


Analyzer = Callable[[str], Sequence[str]]


def _load_analyzer(analyzer: Union[None, str, Path, Analyzer]) -> Optional[Analyzer]:
    """Accept a callable ``word -> ["lemma+TAG+TAG", ...]`` or a path to an HFST optimized-lookup
    (.hfstol) analyser, which is read with the optional pyhfst package."""
    if analyzer is None or callable(analyzer):
        return analyzer
    try:
        import pyhfst
    except ImportError as exc:
        raise ImportError("Reading an .hfstol analyser requires pyhfst (pip install pyhfst).") from exc
    transducer = pyhfst.HfstInputStream(str(analyzer)).read()
    return lambda word: [result[0] for result in transducer.lookup(word)]


def _parse_analysis(analysis: str) -> Tuple[str, List[str]]:
    """'kudo+N+Sg+Ine' -> ('kudo', ['N', 'Sg', 'Ine']); for compounds the last part is used."""
    parts = [part for part in analysis.split("#")[-1].split("+") if part]
    return (parts[0], parts[1:]) if parts else (analysis, [])


def _ngrams(word: str, minimum: int, maximum: int) -> List[str]:
    marked = f"<{word}>"
    return [marked[i : i + size] for size in range(minimum, maximum + 1)
            for i in range(len(marked) - size + 1)]


class _SkipGram(nn.Module):
    def __init__(self, vocabulary_size: int, dimension: int, model_type: str, bucket_size: int,
                 lemma_size: int = 0, tag_size: int = 0) -> None:
        super().__init__()
        self.model_type = model_type
        self.input = nn.Embedding(vocabulary_size, dimension)
        self.output = nn.Embedding(vocabulary_size, dimension)
        self.subwords = nn.EmbeddingBag(bucket_size, dimension, mode="mean") if model_type in {"fasttext", "morph"} else None
        self.lemmas = nn.Embedding(max(lemma_size, 1), dimension) if model_type == "morph" else None
        self.tags = nn.EmbeddingBag(max(tag_size, 1), dimension, mode="mean") if model_type == "morph" else None
        bound = 0.5 / max(dimension, 1)
        nn.init.uniform_(self.input.weight, -bound, bound)
        nn.init.zeros_(self.output.weight)
        for table in (self.subwords, self.lemmas, self.tags):
            if table is not None:
                nn.init.uniform_(table.weight, -bound, bound)

    def vectors(self, words: Tensor, subword_ids: Optional[Tensor] = None, offsets: Optional[Tensor] = None,
                morphology: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None) -> Tensor:
        # Negative ids denote unseen words: only their subword/morphological
        # features contribute, with no learned vocabulary-word embedding.
        result = self.input(words.clamp_min(0))
        result = result.masked_fill(words.lt(0).unsqueeze(1), 0)
        if self.subwords is not None and subword_ids is not None and offsets is not None:
            result = result + self.subwords(subword_ids, offsets)
        if self.lemmas is not None and morphology is not None and len(morphology[0]):
            # morphology = (lemma ids, tag ids, tag offsets, owner word index), one row per analysis
            lemma_ids, tag_ids, tag_offsets, owners = morphology
            analyses = self.lemmas(lemma_ids) + self.tags(tag_ids, tag_offsets)
            total = torch.zeros_like(result).index_add_(0, owners, analyses)
            count = torch.zeros(len(result), device=result.device).index_add_(0, owners, torch.ones_like(owners, dtype=result.dtype))
            result = result + total / count.clamp_min(1).unsqueeze(1)
        return result


class WordEmbeddings:
    """A trained word embedding model returned by :func:`train`.

    Use :meth:`embed`, :meth:`similarity`, :meth:`nearest_neighbors`, and
    :meth:`cluster` to explore vocabulary in a corpus.
    """
    def __init__(self, network: _SkipGram, vocabulary: List[str], config: Dict, folder: Union[str, Path],
                 analyzer: Optional[Analyzer] = None, morph_vocab: Optional[Dict[str, Dict[str, int]]] = None) -> None:
        self.network, self.vocabulary, self.config, self.folder = network, vocabulary, config, Path(folder)
        self.word_to_id = {word: index for index, word in enumerate(vocabulary)}
        self.analyzer, self.morph_vocab = analyzer, morph_vocab or {"lemmas": {}, "tags": {}}
        self.device = torch.device(config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.network.to(self.device).eval()

    def _subword_data(self, words: Sequence[str]) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        if self.config["model_type"] not in {"fasttext", "morph"}:
            return None, None
        buckets = self.config["bucket_size"]
        all_ids, offsets = [], []
        for word in words:
            offsets.append(len(all_ids))
            grams = _ngrams(word, self.config["min_n"], self.config["max_n"])
            # A stable hash, unlike Python's process-randomised hash().
            all_ids.extend(_stable_hash(gram) % buckets for gram in grams or [word])
        return torch.tensor(all_ids, device=self.device), torch.tensor(offsets, device=self.device)

    def _morphology(self, words: Sequence[str]) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor]]:
        if self.config["model_type"] != "morph" or self.analyzer is None:
            return None
        return _morphology_tensors(words, self.analyzer, self.morph_vocab, self.device, self.config.get("max_analyses", 6))

    def _vectors(self, words: Sequence[str]) -> Tensor:
        ids = torch.tensor([self.word_to_id.get(word, -1) for word in words], device=self.device)
        subword_ids, offsets = self._subword_data(words)
        with torch.no_grad():
            vectors = self.network.vectors(ids, subword_ids, offsets, self._morphology(words))
        # word2vec has no useful vector for out-of-vocabulary words.
        missing = [word for word in words if word not in self.word_to_id]
        if missing and self.config["model_type"] == "word2vec":
            raise KeyError(f"Unknown word: {missing[0]!r}. (fasttext can embed unseen words.)")
        return vectors

    def embed(self, word: str) -> Tensor:
        """Return a detached CPU vector for ``word``."""
        if not isinstance(word, str):
            raise TypeError("word must be a string.")
        return self._vectors([word.lower() if self.config["lowercase"] else word])[0].cpu()

    def word(self, embedding: Union[Tensor, Sequence[float]]) -> str:
        """Return the vocabulary word closest by cosine similarity to a vector."""
        vector = torch.as_tensor(embedding, dtype=torch.float, device=self.device).flatten()
        if vector.numel() != self.config["dimension"]:
            raise ValueError(f"embedding must contain {self.config['dimension']} values.")
        matrix = self._vectors(self.vocabulary)
        scores = F.cosine_similarity(matrix, vector.unsqueeze(0), dim=1)
        return self.vocabulary[int(scores.argmax())]

    def similarity(self, first: str, second: str) -> float:
        """Return cosine similarity between two words (from -1 to 1)."""
        vectors = self._vectors([self._normalise(first), self._normalise(second)])
        return float(F.cosine_similarity(vectors[0], vectors[1], dim=0).cpu())

    def _normalise(self, word: str) -> str:
        if not isinstance(word, str):
            raise TypeError("words must be strings.")
        return word.lower() if self.config["lowercase"] else word

    def nearest_neighbors(self, word: str, n: int = 10) -> List[Tuple[str, float]]:
        """Return up to ``n`` nearest vocabulary words and cosine similarities."""
        if n < 1:
            raise ValueError("n must be at least 1.")
        word = self._normalise(word)
        query = self._vectors([word])[0]
        matrix = self._vectors(self.vocabulary)
        scores = F.cosine_similarity(matrix, query.unsqueeze(0), dim=1)
        if word in self.word_to_id:
            scores[self.word_to_id[word]] = -float("inf")
        count = min(n, max(0, len(self.vocabulary) - (word in self.word_to_id)))
        values, ids = torch.topk(scores, count)
        return [(self.vocabulary[int(i)], float(score)) for score, i in zip(values.cpu(), ids.cpu())]

    def cluster(self, words: Sequence[str], *, method: str = "affinity_propagation", preference: Optional[float] = None,
                damping: float = 0.7, max_iter: int = 200, eps: float = 0.25, min_samples: int = 2) -> Dict[int, List[str]]:
        """Cluster words using built-in affinity propagation or DBSCAN.

        ``method`` is ``"affinity_propagation"`` (the default) or ``"dbscan"``.
        DBSCAN's ``eps`` is cosine distance; noise is returned under key ``-1``.
        """
        if not words:
            return {}
        normalised = [self._normalise(word) for word in words]
        vectors = F.normalize(self._vectors(normalised), dim=1).cpu()
        if method == "affinity_propagation":
            labels = _affinity_propagation(vectors @ vectors.T, preference, damping, max_iter)
        elif method == "dbscan":
            labels = _dbscan(1 - vectors @ vectors.T, eps, min_samples)
        else:
            raise ValueError("method must be 'affinity_propagation' or 'dbscan'.")
        groups: Dict[int, List[str]] = defaultdict(list)
        for label, word in zip(labels, words): groups[label].append(word)
        return dict(groups)


def _morphology_tensors(words: Sequence[str], analyzer: Analyzer, vocab: Dict[str, Dict[str, int]], device,
                        max_analyses: int = 6, grow: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build (lemma ids, tag ids, tag offsets, owner index) for the analyses of ``words``.
    Unknown lemmas/tags are skipped unless ``grow`` adds them to ``vocab`` (training time)."""
    lemma_ids, tag_ids, tag_offsets, owners = [], [], [], []
    for index, word in enumerate(words):
        for analysis in list(dict.fromkeys(analyzer(word)))[:max_analyses]:
            lemma, tags = _parse_analysis(analysis)
            if lemma not in vocab["lemmas"]:
                if not grow: continue
                vocab["lemmas"][lemma] = len(vocab["lemmas"])
            ids = []
            for tag in tags or ["<none>"]:
                if tag not in vocab["tags"] and grow: vocab["tags"][tag] = len(vocab["tags"])
                if tag in vocab["tags"]: ids.append(vocab["tags"][tag])
            tag_offsets.append(len(tag_ids)); tag_ids.extend(ids or [0])
            lemma_ids.append(vocab["lemmas"][lemma]); owners.append(index)
    as_tensor = lambda values: torch.tensor(values, dtype=torch.long, device=device)
    return as_tensor(lemma_ids), as_tensor(tag_ids), as_tensor(tag_offsets), as_tensor(owners)


def _stable_hash(value: str) -> int:
    result = 2166136261
    for byte in value.encode("utf-8"):
        result = (result ^ byte) * 16777619 & 0xFFFFFFFF
    return result


def _affinity_propagation(similarity: Tensor, preference: Optional[float], damping: float, max_iter: int) -> List[int]:
    if not 0.5 <= damping < 1: raise ValueError("damping must be in [0.5, 1).")
    n = similarity.size(0)
    if n == 1: return [0]
    s = similarity.clone(); s.fill_diagonal_(float(s[~torch.eye(n, dtype=torch.bool)].median()) if preference is None else preference)
    responsibility, availability = torch.zeros_like(s), torch.zeros_like(s)
    for _ in range(max_iter):
        values = availability + s
        top_values, top_ids = values.max(dim=1)
        replacement = values.clone(); replacement[torch.arange(n), top_ids] = -float("inf")
        second = replacement.max(dim=1).values
        updated_r = s - top_values.unsqueeze(1)
        updated_r[torch.arange(n), top_ids] = s[torch.arange(n), top_ids] - second
        responsibility = damping * responsibility + (1 - damping) * updated_r
        positive = responsibility.clamp_min(0); positive.fill_diagonal_(0)
        sums = positive.sum(dim=0)
        updated_a = torch.minimum(torch.zeros_like(s), responsibility.diag().unsqueeze(0) + sums.unsqueeze(0) - positive)
        updated_a[torch.arange(n), torch.arange(n)] = sums
        availability = damping * availability + (1 - damping) * updated_a
    exemplars = (availability + responsibility).diag().gt(0).nonzero().flatten()
    if not len(exemplars): exemplars = torch.tensor([int((availability + responsibility).diag().argmax())])
    assigned = (s[:, exemplars] + availability[:, exemplars]).argmax(dim=1)
    return [int(exemplars[index]) for index in assigned]


def _dbscan(distance: Tensor, eps: float, min_samples: int) -> List[int]:
    if eps < 0 or min_samples < 1: raise ValueError("eps must be non-negative and min_samples must be at least 1.")
    n, labels, visited, label = distance.size(0), [-1] * distance.size(0), set(), 0
    neighbors = [set(torch.where(distance[i] <= eps)[0].tolist()) for i in range(n)]
    for point in range(n):
        if point in visited: continue
        visited.add(point)
        if len(neighbors[point]) < min_samples: continue
        labels[point] = label; queue = list(neighbors[point])
        while queue:
            other = queue.pop()
            if other not in visited:
                visited.add(other)
                if len(neighbors[other]) >= min_samples: queue.extend(neighbors[other] - set(queue))
            if labels[other] == -1: labels[other] = label
        label += 1
    return labels


def train(folder: Union[str, Path], *, text: Optional[str] = None, files: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
          model_type: str = "word2vec", dimension: int = 100, window: int = 5, min_frequency: int = 1,
          epochs: int = 5, learning_rate: float = 0.025, negative_samples: int = 5, batch_size: int = 512,
          lowercase: bool = True, min_n: int = 3, max_n: int = 6, bucket_size: int = 100_000,
          analyzer: Union[None, str, Path, Analyzer] = None, max_analyses: int = 6,
          seed: Optional[int] = 42, device: Optional[Union[str, torch.device]] = None) -> WordEmbeddings:
    """Train and save word2vec, fastText-style or morphology-aware embeddings.

    Supply ``text``, ``files`` (files and/or folders), or both.  ``model_type``
    is ``"word2vec"``, ``"fasttext"`` or ``"morph"``.  ``morph`` needs an
    ``analyzer``: a function returning analyses such as ``["kudo+N+Sg+Ine"]``
    for a word (for example UralicNLP's ``analyze``), or a path to an ``.hfstol``
    FST (read with pyhfst).  The resulting model is saved in ``folder`` and returned.
    """
    if text is None and files is None: raise ValueError("Provide text, files, or both.")
    if text is not None and not isinstance(text, str): raise TypeError("text must be a string.")
    if not isinstance(model_type, str): raise TypeError("model_type must be a string.")
    model_type = model_type.lower().replace("-", "")
    if model_type not in {"word2vec", "fasttext", "morph"}: raise ValueError("model_type must be 'word2vec', 'fasttext' or 'morph'.")
    analyze = _load_analyzer(analyzer)
    if model_type == "morph" and analyze is None: raise ValueError("model_type='morph' needs an analyzer (a callable or an .hfstol path).")
    if min_frequency < 1 or dimension < 1 or window < 1 or epochs < 1 or negative_samples < 1 or batch_size < 1: raise ValueError("dimension, window, epochs, negative_samples, batch_size, and min_frequency must be positive.")
    if min_n < 1 or max_n < min_n or bucket_size < 1: raise ValueError("Require 1 <= min_n <= max_n and a positive bucket_size.")
    corpus = "\n".join(part for part in (text or "", _read_files(files) if files is not None else "") if part)
    tokens = _tokenize(corpus, lowercase)
    counts = Counter(tokens)
    vocabulary = sorted(word for word, count in counts.items() if count >= min_frequency)
    if len(vocabulary) < 2: raise ValueError("The corpus needs at least two words after min_frequency filtering.")
    stoi = {word: i for i, word in enumerate(vocabulary)}
    ids = [stoi[word] for word in tokens if word in stoi]
    pairs = [(center, ids[context]) for i, center in enumerate(ids) for context in range(max(0, i-window), min(len(ids), i+window+1)) if context != i]
    if not pairs: raise ValueError("The corpus is too short to form training contexts.")
    if seed is not None: random.seed(seed); torch.manual_seed(seed)
    config = {"model_type": model_type, "dimension": dimension, "window": window, "min_frequency": min_frequency, "lowercase": lowercase, "min_n": min_n, "max_n": max_n, "bucket_size": bucket_size, "max_analyses": max_analyses, "device": str(device) if device else None}
    if isinstance(analyzer, (str, Path)): config["analyzer"] = str(analyzer)
    morph_vocab: Dict[str, Dict[str, int]] = {"lemmas": {}, "tags": {"<none>": 0}}
    word_morphology: Dict[int, List[Tuple[int, List[int]]]] = {}
    if model_type == "morph":
        for index, word in enumerate(tqdm(vocabulary, desc="Analysing vocabulary", unit="word")):
            lemma_ids, tag_ids, tag_offsets, _ = _morphology_tensors([word], analyze, morph_vocab, "cpu", max_analyses, grow=True)
            bounds = tag_offsets.tolist() + [len(tag_ids)]
            word_morphology[index] = [(int(lemma), tag_ids[bounds[k]:bounds[k + 1]].tolist()) for k, lemma in enumerate(lemma_ids)]
    network = _SkipGram(len(vocabulary), dimension, model_type, bucket_size, len(morph_vocab["lemmas"]), len(morph_vocab["tags"]))
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")); network.to(target_device)
    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)
    weights = torch.tensor([counts[word] ** 0.75 for word in vocabulary], dtype=torch.float); weights /= weights.sum()
    for epoch in range(epochs):
        random.shuffle(pairs); network.train()
        progress = tqdm(range(0, len(pairs), batch_size), desc=f"Embedding epoch {epoch + 1}/{epochs}", unit="batch")
        for start in progress:
            chunk = pairs[start:start + batch_size]
            centers = torch.tensor([pair[0] for pair in chunk], device=target_device)
            contexts = torch.tensor([pair[1] for pair in chunk], device=target_device)
            negative = torch.multinomial(weights, len(chunk) * negative_samples, replacement=True).view(len(chunk), negative_samples).to(target_device)
            subword_ids = offsets = morphology = None
            if model_type == "morph":
                lemma_ids, tag_ids, tag_offsets, owners = [], [], [], []
                for row, center in enumerate(centers.tolist()):
                    for lemma, tags in word_morphology[center]:
                        tag_offsets.append(len(tag_ids)); tag_ids.extend(tags or [0]); lemma_ids.append(lemma); owners.append(row)
                as_tensor = lambda values: torch.tensor(values, dtype=torch.long, device=target_device)
                morphology = (as_tensor(lemma_ids), as_tensor(tag_ids), as_tensor(tag_offsets), as_tensor(owners))
            if model_type in {"fasttext", "morph"}:
                ids_flat, starts = [], []
                for center in centers.tolist():
                    starts.append(len(ids_flat)); ids_flat.extend(_stable_hash(g) % bucket_size for g in _ngrams(vocabulary[center], min_n, max_n))
                subword_ids, offsets = torch.tensor(ids_flat, device=target_device), torch.tensor(starts, device=target_device)
            inputs = network.vectors(centers, subword_ids, offsets, morphology)
            positive = (inputs * network.output(contexts)).sum(dim=1)
            negative_score = torch.bmm(network.output(negative), inputs.unsqueeze(2)).squeeze(2)
            loss = -(F.logsigmoid(positive).mean() + F.logsigmoid(-negative_score).mean())
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            progress.set_postfix(loss=f"{loss.item():.3f}")
    output = Path(folder); output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": network.cpu().state_dict(), "vocabulary": vocabulary, "config": config, "morph_vocab": morph_vocab}, output / _STATE_FILE)
    (output / _INFO_FILE).write_text(json.dumps({"model_type": model_type, "vocabulary_size": len(vocabulary), **config}, indent=2), encoding="utf-8")
    return WordEmbeddings(network, vocabulary, config, output, analyze, morph_vocab)


def load_model(folder: Union[str, Path], *, device: Optional[Union[str, torch.device]] = None,
               analyzer: Union[None, str, Path, Analyzer] = None) -> WordEmbeddings:
    """Load a model previously saved by :func:`train`.

    ``morph`` models trained with an ``.hfstol`` path reload it automatically; pass
    ``analyzer`` again if the model was trained with a Python function.
    """
    folder = Path(folder); state_path = folder / _STATE_FILE
    if not state_path.is_file(): raise FileNotFoundError(f"No embedding model found at {state_path}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    config = state["config"].copy()
    if device is not None: config["device"] = str(device)
    morph_vocab = state.get("morph_vocab") or {"lemmas": {}, "tags": {}}
    network = _SkipGram(len(state["vocabulary"]), config["dimension"], config["model_type"], config["bucket_size"],
                        len(morph_vocab["lemmas"]), len(morph_vocab["tags"]))
    network.load_state_dict(state["state_dict"])
    analyze = _load_analyzer(analyzer if analyzer is not None else config.get("analyzer"))
    return WordEmbeddings(network, state["vocabulary"], config, folder, analyze, morph_vocab)

