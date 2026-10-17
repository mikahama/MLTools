"""Simple conditional image generators trained from captioned images.

Use data in the same ``[image_path, caption]`` shape as :mod:`image2text`.
The module includes a compact conditional GAN and a small DDPM-style diffusion
model.  It is intended as an approachable starting point, rather than a
replacement for very large, pre-trained text-to-image systems.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from . import image2text, seq2seq

split = seq2seq.split
Pair = Tuple[str, str]


def _image_tensor(path: str, size: Tuple[int, int]) -> Tensor:
    """Load a training image as an RGB tensor in the GAN/DDPM [-1, 1] range."""
    with Image.open(path) as image:
        image = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        width, height = image.size
        pixels = torch.frombuffer(memoryview(image.tobytes()), dtype=torch.uint8).clone()
    return pixels.reshape(height, width, 3).permute(2, 0, 1).float().div_(127.5).sub_(1)


def _to_image(tensor: Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(-1, 1).add(1).mul(127.5).to(torch.uint8)
    height, width = tensor.shape[1:]
    data = bytes(tensor.permute(1, 2, 0).contiguous().flatten().tolist())
    return Image.frombytes("RGB", (width, height), data)


class _GenerationData(Dataset):
    def __init__(self, pairs: Sequence[Pair], vocabulary: seq2seq.Vocabulary, size: Tuple[int, int], tokenizer: seq2seq.Tokenize) -> None:
        self.pairs, self.vocabulary, self.size, self.tokenizer = list(pairs), vocabulary, size, tokenizer

    def __len__(self) -> int: return len(self.pairs)

    def __getitem__(self, index: int):
        path, caption = self.pairs[index]
        return _image_tensor(path, self.size), torch.tensor(self.vocabulary.encode(caption, self.tokenizer, add_eos=False))


def _collate(items, pad_id: int):
    images, captions = zip(*items)
    return torch.stack(images), nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=pad_id)


class _TextCondition(nn.Module):
    """A small mean-pooled text encoder, trained together with the generator."""
    def __init__(self, vocabulary_size: int, embedding_size: int, pad_id: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_size, padding_idx=pad_id)
        self.pad_id = pad_id

    def forward(self, caption: Tensor) -> Tensor:
        vectors = self.embedding(caption)
        mask = caption.ne(self.pad_id).unsqueeze(-1)
        return (vectors * mask).sum(1) / mask.sum(1).clamp_min(1)


class _Generator(nn.Module):
    def __init__(self, condition_size: int, latent_size: int, size: Tuple[int, int], channels: int = 64) -> None:
        super().__init__()
        self.latent_size, self.size = latent_size, size
        width, height = size
        self.start_width, self.start_height = math.ceil(width / 16), math.ceil(height / 16)
        self.initial = nn.Linear(latent_size + condition_size, channels * 8 * self.start_width * self.start_height)
        self.blocks = nn.Sequential(
            nn.BatchNorm2d(channels * 8), nn.ReLU(True),
            nn.ConvTranspose2d(channels * 8, channels * 4, 4, 2, 1), nn.BatchNorm2d(channels * 4), nn.ReLU(True),
            nn.ConvTranspose2d(channels * 4, channels * 2, 4, 2, 1), nn.BatchNorm2d(channels * 2), nn.ReLU(True),
            nn.ConvTranspose2d(channels * 2, channels, 4, 2, 1), nn.BatchNorm2d(channels), nn.ReLU(True),
            nn.ConvTranspose2d(channels, 3, 4, 2, 1), nn.Tanh(),
        )

    def forward(self, noise: Tensor, condition: Tensor) -> Tensor:
        batch = noise.size(0)
        image = self.initial(torch.cat((noise, condition), 1)).reshape(batch, -1, self.start_height, self.start_width)
        image = self.blocks(image)
        return F.interpolate(image, size=(self.size[1], self.size[0]), mode="bilinear", align_corners=False)


class _Discriminator(nn.Module):
    def __init__(self, condition_size: int, channels: int = 64) -> None:
        super().__init__()
        def block(incoming, outgoing): return [nn.Conv2d(incoming, outgoing, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        self.features = nn.Sequential(*block(3, channels), *block(channels, channels * 2), nn.BatchNorm2d(channels * 2), *block(channels * 2, channels * 4), nn.BatchNorm2d(channels * 4), *block(channels * 4, channels * 8), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.output = nn.Linear(channels * 8 + condition_size, 1)

    def forward(self, images: Tensor, condition: Tensor) -> Tensor:
        return self.output(torch.cat((self.features(images), condition), 1)).squeeze(1)


class _ConditionalGAN(nn.Module):
    def __init__(self, vocabulary_size: int, pad_id: int, embedding_size: int, latent_size: int, size: Tuple[int, int]) -> None:
        super().__init__()
        self.latent_size = latent_size
        self.text = _TextCondition(vocabulary_size, embedding_size, pad_id)
        self.generator = _Generator(embedding_size, latent_size, size)
        self.discriminator = _Discriminator(embedding_size)

    def generate(self, captions: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        noise = noise if noise is not None else torch.randn(len(captions), self.latent_size, device=captions.device)
        return self.generator(noise, self.text(captions))


def _time_embedding(times: Tensor, width: int) -> Tensor:
    frequencies = torch.exp(torch.arange(0, width, 2, device=times.device) * (-math.log(10000.0) / width))
    angles = times.float().unsqueeze(1) * frequencies.unsqueeze(0)
    return torch.cat((angles.sin(), angles.cos()), dim=1)


class _ConditionedBlock(nn.Module):
    def __init__(self, channels: int, condition_size: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.condition = nn.Linear(condition_size, channels)
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, image: Tensor, condition: Tensor) -> Tensor:
        return F.silu(self.conv(F.silu(self.norm(image)) + self.condition(condition).unsqueeze(-1).unsqueeze(-1)))


class _TinyUNet(nn.Module):
    """A modest U-Net for educational/small-scale diffusion training."""
    def __init__(self, vocabulary_size: int, pad_id: int, embedding_size: int = 128, channels: int = 64) -> None:
        super().__init__()
        self.text = _TextCondition(vocabulary_size, embedding_size, pad_id)
        self.time = nn.Sequential(nn.Linear(embedding_size, embedding_size), nn.SiLU(), nn.Linear(embedding_size, embedding_size))
        self.stem = nn.Conv2d(3, channels, 3, padding=1)
        self.one = _ConditionedBlock(channels, embedding_size)
        self.down_one = nn.Conv2d(channels, channels * 2, 4, 2, 1)
        self.two = _ConditionedBlock(channels * 2, embedding_size)
        self.down_two = nn.Conv2d(channels * 2, channels * 4, 4, 2, 1)
        self.middle = _ConditionedBlock(channels * 4, embedding_size)
        self.up_one = nn.ConvTranspose2d(channels * 4, channels * 2, 4, 2, 1)
        self.after_one = nn.Conv2d(channels * 4, channels * 2, 3, padding=1)
        self.up_two = nn.ConvTranspose2d(channels * 2, channels, 4, 2, 1)
        self.after_two = nn.Conv2d(channels * 2, channels, 3, padding=1)
        self.output = nn.Conv2d(channels, 3, 3, padding=1)
        self.embedding_size = embedding_size

    def forward(self, image: Tensor, times: Tensor, captions: Tensor) -> Tensor:
        condition = self.text(captions) + self.time(_time_embedding(times, self.embedding_size))
        first = self.one(self.stem(image), condition)
        second = self.two(self.down_one(first), condition)
        middle = self.middle(self.down_two(second), condition)
        up = F.silu(self.after_one(torch.cat((self.up_one(middle), second), 1)))
        up = F.silu(self.after_two(torch.cat((self.up_two(up), first), 1)))
        return self.output(up)


class _Diffusion(nn.Module):
    def __init__(self, vocabulary_size: int, pad_id: int, embedding_size: int, steps: int) -> None:
        super().__init__()
        self.unet = _TinyUNet(vocabulary_size, pad_id, embedding_size)
        betas = torch.linspace(0.0001, 0.02, steps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1 - betas)
        self.register_buffer("alpha_bars", torch.cumprod(1 - betas, dim=0))

    def loss(self, clean: Tensor, captions: Tensor) -> Tensor:
        times = torch.randint(0, len(self.betas), (len(clean),), device=clean.device)
        noise = torch.randn_like(clean)
        alpha = self.alpha_bars[times].view(-1, 1, 1, 1)
        noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
        return F.mse_loss(self.unet(noisy, times, captions), noise)

    @torch.no_grad()
    def generate(self, captions: Tensor, size: Tuple[int, int]) -> Tensor:
        images = torch.randn(len(captions), 3, size[1], size[0], device=captions.device)
        for step in reversed(range(len(self.betas))):
            times = torch.full((len(captions),), step, device=captions.device, dtype=torch.long)
            predicted_noise = self.unet(images, times, captions)
            alpha, alpha_bar, beta = self.alphas[step], self.alpha_bars[step], self.betas[step]
            images = (images - beta / (1 - alpha_bar).sqrt() * predicted_noise) / alpha.sqrt()
            if step:
                images += beta.sqrt() * torch.randn_like(images)
        return images


@dataclass
class _Config:
    model_type: str = "diffusion"
    input_size: Tuple[int, int] = (224, 224)
    embedding_size: int = 128
    latent_size: int = 128
    diffusion_steps: int = 100
    min_frequency: int = 1


def _network(config: _Config, vocabulary: seq2seq.Vocabulary) -> nn.Module:
    if config.model_type == "gan": return _ConditionalGAN(len(vocabulary.itos), vocabulary.pad_id, config.embedding_size, config.latent_size, config.input_size)
    if config.model_type == "diffusion": return _Diffusion(len(vocabulary.itos), vocabulary.pad_id, config.embedding_size, config.diffusion_steps)
    raise ValueError("model_type must be 'gan' or 'diffusion'.")


class Text2ImageModel:
    """A trained conditional image generator returned by :func:`train`."""
    def __init__(self, network: nn.Module, vocabulary: seq2seq.Vocabulary, config: _Config, folder: Union[str, Path], tokenizer: seq2seq.Tokenize = seq2seq._default_tokenizer, device: Optional[Union[str, torch.device]] = None) -> None:
        self.network, self.vocabulary, self.config = network, vocabulary, config
        self.folder, self.tokenizer = Path(folder), tokenizer
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.network.to(self.device); self.epoch = 0

    @property
    def input_size(self) -> Tuple[int, int]: return tuple(self.config.input_size)

    def _loader(self, data: Sequence[Sequence[str]], batch_size: int, shuffle: bool) -> DataLoader:
        pairs = image2text._image_pairs(data)
        return DataLoader(_GenerationData(pairs, self.vocabulary, self.input_size, self.tokenizer), batch_size=batch_size, shuffle=shuffle, collate_fn=lambda items: _collate(items, self.vocabulary.pad_id))

    def _validation_loss(self, data: Sequence[Sequence[str]], batch_size: int) -> float:
        if self.config.model_type == "gan": return float("nan")
        self.network.eval(); total = count = 0
        with torch.no_grad():
            for images, captions in self._loader(data, batch_size, False):
                total += self.network.loss(images.to(self.device), captions.to(self.device)).item() * len(images)  # type: ignore[attr-defined]
                count += len(images)
        return total / max(count, 1)

    def _save(self, optimizer: object, validation_loss: float, *, best: bool) -> None:
        state = {"state_dict": self.network.state_dict(), "optimizer": optimizer, "epoch": self.epoch, "validation_loss": validation_loss, "config": asdict(self.config), "vocabulary": self.vocabulary.itos}
        torch.save(state, self.folder / "model_latest.pt")
        torch.save(state, self.folder / f"checkpoint_epoch_{self.epoch:03d}.pt")
        if best: torch.save(state, self.folder / "model_best.pt")

    def fit(self, train_data: Sequence[Sequence[str]], validation_data: Sequence[Sequence[str]], *, epochs: int = 20, batch_size: int = 16, learning_rate: float = 0.0002, patience: int = 3, min_delta: float = 0.0, optimizer_state: Optional[object] = None) -> "Text2ImageModel":
        """Train the generator. Diffusion uses validation early stopping; GANs save every epoch."""
        if self.config.model_type == "diffusion" and not validation_data:
            raise ValueError("validation_data cannot be empty for diffusion early stopping.")
        if self.config.model_type == "gan":
            generator = torch.optim.Adam(list(self.network.text.parameters()) + list(self.network.generator.parameters()), lr=learning_rate, betas=(0.5, 0.999))  # type: ignore[attr-defined]
            critic = torch.optim.Adam(self.network.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))  # type: ignore[attr-defined]
            if optimizer_state:
                generator.load_state_dict(optimizer_state["generator"]); critic.load_state_dict(optimizer_state["critic"])
        else:
            optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
            if optimizer_state: optimizer.load_state_dict(optimizer_state)  # type: ignore[arg-type]
        best_loss, stalled = float("inf"), 0
        for _ in range(epochs):
            self.epoch += 1; self.network.train(); running = 0.0
            progress = tqdm(self._loader(train_data, batch_size, True), desc=f"Epoch {self.epoch}", unit="batch")
            for images, captions in progress:
                images, captions = images.to(self.device), captions.to(self.device)
                if self.config.model_type == "gan":
                    condition = self.network.text(captions)  # type: ignore[attr-defined]
                    critic.zero_grad()
                    real_loss = F.binary_cross_entropy_with_logits(self.network.discriminator(images, condition), torch.ones(len(images), device=self.device))  # type: ignore[attr-defined]
                    fake = self.network.generator(torch.randn(len(images), self.config.latent_size, device=self.device), condition).detach()  # type: ignore[attr-defined]
                    fake_loss = F.binary_cross_entropy_with_logits(self.network.discriminator(fake, condition), torch.zeros(len(images), device=self.device))  # type: ignore[attr-defined]
                    (real_loss + fake_loss).backward(); critic.step()
                    generator.zero_grad()
                    generated = self.network.generate(captions)  # type: ignore[attr-defined]
                    loss = F.binary_cross_entropy_with_logits(self.network.discriminator(generated, self.network.text(captions)), torch.ones(len(images), device=self.device))  # type: ignore[attr-defined]
                    loss.backward(); generator.step()
                else:
                    optimizer.zero_grad(); loss = self.network.loss(images, captions)  # type: ignore[attr-defined]
                    loss.backward(); torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0); optimizer.step()
                running += loss.item(); progress.set_postfix(loss=f"{running / max(progress.n, 1):.3f}")
            validation_loss = self._validation_loss(validation_data, batch_size)
            improved = self.config.model_type == "gan" or validation_loss < best_loss - min_delta
            best_loss, stalled = (validation_loss, 0) if improved else (best_loss, stalled + 1)
            optimizer_data = {"generator": generator.state_dict(), "critic": critic.state_dict()} if self.config.model_type == "gan" else optimizer.state_dict()
            self._save(optimizer_data, validation_loss, best=improved)
            label = "GAN loss is not comparable across epochs" if self.config.model_type == "gan" else f"validation loss {validation_loss:.4f}"
            print(f"Epoch {self.epoch}: {label}" + (" (best)" if improved and self.config.model_type != "gan" else ""))
            if self.config.model_type == "diffusion" and stalled >= patience:
                print(f"Early stopping after {self.epoch} epochs (no validation improvement for {patience} epochs).")
                break
        return self

    def generate(self, prompt: Union[str, Sequence[str]], *, output_folder: Optional[Union[str, Path]] = None, seed: Optional[int] = None):
        """Generate a PIL image for one prompt or a list of prompts.

        Set ``output_folder`` to also save images as ``generated_001.png`` and
        so on.  A seed makes GAN noise (and diffusion noise) reproducible.
        """
        prompts, one = ([prompt], True) if isinstance(prompt, str) else (list(prompt), False)
        if not prompts or not all(isinstance(item, str) for item in prompts): raise TypeError("generate expects one prompt or a non-empty sequence of prompts.")
        if seed is not None: torch.manual_seed(seed)
        captions = nn.utils.rnn.pad_sequence([torch.tensor(self.vocabulary.encode(text, self.tokenizer, add_eos=False)) for text in prompts], batch_first=True, padding_value=self.vocabulary.pad_id).to(self.device)
        self.network.eval()
        with torch.no_grad():
            tensors = self.network.generate(captions, self.input_size) if self.config.model_type == "diffusion" else self.network.generate(captions)  # type: ignore[attr-defined]
        images = [_to_image(tensor) for tensor in tensors]
        if output_folder is not None:
            destination = Path(output_folder); destination.mkdir(parents=True, exist_ok=True)
            for index, image in enumerate(images, 1): image.save(destination / f"generated_{index:03d}.png")
        return images[0] if one else images


def train(output_folder: Union[str, Path], train_data: Sequence[Sequence[str]], validation_data: Sequence[Sequence[str]], *, model_type: str = "diffusion", input_size: Optional[Tuple[int, int]] = None, embedding_size: int = 128, latent_size: int = 128, diffusion_steps: int = 100, min_frequency: int = 1, epochs: int = 20, batch_size: int = 16, learning_rate: float = 0.0002, patience: int = 3, min_delta: float = 0.0, tokenizer: seq2seq.Tokenize = seq2seq._default_tokenizer, device: Optional[Union[str, torch.device]] = None) -> Text2ImageModel:
    """Train a conditional ``gan`` or ``diffusion`` generator from captioned images."""
    folder = Path(output_folder)
    if folder.exists() and any(folder.iterdir()): raise ValueError(f"Output folder must be empty: {folder}. Use continue_train for an existing model.")
    folder.mkdir(parents=True, exist_ok=True)
    pairs = image2text._image_pairs(train_data)
    size = tuple(input_size) if input_size else image2text.infer_input_size([path for path, _ in pairs])
    if len(size) != 2 or min(size) < 32 or any(side % 4 for side in size): raise ValueError("input_size must be a (width, height) pair, at least 32 pixels, with both sides divisible by 4.")
    print(f"Using image input size {size[0]}×{size[1]} pixels.")
    config = _Config(model_type.lower(), size, embedding_size, latent_size, diffusion_steps, min_frequency)
    if config.diffusion_steps < 2: raise ValueError("diffusion_steps must be at least 2.")
    vocabulary = seq2seq.Vocabulary.build([caption for _, caption in pairs], tokenizer, min_frequency)
    model = Text2ImageModel(_network(config, vocabulary), vocabulary, config, folder, tokenizer, device)
    (folder / "training_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    return model.fit(pairs, validation_data, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, patience=patience, min_delta=min_delta)


def load_model(path: Union[str, Path], *, device: Optional[Union[str, torch.device]] = None, tokenizer: seq2seq.Tokenize = seq2seq._default_tokenizer) -> Text2ImageModel:
    """Load the latest saved generator from a folder, or a checkpoint file."""
    path = Path(path); checkpoint_path = path / "model_latest.pt" if path.is_dir() else path
    if not checkpoint_path.is_file(): raise FileNotFoundError(f"No model checkpoint found at {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
    config_data = checkpoint["config"]; config_data["input_size"] = tuple(config_data["input_size"])
    config = _Config(**config_data); vocabulary = seq2seq.Vocabulary(checkpoint["vocabulary"])
    model = Text2ImageModel(_network(config, vocabulary), vocabulary, config, checkpoint_path.parent, tokenizer, device)
    model.network.load_state_dict(checkpoint["state_dict"]); model.epoch = checkpoint.get("epoch", 0)
    return model


def continue_train(output_folder: Union[str, Path], train_data: Sequence[Sequence[str]], validation_data: Sequence[Sequence[str]], **kwargs) -> Text2ImageModel:
    """Resume text-to-image training from the latest checkpoint."""
    folder = Path(output_folder); checkpoint = torch.load(folder / "model_latest.pt", map_location=kwargs.get("device", "cpu"))
    model = load_model(folder, device=kwargs.pop("device", None), tokenizer=kwargs.pop("tokenizer", seq2seq._default_tokenizer))
    return model.fit(train_data, validation_data, optimizer_state=checkpoint.get("optimizer"), **kwargs)
