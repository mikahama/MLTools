# MLTools

MLTools is a deliberately small, friendly Python library for machine-learning tasks.

## Word embeddings

Train dependency-light word2vec or fastText-style embeddings directly from a
string, text files, and/or folders (folders are searched recursively; hidden
files are ignored):

```python
from MLTools.LM import word_embeddings

model = word_embeddings.train("my_embeddings", text="A dog chases a cat.",
                              files=["books", "notes.txt"], model_type="fasttext")
model.similarity("dog", "cat")
model.nearest_neighbors("dog")
model.embed("dog")
model.word(model.embed("dog"))
model.cluster(["dog", "cat", "car", "bike"], method="dbscan")

# Load it later.
model = word_embeddings.load_model("my_embeddings")
```

## Foundation models

`foundation_models` is a small wrapper around Hugging Face encoder models. It
trains BERT by default with masked-language modelling; set `model_type="roberta"`
or pass a compatible Hugging Face `model_name` to use another BERT-style model.

```python
from MLTools.LM import foundation_models

model = foundation_models.train("my_bert", text="My corpus...", files=["books"])
model.similarity("dog is funny", "cat is crazy")
embedding = model.embed("dog is funny")
model.cluster(["dog is funny", "cat is crazy", "cars are big", "bikes are fast"])

model = foundation_models.load_model("my_bert")
# Hub IDs work too: foundation_models.load_model("bert-base-uncased")

data = [["this is cool", "positive"], ["this is bad", "negative"]]
classifier = model.fine_tune("sentiment_model", data, "classification")
classifier.predict("this dog is funny")
```

For sentence-pair classification, use `[first_text, second_text, label]`; for
regression, use `[text, number]` with `task="regression"`.

## GPT-style language models

Fine-tune GPT-2 (the default) or another Hugging Face causal model directly on
your files and generate continuations. `fine_tune` updates and returns the
model, so either call style is valid.

```python
from MLTools.LM import gpts

model = gpts.train("my_gpt", text="My corpus...", files=["books"])
model.predict("Hi, how are you?")
model.similarity("dog is funny", "cat is crazy")
model.embed("dog is funny")
model.cluster(["dog is funny", "cat is crazy", "cars are big", "bikes are fast"])

model = gpts.load_model("my_gpt")
model.fine_tune("new_gpt", text="More corpus text")
model.predict("Hi, how are you?")
```

## Text to text models

```python
from MLTools import seq2seq

corpus = [["hei", "hello"], ["Minulla on koira", "I have a dog"]]
train, validation, test = seq2seq.split(corpus)

model = seq2seq.train("output_folder", train, validation)
model.predict("Kissa juoksee")
model.predict(["kissa juoksee", "koira hyppii"], n_best=3, return_scores=True)
model.evaluate(test)

# Later, resume training or load the most recent checkpoint.
model = seq2seq.continue_train("output_folder", train, validation, epochs=10)
model = seq2seq.load_model("output_folder")
```

The default is a bidirectional-GRU encoder with attention and a GRU decoder.
Choose `model_type="transformer"` to train a Transformer instead. Training saves
`model_latest.pt` after every epoch, plus OpenNMT-style numbered checkpoints and
`model_best.pt` when validation loss improves. Validation early stopping defaults
to `patience=3` epochs.

## Images to text

Give `image2text` pairs containing an image filename and its description. It
checks the image files, samples their dimensions, and chooses a sensible common
size automatically. Every image is then resized before it reaches the model.

```python
from MLTools import image2text

captions = [
    ["photos/cat.jpg", "a cat sits on a chair"],
    ["photos/dog.jpg", "a dog runs in the grass"],
]
train, validation, test = image2text.split(captions)

# Default: a compact CNN, a good starting point for small custom datasets.
model = image2text.train("caption_model", train, validation)
model.predict("photos/cat.jpg")
model.evaluate(test)

# Other encoders: "resnet18", "resnet50", and "mobilenet_v3_small".
# pretrained=True downloads ImageNet weights the first time it is used.
model = image2text.train("better_caption_model", train, validation,
                         model_type="resnet18", pretrained=True)
```

Pass `input_size=(width, height)` to override the automatic choice. The chosen
size is printed, exposed as `model.input_size`, and stored in each checkpoint;
prediction always applies the same resize. The module also has
`continue_train()` and `load_model()` with the same behavior as `seq2seq`.

## Text to images

`text2image` learns from the same `[image_path, caption]` format and makes a
new image from a prompt. It also detects a suitable input resolution from the
training image files, preserves their usual aspect ratio, and resizes images
automatically during training.

```python
from MLTools import text2image

examples = [
    ["art/sunrise.png", "an orange sunrise over mountains"],
    ["art/forest.png", "a green forest in summer"],
]
train, validation, test = text2image.split(examples)

# Default: a compact diffusion model.
model = text2image.train("my_generator", train, validation)
image = model.generate("an orange sunrise over mountains")
image.save("sunrise.png")

# A faster, adversarial alternative.
gan = text2image.train("my_gan", train, validation, model_type="gan")
gan.generate(["a summer forest", "a snowy forest"], output_folder="generated")
```

The included models are intentionally compact conditional models, suited to
learning a visually consistent dataset rather than producing general-purpose,
photorealistic images from a handful of examples. Diffusion uses validation
early stopping; GAN checkpoints are saved every epoch because adversarial loss
is not a reliable early-stopping signal. Use `input_size=(width, height)` to
override automatic sizing; `model.input_size` shows the selected resolution.

## Audio to text (speech recognition)

`audio2text` learns transcriptions from `[audio_path, transcription]` pairs.
It automatically converts stereo recordings to mono, inspects the recording
sample rates, chooses a common standard rate, resamples the audio, and produces
log-mel speech features. Transcription is character based, so it works without
a language-specific tokenizer.

```python
from MLTools import audio2text

recordings = [
    ["audio/hello.wav", "hello world"],
    ["audio/good_morning.wav", "good morning"],
]
train, validation, test = audio2text.split(recordings)

# Default: a bidirectional-GRU recognizer with CTC training.
model = audio2text.train("speech_model", train, validation)
model.predict("audio/hello.wav")
model.evaluate(test)  # WER and CER

# A Transformer acoustic model is also available.
transformer = audio2text.train("transformer_speech", train, validation,
                               model_type="transformer")
```

Pass `sample_rate=16000` to override automatic sample-rate selection. The
selected rate is printed and saved in the model, so predictions receive the
same preprocessing. `predict(..., return_scores=True)`, `load_model()`, and
`continue_train()` are supported too.

## Text to audio (speech synthesis)

`text2audio` learns a single voice from `[audio_path, text]` pairs. It detects
and standardizes the recordings' sample rate automatically, predicts a mel
spectrogram from characters, and turns that spectrogram into a WAV file. This
is a straightforward baseline for a consistent, single-speaker dataset.

```python
from MLTools import text2audio

voice = [
    ["voice/hello.wav", "hello world"],
    ["voice/morning.wav", "good morning"],
]
train, validation, test = text2audio.split(voice)

model = text2audio.train("my_voice", train, validation)
speech = model.synthesize("hello world")
speech.save("hello.wav")

# Or save during synthesis.
model.synthesize("good morning", output_path="morning.wav")
```

`GeneratedAudio` contains the waveform and selected `sample_rate`. Use
`sample_rate=16000` to override automatic selection. Training has progress
bars, early stopping, checkpoints, `load_model()`, and `continue_train()` like
the other modules.

Install locally with:

```bash
pip install -e .
```
