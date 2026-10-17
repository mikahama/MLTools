"""MLTools: approachable machine learning utilities.

Public modules include :mod:`MLTools.seq2seq` for text-to-text models,
:mod:`MLTools.image2text` for image captioning, :mod:`MLTools.text2image` for
simple conditional image generation, and :mod:`MLTools.audio2text` for speech
recognition, and :mod:`MLTools.text2audio` for speech synthesis.
"""

import importlib

__all__ = ["LM", "seq2seq", "image2text", "text2image", "audio2text", "text2audio"]


def __getattr__(name):
    """Import task modules only when requested.

    This lets, for example, the text-only language-model tools work in an
    environment where an optional audio backend has not been installed.
    """
    if name in __all__:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
