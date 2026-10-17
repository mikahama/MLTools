"""Language-modelling tools, including dependency-light word embeddings.

Typical use::

    from MLTools.LM import word_embeddings
    model = word_embeddings.train("my_embeddings", text="A small corpus")
    model.nearest_neighbors("corpus")
"""

from . import foundation_models, gpts, word_embeddings

__all__ = ["word_embeddings", "foundation_models", "gpts"]
