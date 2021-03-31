"""TFX preprocessing.

This file defines a template for TFX Transform component.
"""

from __future__ import division
from __future__ import print_function

from absl import logging

import tensorflow as tf
import tensorflow_transform as tft

from models import features


def preprocessing_fn(inputs):
    """
    Preprocess data inputs.

    Parameters
    ----------
    inputs : dict-like
        map from feature keys to raw not-yet-transformed features.

    Returns
    -------
    output: dict
        Map from string feature key to transformed feature operations.
    """
    # String to integer indexing
    content = inputs["InSeasonSeries_Id"]
    token = inputs["token"]
    vocab_uri = tft.vocabulary(
        tf.concat([content, token], axis=0),
        vocab_filename="node_vocab.txt",
        name="node_vocab",
    )
    # Logging
    logging.info(f"graph vocabulary uri: {vocab_uri}")

    # output as a dict
    output = {}
    output["InSeasonSeries_Id"] = tft.apply_vocabulary(
        content, deferred_vocab_filename_tensor=vocab_uri, default_value=0
    )
    output["token"] = tft.apply_vocabulary(
        token, deferred_vocab_filename_tensor=vocab_uri, default_value=0
    )
    output["weight"] = 1.0 / inputs["token_count"]
    return output