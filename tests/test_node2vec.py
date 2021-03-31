"""Test functions for node2vec.py"""

import numpy as np
import pandas as pd
import tensorflow as tf

from models.node2vec.node2vec import random_walk_sampling_step_tf
from models.node2vec.node2vec import SkipGram


def test_random_walk_sampling_step_tf():
    """Test tensorflow.sparse implementation of random walk."""
    p = 0.2
    q = 0.8
    df = pd.DataFrame(
        {
            "content": [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "cast": [1, 2, 3, 4, 0, 2, 4, 0, 1, 3, 0, 2, 4, 0, 1, 3],
            "weight": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
    )
    W = tf.sparse.SparseTensor(
        df[["content", "cast"]].values, df["weight"].values, dense_shape=(5, 5)
    )
    s0 = tf.constant([3, 5, 2, 1, 4], dtype="int64") - 1
    s1 = tf.constant([2, 4, 1, 5, 3], dtype="int64") - 1
    W_sample, _, cdf_sample, s_next = random_walk_sampling_step_tf(W, s0, s1, p, q)

    # print(tf.sparse.to_dense(W_sample))
    # print(s_next)

    assert W_sample.shape[0] == 5
    assert len(s_next) == 5
    assert np.all(
        np.array([np.where(x)[0][0] for x in tf.sparse.to_dense(cdf_sample).numpy()])
        == s_next.numpy()
    )


def test_skipgram_full_softmax():
    """Test SkipGram model with full softmax output."""
    batch_size = 16
    embed_size = 32
    vocab_size = 1024
    full_softmax_model = SkipGram(
        vocab_size=vocab_size, embed_size=embed_size, num_samples=-1
    )
    full_softmax_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    example_target = tf.random.uniform(
        [batch_size, 1], minval=1, maxval=1000, dtype="int32"
    )
    y_hat = full_softmax_model(example_target)
    print(full_softmax_model.model().summary())

    assert y_hat.shape[0] == batch_size
    assert y_hat.shape[1] == vocab_size
    assert full_softmax_model.embeddings.embeddings.shape[0] == vocab_size
    assert full_softmax_model.embeddings.embeddings.shape[1] == embed_size


def test_skipgram_sampled_softmax():
    """Test SkipGram model with sampled softmax output."""
    batch_size = 16
    embed_size = 32
    vocab_size = 1024
    num_samples = 12
    sampled_softmax_model = SkipGram(
        vocab_size=vocab_size, embed_size=embed_size, num_samples=num_samples
    )
    sampled_softmax_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    example_target = tf.random.uniform(
        [batch_size, 1], minval=1, maxval=1000, dtype="int32"
    )
    example_context = tf.random.uniform(
        [batch_size, 1], minval=1, maxval=1000, dtype="int32"
    )
    y_hat = sampled_softmax_model([example_target, example_context])
    print(sampled_softmax_model.model().summary())

    assert y_hat.shape[0] == batch_size
    assert y_hat.shape[1] == num_samples + 1  # 1 pos sample + num negative samples
    assert sampled_softmax_model.embeddings.embeddings.shape[0] == vocab_size
    assert sampled_softmax_model.embeddings.embeddings.shape[1] == embed_size
