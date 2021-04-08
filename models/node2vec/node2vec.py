"""Node2Vec core algorithm."""

from absl import logging
import time
from typing import Text, Optional, Union, Callable, List
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Embedding,
    Lambda,
    Concatenate,
)
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.sequence import skipgrams
import tensorflow.keras.backend as K
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops

import apache_beam as beam
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils
try:
    from tensorflow.sparse import map_values
except:
    from tensorflow.python.framework import sparse_tensor

    def map_values(op, *args):
        """
        Applies the `op` to the `.values` tensor of one or more `SparseTensor`s.
        For tensorflow versions below 2.4. For versions above, use function
        `tf.sparse.map_values`.
        """
        return sparse_tensor.SparseTensor(
            args[0].indices, op(*[a.values for a in args]), args[0].dense_shape
        )


# %% Core module for node2vec sampling
def tf_sparse_multiply(a: tf.SparseTensor, b: tf.SparseTensor):
    """
    Sparse matrix multiplication between 2 SparseTensors in tensorflow.

    Parameters
    ----------
    a : tf.SparseTensor
    b : tf.SparseTensor

    Returns
    -------
    c : tf.SparseTensor
    """
    a_sm = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
        a.indices, a.values, a.dense_shape
    )

    b_sm = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
        b.indices, b.values, b.dense_shape
    )

    c_sm = sparse_csr_matrix_ops.sparse_matrix_sparse_mat_mul(
        a=a_sm, b=b_sm, type=tf.float32
    )

    c = sparse_csr_matrix_ops.csr_sparse_matrix_to_sparse_tensor(c_sm, tf.float32)

    return tf.SparseTensor(c.indices, c.values, dense_shape=c.dense_shape)


def sample_from_sparse(W_sample):
    """Take a sample given unnormalized sparse weight matrix."""
    # Normalize each row
    row_sum = tf.sparse.reduce_sum(W_sample, axis=1, keepdims=True)
    W_sample = W_sample.__div__(row_sum)
    W_sample = tf.sparse.reorder(W_sample)  # Make sure the indices are sorted

    # uniform_sample = tf.random.uniform((num_nodes, 1), minval=0, maxval=1)
    cdf = tf.map_fn(
        lambda x: map_values(
            lambda y: tf.cumsum(y) - tf.random.uniform((1,), minval=0, maxval=1), x
        ),
        W_sample,
    )  # map to each row
    is_pos = tf.greater_equal(cdf.values, 0)
    cdf_sample = tf.sparse.retain(cdf, is_pos)
    cdf_sample = tf.sparse.reorder(cdf_sample)

    # Materialize the samples: Take the first nonzero col of each row
    # s_next = tf.constant([list(item)[0][1] for _, item in \
    #    itertools.groupby(cdf_sample.indices.numpy(), lambda x: x[0])])
    # Casting to csr matrix
    index = cdf_sample.indices  # assuming sorted already
    indices = tf.concat(
        [tf.constant([1], dtype="int64"), index[1:, 0] - index[:-1, 0]], axis=0
    )
    s_next = index[:, 1][tf.greater(indices, 0)]

    return W_sample, cdf, cdf_sample, s_next


def random_walk_sampling_step_tf(
    W: tf.SparseTensor, s0: tf.Tensor, s1: tf.Tensor, p: float, q: float
):
    """
    Perform a 1-step sample of random walk.

    Parameters
    ----------
    W : tf.SparseTensor
        Adjacency weight matrix of the graph.
    s0 : tf.Tensor
        An array of samples (indices of nodes) in the prior step.
    s1 : tf.Tensor
        An array of samples (indices of nodes) in the current step.
    p : float
        node2vec Return Parameter
    q : float
        node2vec In-Out Parameter

    Returns
    -------
    W_sample : tf.SparseTensor
        The weight matrix used for sampling
    cdf : tf.SparseTensor
        The cumulative probability density matrix after
        subtracting the random uniform.
    cdf_sample : tf.SparseTensor
        The cumulative weight matrix after masking any
        negative values from cdf.
    s_next: tf.Tensor
        An array of samples (indices of nodes) drawn for the next step.
    """
    # Get dimension
    num_nodes = W.shape[0]

    # alpha_1 / P
    P = tf.sparse.SparseTensor(
        tf.cast(
            tf.stack([tf.range(num_nodes, dtype="int64"), s0], axis=1), dtype="int64"
        ),
        tf.ones(num_nodes),
        dense_shape=(num_nodes, num_nodes),
    )

    # alpha_2 / R
    A_0 = tf.sparse.SparseTensor(
        W.indices, tf.ones(W.indices.shape[0], dtype="float32"), dense_shape=W.shape
    )
    A_i_1 = tf_sparse_multiply(P, A_0)

    I = tf.sparse.SparseTensor(
        tf.cast(tf.stack([tf.range(num_nodes, dtype="int64"), s1], axis=1), "int64"),
        tf.ones(num_nodes),
        dense_shape=(num_nodes, num_nodes),
    )  # permutation matrix
    A_i = tf_sparse_multiply(I, A_0)

    ## intersection
    R = tf.sparse.minimum(A_i_1, A_i)
    is_nonzero = tf.not_equal(R.values, 0)
    R = tf.sparse.retain(R, is_nonzero)

    # alpha3 / Q
    Q = tf.sparse.add(A_i, P.__mul__(tf.constant([-1], dtype="float32")))
    Q = tf.sparse.add(Q, R.__mul__(tf.constant([-1], dtype="float32")))
    is_nonzero = tf.not_equal(Q.values, 0)
    Q = tf.sparse.retain(Q, is_nonzero)

    # Combine to get the final weight
    W_sample = tf.sparse.add(P.__mul__(tf.constant([1 / p], dtype="float32")), R)
    W_sample = tf.sparse.add(W_sample, Q.__mul__(tf.constant([1 / q], dtype="float32")))
    is_nonzero = tf.not_equal(W_sample.values, 0)
    W_sample = tf.sparse.retain(W_sample, is_nonzero)
    W_sample = tf.sparse.reorder(W_sample)

    # Make sure the orders of indices are the same
    W_new = tf_sparse_multiply(I, tf.cast(tf.sparse.reorder(W), dtype="float32"))
    W_new = tf.sparse.reorder(W_new)

    # Multiply the weights by creating a new sparse matrix
    W_sample = tf.sparse.SparseTensor(
        W_sample.indices,
        tf.multiply(W_sample.values, W_new.values),
        dense_shape=W_sample.shape,
    )

    # Taking samples from the sparse matrix
    W_sample, cdf, cdf_sample, s_next = sample_from_sparse(W_sample)

    return W_sample, cdf, cdf_sample, s_next


def sample_1_iteration(W, p, q, walk_length=80, symmetrify=True):
    W = tf.cast(W, "float32")
    if symmetrify:
        W = tf.sparse.maximum(W, tf.sparse.transpose(W))
    else:
        # Make sure each row has at least 1 entry. The case where
        # a row does not have a weight could happen when this is a
        # directed graph (A -> B but not B -> A). In this case,
        # we set the weight to itself as 1.
        indices = tf.sparse.to_dense(
            tf.sets.difference(
                [tf.range(W.shape[0], dtype="int64")], [tf.unique(W.indices[:, 0]).y]
            )
        )
        indices = tf.transpose(tf.concat([indices, indices], axis=0))
        terms = tf.sparse.SparseTensor(
            indices, tf.ones(indices.shape[0]), dense_shape=(10, 10)
        )
        W = tf.sparse.add(W, terms)
    W = tf.sparse.reorder(W)  # make sure the indices are sorted

    # First step
    s0 = tf.range(W.shape[0], dtype="int64")
    W_sample_1, cdf_1, cdf_sample_1, s1 = sample_from_sparse(W)
    S = [s0, s1]

    for _ in range(walk_length - 1):
        _, _, _, s_next = random_walk_sampling_step_tf(W, S[-2], S[-1], p, q)
        S.append(s_next)

    for ii, ss in enumerate(S):  # verbose print
        logging.info(f"s{ii}: {ss}")

    return S


# %% Numpy procedure to generate skipgrams
def generate_skipgram_numpy(S, vocab_size=10, window_size=4, negative_samples=0.0):
    pairs_mat, labels_arr = [], []
    for s in tqdm(S):  # each row
        pairs, labels = skipgrams(
            s,
            vocabulary_size=vocab_size,
            window_size=window_size,
            negative_samples=negative_samples,
        )
        pairs_mat.append(tf.convert_to_tensor(pairs))
        labels_arr.append(tf.convert_to_tensor(labels))

    pairs_mat = tf.concat(pairs_mat, axis=0)
    labels_arr = tf.concat(labels_arr, axis=0)

    # Target, context, label
    return pairs_mat[:, 0], pairs_mat[:, 1], labels_arr


# %% Generate skipgram with beam pipeline
def make_preproc_func(vocabulary_size, window_size, negative_samples):
    """Returns a preprocessing_fn to make skipgrams given the parameters."""

    def _make_skipgrams(s):
        """Numpy function to make skipgrams."""
        pairs, labels = skipgrams(
            s,
            vocabulary_size=vocabulary_size,
            window_size=window_size,
            negative_samples=negative_samples,
        )
        samples = np.concatenate(
            [np.asarray(pairs), np.asarray(labels)[:, None]], axis=1
        )
        return samples

    @tf.function
    def _tf_make_skipgrams(s):
        """tf nump / function wrapper."""
        y = tf.numpy_function(_make_skipgrams, [s], tf.int64)
        return y

    def _fn(inputs):
        """Preprocess input columns into transformed columns."""
        S = tf.stack(list(inputs.values()), axis=1)  # tf tensor

        out = tf.map_fn(
            _tf_make_skipgrams,
            S,
            fn_output_signature=tf.RaggedTensorSpec(
                shape=[None, 3], ragged_rank=0, dtype=tf.int64
            ),
        )

        out = out.to_tensor(default_value=-1)
        out = tf.reshape(out, (-1, 3))
        index = tf.reduce_all(tf.greater(out, -1), axis=1)
        out = tf.boolean_mask(out, index, axis=0)

        output = {}
        output["target"] = out[:, 0]
        output["context"] = out[:, 1]
        output["label"] = out[:, 2]

        return output

    return _fn


def generate_skipgram_beam(
    features,
    vocabulary_size=10,
    window_size=2,
    negative_samples=0.0,
    feature_names=None,
    temp_dir="/tmp",
    save_path="temp",
):
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(features.shape[1])]
    assert len(feature_names) == features.shape[1]

    # Convert to list of dict dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        {f"s{i}": features[:, i] for i in range(features.shape[1])}
    )
    dataset_schema = dataset_metadata.DatasetMetadata(
        schema_utils.schema_from_feature_spec(
            {
                f"s{i}": tf.io.FixedLenFeature([], tf.int64)
                for i in range(features.shape[1])
            }
        )
    )

    # Make the preprocessing_fn
    preprocessing_fn = make_preproc_func(vocabulary_size, window_size, negative_samples)

    # Run the beam pipeline
    with tft_beam.Context(temp_dir=temp_dir):
        transformed_dataset, transform_fn = (  # pylint: disable=unused-variable
            dataset.as_numpy_iterator(),
            dataset_schema,
        ) | "Make Skipgrams " >> tft_beam.AnalyzeAndTransformDataset(preprocessing_fn)

    # pylint: disable=unused-variable
    transformed_data, transformed_metadata = transformed_dataset
    saved_results = (
        transformed_data
        | "Write to TFRecord"
        >> beam.io.tfrecordio.WriteToTFRecord(
            file_path_prefix=save_path,
            file_name_suffix=".tfrecords",
            coder=tft.coders.example_proto_coder.ExampleProtoCoder(
                transformed_metadata.schema
            ),
        )
    )
    # print('\nRaw data:\n{}\n'.format(pprint.pformat(dataset)))
    # print('Transformed data:\n{}'.format(pprint.pformat(transformed_data)))
    # Return the list of paths of tfrecords
    num_rows_saved = len(transformed_data)

    return saved_results, num_rows_saved


# %% Skipgram keras model
class SkipGram(tf.keras.Model):
    """
    SkipGram model for Word2Vec.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    embed_size : int
        Embedding size.
    num_neg_samples : int, optional
        Number of negative samples, by default -1, which
        does not take any negative samples.
    """

    def __init__(self, vocab_size, embed_size, num_neg_samples=-1):
        """
        Construct a SkipGram model for Word2Vec.

        Parameters
        ----------
        vocab_size : int
            Size of the vocabulary.
        embed_size : int
            Embedding size.
        num_neg_samples : int, optional
            Number of negative samples, by default -1, which
            does not take any negative samples.
        """
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_neg_samples = num_neg_samples

        self.embeddings = Embedding(
            input_dim=vocab_size, output_dim=embed_size, name="Embedding"
        )
        self.pool_layer = Lambda(lambda x: K.sum(x, axis=1), name="AvgPool")

        if num_neg_samples > 0:
            self.softmax_weights = Embedding(
                input_dim=vocab_size, output_dim=embed_size, name="softmax_weight"
            )
            self.softmax_biases = Embedding(
                input_dim=vocab_size, output_dim=1, name="softmax_bias"
            )

            self.negatives = Lambda(
                lambda x: K.random_uniform(
                    (K.shape(x)[0], num_neg_samples),
                    minval=tf.cast(0, "int64"),
                    maxval=tf.cast(vocab_size, "int64"),
                    dtype="int64",
                ),
                name="negative_sampling",
            )
            self.concat_samples = Concatenate(name="samples_concat")

            # Sampled softmax dense layer
            self.dense = Lambda(
                lambda x: K.softmax(
                    (K.batch_dot(x[1], K.expand_dims(x[0], 2)) + x[2])[:, :, 0]
                ),
                name="sampled_softmax_dense",
            )  # x = [embed_pool, softmax_w, softmax_b]
        else:
            self.dense = Dense(
                vocab_size, activation="softmax", name="full_softmax_dense"
            )

    def call(self, input_data):
        """Model call."""
        if self.num_neg_samples > 0:  # use sampled softmax
            target_word, context_word = input_data[0], input_data[1]
            if len(target_word.shape) < 2:  # make sure it's 2D
                target_word = K.expand_dims(target_word, axis=1)
            if len(context_word.shape) < 2:  # make sure it's 2D
                context_word = K.expand_dim(context_word, axis=1)
            # Get embeddings of the input
            embed = self.embeddings(context_word)
            embed_pool = self.pool_layer(embed)
            # Draw negative samples
            negatives = self.negatives(target_word)
            # Concatenate all the samples
            samples = self.concat_samples([target_word, negatives])
            # Get embeddings of all samples
            softmax_w = self.softmax_weights(samples)
            softmax_b = self.softmax_biases(samples)
            # Activation for softmax
            # Using Embedding to save the parameters, and then
            # use matmul to make the sub-sampled Dense() layer
            y_hat = self.dense([embed_pool, softmax_w, softmax_b])

        else:  # use full softmax
            embed = self.embeddings(input_data)
            embed_pool = self.pool_layer(embed)  # pooling
            y_hat = self.dense(embed_pool)  # predicted class

        return y_hat

    def model(self):
        """Construct the model."""
        x = Input(shape=(1,), dtype="int64", name="target")
        if self.num_neg_samples > 0:
            y = Input(shape=(1,), dtype="int64", name="context")
            return tf.keras.Model(inputs=[x, y], outputs=self.call([x, y]))
        else:
            return tf.keras.Model(inputs=x, outputs=self.call(x))


def build_keras_model(
    vocab_size: int,
    embed_size: int,
    num_neg_samples: Optional[int] = -1,
    loss: Optional[Union[Text, Callable]] = "sparse_categorical_crossentropy",
    optimizer: Optional[Union[Text, Callable]] = "adam",
    metrics: Optional[Union[List, Callable]] = ["accuracy"],
):
    """
    Build keras model

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    embed_size : int
        Embedding size.
    num_neg_samples : int, optional
        Number of negative samples, by default -1, which
        does not take any negative samples.
    loss : str, keras.losses, optional
        Loss function of training, by default "sparse_categorical_crossentropy"
    optimizer : str, keras.optimizers, optional
        Training optimizer, by default "adam"
    metrics : list(str), optional
        List of monitoring metrics, by default ["accuracy"]

    Returns
    -------
        Returns the compiled keras model
    """
    skipgram = SkipGram(
        vocab_size=vocab_size,
        embed_size=embed_size,
        num_neg_samples=num_neg_samples,
    )
    model = skipgram.model()
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    logging.info(model.summary())

    return model


class LossMetricsPrintCallback(Callback):
    """Print loss / metrics at epoch ends."""

    def __init__(self, epochs, metrics=["accuracy"], verbose=0):
        """Instantiate LossErrorPrintCallback object and initial attributes."""
        super(LossMetricsPrintCallback, self).__init__()
        self.epochs = epochs
        self.metrics = metrics
        self.verbose = verbose
        self.start_time = 0

    def on_epoch_begin(self, epoch, logs=None):
        """Operations at the start of the epoch."""
        # pylint: disable=unused-argument
        self.start_time = time.time()  # epoch start time

    def on_epoch_end(self, epoch, logs=None):
        """Print metrics at epoch end."""
        # pylint: disable=no-self-use
        epoch_dur = time.time() - self.start_time  # sec
        # Construct the string
        out_string = [f"Epoch {epoch + 1}/{self.epochs}: {epoch_dur:.0f}s\t"]
        iterated_fields = ["", "val_"] if "val_loss" in logs else [""]
        for prefix in iterated_fields:
            field = f"{prefix}loss"
            value = logs[field]
            out_string.append(f"{field}: {value:.4f}")
            for m in self.metrics:
                field = f"{prefix}{m}"
                value = logs[field]
                out_string.append(f"{field}: {value:.4f}")
        out_string = ",\t".join(out_string)

        logging.info(out_string)