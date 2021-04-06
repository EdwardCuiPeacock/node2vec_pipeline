"""Node2Vec core algorithm."""
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Lambda
import tensorflow.keras.backend as K
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops


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
        lambda x: tf.sparse.map_values(
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


def random_walk_sampling_step_tf(W: tf.SparseTensor, s0: tf.Tensor, s1: tf.Tensor, p: float, q: float):
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
    A_0 = tf.cast(tf.sparse.map_values(tf.ones_like, W), "float32")
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

    # Make sure the orders of indices are the same
    W_sample = tf.sparse.reorder(W_sample)
    W_new = tf_sparse_multiply(I, tf.cast(tf.sparse.reorder(W), dtype="float32"))
    W_new = tf.sparse.reorder(W_new)

    # Multiply the weights by creating a new sparse matrix
    W_sample = tf.sparse.map_values(tf.multiply, W_sample, W_new)

    # Taking samples from the sparse matrix
    W_sample, cdf, cdf_sample, s_next = sample_from_sparse(W_sample)

    return W_sample, cdf, cdf_sample, s_next


def sample_1_iteration(W, p, q, walk_length=80):
    W = tf.cast(W, "float32")

    # First step
    s0 = tf.range(W.shape[0], dtype="int64")
    _, _, _, s1 = sample_from_sparse(W)
    S = [s0, s1]

    for _ in range(walk_length - 1):
        _, _, _, s_next = random_walk_sampling_step_tf(W, S[-2], S[-1], p, q)
        S.append(s_next)
        
    return S


class SkipGram(tf.keras.Model):
    """
    SkipGram model for Word2Vec.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    embed_size : int
        Embedding size.
    num_samples : int, optional
        Number of negative samples, by default -1, which
        does not take any negative samples.
    """

    def __init__(self, vocab_size, embed_size, num_samples=-1):
        """
        Construct a SkipGram model for Word2Vec.

        Parameters
        ----------
        vocab_size : int
            Size of the vocabulary.
        embed_size : int
            Embedding size.
        num_samples : int, optional
            Number of negative samples, by default -1, which
            does not take any negative samples.
        """
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_samples = num_samples

        self.embeddings = Embedding(
            input_dim=vocab_size, output_dim=embed_size, name="Embedding"
        )
        self.pool_layer = Lambda(lambda x: K.sum(x, axis=1), name="AvgPool")

        if num_samples > 0:
            self.softmax_weights = Embedding(
                input_dim=vocab_size, output_dim=embed_size, name="softmax_weight"
            )
            self.softmax_biases = Embedding(
                input_dim=vocab_size, output_dim=1, name="softmax_bias"
            )

            self.negatives = Lambda(
                lambda x: K.random_uniform(
                    (K.shape(x)[0], num_samples),
                    minval=0,
                    maxval=vocab_size,
                    dtype="int32",
                ),
                name="negative_sampling",
            )
            self.concat_samples = Lambda(
                lambda x: K.concatenate(x), name="samples_concat"
            )

            # Sampled softmax dense layer
            self.dense = Lambda(
                lambda x: K.softmax(
                    (K.batch_dot(x[1], K.expand_dims(x[0], 2)) + x[2])[:, :, 0]
                ),
                name="sampled_softmax_dense",
            )
        else:
            self.dense = Dense(
                vocab_size, activation="softmax", name="full_softmax_dense"
            )

    def call(self, input_data):
        """Model call."""
        if self.num_samples > 0:  # use sampled softmax
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
        x = Input(shape=(1,), dtype="int32", name="target")
        if self.num_samples > 0:
            y = Input(shape=(1,), dtype="int32", name="context")
            return tf.keras.Model(inputs=[x, y], outputs=self.call([x, y]))
        else:
            return tf.keras.Model(inputs=x, outputs=self.call(x))
