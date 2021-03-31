"""Routines for model running."""

from __future__ import division
from __future__ import print_function

import os
from absl import logging

import tensorflow as tf
import tensorflow_transform as tft

from models.node2vec.node2vec import SkipGram, sample_1_iteration

def tensor2tfrecord(S, data_uri='temp.tfrecord'):
    """Write a tensor to a single TFRecord file."""
    ds = (tf.data.Dataset.from_tensor_slices(S)
      .map(tf.io.serialize_tensor))
    writer = tf.data.experimental.TFRecordWriter(data_uri)
    writer.write(ds)

def record2tensor(data_uri='temp.tfrecord'):
    """Read from a TFRecord file or a list of files."""
    parse_tensor_f = lambda x: tf.io.parse_tensor(x, tf.int64)
    dataset = (tf.data.TFRecordDataset(data_uri)
        .map(parse_tensor_f))
    return dataset

def _create_sampled_training_data(dataset, num_nodes, p, q, walk_length, repetitions, storage_path):
    # dataset = [""]
    # Build the graph from the entire dataset
    indices = tf.concat([tf.expand_dims(dataset["content"], axis=1), 
                         tf.expand_dims(dataset["token"],   axis=1)], 
                         axis=1)
    values = dataset["weight"]
    W = tf.sparse.SparseTensor(indices, values, dense_shape=(num_nodes, num_nodes))

    data_uri_list = []
    # Create the TFRecords
    for r in range(repetitions):
        # Take the sample
        S = sample_1_iteration(W, p, q, walk_length)

        # Write the tensor to a TFRecord file
        S = tf.transpose(tf.stack(S, axis=0))
        data_uri = os.path.join(storage_path)
        tensor2tfrecord(S, data_uri=data_uri)
        data_uri_list.append(data_uri)

    return data_uri_list
    

def _input_fn(file_pattern, data_accessor, tf_transform_output, batch_size=128):
    """
    Generates features and label for tuning/training.

    Parameters
    ----------
    file_pattern : list(str)
        List of paths or patterns of input tfrecord files.
    data_accessor : tfx.components.trainer.fn_args_utils.DataAccessor
        DataAccessor for converting input to RecordBatch.
    tf_transform_output : tft.TFTransformOutput
        A TFTransformOutput.
    batch_size : int, optional
        Batch size of data generator, by default 128

    Returns
    -------
    tf.Dataset
        A dataset that contains (features, indices) tuple where features is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    return data_accessor.tf_dataset_factory(
        file_pattern,
        dataset_options.TensorFlowDatasetOptions(
            batch_size=batch_size,
            label_key=features.transformed_name(features.LABEL_KEY),
        ),
        tf_transform_output.transformed_metadata.schema,
    )


def _build_keras_model(
    vocab_size,
    embed_size,
    num_samples=-1,
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
):
    """
    Build keras model

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    embed_size : int
        Embedding size.
    num_samples : int, optional
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
    SkipGram
        Returns the compiled skipgra model
    """
    skipgram = SkipGram(
        vocab_size=vocab_size, embed_size=embed_size, num_samples=num_samples
    )
    model = skipgram.model()
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example and applies TFT."""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(features.LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

        transformed_features = model.tft_layer(parsed_features)

        return model(transformed_features)

    return serve_tf_examples_fn


# TFX Trainer will call this function.
def run_fn(fn_args):
    """
    Train the model based on given args.

    Parameters
    ----------
    fn_args : custom class
        Holds args used to train the model as attributes.

        - fn_args.config: dict
            Model configurations read from the "model_configurations"
            section of the "metadata.yaml" file.
        - fn_args.train_files: str
            File path for training data.
        - fn_args.eval_files: str
            File path for evaluation data.
        - fn_args.data_accessor: DataAccessor 
            DataAccessor for converting input to RecordBatch.
        - fn_args.model_run_dir: str
            Path to model running directory.
        - fn_args.model_serve_dir: str
            Path to model serving directory.
    """
    # ?? tfx.components.trainer.fn_args_utils.FnArgs
    fn_type = type(fn_args)
    logging.info(f"fn_args type: {fn_type}")
    

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        fn_args.config.get("batch_size", 128),
    )
    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        fn_args.config.get("eval_batch_size") or \
            fn_args.config.get("batch_size", 128), # default to train batch_size
    )

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = _build_keras_model(
            vocab_size=fn_args.vocab_size, 
            embed_size=fn_args.config.get("embed_size", 32),
            num_samples=fn_args.config.get("num_samples", 12),
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )

    # Write logs to path
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq="batch"
    )

    # Do the fitting
    model.fit(
        train_dataset,
        epochs=fn_args.config.get("num_epochs", 30),
        steps_per_epoch=30, # TODO: this can be precomputed based on data size
        validation_data=eval_dataset,
        validation_steps=150, # TODO: this can be preocmputed based on the data size
        callbacks=[tensorboard_callback],
    )

    # Save and serve the model
    # signatures = {
    #     "serving_default": _get_serve_tf_examples_fn(
    #         model, tf_transform_output
    #     ).get_concrete_function(
    #         tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
    #     ),
    # }
    model.save(fn_args.model_serve_dir, save_format="tf", signatures={})
