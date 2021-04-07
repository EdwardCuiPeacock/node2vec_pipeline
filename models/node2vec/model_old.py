"""Routines for model running."""

from __future__ import division
from __future__ import print_function

import os
from absl import logging

import tensorflow as tf
import tensorflow_transform as tft

from models.node2vec.node2vec import (
    SkipGram,
    sample_1_iteration,
    generate_skipgram_numpy,
    build_keras_model,
    LossMetricsPrintCallback,
)

from tfx_bsl.tfxio import dataset_options


def tensor2tfrecord(S, data_uri="temp.tfrecord", compression_type="GZIP"):
    """Write a tensor to a single TFRecord file."""
    ds = tf.data.Dataset.from_tensor_slices(S).map(tf.io.serialize_tensor)
    writer = tf.data.experimental.TFRecordWriter(
        data_uri, compression_type=compression_type
    )
    writer.write(ds)


def record2dataset(
    data_uri="temp.tfrecord", compression_type="GZIP", decode_as=tf.int64
):
    """Read from a TFRecord file or a list of files."""

    def parse_tensor_f(x):
        xp = tf.io.parse_tensor(x, decode_as)
        xp.set_shape([None])
        return (xp[0], xp[1]), xp[2]

    dataset = tf.data.TFRecordDataset(data_uri, compression_type=compression_type).map(
        parse_tensor_f
    )
    return dataset


def _read_transformed_dataset(file_pattern, data_accessor, tf_transform_output):
    """
    Read data coming out of Transformation component.

    Parameters
    ----------
    file_pattern : list(str)
        List of paths or patterns of input tfrecord files.
    data_accessor : tfx.components.trainer.fn_args_utils.DataAccessor
        DataAccessor for converting input to RecordBatch.
    tf_transform_output : tft.TFTransformOutput
        A TFTransformOutput.

    Returns
    -------
    tf.Dataset (iterable)
        An iterable dataset where each iteration returns a data batch
        as dictionary {"field1": array[...], "field1": array[...], ...}
    """
    return data_accessor.tf_dataset_factory(
        file_pattern,
        dataset_options.TensorFlowDatasetOptions(
            num_epochs=1,
            shuffle=False,
            sloppy_ordering=True,
            batch_size=int(1e5),
        ),
        tf_transform_output.transformed_metadata.schema,
    )
    return dataset


def _create_sampled_training_data(
    file_pattern,
    storage_path,
    data_accessor,
    tf_transform_output,
    window_size,
    negative_samples,
    p,
    q,
    walk_length,
    train_repetitions,
    eval_repetitions,
):
    """Sample from the graph and save the samples.
    file_pattern: list(str)
        list of files (patterns) coming out of the Transform step,
        with pattern "{{ PIPELINE ROOT }}/Transform/transformed_examples/{{ run_id }}/[train:eval]/*.gz"
    storage_path: str
        Output directory of the graph samples
    data_accessor : tfx.components.trainer.fn_args_utils.DataAccessor
        DataAccessor for converting input to RecordBatch.
    tf_transform_output : tft.TFTransformOutput
        A TFTransformOutput.
    window_size, negative_samples, p, q, walk_length, repetitions: node2vec parameters
    """
    dataset_iterable = _read_transformed_dataset(
        file_pattern, data_accessor, tf_transform_output
    )
    logging.info("Loaded dataset schema ...")
    logging.info(dataset_iterable)
    # Iterate over the batches and build the final dict
    dataset = {"indices": [], "weight": []}
    for batch_num, batch_data in enumerate(dataset_iterable):
        dataset["indices"].append(
            tf.concat([batch_data["InSeasonSeries_Id"], batch_data["token"]], axis=1)
        )
        dataset["weight"].append(batch_data["weight"])
    # Merge into a single tensor
    dataset = {k: tf.concat(v, axis=0) for k, v in dataset.items()}
    dataset["weight"] = tf.reshape(dataset["weight"], shape=(-1,))  # flatten
    # logging.info(dataset) # verbose print

    # Build the graph from the entire dataset
    num_nodes = int(
        tf.shape(tf.unique(tf.reshape(dataset["indices"], shape=(-1,)))[0])[0]
    )
    W = tf.sparse.SparseTensor(
        dataset["indices"], dataset["weight"], dense_shape=(num_nodes, num_nodes)
    )

    logging.info(f"Num nodes: {num_nodes}")

    train_data_uri_list = []
    train_data_size = 0
    # Create the TFRecords for training data
    for r in range(train_repetitions):
        # Take the sample
        S = sample_1_iteration(W, p, q, walk_length)

        # Write the tensor to a TFRecord file
        S = tf.transpose(tf.stack(S, axis=0))
        targets, contexts, labels = generate_skipgram_numpy(
            S, num_nodes, window_size, negative_samples
        )
        data_uri = os.path.join(storage_path, "train", f"graph_sample_{r:05}.tfrecord")
        sample_out = tf.stack(
            [tf.cast(xx, "int64") for xx in [targets, contexts, labels]], axis=1
        )
        tensor2tfrecord(sample_out, data_uri=data_uri)
        train_data_uri_list.append(data_uri)
        train_data_size += len(labels)

    eval_data_uri_list = []
    eval_data_size = 0
    for r in range(eval_repetitions):
        # Take the sample
        S = sample_1_iteration(W, p, q, walk_length)

        # Write the tensor to a TFRecord file
        S = tf.transpose(tf.stack(S, axis=0))
        targets, contexts, labels = generate_skipgram(
            S, num_nodes, window_size, negative_samples
        )
        data_uri = os.path.join(storage_path, "eval", f"graph_sample_{r:05}.tfrecord")
        sample_out = tf.stack(
            [tf.cast(xx, "int64") for xx in [targets, contexts, labels]], axis=1
        )
        tensor2tfrecord(sample_out, data_uri=data_uri)
        eval_data_uri_list.append(data_uri)
        eval_data_size += len(labels)

    logging.info(f"Successfully created graph sampled dataset {storage_path}")
    logging.info("train data")
    logging.info(train_data_uri_list)
    logging.info(f"train data size: {train_data_size}")
    logging.info("eval data")
    logging.info(eval_data_uri_list)
    logging.info(f"eval data size: {eval_data_size}")

    return (
        train_data_uri_list,
        eval_data_uri_list,
        train_data_size,
        eval_data_size,
        num_nodes,
    )


def _input_fn(data_uri_list, batch_size=128, num_epochs=5, shuffle=False):
    # Load the raw "sentences" data
    dataset = record2dataset(data_uri_list, decode_as=tf.int64)
    if shuffle:
        dataset = (
            dataset.shuffle(buffer_size=10000).batch(batch_size).repeat(num_epochs)
        )
    else:
        dataset = dataset.batch(batch_size).repeat(num_epochs)

    return dataset


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example and applies TFT."""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop("weight")  # features.LABEL_KEY
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
    fn_args : tfx.components.trainer.fn_args_utils.FnArgs
        Holds args used to train the model as attributes.

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
        - fn_args.transform_output: str
            Transformed output uri
        - fn_args.custom_config: dict
            A dictionary of additional custom configs for modeling.
            - "model_config": model configurations
            - "system_config": system configurations
    """
    fn_lists = str(fn_args.__dict__)
    logging.info(f"fn attributes {fn_lists}")
    logging.info(f"tansformed_oputput {fn_args.transform_output}")

    # Get some parameters
    system_config = fn_args.custom_config["system_config"]
    model_config = fn_args.custom_config["model_config"]
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    # Create the dataset
    graph_sample_path = os.path.join(system_config["PIPELINE_ROOT"], "graph_samples")
    train_sample_path = os.path.join(graph_sample_path, "train")
    eval_sample_path = os.path.join(graph_sample_path, "eval")
    (
        train_data_uri_list,
        eval_data_uri_list,
        train_data_size,
        eval_data_size,
        num_nodes,
    ) = _create_sampled_training_data(
        fn_args.train_files,
        train_sample_path,
        fn_args.data_accessor,
        tf_transform_output,
        model_config["window_size"],
        0.0,  # not generating negative samples from the preprocessing; use softmax negative sampling
        model_config["p"],
        model_config["q"],
        model_config["walk_length"],
        model_config["train_repetitions"],
        model_config["eval_repetitions"],
    )

    # TODO: Left off here Friday
    # Load the created dataset
    train_batch_size = model_config.get("train_batch_size", 128)
    train_dataset = _input_fn(
        train_data_uri_list,
        batch_size=train_batch_size,
        num_epochs=model_config.get("num_epochs", 10),
    )
    eval_batch_size = model_config.get("eval_batch_size") or train_batch_size
    eval_dataset = _input_fn(
        eval_data_uri_list,
        batch_size=eval_batch_size,  # default to train batch_size
        num_epochs=1,
        shuffle=False,
    )

    # Build the model
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = build_keras_model(
            vocab_size=int(num_nodes),
            embed_size=model_config.get("embed_size", 32),
            num_neg_samples=model_config.get("num_neg_samples", 12),
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )

    # Write logs to path
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq="batch"
    )
    
    logging.info("See if GPU is available")
    logging.info(tf.config.list_physical_devices("GPU"))

    # Do the fitting
    train_steps = train_data_size // train_batch_size + 1 * (
        train_data_size % train_batch_size > 0
    )
    eval_steps = eval_data_size // eval_batch_size + 1 * (
        eval_data_size % eval_batch_size > 0
    )
    num_epochs = model_config.get("num_epochs", 30)
    model.fit(
        train_dataset,
        epochs=num_epochs,
        steps_per_epoch=train_steps,
        validation_data=eval_dataset,
        validation_steps=eval_steps,
        callbacks=[tensorboard_callback, LossMetricsPrintCallback(epochs=num_epochs, metrics=["accuracy"])],
        verbose=0,
    )

    # Save and serve the model
    # signatures = {
    #     "serving_default": _get_serve_tf_examples_fn(
    #         model, tf_transform_output
    #     ).get_concrete_function(
    #         tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
    #     ),
    # }

    model.save(fn_args.serving_model_dir, save_format="tf", signatures={})

    #raise(ValueError("Artificial Error: Attempting to rerun the model with cache ..."))