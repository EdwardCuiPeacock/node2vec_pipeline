"""Routines for model running."""

from __future__ import division
from __future__ import print_function
from itertools import compress

import os

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras.callbacks import EarlyStopping

from models.node2vec.node2vec import (
    generate_skipgram_beam,
    generate_skipgram_numpy,
    build_keras_model,
    sample_1_iteration_numpy,
)

# from models.node2vec.node2vec_test import (
#    sample_1_iteration_tf,
#    sample_1_iteration_numpy,
# )

from tfx_bsl.tfxio import dataset_options


def _read_transformed_dataset(
    file_pattern,
    data_accessor,
    tf_transform_output,
    num_epochs=1,
    shuffle=False,
    sloppy_ordering=True,
    batch_size=100000,
):
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
            num_epochs=num_epochs,
            shuffle=shuffle,
            sloppy_ordering=sloppy_ordering,
            batch_size=int(batch_size),
        ),
        tf_transform_output.transformed_metadata.schema,
    )


def _create_sampled_training_data(
    file_pattern,
    storage_path,
    data_accessor,
    tf_transform_output,
    window_size,
    p,
    q,
    walk_length,
    train_repetitions,
    eval_repetitions,
    temp_dir="/tmp",
    seed=None,
    beam_pipeline_args=None,
):
    """
    Sample from the graph and save the samples.

    Parameters
    ----------
    file_pattern: list(str)
        list of files (patterns) coming out of the Transform step,
        with pattern "{{ PIPELINE ROOT }}/Transform/transformed_examples/{{ run_id }}/[train:eval]/*.gz"
    storage_path: str
        Output directory of the graph samples
    data_accessor : tfx.components.trainer.fn_args_utils.DataAccessor
        DataAccessor for converting input to RecordBatch.
    tf_transform_output : tft.TFTransformOutput
        A TFTransformOutput.
    window_size : int, optional
        Window size of skipgram, by default 2
    p : float
        node2vec Return Parameter
    q : float
        node2vec In-Out Parameter
     walk_length : int, optional
        Number of steps to take in the random walk,
        by default 80
    train_repetitions : int
        Number of times to sample the graph to make training data
    eval_repetitions : int
        Number of times to sample the graph to make evaluation data
    temp_dir: str, optional
        Temporary directory to store the intermediate results for beam
    seed: int, optional
        Starting seed of graph sample generation. The default is None.
    beam_pipeline_args: dict, optional.
        Beam pipeline arguments.

    Returns
    -------
    [type]
        [description]
    """
    dataset_iterable = _read_transformed_dataset(
        file_pattern, data_accessor, tf_transform_output
    )
    print("Loaded dataset schema ...")
    print(dataset_iterable)
    # Iterate over the batches and build the final dict
    dataset = {"indices": [], "weight": []}
    for batch_data in dataset_iterable:
        dataset["indices"].append(
            tf.concat([batch_data["InSeasonSeries_Id"], batch_data["token"]], axis=1)
        )
        dataset["weight"].append(batch_data["weight"])

    # Merge into a single tensor
    dataset = {k: tf.concat(v, axis=0) for k, v in dataset.items()}
    dataset["weight"] = tf.reshape(dataset["weight"], shape=(-1,))  # flatten

    # Remove any unmatched vocabularies
    mask = tf.reduce_all(dataset["indices"] >= 0, axis=1)
    dataset["indices"] = tf.boolean_mask(dataset["indices"], mask, axis=0)
    dataset["weight"] = tf.boolean_mask(dataset["weight"], mask, axis=0)

    # print(dataset) # verbose print
    count_unique_nodes = int(
        tf.shape(tf.unique(tf.reshape(dataset["indices"], shape=(-1,)))[0])[0]
    )

    num_nodes = int(tf.reduce_max(dataset["indices"])) + 1

    print(f"Max index / Num unique nodes: {num_nodes} / {count_unique_nodes}")
    num_edges = len(dataset["weight"])
    print(f"num edges: {num_edges}")

    # Build the graph from the entire dataset
    # W = tf.sparse.SparseTensor(
    #    dataset["indices"], dataset["weight"], dense_shape=(num_nodes, num_nodes)
    # )
    W = coo_matrix(
        (
            dataset["weight"].numpy(),
            (
                dataset["indices"][:, 0].numpy().astype(np.int32),
                dataset["indices"][:, 1].numpy().astype(np.int32),
            ),
        ),
        shape=(num_nodes, num_nodes),
    )

    # Check to see if all rows have at least 1 neighbor
    sample_metadata = {
        "train": {
            "random_walk_uri_list": [],
            "skipgram_uri_list": [],
            "data_size": 0,
            "repetitions": train_repetitions,
        },
        "eval": {
            "random_walk_uri_list": [],
            "skipgram_uri_list": [],
            "data_size": 0,
            "repetitions": eval_repetitions,
        },
    }
    # Perform the node2vec random walk
    print("Perform the node2vec random walk")
    for phase in ["train", "eval"]:
        for r in range(sample_metadata[phase]["repetitions"]):
            # Take the sample
            cur_seed = (
                (
                    seed
                    + (r + (train_repetitions if phase == "eval" else 0)) * walk_length
                )
                if seed is not None
                else None
            )
            S = sample_1_iteration_numpy(W, p, q, walk_length, seed=cur_seed)
            S = tf.cast(tf.transpose(tf.stack(S, axis=0)), "int32")

            # Write the tensor to a TFRecord file
            data_uri = os.path.join(
                storage_path, f"random_walk_{phase}", f"graph_sample_{r:05}.tfrecord"
            )
            print(f"Phase {phase} r={r} random walk data_uri: {data_uri}")
            sample_metadata[phase]["random_walk_uri_list"].append(data_uri)

            ds = tf.data.Dataset.from_tensor_slices(S).map(tf.io.serialize_tensor)
            writer = tf.data.experimental.TFRecordWriter(
                data_uri, compression_type="GZIP"
            )
            writer.write(ds)

    print(f"Successfully created random walk datasets")

    # Generate Skipgrams
    print("Generate Skipgrams")
    print(f"Storing graph sampled skipgrams at {storage_path}")
    for phase in ["train", "eval"]:
        cur_seed = (cur_seed + 1) if seed is not None else None
        print("Current phase is: ", phase)
        saved_results, num_rows_saved = generate_skipgram_numpy(
            sample_metadata[phase]["random_walk_uri_list"],
            window_size=window_size,
            buffer_size=10000,
            save_path=os.path.join(storage_path, phase),
            num_targets=1,
        )

        sample_metadata[phase]["skipgram_uri_list"] = saved_results
        sample_metadata[phase]["data_size"] = num_rows_saved

    train_data_uri_list = sample_metadata["train"]["skipgram_uri_list"]
    train_data_size = sample_metadata["train"]["data_size"]
    eval_data_uri_list = sample_metadata["eval"]["skipgram_uri_list"]
    eval_data_size = sample_metadata["eval"]["data_size"]

    print(f"Successfully created graph sampled skipgrams {storage_path}")
    print("train data")
    print(train_data_uri_list)
    print(f"train data size: {train_data_size}")
    print("eval data")
    print(eval_data_uri_list)
    print(f"eval data size: {eval_data_size}")

    return (
        train_data_uri_list,
        eval_data_uri_list,
        train_data_size,
        eval_data_size,
        num_nodes,
    )


def _input_fn(data_uri_list, batch_size=128, num_epochs=10, shuffle=False, seed=None):
    """
    Create a dataset that contains ((target, context), label)

    Parameters
    ----------
    data_uri_list : list(str)
        List of data uris / path to the TFRecord files
    batch_size : int, optional
        Batch size of the sample, by default 128
    num_epochs : int, optional
        Number of epochs to train, by default 10
    shuffle : bool, optional
        Whether or not to shuffle, by default True
    seed : int
        Sampling seed. The default is None

    Returns
    -------
    tf.data.Dataset
        SkipGram training dataset of ((target, context), label)
    """

    def map_fn(x):
        x = tf.io.parse_tensor(x, tf.int64)
        x.set_shape([None])
        # (target, context), (softmax sparse label index = 0)
        return (tf.cast(x[0], "int32"), tf.cast(x[1], "int32")), tf.cast(
            tf.constant([0]), "int32"
        )

    # Return as (target, context), label
    dataset = tf.data.TFRecordDataset(data_uri_list, compression_type="GZIP").map(
        map_fn
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=10000, seed=seed, reshuffle_each_iteration=True
        )

    dataset = dataset.batch(batch_size).repeat(num_epochs)

    return dataset


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
        - fn_args.serving_model_dir: str
            Path to model serving directory.
        - fn_args.transform_output: str
            Transformed output uri
        - fn_args.custom_config: dict
            A dictionary of additional custom configs for modeling.
            - "model_config": model configurations
            - "system_config": system configurations
    """
    fn_lists = str(fn_args.__dict__)
    print(f"fn attributes {fn_lists}")
    print(f"tansformed_oputput {fn_args.transform_output}")

    print("See if GPU is available")
    print(tf.config.list_physical_devices("GPU"))

    print(f"Tensorflow version: {tf.__version__}")

    # Get some parameters
    system_config = fn_args.custom_config["system_config"]
    model_config = fn_args.custom_config["model_config"]
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    num_epochs = model_config.get("num_epochs", 30)

    # Create the dataset
    graph_sample_path = fn_args.model_run_dir.replace("model_run", "graph_samples")
    # Generate the samples
    (
        train_data_uri_list,
        eval_data_uri_list,
        train_data_size,
        eval_data_size,
        num_nodes,
    ) = _create_sampled_training_data(
        file_pattern=fn_args.train_files,
        storage_path=graph_sample_path,
        data_accessor=fn_args.data_accessor,
        tf_transform_output=tf_transform_output,
        window_size=model_config["window_size"],
        p=model_config["p"],
        q=model_config["q"],
        walk_length=model_config["walk_length"],
        train_repetitions=model_config["train_repetitions"],
        eval_repetitions=model_config["eval_repetitions"],
        seed=model_config["seed"],
        beam_pipeline_args=system_config["DATAFLOW_BEAM_PIPELINE_ARGS"],
    )

    # Load the created dataset
    train_batch_size = model_config.get("train_batch_size", 128)
    train_dataset = _input_fn(
        train_data_uri_list,
        batch_size=train_batch_size,
        num_epochs=num_epochs,
        shuffle=True,
        seed=model_config["seed"],
    )

    eval_batch_size = model_config.get("eval_batch_size") or train_batch_size
    eval_dataset = _input_fn(
        eval_data_uri_list,
        batch_size=eval_batch_size,  # default to train batch_size
        num_epochs=1,
        shuffle=False,
    )

    # Build the model
    previous_model_path = model_config["continue_training"]
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        if previous_model_path is not None and previous_model_path != "":
            model = tf.keras.models.load_model(previous_model_path)
            print("Continue training from:", previous_model_path)
            print(model.summary())
        else:
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
        log_dir=fn_args.model_run_dir,
        update_freq=1000,
    )

    # Model checkpoint
    check_points = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(system_config["GCS_BUCKET_NAME"], "tmp", "checkpoint"),
        monitor="val_loss",
        save_freq="epoch",
    )

    # Do the fitting
    train_steps = train_data_size // train_batch_size + 1 * (
        train_data_size % train_batch_size > 0
    )
    eval_steps = eval_data_size // eval_batch_size + 1 * (
        eval_data_size % eval_batch_size > 0
    )
    print(f"Train steps: {train_steps}, Eval steps: {eval_steps}")

    model.fit(
        train_dataset,
        epochs=num_epochs,
        steps_per_epoch=train_steps,
        validation_data=eval_dataset,
        validation_steps=eval_steps,
        callbacks=[tensorboard_callback, check_points],
        verbose=1,
    )

    model.save(fn_args.serving_model_dir, save_format="tf", signatures={})

    # Artificially raise an error so that the training can be rerun
    # raise (ValueError("Artificial Error: Attempting to rerun the model with cache ..."))
