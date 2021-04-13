#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 19:20:05 2021

@author: edwardcui
"""
import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def tensors2tfrecord(filename, **kwargs):
    """
    Write a {feature_name: feature_val} dict to a TFRecord file.

    Parameters
    ----------
    filename : str
        Filename of the tfrecord to save to
    kwargs: dict
        Input the feature data as {feature_name: feature_val},
        or feature_name=feature_val

    The saved serialized tfrecord can be read with 
    tf.data.experimental.make_batched_features_dataset
    """
    feature_name, features = tuple(zip(*kwargs.items()))

    def serialize_example(*args):
        """
        Creates a tf.train.Example message ready to be written to a file.
        """
        # Create a dictionary mapping the feature name to the tf.train.Example-compatible
        # data type.
        feature = {}
        for i, val in enumerate(args):
            if val.dtype in [tf.int32, tf.int64]:
                casted_val = _int64_feature(val)
            elif val.dtype in [tf.float16, tf.float32, tf.float64]:
                casted_val = _float_feature(val)
            else:
                casted_val = _bytes_feature(val)
                
            key = feature_name[i]
            feature[key] = casted_val
    
        # Create a Features message using tf.train.Example
        example_proto = tf.train.Example(
            features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()


    def tf_serialize_example(*args):
        tf_string = tf.py_function(
          serialize_example,
          args,  # pass these args to the above function.
          tf.string)      # the return type is `tf.string`.
        return tf.reshape(tf_string, ()) # The result is a scalar
    
    
    feature_dataset = tf.data.Dataset.from_tensor_slices(features)
    serialized_features_dataset = feature_dataset.map(tf_serialize_example)
    
    # write to tf record
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(serialized_features_dataset)
    
    
def tfrecord2dataset(file_pattern, feature_spec, label_key, batch_size=5, 
                       num_epochs=2):
    """Returns:
        A dataset that contains (features, indices) tuple where features is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    dataset = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=batch_size,
      num_epochs=num_epochs,
      features=feature_spec,
      label_key=label_key)
    #dataset = tf.data.TFRecord()
    return dataset


def test_tensors2tfrecord():
    S = tf.constant([[1,2, 3], [4, 5, 6], [1, 2, 3], [6, 7, 8], [2, 3, 5], [3, 5, 7]])
    tensors2tfrecord("temp.tfrecord", name=S[:, 0], 
                       content=S[:, 1], 
                       weight=tf.cast(S[:, 2], "float32"))


def test_tfrecord2dataset():
    # Read
    feature_spec = {
        "name":    tf.io.FixedLenFeature([], dtype=tf.int64),
        "content": tf.io.FixedLenFeature([], dtype=tf.int64),
        "weight":  tf.io.FixedLenFeature([], dtype=tf.float32)}    
    def map_fn(x, y):
        return (x["content"], x["name"]), y
    loaded_dataset = tfrecord2dataset(["temp.tfrecord"], 
                                        feature_spec, 
                                        label_key="weight",
                                        batch_size=5
                                        ).map(map_fn)

    for i, d in enumerate(loaded_dataset):
        print(i)
        print(d)